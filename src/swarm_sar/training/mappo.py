"""Multi-Agent PPO with Centralized Training, Decentralized Execution (CTDE).

A compact, readable from-scratch MAPPO implementation:
 * per-agent actor + encoder (shared or independent parameters),
 * a *centralized* critic that consumes the global state (concatenated agent
   observations) during training only,
 * Generalized Advantage Estimation, clipped surrogate objective, entropy bonus.
 * FAST Vectorized environment execution using multiprocessing SubprocVecEnv.

For large-scale distributed training we also ship an RLlib configuration
(:mod:`swarm_sar.training.rllib_trainer`); this module is the self-contained,
dependency-light reference used for unit tests and small experiments.
"""
from __future__ import annotations
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np

from swarm_sar.config import Config
from swarm_sar.drone.drone import ACTIONS
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.environment.vec_env import SubprocVecEnv
from swarm_sar.training.graph import (
    block_diag_adjacency,
    comm_adjacency,
    policy_uses_pyg_graph,
    torch_graph_from_adjacency,
)
from swarm_sar.mission.planner import HeuristicSwarmController
from swarm_sar.policies.base import HAS_TORCH, build_policy, PolicySpec

if HAS_TORCH:
    import torch
    import torch.nn as nn


@dataclass
class RolloutBuffer:
    obs: list; actions: list; logps: list; rewards: list
    values: list; dones: list; global_state: list
    graphs: list = None; action_masks: list = None; expert_actions: list = None
    bootstrap_value: float = 0.0

    @classmethod
    def empty(cls):
        return cls([], [], [], [], [], [], [], [], [], [], 0.0)


def merge_rollout_buffers(bufs: List[RolloutBuffer], group_size: int = 1) -> RolloutBuffer:
    """Merge per-agent buffers into one stream, TIME-MAJOR within each env group.

    Buffers arrive ordered [env0_agent0, env0_agent1, ..., env1_agent0, ...].
    The merged layout is [env0: (t0, agents 0..n-1), (t1, agents 0..n-1), ...]
    [env1: ...] so transition ``t * group_size + i`` is (env, step t, agent i).
    ``merged_advantages_returns`` emits advantages in this exact order — the
    two MUST stay in lockstep or PPO trains on misaligned advantages.
    """
    merged = RolloutBuffer.empty()
    if not bufs:
        return merged

    fields = ("obs", "actions", "logps", "rewards", "values",
              "dones", "global_state", "graphs", "action_masks", "expert_actions")

    num_envs = max(1, len(bufs) // group_size)
    for env_idx in range(num_envs):
        env_bufs = bufs[env_idx * group_size: (env_idx + 1) * group_size]
        max_len = max((len(b.obs) for b in env_bufs), default=0)
        for t in range(max_len):
            for b in env_bufs:
                if t < len(b.obs):
                    for field in fields:
                        vals = getattr(b, field) or []
                        if t < len(vals):
                            getattr(merged, field).append(vals[t])
    return merged


def _gae(rewards, values, dones, gamma, lam, bootstrap=0.0):
    adv = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    values = list(values) + [bootstrap]
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + np.asarray(values[:-1], dtype=np.float32)
    return adv, ret


def merged_advantages_returns(bufs: List[RolloutBuffer], gamma: float, lam: float,
                              group_size: int = 1):
    """Per-buffer GAE, interleaved in the same time-major order as
    :func:`merge_rollout_buffers` (advantage k belongs to merged transition k).

    Returns (normalized advantages, raw returns) as float32 arrays.
    """
    advs, rets = [], []
    for b in bufs:
        adv, ret = _gae(b.rewards, b.values, b.dones, gamma, lam, bootstrap=b.bootstrap_value)
        advs.append(adv); rets.append(ret)

    merged_adv, merged_ret = [], []
    num_envs = max(1, len(bufs) // group_size)
    for env_idx in range(num_envs):
        env_advs = advs[env_idx * group_size: (env_idx + 1) * group_size]
        env_rets = rets[env_idx * group_size: (env_idx + 1) * group_size]
        max_len = max((len(a) for a in env_advs), default=0)
        for t in range(max_len):
            for i in range(len(env_advs)):
                if t < len(env_advs[i]):
                    merged_adv.append(env_advs[i][t])
                    merged_ret.append(env_rets[i][t])

    adv = np.asarray(merged_adv, dtype=np.float32)
    ret = np.asarray(merged_ret, dtype=np.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


class MAPPOTrainer:
    def __init__(self, cfg: Config, resume: str = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MAPPO. `pip install -e '.[rl]'`")
        self.cfg = cfg
        self.device = ("cuda" if (cfg.train.device in ("auto", "cuda")
                                  and torch.cuda.is_available()) else "cpu")
        self._base_env_cfg = copy.deepcopy(cfg.env)
        self.num_envs = max(1, int(getattr(cfg.train, "num_envs", 1)))
        
        # Local dummy env for metadata
        self.env = SARSwarmEnv(copy.deepcopy(cfg.env), seed=cfg.train.seed)
        self.env.reset(seed=cfg.train.seed)
        self.n = self.env.n
        
        if self.num_envs > 0:
            self.vec_env = SubprocVecEnv(
                [copy.deepcopy(cfg.env) for _ in range(self.num_envs)],
                [cfg.train.seed + i * 1009 for i in range(self.num_envs)]
            )
        else:
            self.vec_env = None

        spec = PolicySpec(obs_dim=self.env.obs_dim, n_actions=self.env.n_actions,
                          global_dim=self.env.obs_dim * self.n, n_agents=self.n)
        self.spec = spec
        # shared vs independent parameters
        centralized = bool(cfg.train.centralized_critic)
        if cfg.train.policy_sharing == "independent":
            self.policies = [build_policy(cfg.model, spec, centralized=centralized).to(self.device)
                             for _ in range(self.n)]
        else:
            shared = build_policy(cfg.model, spec, centralized=centralized).to(self.device)
            self.policies = [shared] * self.n
        params = {id(p): p for p in self.policies}.values()
        self.opt = torch.optim.Adam([pp for p in params for pp in p.parameters()],
                                    lr=cfg.train.lr, foreach=False)
        self.last_collect_stats: dict = {}
        if resume:
            import os
            if os.path.exists(resume):
                ckpt = torch.load(resume, map_location=self.device, weights_only=False)
                if cfg.train.policy_sharing == "independent":
                    for i, policy in enumerate(self.policies):
                        key = f"state_dict_{i}" if f"state_dict_{i}" in ckpt else "state_dict"
                        policy.load_state_dict(ckpt[key], strict=False)
                else:
                    self.policies[0].load_state_dict(ckpt["state_dict"], strict=False)
                print(f"resumed from {resume}")

    def _global_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs[a].flatten() for a in self.env.possible_agents], dtype=np.float32)

    def _adjacency(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        # Observations contain a flattened temporal stack.  Use the newest GPS
        # frame and convert its [0, 1] coordinates back to world units before
        # applying the radio-range threshold.  Comparing normalized positions
        # directly with a metre-valued range made every drone appear connected.
        frame_dim = self.env._compute_obs_dim()
        pos = np.array([
            (obs[a][-frame_dim:-frame_dim + 2] if obs[a].ndim == 1 else obs[a][-1, :2])
            * self.cfg.env.world.size
            for a in self.env.possible_agents
        ], dtype=np.float32)
        return comm_adjacency(pos, self.cfg.env.comm.range_m)

    def _graph_tensor(self, adj: np.ndarray, policy):
        return torch_graph_from_adjacency(adj, self.device, policy_uses_pyg_graph(policy))

    def _mask_tensor(self, masks: np.ndarray):
        return torch.tensor(masks, dtype=torch.float32, device=self.device)

    def _expert_moves(self, expert_actions: Dict[str, object], masks_np: np.ndarray) -> List[int]:
        moves = []
        for i, a in enumerate(self.env.possible_agents):
            move = self.env._split_action(expert_actions[a])[0]
            if masks_np[i, move] <= 0:
                move = int(np.argmax(masks_np[i]))
            moves.append(move)
        return moves

    def collect(self, steps: int):
        bufs = [RolloutBuffer.empty() for _ in range(self.num_envs * self.n)]
        ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        completed = []
        action_counts = np.zeros(self.env.n_actions, dtype=np.int64)
        started = time.perf_counter()
        env_steps = 0
        
        self.vec_env.init_expert(
            [self.cfg.task_alloc.strategy] * self.num_envs,
            [self.cfg.train.seed + env_idx * 1009 for env_idx in range(self.num_envs)]
        )
        observations, _ = self.vec_env.reset()

        for _ in range(steps):
            acts_by_env: List[Dict[str, int]] = [{} for _ in range(self.num_envs)]
            step_cache_by_env: List[list] = [[] for _ in range(self.num_envs)]
            
            masks_list = self.vec_env.action_masks()
            expert_acts_list = self.vec_env.expert_act()
            
            if self.cfg.train.policy_sharing == "independent":
                for env_idx in range(self.num_envs):
                    obs = observations[env_idx]
                    gs_np = self._global_state(obs)
                    gs = torch.tensor(gs_np, device=self.device).float()
                    adj = self._adjacency(obs)
                    masks_np = masks_list[env_idx]
                    expert_moves = self._expert_moves(expert_acts_list[env_idx], masks_np)
                    for i, a in enumerate(self.env.possible_agents):
                        o = torch.tensor(obs[a], device=self.device).float().unsqueeze(0)
                        mask = self._mask_tensor(masks_np[i:i + 1])
                        action, logp, value = self.policies[i].act(
                            o, global_state=gs.unsqueeze(0), action_mask=mask
                        )
                        ac = int(action.item())
                        acts_by_env[env_idx][a] = ac
                        action_counts[ac] += 1
                        step_cache_by_env[env_idx].append(
                            (i, obs[a], ac, float(logp.item()), float(value.item()),
                             gs_np, adj, masks_np[i], expert_moves[i])
                        )
            else:
                policy = self.policies[0]
                rows, states, masks, graphs, meta = [], [], [], [], []
                for env_idx in range(self.num_envs):
                    obs = observations[env_idx]
                    gs_np = self._global_state(obs)
                    adj = self._adjacency(obs)
                    masks_np = masks_list[env_idx]
                    expert_moves = self._expert_moves(expert_acts_list[env_idx], masks_np)
                    for i, a in enumerate(self.env.possible_agents):
                        rows.append(obs[a])
                        states.append(gs_np)
                        masks.append(masks_np[i])
                        graphs.append(adj)
                        meta.append((env_idx, i, a, obs[a], gs_np, adj, masks_np[i], expert_moves[i]))
                obs_batch = torch.tensor(np.array(rows), device=self.device).float()
                gs_batch = torch.tensor(np.array(states), device=self.device).float()
                adj_batch = block_diag_adjacency(graphs, self.n)
                graph = self._graph_tensor(adj_batch, policy) if adj_batch is not None else None
                mask = self._mask_tensor(np.array(masks))
                action, logp, value = policy.act(
                    obs_batch, graph=graph, global_state=gs_batch,
                    action_mask=mask
                )
                for row_idx, item in enumerate(meta):
                    env_idx, i, a, o_np, gs_np, adj, mask_np, expert_ac = item
                    ac = int(action[row_idx].item())
                    acts_by_env[env_idx][a] = ac
                    action_counts[ac] += 1
                    step_cache_by_env[env_idx].append(
                        (i, o_np, ac, float(logp[row_idx].item()),
                         float(value[row_idx].item()), gs_np, adj, mask_np, expert_ac)
                    )
            
            # SubprocVecEnv batches the execution
            nobs_list, rews_list, terms_list, truncs_list, infos_list = self.vec_env.step(acts_by_env)
            
            for env_idx in range(self.num_envs):
                nobs = nobs_list[env_idx]
                rew = rews_list[env_idx]
                term = terms_list[env_idx]
                trunc = truncs_list[env_idx]
                env_steps += 1
                ep_returns[env_idx] += float(sum(rew.values()))
                done_flags = {a: float(term.get(a, False) or trunc.get(a, False))
                              for a in self.env.possible_agents}

                # On episode end the worker auto-resets and `nobs` is already the
                # fresh episode's first observation; the terminal observation is
                # not needed here (GAE zeroes the bootstrap via the done flag).
                observations[env_idx] = nobs
                for (i, o, ac, lp, v, gs_np, graph_np, mask_np, expert_ac) in step_cache_by_env[env_idx]:
                    b = bufs[env_idx * self.n + i]
                    agent_name = self.env.possible_agents[i]
                    b.obs.append(o); b.actions.append(ac); b.logps.append(lp)
                    b.values.append(v)
                    b.rewards.append(rew.get(agent_name, 0.0) * self.cfg.train.reward_scale)
                    b.dones.append(done_flags.get(agent_name, 0.0)); b.global_state.append(gs_np)
                    b.graphs.append(graph_np); b.action_masks.append(mask_np)
                    b.expert_actions.append(expert_ac)
                # Check if episode ended based on `term` and `trunc`
                # If so, SubprocVecEnv already auto-reset the env behind the scenes.
                is_done = all(term.values()) or all(trunc.values())
                if is_done:
                    completed.append(float(ep_returns[env_idx]))
                    ep_returns[env_idx] = 0.0
                    
        # Compute bootstrap values for GAE
        bootstraps = [[0.0] * self.n for _ in range(self.num_envs)]
        masks_list = self.vec_env.action_masks()
        
        if self.cfg.train.policy_sharing == "independent":
            for env_idx in range(self.num_envs):
                obs = observations[env_idx]
                gs_np = self._global_state(obs)
                gs = torch.tensor(gs_np, device=self.device).float().unsqueeze(0)
                masks_np = masks_list[env_idx]
                for i, a in enumerate(self.env.possible_agents):
                    o = torch.tensor(obs[a], device=self.device).float().unsqueeze(0)
                    mask = self._mask_tensor(masks_np[i:i + 1])
                    with torch.no_grad():
                        _, _, value = self.policies[i].act(o, global_state=gs, action_mask=mask)
                    bootstraps[env_idx][i] = float(value.item())
        else:
            policy = self.policies[0]
            rows, states, masks, graphs = [], [], [], []
            for env_idx in range(self.num_envs):
                obs = observations[env_idx]
                gs_np = self._global_state(obs)
                adj = self._adjacency(obs)
                masks_np = masks_list[env_idx]
                for i, a in enumerate(self.env.possible_agents):
                    rows.append(obs[a])
                    states.append(gs_np)
                    masks.append(masks_np[i])
                    graphs.append(adj)
            obs_batch = torch.tensor(np.array(rows), device=self.device).float()
            gs_batch = torch.tensor(np.array(states), device=self.device).float()
            adj_batch = block_diag_adjacency(graphs, self.n)
            graph = self._graph_tensor(adj_batch, policy) if adj_batch is not None else None
            mask = self._mask_tensor(np.array(masks))
            with torch.no_grad():
                _, _, value = policy.act(obs_batch, graph=graph, global_state=gs_batch, action_mask=mask)
            
            for env_idx in range(self.num_envs):
                for i in range(self.n):
                    bootstraps[env_idx][i] = float(value[env_idx * self.n + i].item())
                    
        # Apply bootstrap values to buffers
        for env_idx in range(self.num_envs):
            for i in range(self.n):
                bufs[env_idx * self.n + i].bootstrap_value = bootstraps[env_idx][i]
                    
        elapsed = max(1e-9, time.perf_counter() - started)
        total_actions = max(1, int(action_counts.sum()))
        hover_frac = float(action_counts[ACTIONS.index("hover")] / total_actions)
        rtb_frac = float(action_counts[ACTIONS.index("return_to_base")] / total_actions)
        self.last_collect_stats = {
            "num_envs": float(self.num_envs),
            "env_steps_per_s": float(env_steps / elapsed),
            "agent_transitions_per_s": float((env_steps * self.n) / elapsed),
            "hover_pct": hover_frac,
            "return_to_base_pct": rtb_frac,
            "policy_collapse_warn": float(
                hover_frac + rtb_frac >= self.cfg.train.collapse_action_threshold
            ),
        }
        for idx, name in enumerate(ACTIONS):
            self.last_collect_stats[f"action_{name}_pct"] = float(action_counts[idx] / total_actions)
        if self.device == "cuda":
            self.last_collect_stats["cuda_memory_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
        mean_ret = float(np.mean(completed)) if completed else float(np.mean(ep_returns))
        return bufs, mean_ret

    def _update_policy_on_buffer(self, policy, b: RolloutBuffer, adv, ret, epochs: int,
                                 group_size: int | None = None, entropy_coef: float = None):
        if entropy_coef is None:
            entropy_coef = self.cfg.train.entropy_coef
        c = self.cfg.train
        obs = torch.tensor(np.array(b.obs), device=self.device).float()
        gs = torch.tensor(np.array(b.global_state), device=self.device).float()
        act = torch.tensor(b.actions, device=self.device).long()
        old_lp = torch.tensor(b.logps, device=self.device).float()
        adv_t = torch.tensor(adv, device=self.device).float()
        ret_t = torch.tensor(ret, device=self.device).float()
        mask = (torch.tensor(np.array(b.action_masks), device=self.device).float()
                if b.action_masks else None)

        pi_loss = v_loss = ent = imitation_loss = torch.tensor(0.0, device=self.device)
        expert_t = (torch.tensor(b.expert_actions, device=self.device).long()
                    if b.expert_actions else None)
        n_total = obs.shape[0]
        group_size = group_size or self.n
        T = max(1, n_total // group_size)
        
        n_minibatches = max(1, getattr(c, 'minibatches', 1))
        mb_size_t = max(1, T // n_minibatches)
        
        for _ in range(epochs):
            perm_t = torch.randperm(T)
            for mb_start in range(0, T, mb_size_t):
                idx_t = perm_t[mb_start:mb_start + mb_size_t]
                
                # Expand rollout-block indices to transition indices
                idx = []
                for t in idx_t.tolist():
                    start = t * group_size
                    idx.extend(range(start, min(start + group_size, n_total)))
                idx = torch.tensor(idx, device=self.device).long()
                
                mb_obs = obs[idx]; mb_gs = gs[idx]; mb_act = act[idx]
                mb_old_lp = old_lp[idx]; mb_adv = adv_t[idx]; mb_ret = ret_t[idx]
                mb_mask = mask[idx] if mask is not None else None
                mb_expert = expert_t[idx] if expert_t is not None else None
                
                graph = None
                if b.graphs and self.cfg.train.policy_sharing != "independent":
                    # Reconstruct block diagonal graph for this minibatch
                    mb_graphs = []
                    for t in idx_t.tolist():
                        start = t * group_size
                        if start < len(b.graphs):
                            mb_graphs.extend([b.graphs[start]] * group_size)
                    adj = block_diag_adjacency(mb_graphs, group_size)
                    graph = self._graph_tensor(adj, policy) if adj is not None else None

                logits, value = policy(
                    mb_obs, graph=graph, global_state=mb_gs, action_mask=mb_mask,
                )
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(mb_act)
                ratio = torch.exp(lp - mb_old_lp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * mb_adv
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = ((value.squeeze(-1) - mb_ret) ** 2).mean()
                ent = dist.entropy().mean()
                imitation_loss = (
                    nn.functional.cross_entropy(logits, mb_expert)
                    if mb_expert is not None and c.imitation_coef > 0 else
                    torch.tensor(0.0, device=self.device)
                )
                loss = (
                    pi_loss +
                    c.value_coef * v_loss -
                    entropy_coef * ent +
                    c.imitation_coef * imitation_loss
                )
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_([p for pol in self.policies for p in pol.parameters()],
                                         c.grad_clip)
                self.opt.step()
        return (float(pi_loss.item()), float(v_loss.item()), float(ent.item()),
                float(imitation_loss.item()))

    def update(self, buffers: List[RolloutBuffer], entropy_coef: float = None) -> dict:
        if entropy_coef is None:
            entropy_coef = self.cfg.train.entropy_coef
        c = self.cfg.train
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "imitation_loss": 0.0}
        if self.cfg.train.policy_sharing != "independent":
            merged = merge_rollout_buffers(buffers, group_size=self.n)
            if not merged.rewards:
                return stats
            adv, ret = merged_advantages_returns(buffers, c.gamma, c.gae_lambda, group_size=self.n)
            pi, v, ent, im = self._update_policy_on_buffer(
                self.policies[0], merged, adv, ret, c.epochs, group_size=self.n, entropy_coef=entropy_coef
            )
            stats["pi_loss"] += pi; stats["v_loss"] += v; stats["entropy"] += ent
            stats["imitation_loss"] = im
            return stats

        for i, b in enumerate(buffers):
            if not b.rewards:
                continue
            adv, ret = _gae(b.rewards, b.values, b.dones, c.gamma, c.gae_lambda, bootstrap=b.bootstrap_value)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pi, v, ent, im = self._update_policy_on_buffer(
                self.policies[i % self.n], b, adv, ret, c.epochs, group_size=1, entropy_coef=entropy_coef
            )
            stats["pi_loss"] += pi
            stats["v_loss"] += v
            stats["entropy"] += ent
            stats["imitation_loss"] += im
        return stats

    def behavior_clone_from_heuristic(self, episodes: int = 8, epochs: int = 2) -> dict:
        """Supervised warm-start from the strong frontier/auction heuristic."""
        if episodes <= 0:
            return {"bc_loss": 0.0}
        # Warm-start every distinct policy (one for shared parameters, one per
        # agent under independent sharing) rather than only policies[0].
        unique_policies = list({id(p): p for p in self.policies}.values())

        # Parallelize behavior cloning rollout too
        self.vec_env.init_expert(
            [self.cfg.task_alloc.strategy] * self.num_envs,
            [self.cfg.train.seed + ep * 1009 for ep in range(self.num_envs)]
        )
        observations, _ = self.vec_env.reset()
        
        rows, actions, masks, graphs, states = [], [], [], [], []
        # We need to collect `episodes` total full episodes
        # Since we have self.num_envs running in parallel, we run until we finish enough episodes.
        episodes_completed = 0
        
        while episodes_completed < episodes:
            masks_list = self.vec_env.action_masks()
            expert_acts_list = self.vec_env.expert_act()
            
            for env_idx in range(self.num_envs):
                obs = observations[env_idx]
                adj = self._adjacency(obs)
                mask_np = masks_list[env_idx]
                expert = expert_acts_list[env_idx]
                # _expert_moves already projects illegal expert moves onto the
                # nearest legal action, so the mask needs no mutation here.
                parsed = self._expert_moves(expert, mask_np)
                rows.extend([obs[a] for a in self.env.possible_agents])
                actions.extend(parsed)
                masks.extend(mask_np)
                graphs.extend([adj for _ in range(self.n)])
                gs = self._global_state(obs)
                states.extend([gs for _ in range(self.n)])
                
            nobs_list, rews_list, terms_list, truncs_list, infos_list = self.vec_env.step(expert_acts_list)
            for env_idx in range(self.num_envs):
                observations[env_idx] = nobs_list[env_idx]
                term = terms_list[env_idx]
                trunc = truncs_list[env_idx]
                if all(term.values()) or all(trunc.values()):
                    episodes_completed += 1

        obs_np = np.array(rows)
        act_np = np.array(actions)
        mask_np_arr = np.array(masks)
        gs_np = np.array(states)
        
        loss_val = 0.0
        batch_blocks = 256
        batch_size = batch_blocks * self.n

        for _ in range(max(1, epochs)):
            epoch_loss = 0.0
            n_batches = 0
            for start_idx in range(0, len(rows), batch_size):
                end_idx = min(start_idx + batch_size, len(rows))

                obs_t = torch.tensor(obs_np[start_idx:end_idx], device=self.device).float()
                act_t = torch.tensor(act_np[start_idx:end_idx], device=self.device).long()
                mask_t = torch.tensor(mask_np_arr[start_idx:end_idx], device=self.device).float()
                gs_t = torch.tensor(gs_np[start_idx:end_idx], device=self.device).float()

                batch_graphs = graphs[start_idx:end_idx]
                adj = block_diag_adjacency(batch_graphs, self.n)

                loss_t = torch.tensor(0.0, device=self.device)
                for policy in unique_policies:
                    graph = self._graph_tensor(adj, policy) if adj is not None else None
                    logits, *_ = policy(obs_t, graph=graph, global_state=gs_t, action_mask=mask_t)
                    loss_t = loss_t + nn.functional.cross_entropy(logits, act_t)
                loss_t = loss_t / len(unique_policies)

                if not torch.isfinite(loss_t):
                    raise RuntimeError("behavior cloning produced a non-finite loss")

                self.opt.zero_grad()
                loss_t.backward()
                nn.utils.clip_grad_norm_([p for pol in self.policies for p in pol.parameters()],
                                         self.cfg.train.grad_clip)
                self.opt.step()
                epoch_loss += loss_t.item()
                n_batches += 1
            loss_val = epoch_loss / max(1, n_batches)

        return {"bc_loss": loss_val}

    def _apply_curriculum(self, iteration: int, total_iters: int) -> None:
        if not self.cfg.train.curriculum:
            return
        progress = iteration / max(1, total_iters - 1)
        base = self._base_env_cfg
        cfg = copy.deepcopy(base)
        
        # Stage 1: Easy (0.0 to 0.2)
        if progress < 0.2:
            cfg.world.size = 32
            cfg.world.n_victims = 1
            cfg.world.n_buildings = 4
            cfg.world.n_trees = 10
            cfg.world.n_dynamic_obstacles = 0
            cfg.world.n_no_fly_zones = 0
            cfg.faults.enable = False
            cfg.comm.packet_loss = 0.0
            cfg.dwell_steps_to_rescue = 1
            cfg.world.weather_enabled = False
        # Stage 2: Medium-Easy (0.2 to 0.4)
        elif progress < 0.4:
            cfg.world.size = 48
            cfg.world.n_victims = 2
            cfg.world.n_buildings = 8
            cfg.world.n_trees = 20
            cfg.world.n_dynamic_obstacles = 0
            cfg.world.n_no_fly_zones = 0
            cfg.faults.enable = False
            cfg.comm.packet_loss = base.comm.packet_loss * 0.25
            cfg.dwell_steps_to_rescue = 1
            cfg.world.weather_enabled = False
        # Stage 3: Medium (0.4 to 0.6)
        elif progress < 0.6:
            cfg.world.size = 64
            cfg.world.n_victims = 4
            cfg.world.n_buildings = 14
            cfg.world.n_trees = 30
            cfg.world.n_dynamic_obstacles = 2
            cfg.world.n_no_fly_zones = 1
            cfg.faults.enable = False
            cfg.comm.packet_loss = base.comm.packet_loss * 0.5
            cfg.dwell_steps_to_rescue = max(1, min(base.dwell_steps_to_rescue, 2))
        # Stage 4: Medium-Hard (0.6 to 0.8)
        elif progress < 0.8:
            cfg.world.size = base.world.size
            cfg.world.n_victims = 6
            cfg.world.n_dynamic_obstacles = max(2, base.world.n_dynamic_obstacles // 2)
            cfg.world.n_no_fly_zones = max(1, base.world.n_no_fly_zones // 2)
            cfg.faults.enable = True
            cfg.comm.packet_loss = base.comm.packet_loss * 0.75
            cfg.dwell_steps_to_rescue = base.dwell_steps_to_rescue
        # Stage 5: Full Benchmark (0.8 to 1.0) -> Uses base settings completely

        self.cfg.env = cfg
        if self.vec_env:
            self.vec_env.set_cfg([copy.deepcopy(cfg) for _ in range(self.num_envs)])

    @torch.no_grad()
    def evaluate_policy(self, episodes: int = 3, seed: int = 10_000) -> dict:
        """Deterministic evaluation on the *base* (non-curriculum) environment."""
        env = SARSwarmEnv(copy.deepcopy(self._base_env_cfg), seed=seed)
        frame_dim = env._compute_obs_dim()
        returns, rescue_rates, coverages = [], [], []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            while env.agents:
                masks_np = env.action_masks()
                acts = {}
                if self.cfg.train.policy_sharing == "independent":
                    for i, a in enumerate(env.possible_agents):
                        o = torch.tensor(obs[a], device=self.device).float().unsqueeze(0)
                        mask = self._mask_tensor(masks_np[i:i + 1])
                        action, _, _ = self.policies[i].act(
                            o, action_mask=mask, deterministic=True)
                        acts[a] = int(action.item())
                else:
                    policy = self.policies[0]
                    pos = np.array([
                        obs[a][-frame_dim:-frame_dim + 2] * env.cfg.world.size
                        for a in env.possible_agents], dtype=np.float32)
                    adj = comm_adjacency(pos, env.cfg.comm.range_m)
                    graph = self._graph_tensor(adj, policy)
                    obs_batch = torch.tensor(
                        np.array([obs[a] for a in env.possible_agents]),
                        device=self.device).float()
                    mask = self._mask_tensor(masks_np)
                    action, _, _ = policy.act(
                        obs_batch, graph=graph, action_mask=mask, deterministic=True)
                    acts = {a: int(action[i].item())
                            for i, a in enumerate(env.possible_agents)}
                obs, rew, term, trunc, _ = env.step(acts)
                total += float(sum(rew.values()))
            summary = env.episode_summary()
            returns.append(total)
            rescue_rates.append(summary["rescue_rate"])
            coverages.append(summary["coverage"])
        return {
            "eval_return": float(np.mean(returns)),
            "eval_rescue_rate": float(np.mean(rescue_rates)),
            "eval_coverage": float(np.mean(coverages)),
        }

    def train(self, logger=None):
        c = self.cfg.train
        rollout_env_steps = c.rollout_len * self.num_envs
        iters = max(1, c.total_timesteps // rollout_env_steps)
        eval_interval = max(1, c.eval_every // rollout_env_steps)
        best_key = (-1.0, -float("inf"))          # (rescue_rate, return)
        pending_stats = {}
        if c.bc_warmstart_episodes > 0:
            pending_stats = self.behavior_clone_from_heuristic(
                c.bc_warmstart_episodes, c.bc_epochs
            )
        for it in range(iters):
            self._apply_curriculum(it, iters)
            
            # Linear entropy decay
            progress = it / max(1, iters - 1)
            current_entropy_coef = max(0.001, c.entropy_coef * (1.0 - progress))
            
            bufs, ep_ret = self.collect(c.rollout_len)
            stats = self.update(bufs, entropy_coef=current_entropy_coef)
            stats["ep_return"] = ep_ret
            stats.update(self.last_collect_stats)
            print(f"Iter {it}/{iters} | Ep Ret: {ep_ret:.2f} | V Loss: {stats.get('v_loss', 0.0):.4f} | Pi Loss: {stats.get('pi_loss', 0.0):.4f} | Ent: {stats.get('entropy', 0.0):.4f}")
            
            # Action distribution telemetry
            hover_pct = stats.get("action_hover_pct", 0.0) * 100
            n_pct = stats.get("action_north_pct", 0.0) * 100
            s_pct = stats.get("action_south_pct", 0.0) * 100
            e_pct = stats.get("action_east_pct", 0.0) * 100
            w_pct = stats.get("action_west_pct", 0.0) * 100
            broadcast_pct = stats.get("action_broadcast_pct", 0.0) * 100
            
            print(f"  Actions: Hover {hover_pct:.1f}% | N {n_pct:.1f}% | S {s_pct:.1f}% | E {e_pct:.1f}% | W {w_pct:.1f}% | Broadcast {broadcast_pct:.1f}%")
            if hover_pct > 60.0:
                print("  [WARNING] Policy collapse detected: Hover > 60%")
            if pending_stats:
                stats.update(pending_stats)
                pending_stats = {}

            checkpoint_dir = Path(logger.dir) / "checkpoints" if logger is not None else \
                Path(self.cfg.log.out_dir) / "checkpoints"
            # Periodic deterministic evaluation on the base env; keep the best.
            if it % eval_interval == 0 or it == iters - 1:
                eval_stats = self.evaluate_policy()
                stats.update(eval_stats)
                key = (eval_stats["eval_rescue_rate"], eval_stats["eval_return"])
                print(f"  Eval: rescue {key[0]:.2f} | coverage "
                      f"{eval_stats['eval_coverage']:.2f} | return {key[1]:.2f}")
                if key > best_key:
                    best_key = key
                    self.save(checkpoint_dir / "best.pt")

            if logger is not None:
                logger.log_scalars(stats, step=it * rollout_env_steps)
            if it % max(1, (c.checkpoint_every // rollout_env_steps)) == 0:
                self.save(checkpoint_dir / f"iter_{it}.pt")
                
        # Close multiprocessing workers after training finishes
        if self.vec_env:
            self.vec_env.close()
            
        return self

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        data = {"model_cfg": self.cfg.model, "spec": self.spec.__dict__}
        if self.cfg.train.policy_sharing == "independent":
            for i, p in enumerate(self.policies):
                data[f"state_dict_{i}"] = p.state_dict()
        else:
            data["state_dict"] = self.policies[0].state_dict()
        torch.save(data, path)
