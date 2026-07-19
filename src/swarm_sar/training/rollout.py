"""Rollout utilities used by evaluation, experiments and figure generation.

Runs full episodes with any *actor* (a callable ``obs_dict -> action_dict``) and
returns a structured trajectory log. The :class:`HeuristicActor` wraps the
non-learned controller so the whole pipeline runs without PyTorch; a neural actor
wrapping a trained :class:`ActorCritic` is used after training.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.mission.planner import HeuristicSwarmController
from swarm_sar.training.graph import (
    comm_adjacency,
    policy_uses_pyg_graph,
    torch_graph_from_adjacency,
)


@dataclass
class EpisodeLog:
    frames: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    total_reward: float = 0.0
    collisions: list[dict] = field(default_factory=list)
    world_grid: np.ndarray | None = None
    victims: list[dict] = field(default_factory=list)
    charging: list[list] = field(default_factory=list)


class HeuristicActor:
    """Adapts :class:`HeuristicSwarmController` to the actor interface."""
    def __init__(self, env: SARSwarmEnv, strategy: str = "auction", seed: int = 0):
        self.ctl = HeuristicSwarmController(env, strategy=strategy, seed=seed)

    def __call__(self, obs: dict[str, np.ndarray]) -> dict[str, int]:
        return self.ctl.act()


def run_episode(env: SARSwarmEnv, actor: Callable, seed: int | None = None,
                record: bool = True) -> EpisodeLog:
    obs, _ = env.reset(seed=seed)
    if hasattr(actor, "bind_env"):
        actor.bind_env(env)
    log = EpisodeLog(world_grid=env.world.grid.copy())
    total = 0.0
    while env.agents:
        actions = actor(obs)
        obs, rewards, term, trunc, _ = env.step(actions)
        total += float(sum(rewards.values()))
    log.frames = env.history
    log.collisions = env.collisions
    log.summary = env.episode_summary()
    log.total_reward = total
    log.victims = [{"idx": v.idx, "pos": v.pos.tolist(), "detected": v.detected,
                    "rescued": v.rescued, "detected_step": v.detected_step,
                    "rescued_step": v.rescued_step, "severity": v.severity}
                   for v in env.world.victims]
    log.charging = [c.pos.tolist() for c in env.world.charging]
    return log


class RolloutWorker:
    """Collects many episodes (optionally multi-seed) and aggregates metrics."""
    def __init__(self, env_fn: Callable[[int], SARSwarmEnv], actor_fn: Callable):
        self.env_fn = env_fn
        self.actor_fn = actor_fn

    def collect(self, seeds: list[int]) -> list[EpisodeLog]:
        logs = []
        for s in seeds:
            env = self.env_fn(s)
            actor = self.actor_fn(env)
            logs.append(run_episode(env, actor, seed=s))
        return logs


# --------------------------------------------------------------------------- #
# Neural actor (used after training). Torch-optional.                          #
# --------------------------------------------------------------------------- #
class NeuralActor:
    """Wraps a trained ActorCritic for decentralized execution (obs->action)."""
    def __init__(self, policy, deterministic: bool = True):
        import torch  # noqa
        self.policy = policy
        self.policy.eval()
        self.deterministic = deterministic
        self._torch = torch
        self.env = None

    def bind_env(self, env: SARSwarmEnv) -> None:
        self.env = env

    def __call__(self, obs):
        acts = {}
        if self.env is not None and len(obs) == self.env.n:
            agents = list(self.env.possible_agents)
            batch = self._torch.tensor(
                np.array([obs[a] for a in agents]), dtype=self._torch.float32
            )
            # Use GPS-believed positions, matching what the trainer derives
            # from observations — evaluation must not get a cleaner graph than
            # training did.
            pos = np.array([d.gps_position() for d in self.env.drones], dtype=np.float32)
            adj = comm_adjacency(pos, self.env.cfg.comm.range_m / self.env.cfg.world.cell_size_m,
                                 packet_loss=self.env.cfg.comm.packet_loss,
                                 rng=self.env.rng)
            graph = torch_graph_from_adjacency(
                adj, batch.device, policy_uses_pyg_graph(self.policy)
            )
            mask = self._torch.tensor(self.env.action_masks(), dtype=self._torch.float32)
            with self._torch.no_grad():
                action, _, _ = self.policy.act(
                    batch, graph=graph, global_state=None, action_mask=mask,
                    deterministic=self.deterministic
                )
            for i, a in enumerate(agents):
                acts[a] = int(action[i].item())
            return acts

        for a, o in obs.items():
            t = self._torch.tensor(o, dtype=self._torch.float32).unsqueeze(0)
            with self._torch.no_grad():
                action, _, _ = self.policy.act(t, global_state=None,
                                               deterministic=self.deterministic)
            acts[a] = int(action.item())
        return acts


def load_neural_actor(checkpoint: str, deterministic: bool = True) -> NeuralActor:
    from swarm_sar.policies import load_policy
    return NeuralActor(load_policy(checkpoint), deterministic=deterministic)
