"""Multiprocessing Vector Environment wrapper."""
import multiprocessing as mp
import traceback

from swarm_sar.config import EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.mission.planner import HeuristicSwarmController


def worker(remote, parent_remote, cfg: EnvConfig, seed: int):
    parent_remote.close()
    import os
    os.environ["IS_WORKER"] = "1"
    try:
        env = SARSwarmEnv(cfg, seed=seed)
        # Keep an actor handy for behavior cloning / expert moves
        actor = None
        actor_strategy = None
        pending_cfg = None                     # applied at the next episode boundary

        def apply_cfg(new_cfg):
            # Swapping configs mid-episode would leave the running episode with
            # a mismatched world/reward state, so this only runs right before a
            # reset (see 'step' auto-reset and 'reset' below). reset() rebuilds
            # comm/faults/world/allocator from env.cfg, so only the config and
            # the objects created in __init__ need touching here.
            env.cfg = new_cfg
            env.reward_engine = env.reward_engine.__class__(new_cfg.reward)
            env.obs_engine.cfg = new_cfg
            env._built = False                 # force world regeneration on reset

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, term, trunc, info = env.step(data)
                # auto-reset on episode completion
                if not env.agents:
                    if pending_cfg is not None:
                        apply_cfg(pending_cfg)
                        pending_cfg = None
                    # Keep the TERMINAL global state (for truncation value
                    # bootstrapping) and replace the outgoing global state with
                    # the fresh episode's, matching the reset observation.
                    info = dict(info)
                    info["terminal_global_state"] = info.get("global_state")
                    obs, reset_info = env.reset()
                    info["global_state"] = reset_info["global_state"]
                    if actor is not None:
                        actor = HeuristicSwarmController(env, strategy=actor_strategy, seed=seed)
                remote.send((obs, reward, term, trunc, info))
            elif cmd == 'reset':
                if pending_cfg is not None:
                    apply_cfg(pending_cfg)
                    pending_cfg = None
                obs, info = env.reset(seed=data)
                if actor is not None:
                    actor = HeuristicSwarmController(env, strategy=actor_strategy, seed=data or seed)
                remote.send((obs, info))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'action_masks':
                remote.send(env.action_masks())
            elif cmd == 'init_expert':
                actor_strategy = data['strategy']
                actor = HeuristicSwarmController(env, strategy=actor_strategy, seed=data['seed'])
                remote.send(True)
            elif cmd == 'expert_act':
                remote.send(actor.act())
            elif cmd == 'set_cfg':
                pending_cfg = data
                remote.send(True)
            else:
                raise NotImplementedError(f"Got unrecognized cmd: {cmd}")
    except EOFError:
        pass
    except Exception as e:
        remote.send(("ERROR", str(e), traceback.format_exc()))

class SubprocVecEnv:
    def __init__(self, cfgs: list[EnvConfig], seeds: list[int]):
        self.waiting = False
        self.closed = False
        self.num_envs = len(cfgs)

        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        import os
        os.environ["IS_WORKER"] = "1"
        for wrk, rem, cfg, seed in zip(self.work_remotes, self.remotes, cfgs, seeds):
            p = ctx.Process(target=worker, args=(wrk, rem, cfg, seed))
            p.daemon = True
            p.start()
            self.ps.append(p)
            wrk.close()
        del os.environ["IS_WORKER"]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [self._receive(remote) for remote in self.remotes]
        obs, rews, terms, truncs, infos = zip(*results)
        return list(obs), list(rews), list(terms), list(truncs), list(infos)

    def reset(self, seeds=None):
        if seeds is None:
            seeds = [None] * self.num_envs
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
        results = [self._receive(remote) for remote in self.remotes]
        obs, infos = zip(*results)
        return list(obs), list(infos)

    def action_masks(self):
        for remote in self.remotes:
            remote.send(('action_masks', None))
        return [self._receive(remote) for remote in self.remotes]

    def init_expert(self, strategies: list[str], seeds: list[int]):
        for remote, strat, seed in zip(self.remotes, strategies, seeds):
            remote.send(('init_expert', {'strategy': strat, 'seed': seed}))
        return [self._receive(remote) for remote in self.remotes]

    def expert_act(self):
        for remote in self.remotes:
            remote.send(('expert_act', None))
        return [self._receive(remote) for remote in self.remotes]

    def set_cfg(self, cfgs: list[EnvConfig]):
        for remote, cfg in zip(self.remotes, cfgs):
            remote.send(('set_cfg', cfg))
        return [self._receive(remote) for remote in self.remotes]

    def _receive(self, remote):
        res = remote.recv()
        if isinstance(res, tuple) and len(res) == 3 and res[0] == "ERROR":
            raise RuntimeError(f"Worker Exception: {res[1]}\n{res[2]}")
        return res

    def close(self):
        if self.closed: return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
