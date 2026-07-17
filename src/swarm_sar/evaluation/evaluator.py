"""High-level evaluation: multi-seed metrics, robustness & generalization scores."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from swarm_sar.config import EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.evaluation.metrics import aggregate, episode_metrics
from swarm_sar.training.rollout import run_episode


class Evaluator:
    def __init__(self, cfg: EnvConfig, actor_factory: Callable):
        self.cfg = cfg
        self.actor_factory = actor_factory

    def _run(self, cfg: EnvConfig, seed: int):
        env = SARSwarmEnv(cfg, seed=seed)
        return run_episode(env, self.actor_factory(env), seed=seed)

    def evaluate(self, seeds: list[int]) -> dict:
        logs = [self._run(self.cfg, s) for s in seeds]
        mets = [episode_metrics(log, grid_size=self.cfg.world.size) for log in logs]
        return {"per_seed": mets, "aggregate": aggregate(mets), "logs": logs}

    def robustness(self, seeds: list[int]) -> dict[str, float]:
        """Ratio of SIS with faults enabled vs disabled (1.0 = fully robust)."""
        # Since config is a pydantic model now:
        base = self.cfg.model_copy(deep=True)
        base.faults.enable = False
        no_fault = [episode_metrics(self._run(base, s), grid_size=self.cfg.world.size) for s in seeds]
        with_fault = [episode_metrics(self._run(self.cfg, s), grid_size=self.cfg.world.size) for s in seeds]
        a = aggregate(no_fault)["swarm_intelligence_score"]["mean"]
        b = aggregate(with_fault)["swarm_intelligence_score"]["mean"]
        return {"sis_nominal": a, "sis_faulted": b,
                "robustness_score": float(np.clip(b / (a + 1e-6), 0, 1))}

    def generalization(self, train_seeds: list[int], test_seeds: list[int]) -> dict[str, float]:
        """Performance drop on unseen maps (1.0 = perfect generalization)."""
        # evaluate on train distribution (seen during MAPPO)
        seen = [episode_metrics(self._run(self.cfg, s), grid_size=self.cfg.world.size) for s in train_seeds]
        unseen = [episode_metrics(self._run(self.cfg, s), grid_size=self.cfg.world.size) for s in test_seeds]
        a = aggregate(seen)["swarm_intelligence_score"]["mean"]
        b = aggregate(unseen)["swarm_intelligence_score"]["mean"]
        return {"sis_seen": a, "sis_unseen": b,
                "generalization_score": float(np.clip(b / (a + 1e-6), 0, 1))}

    def inference_time(self, seed: int = 42, steps: int = 100) -> dict[str, float]:
        env = SARSwarmEnv(self.cfg, seed=seed)
        actor = self.actor_factory(env)
        obs, _ = env.reset(seed=seed)
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, _, term, trunc, _ = env.step(actor(obs))
            if any(term.values()) or any(trunc.values()):
                obs, _ = env.reset()
        dt = (time.perf_counter() - t0) / steps
        return {"ms_per_step": dt * 1000.0, "hz": 1.0 / dt}
