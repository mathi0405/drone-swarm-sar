"""Frozen evaluation benchmark: a *fixed*, versioned set of seeded maps so every
method is compared on identical instances (in-distribution and OOD).

Randomizing maps is right for training but wrong for comparison. Here we pin the
eval seeds and configs; results are therefore directly comparable across methods
and reproducible across machines.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple
import numpy as np

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.training.rollout import run_episode
from swarm_sar.evaluation.metrics import episode_metrics
from swarm_sar.evaluation.stats import iqm, bootstrap_ci

BENCHMARK_VERSION = "v1"
IN_DIST_SEEDS = list(range(200, 220))          # 20 held-out maps
OOD_SEEDS = list(range(300, 315))              # 15 harder maps


def suite(regime: str = "in_dist") -> List[Tuple[int, EnvConfig]]:
    """Return the frozen (seed, EnvConfig) pairs for a benchmark regime."""
    if regime == "in_dist":
        base = EnvConfig(num_drones=3, max_steps=200, world=WorldConfig(size=64, n_victims=8))
        return [(s, base) for s in IN_DIST_SEEDS]
    if regime == "ood":                        # larger, denser, more victims
        base = EnvConfig(num_drones=3, max_steps=250,
                         world=WorldConfig(size=80, n_victims=14, n_buildings=22,
                                           n_dynamic_obstacles=10))
        return [(s, base) for s in OOD_SEEDS]
    if regime == "scale":                      # 10 drones, large map
        base = EnvConfig(num_drones=10, max_steps=300,
                         world=WorldConfig(size=96, n_victims=16, n_charging_stations=5))
        return [(s, base) for s in IN_DIST_SEEDS[:10]]
    raise ValueError(regime)


def run_method(actor_factory: Callable[[SARSwarmEnv], Callable],
               regime: str = "in_dist") -> Dict[str, np.ndarray]:
    """Run one method across the frozen suite; return arrays of per-map metrics."""
    rows = []
    for seed, cfg in suite(regime):
        env = SARSwarmEnv(cfg, seed=seed)
        log = run_episode(env, actor_factory(env), seed=seed)
        rows.append(episode_metrics(log, grid_size=cfg.world.size))
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], dtype=float) for k in keys}


def summarize_method(arrays: Dict[str, np.ndarray], key: str = "swarm_intelligence_score") -> Dict:
    x = arrays[key]
    lo, hi = bootstrap_ci(x)
    return {"iqm": iqm(x), "ci95_low": lo, "ci95_high": hi,
            "mean": float(x.mean()), "std": float(x.std()), "n": int(x.size)}


def compare(methods: Dict[str, Callable], regime: str = "in_dist",
            key: str = "swarm_intelligence_score") -> Dict[str, Dict]:
    """Run every method on the frozen suite and summarize the target metric."""
    out = {}
    for name, factory in methods.items():
        arr = run_method(factory, regime)
        out[name] = {"summary": summarize_method(arr, key),
                     "means": {k: float(v.mean()) for k, v in arr.items()},
                     "raw_sis": arr[key].tolist()}
    return out
