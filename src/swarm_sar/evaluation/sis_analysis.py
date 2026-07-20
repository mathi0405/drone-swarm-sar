"""Validation of the Swarm Intelligence Score (SIS).

Answers the reviewer questions "why these weights?" and "why geometric?":
 * weight sensitivity - is the *ranking* of methods stable under weight perturbation?
 * geometric vs linear - does the geometric mean actually penalize imbalance?
 * Pareto analysis    - which methods are non-dominated across the 5 raw objectives?
"""
from __future__ import annotations

import numpy as np

from swarm_sar.evaluation.metrics import COLLISION_RATE_CAP, SIS_WEIGHTS, swarm_intelligence_score

DIMS = ["coverage", "rescue", "energy", "communication", "safety"]


def sis_components(metrics: dict[str, float], n_victims: int = 8,
                   energy_budget: float = 270.0) -> dict[str, float]:
    """Map raw episode metrics onto the 5 normalized SIS dimensions."""
    return {
        "coverage": float(np.clip(metrics["coverage"], 0, 1)),
        "rescue": float(np.clip(metrics["victims_rescued"] / max(1, n_victims), 0, 1)),
        "energy": float(np.clip(1 - metrics["energy_wh"] / energy_budget, 0, 1)),
        # The SIS communication dimension is UTILITY (assisted rescues +
        # coverage uniqueness), not message activity; fall back to the old
        # activity signal only for metrics dicts predating the change.
        "communication": float(np.clip(
            metrics.get("communication_utility",
                        metrics["communication_efficiency"]), 0, 1)),
        "safety": float(np.clip(1 - metrics["collision_rate"] / COLLISION_RATE_CAP, 0, 1)),
    }


def _rank(order: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(order)}


def _spearman(a: dict[str, int], b: dict[str, int]) -> float:
    keys = list(a); n = len(keys)
    if n < 2:
        return 1.0
    d2 = sum((a[k] - b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def weight_sensitivity(components_by_method: dict[str, dict[str, float]],
                       n_samples: int = 2000, concentration: float = 20.0,
                       seed: int = 0) -> dict:
    """Perturb the weights (Dirichlet around the default) and measure how stable the
    method ranking is (mean Spearman rho vs default ranking; top-1 retention)."""
    rng = np.random.default_rng(seed)
    w0 = np.array([SIS_WEIGHTS[d] for d in DIMS])
    def score(weights):
        w = {d: weights[i] for i, d in enumerate(DIMS)}
        return {m: swarm_intelligence_score(c, w) for m, c in components_by_method.items()}
    base = score(w0)
    base_order = sorted(base, key=base.get, reverse=True)
    base_rank = _rank(base_order); base_top = base_order[0]
    rhos, top_hits = [], 0
    for _ in range(n_samples):
        w = rng.dirichlet(w0 * concentration)
        s = score(w)
        order = sorted(s, key=s.get, reverse=True)
        rhos.append(_spearman(base_rank, _rank(order)))
        top_hits += int(order[0] == base_top)
    return {"mean_spearman": float(np.mean(rhos)), "std_spearman": float(np.std(rhos)),
            "top1_retention": top_hits / n_samples, "base_order": base_order,
            "rhos": rhos}


def pareto_front(components_by_method: dict[str, dict[str, float]]) -> list[str]:
    """Non-dominated methods across all 5 objectives (higher is better)."""
    names = list(components_by_method)
    M = np.array([[components_by_method[n][d] for d in DIMS] for n in names])
    front = []
    for i, n in enumerate(names):
        dominated = any((M[j] >= M[i]).all() and (M[j] > M[i]).any() for j in range(len(names)) if j != i)
        if not dominated:
            front.append(n)
    return front


def geometric_vs_linear(component: dict[str, float]) -> dict[str, float]:
    return {"geometric": swarm_intelligence_score(component, mode="geometric"),
            "linear": swarm_intelligence_score(component, mode="linear")}
