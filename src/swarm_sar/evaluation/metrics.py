"""Evaluation metrics for cooperative SAR, plus the Swarm Intelligence Score.

All metrics are computed from an :class:`~swarm_sar.training.rollout.EpisodeLog`
so they are backend-agnostic (self-contained sim or AirSim). Aggregation reports
mean +/- std across seeds, as required for publication-grade results.
"""
from __future__ import annotations

import numpy as np

from swarm_sar.drone.drone import ACTIONS

# weights for the Swarm Intelligence Score (sum to 1)
SIS_WEIGHTS = {"coverage": 0.25, "rescue": 0.30, "energy": 0.15,
               "communication": 0.10, "safety": 0.20}

# A swarm colliding at or above this rate (collisions per step) scores 0 on safety.
# Chosen as ~1 collision per 5 steps; documented rather than a magic constant.
COLLISION_RATE_CAP = 0.20


def _path_length(frames) -> float:
    total = 0.0
    for a, b in zip(frames[:-1], frames[1:]):
        for pa, pb in zip(a["pos"], b["pos"]):
            total += float(np.linalg.norm(np.asarray(pb) - np.asarray(pa)))
    return total


def exploration_entropy(frames, grid_size: int, bins: int = 16) -> float:
    """Shannon entropy of the spatial visitation distribution (bits), normalized
    to [0,1]. High = drones spread out; low = redundant, clustered search."""
    counts = np.zeros((bins, bins), dtype=float)
    for f in frames:
        for p in f["pos"]:
            x = max(0, min(bins - 1, int(p[0] / grid_size * bins)))
            y = max(0, min(bins - 1, int(p[1] / grid_size * bins)))
            counts[y, x] += 1
    p = counts.flatten()
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    nz = p[p > 0]
    H = -np.sum(nz * np.log2(nz))
    return float(H / np.log2(bins * bins))            # normalize by max entropy


def episode_metrics(log, energy_budget: float | None = None, grid_size: int = 64,
                    sensor_swath_cells: float = 16.0) -> dict[str, float]:
    s = log.summary
    frames = log.frames
    steps = max(1, s["steps"])
    n_v = max(1, s["victims_total"])
    path = _path_length(frames)
    rescued_steps = [v["rescued_step"] for v in log.victims if v["rescued"] and v["rescued_step"] >= 0]

    # Budget = combined initial battery capacity, recorded by the env so the
    # metric adapts to fleet size instead of assuming 3 drones.
    if energy_budget is None:
        energy_budget = float(s.get("energy_budget_wh", 270.0))

    coverage = s["coverage"]
    victims_rescued = s["victims_rescued"]
    rescue_ratio = victims_rescued / n_v
    comm_eff = float(np.clip(s["delivery_ratio"], 0, 1))

    completion_score = (
        0.40 * coverage +
        0.40 * rescue_ratio +
        0.20 * comm_eff
    )

    # mission_success uses the environment's single definition (all victims
    # rescued); the graded *_success flags grade the composite completion score.
    success = 1.0 if s.get("success") else 0.0
    partial_success = 1.0 if completion_score >= 0.40 else 0.0
    good_success = 1.0 if completion_score >= 0.60 else 0.0
    excellent_success = 1.0 if completion_score >= 0.80 else 0.0

    avg_rescue_time = float(np.mean(rescued_steps)) if rescued_steps else float(steps)
    # path efficiency vs a lawnmower lower bound: the minimum distance to sweep the
    # explored area at the sensor swath is (explored_cells / swath). Efficiency is
    # that ideal distance over the distance actually travelled, in [0, 1].
    explored_cells = coverage * grid_size * grid_size
    ideal_distance = explored_cells / max(1.0, sensor_swath_cells)
    path_efficiency = float(np.clip(ideal_distance / (path + 1e-6), 0, 1))
    collision_rate = s["collisions"] / steps
    energy = s["energy_wh"]
    expl_entropy = exploration_entropy(frames, grid_size)
    alive_frac = float(np.mean(frames[-1]["alive"])) if frames else 1.0
    robustness = float(np.clip(0.5 * alive_frac + 0.5 * (victims_rescued / n_v), 0, 1))

    return {
        "coverage": float(coverage),
        "completion_score": float(completion_score),
        "mission_success": success,
        "partial_success": partial_success,
        "good_success": good_success,
        "excellent_success": excellent_success,
        "victims_rescued": float(victims_rescued),
        "victims_detected": float(s["victims_detected"]),
        "avg_rescue_time": avg_rescue_time,
        "path_efficiency": path_efficiency,
        "collision_rate": float(collision_rate),
        "energy_wh": float(energy),
        "communication_efficiency": comm_eff,
        "exploration_entropy": expl_entropy,
        "robustness": robustness,
        "swarm_intelligence_score": swarm_intelligence_score({
            "coverage": coverage,
            "rescue": victims_rescued / n_v,
            "energy": np.clip(1 - energy / energy_budget, 0, 1),
            "communication": comm_eff,
            "safety": np.clip(1 - collision_rate / COLLISION_RATE_CAP, 0, 1),
        }),
    }


def episode_diagnostics(log) -> dict[str, object]:
    """Low-level diagnostics that expose collapse, silence and unsafe behavior."""
    hist = {name: 0 for name in ACTIONS}
    comm_edges = 0
    frames = getattr(log, "frames", [])
    for f in frames:
        for action in f.get("actions", []):
            idx = int(action)
            name = ACTIONS[idx] if 0 <= idx < len(ACTIONS) else "unknown"
            hist[name] = hist.get(name, 0) + 1
        comm_edges += len(f.get("comm_edges", []))
    summary = getattr(log, "summary", {})
    return {
        "action_histogram": hist,
        "comm_sent": int(summary.get("comm_sent", 0)),
        "comm_delivered": int(summary.get("comm_delivered", 0)),
        "comm_edges": int(comm_edges),
        "delivery_ratio": float(summary.get("delivery_ratio", 0.0)),
        "collisions": int(summary.get("collisions", 0)),
        "steps": int(summary.get("steps", len(frames))),
    }


def swarm_intelligence_score(components: dict[str, float],
                             weights: dict[str, float] = None,
                             mode: str = "geometric") -> float:
    """Composite Swarm Intelligence Score (SIS) in [0, 100].

    Combines coverage, rescue efficiency, energy efficiency, communication and
    safety. A *weighted geometric mean* (default) rewards balanced competence and
    penalizes swarms that sacrifice any one dimension (e.g. high coverage bought
    with many collisions), which we argue better reflects "swarm intelligence"
    than a linear sum.
    """
    w = weights or SIS_WEIGHTS
    keys = list(w.keys())
    vals = np.array([np.clip(components.get(k, 0.0), 1e-4, 1.0) for k in keys])
    ww = np.array([w[k] for k in keys]); ww = ww / ww.sum()
    if mode == "geometric":
        score = float(np.exp(np.sum(ww * np.log(vals))))
    else:
        score = float(np.sum(ww * vals))
    return round(100.0 * score, 2)


def aggregate(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean/std across seeds for every metric."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "n": len(vals)}
    return out
