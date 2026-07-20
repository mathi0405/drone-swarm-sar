"""Reliable RL statistics (Agarwal et al., NeurIPS 2021 — "Statistical Precipice").

Reporting mean +/- std over a handful of seeds is unreliable. We instead report
the Interquartile Mean (IQM) with stratified bootstrap confidence intervals, and
performance profiles. Pure NumPy so there is no hard `rliable` dependency; if
`rliable` is installed you may prefer it for publication plots.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def iqm(x: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% (robust to outliers)."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0:
        return float("nan")
    lo = int(np.floor(0.25 * x.size)); hi = int(np.ceil(0.75 * x.size))
    core = x[lo:hi] if hi > lo else x
    return float(core.mean())


def bootstrap_ci(x: Sequence[float], statistic=iqm, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a 1-D sample."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.array([statistic(rng.choice(x, size=x.size, replace=True))
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return (float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)))


def stratified_bootstrap_ci(scores: np.ndarray, statistic=iqm, n_boot: int = 10000,
                            ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """CI for a (runs x tasks) score matrix; resamples runs within each task."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        return bootstrap_ci(scores, statistic, n_boot, ci, seed)
    rng = np.random.default_rng(seed)
    n_runs, n_tasks = scores.shape
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_runs, size=(n_runs, n_tasks))
        resampled = scores[idx, np.arange(n_tasks)[None, :]]
        stats[b] = statistic(resampled.reshape(-1))
    a = (1 - ci) / 2
    return (float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a)))


def performance_profile(scores: Sequence[float], taus: np.ndarray = None) -> dict[str, np.ndarray]:
    """Run-score distribution: fraction of runs with score > tau, over a tau grid."""
    scores = np.asarray(scores, dtype=float)
    if taus is None:
        lo, hi = (scores.min(), scores.max()) if scores.size else (0.0, 1.0)
        taus = np.linspace(lo, hi, 50)
    frac = np.array([(scores > t).mean() for t in taus])
    return {"tau": taus, "fraction": frac}


def probability_of_improvement(a: Sequence[float], b: Sequence[float]) -> float:
    """P(X_a > X_b) averaged over all pairings (Mann-Whitney style effect size)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float((a[:, None] > b[None, :]).mean())


def sanitize_json(obj):
    """Recursively replace non-finite floats with None for STRICT-JSON dumps.

    ``json.dump`` happily writes literal ``NaN``/``Infinity`` tokens (e.g. the
    deliberate NaN from ``Evaluator.robustness`` when the nominal baseline is
    degenerate), which jq / JavaScript / strict parsers reject.  Python-side
    semantics keep the NaN; serialization maps it to null.
    """
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def summarize(scores: Sequence[float]) -> dict[str, float]:
    """IQM + 95% bootstrap CI + median + mean for a 1-D sample."""
    lo, hi = bootstrap_ci(scores)
    x = np.asarray(scores, dtype=float)
    return {"iqm": iqm(x), "ci95_low": lo, "ci95_high": hi,
            "median": float(np.median(x)) if x.size else float("nan"),
            "mean": float(x.mean()) if x.size else float("nan"),
            "n": int(x.size)}
