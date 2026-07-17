"""Illustrative learning-curve generator.

IMPORTANT (scientific integrity): the functions here produce *illustrative*
learning curves for the figure templates (reward vs steps, coverage vs episodes,
architecture comparison) so the plotting pipeline and paper compile end-to-end
before a multi-day GPU training run is available. They are deterministic given a
seed and are always labelled "illustrative" in the figure. Replace them with real
TensorBoard/CSV logs from ``scripts/train.py`` for publication.
"""
from __future__ import annotations
import numpy as np

ARCHS = ["PPO", "PPO+GNN", "PPO+Transformer", "PPO+Transformer+GNN"]
# asymptotic performance & learning speed we *expect* per architecture
_PROFILE = {
    "PPO":                  (0.62, 1.0),
    "PPO+GNN":              (0.74, 1.2),
    "PPO+Transformer":      (0.78, 1.15),
    "PPO+Transformer+GNN":  (0.86, 1.35),
}


def learning_curve(arch: str, steps: int = 60, seed: int = 0, metric_max: float = 1.0):
    rng = np.random.default_rng(seed)
    asymptote, speed = _PROFILE[arch]
    x = np.linspace(0, 1, steps)
    y = asymptote * (1 - np.exp(-3.0 * speed * x))
    y += rng.normal(0, 0.02, size=steps)              # seed noise
    return np.clip(y, 0, metric_max)


def multi_seed(arch: str, steps: int = 60, seeds=(0, 1, 2, 3, 4)):
    ys = np.stack([learning_curve(arch, steps, s) for s in seeds])
    return ys.mean(0), ys.std(0)
