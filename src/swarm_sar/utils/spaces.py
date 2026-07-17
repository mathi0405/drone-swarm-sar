"""Lightweight space descriptors (mirror gymnasium.spaces without hard dep)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Discrete:
    n: int
    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.n))


@dataclass
class Box:
    low: float
    high: float
    shape: tuple
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=self.shape).astype(np.float32)
