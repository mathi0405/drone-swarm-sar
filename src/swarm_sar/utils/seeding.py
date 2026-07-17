"""Deterministic seeding utilities (critical for reproducible experiments)."""
from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Return a per-object NumPy Generator; avoids global-state coupling."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int) -> None:
    """Seed all common RNGs. Torch is seeded only if installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def set_deterministic(seed: int = 0) -> None:
    """Fully deterministic mode (reproducible experiments). Slower on GPU."""
    set_global_seed(seed)
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
