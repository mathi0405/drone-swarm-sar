"""Region partitioning utilities for cooperative search."""
from __future__ import annotations

import numpy as np


def voronoi_partition(size: int, drone_pos: np.ndarray, flyable_mask: np.ndarray | None = None) -> np.ndarray:
    """Assign each map cell to the nearest drone index, or -1 for non-flyable cells."""
    positions = np.asarray(drone_pos, dtype=float)
    yy, xx = np.mgrid[0:size, 0:size]
    cells = np.stack([xx, yy], axis=-1)
    dist = np.linalg.norm(cells[..., None, :] - positions[None, None, :, :], axis=-1)
    regions = np.argmin(dist, axis=-1).astype(np.int32)
    if flyable_mask is not None:
        regions = np.where(flyable_mask, regions, -1)
    return regions


def frontier_targets(regions: np.ndarray, explored: np.ndarray) -> dict[int, np.ndarray]:
    """Return one unexplored target per region owner."""
    targets: dict[int, np.ndarray] = {}
    for owner in sorted(int(v) for v in np.unique(regions) if v >= 0):
        mask = (regions == owner) & (~explored)
        ys, xs = np.nonzero(mask)
        if len(xs):
            targets[owner] = np.array([xs[0], ys[0]], dtype=float)
    return targets
