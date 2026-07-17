"""Small, dependency-free geometry helpers used across the sim."""
from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple
import numpy as np


def l2(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def clip_to_bounds(xy, low, high):
    return np.clip(np.asarray(xy, dtype=float), low, high)


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Integer grid cells on the line (x0,y0)->(x1,y1). Used for line-of-sight."""
    cells: List[Tuple[int, int]] = []
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
    return cells


@lru_cache(maxsize=32)
def disk_visibility_offsets(radius: int):
    """Precomputed geometry for exact Bresenham line-of-sight over a disk.

    For every cell offset (dy, dx) within ``radius`` of the origin (the closed
    disk, center included), this precomputes the *intermediate* cells of the
    Bresenham line from the origin to that offset — exactly the cells
    :meth:`SARWorld.line_of_sight` inspects. Because the lines are
    translation-invariant, a sensing sweep reduces to fancy indexing into an
    occlusion mask instead of per-cell Python raycasts.

    Returns ``(offs, line_y, line_x, line_valid)``:
      * ``offs``       (N, 2) int32 — (dy, dx) of each disk cell,
      * ``line_y/x``   (N, L) int32 — padded intermediate-cell offsets,
      * ``line_valid`` (N, L) bool  — mask of real (non-padding) entries.
    """
    cells: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                cells.append((dy, dx))
    lines = [bresenham(0, 0, dx, dy)[1:-1] for (dy, dx) in cells]
    n = len(cells)
    max_len = max((len(line) for line in lines), default=0) or 1
    line_y = np.zeros((n, max_len), dtype=np.int32)
    line_x = np.zeros((n, max_len), dtype=np.int32)
    line_valid = np.zeros((n, max_len), dtype=bool)
    for k, line in enumerate(lines):
        for j, (x, y) in enumerate(line):
            line_x[k, j] = x
            line_y[k, j] = y
            line_valid[k, j] = True
    offs = np.asarray(cells, dtype=np.int32)
    return offs, line_y, line_x, line_valid
