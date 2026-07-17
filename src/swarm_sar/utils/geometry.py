"""Small, dependency-free geometry helpers used across the sim."""
from __future__ import annotations
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
