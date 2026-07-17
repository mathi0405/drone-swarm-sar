"""Swarm coordination helpers: leaders, formations and failure reassignment."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def elect_leader(positions: np.ndarray, alive: Iterable[bool],
                 comm_ok: Iterable[bool], battery: Iterable[float],
                 comm_range: float) -> int:
    """Choose an alive, reachable drone with high battery and central position."""
    pos = np.asarray(positions, dtype=float)
    alive_arr = np.asarray(list(alive), dtype=bool)
    comm_arr = np.asarray(list(comm_ok), dtype=bool)
    batt = np.asarray(list(battery), dtype=float)
    candidates = np.where(alive_arr & comm_arr)[0]
    if len(candidates) == 0:
        return -1
    scores = []
    for i in candidates:
        d = np.linalg.norm(pos[candidates] - pos[i], axis=1)
        reachable = float(np.mean(d <= comm_range)) if len(d) else 0.0
        centrality = 1.0 / (1.0 + (float(d.mean()) if len(d) else 1.0))
        scores.append(batt[i] + reachable + centrality)
    return int(candidates[int(np.argmax(scores))])


def reassign_failed_tasks(assignments: dict[int, int], alive: Iterable[bool],
                          positions: np.ndarray, target_pos: np.ndarray) -> dict[int, int]:
    """Move tasks owned by failed drones to nearest alive drones."""
    alive_arr = np.asarray(list(alive), dtype=bool)
    pos = np.asarray(positions, dtype=float)
    targets = np.asarray(target_pos, dtype=float)
    out = {int(k): int(v) for k, v in assignments.items() if alive_arr[int(k)]}
    failed_targets = [v for k, v in assignments.items() if not alive_arr[int(k)] and int(v) >= 0]
    # Only drones without a task are candidates: overwriting a busy drone's
    # assignment would silently drop its existing task.
    available = [i for i, ok in enumerate(alive_arr) if ok and i not in out]
    for victim_idx in failed_targets:
        if not available:
            break
        victim = targets[int(victim_idx)]
        owner = min(available, key=lambda i: float(np.linalg.norm(pos[i] - victim)))
        out[int(owner)] = int(victim_idx)
        available.remove(owner)
    return out


def line_formation(center: np.ndarray, n: int, spacing: float = 3.0) -> np.ndarray:
    """Generate simple formation offsets for scan-line search."""
    offsets = np.zeros((n, 2), dtype=float)
    start = -(n - 1) * spacing * 0.5
    for i in range(n):
        offsets[i] = np.asarray(center, dtype=float) + np.array([start + i * spacing, 0.0])
    return offsets
