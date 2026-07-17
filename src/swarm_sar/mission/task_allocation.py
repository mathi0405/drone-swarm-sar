"""Multi-robot task allocation: assign detected victims to drones.

Five interchangeable strategies are provided so the effect of coordination
quality can be ablated independently of the RL policy:

* ``random``    - random feasible assignment (lower bound baseline)
* ``nearest``   - greedy nearest-drone (myopic, fast)
* ``hungarian`` - optimal one-shot assignment minimizing total distance
* ``auction``   - decentralized auction (Bertsekas), robust to comm loss
* ``rl``        - learned allocation head (delegates to the policy at runtime)

The Hungarian solver uses ``scipy.optimize.linear_sum_assignment`` when available
and otherwise falls back to a self-contained O(n^3) implementation, so the module
has no hard SciPy dependency.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# cost matrix                                                                  #
# --------------------------------------------------------------------------- #
def cost_matrix(drone_pos: np.ndarray, victim_pos: np.ndarray,
                battery: np.ndarray | None = None,
                charger_pos: np.ndarray | None = None,
                min_battery_for_task: float = 0.18) -> np.ndarray:
    """(n_drones x n_victims) travel-cost matrix, optionally battery-weighted."""
    d = np.linalg.norm(drone_pos[:, None, :] - victim_pos[None, :, :], axis=-1)
    if battery is not None:
        battery = np.asarray(battery, dtype=float)
        d = d / np.clip(battery[:, None], 0.05, 1.0)   # low battery -> higher cost
        if charger_pos is not None:
            chargers = np.asarray(charger_pos, dtype=float)
            if chargers.ndim == 1:
                chargers = chargers[None, :]
            home = np.min(
                np.linalg.norm(drone_pos[:, None, :] - chargers[None, :, :], axis=-1),
                axis=1,
            )
            rescue = np.linalg.norm(victim_pos[None, :, :] - drone_pos[:, None, :], axis=-1)
            return_leg = np.min(
                np.linalg.norm(victim_pos[:, None, :] - chargers[None, :, :], axis=-1),
                axis=1,
            )
            energy_risk = rescue + return_leg[None, :] + home[:, None] * 0.25
            low = battery < min_battery_for_task
            d = d + energy_risk / np.clip(battery[:, None], 0.05, 1.0)
            d[low, :] += energy_risk[low, :] * 100.0
    return d


# --------------------------------------------------------------------------- #
# Hungarian (pure-numpy fallback)                                              #
# --------------------------------------------------------------------------- #
def _hungarian_numpy(cost: np.ndarray) -> list[int]:
    """Correct O(n^3) Hungarian (Kuhn-Munkres) with potentials, no SciPy needed.

    Rectangular inputs are padded to a square matrix; rows assigned to a dummy
    (padding) column are returned as -1. Verified against brute-force optimum.
    """
    C = np.asarray(cost, dtype=float)
    n, m = C.shape
    size = max(n, m)
    big = (C.max() + 1.0) if C.size else 1.0
    A = np.full((size, size), big, dtype=float)
    A[:n, :m] = C

    INF = float("inf")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)        # p[j] = row matched to column j (1-indexed)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = A[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assign = [-1] * n
    for j in range(1, size + 1):
        i = p[j]
        if 1 <= i <= n and (j - 1) < m:
            assign[i - 1] = j - 1
    return assign


def hungarian(cost: np.ndarray) -> list[int]:
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        out = [-1] * cost.shape[0]
        for ri, ci in zip(r, c):
            out[ri] = int(ci)
        return out
    except Exception:
        return _hungarian_numpy(cost)


# --------------------------------------------------------------------------- #
# auction (Bertsekas)                                                          #
# --------------------------------------------------------------------------- #
def auction(cost: np.ndarray, eps: float = 0.05, max_iter: int = 1000) -> list[int]:
    n, m = cost.shape
    # Pad with dummy columns when bidders outnumber tasks: without them the
    # eviction war between surplus drones never converges and every call
    # burned the full max_iter budget (the dominant env hotspot in profiling).
    n_dummy = max(0, n - m)
    if n_dummy:
        big = (float(cost.max()) if cost.size else 0.0) + 1.0
        cost = np.concatenate([cost, np.full((n, n_dummy), big)], axis=1)
    m_eff = cost.shape[1]
    value = -cost                      # maximize negative cost
    prices = np.zeros(m_eff)
    assigned_col = -np.ones(n, dtype=int)
    col_owner = -np.ones(m_eff, dtype=int)
    unassigned = list(range(n))
    it = 0
    while unassigned and it < max_iter:
        it += 1
        i = unassigned.pop(0)
        net = value[i] - prices
        j = int(np.argmax(net))
        best = net[j]
        net[j] = -np.inf
        second = net.max() if m_eff > 1 else best
        prices[j] += (best - second) + eps
        prev = col_owner[j]
        if prev != -1:
            assigned_col[prev] = -1
            unassigned.append(prev)
        col_owner[j] = i
        assigned_col[i] = j
    assigned_col[assigned_col >= m] = -1          # dummy column -> unassigned
    return assigned_col.tolist()


# --------------------------------------------------------------------------- #
# public interface                                                            #
# --------------------------------------------------------------------------- #
def _random(cost, rng):
    n, m = cost.shape
    cols = list(range(m)); rng.shuffle(cols)
    out = -np.ones(n, dtype=int)
    for i in range(min(n, m)):
        out[i] = cols[i]
    return out.tolist()


def _nearest(cost):
    n, m = cost.shape
    out = -np.ones(n, dtype=int)
    taken = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for i, j in order:
        if out[i] == -1 and j not in taken:
            out[i] = j; taken.add(j)
    return out.tolist()


# --------------------------------------------------------------------------- #
# Consensus-Based Auction Algorithm (CBAA)                                     #
# --------------------------------------------------------------------------- #
def cbaa(cost: np.ndarray, max_iter: int = 100) -> list[int]:
    n, m = cost.shape
    value = -cost
    y = np.full((n, m), -np.inf)
    z = -np.ones((n, m), dtype=int)

    for _ in range(max_iter):
        # 1. Phase 1: Auction
        for i in range(n):
            # Check if drone i is currently assigned to any task
            if i not in z[i]:
                # Valid tasks are those where drone i's value > current winning bid
                valid = value[i] > y[i]
                if valid.any():
                    # Bid on the task with the highest utility
                    h = np.where(valid, value[i], -np.inf)
                    best_j = np.argmax(h)
                    y[i, best_j] = value[i, best_j]
                    z[i, best_j] = i

        # 2. Phase 2: Consensus (simulate all-to-all broadcast)
        y_next = y.copy()
        z_next = z.copy()

        for j in range(m):
            # Find the globally maximum bid for task j
            max_bid = np.max(y[:, j])
            if max_bid > -np.inf:
                # Find all drones that have this max bid
                bidders = np.where(y[:, j] == max_bid)[0]
                # Identify the winning drone for this bid (tie break by min index)
                candidates = z[bidders, j]
                winner = np.min(candidates[candidates != -1]) if np.any(candidates != -1) else -1
                if winner != -1:
                    y_next[:, j] = max_bid
                    z_next[:, j] = winner

        if np.array_equal(y, y_next) and np.array_equal(z, z_next):
            break

        y = y_next
        z = z_next

    assign = [-1] * n
    for j in range(m):
        if z[0, j] != -1:
            assign[z[0, j]] = j
    return assign


ALLOCATORS = ("random", "nearest", "hungarian", "auction", "cbaa", "rl")


class TaskAllocator:
    def __init__(self, strategy: str = "auction", rng: np.random.Generator | None = None):
        assert strategy in ALLOCATORS, f"unknown strategy {strategy}"
        self.strategy = strategy
        self.rng = rng or np.random.default_rng(0)

    def assign(self, drone_pos: np.ndarray, victim_pos: np.ndarray,
               battery: np.ndarray | None = None,
               rl_scores: np.ndarray | None = None,
               charger_pos: np.ndarray | None = None,
               min_battery_for_task: float = 0.18) -> dict[int, int]:
        """Return {drone_idx: victim_idx} (victim_idx == -1 means unassigned)."""
        if len(victim_pos) == 0 or len(drone_pos) == 0:
            return {i: -1 for i in range(len(drone_pos))}
        C = cost_matrix(
            np.asarray(drone_pos, float),
            np.asarray(victim_pos, float),
            battery,
            charger_pos=charger_pos,
            min_battery_for_task=min_battery_for_task,
        )
        if self.strategy == "random":
            res = _random(C, self.rng)
        elif self.strategy == "nearest":
            res = _nearest(C)
        elif self.strategy == "hungarian":
            res = hungarian(C)
        elif self.strategy == "auction":
            res = auction(C)
        elif self.strategy == "cbaa":
            res = cbaa(C)
        else:  # rl - argmax over learned compatibility scores (falls back to nearest)
            res = list(np.argmax(rl_scores, axis=1)) if rl_scores is not None else _nearest(C)
        return {i: int(res[i]) for i in range(len(res))}
