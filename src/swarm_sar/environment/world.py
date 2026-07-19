"""Procedural disaster-response world.

A new, randomized map is generated every episode (unless disabled) to force the
policy to *generalize* rather than memorize a single layout. The world is a 2.5-D
occupancy grid: obstacles have footprints on a 2-D grid, and drones fly at a
continuous altitude above it. This keeps the simulation fast enough to run
thousands of episodes on a CPU while retaining the structure needed to study
cooperative exploration under occlusion, fire/smoke and no-fly constraints.
"""
from __future__ import annotations

import numpy as np

from swarm_sar.config import WorldConfig
from swarm_sar.environment.entities import (
    CellType,
    ChargingStation,
    DynamicObstacle,
    Victim,
)
from swarm_sar.utils.geometry import bresenham

# Integer cell values derived from the enum ONCE, so flight/vision rules can
# never drift from CellType.blocks_flight / CellType.occludes_vision while the
# hot-path grid checks stay plain integer comparisons.
BLOCKS_FLIGHT_VALUES = tuple(ct.value for ct in CellType if ct.blocks_flight)
OCCLUDES_VALUES = tuple(ct.value for ct in CellType if ct.occludes_vision)
_BLOCKS_FLIGHT = BLOCKS_FLIGHT_VALUES
_OCCLUDES = OCCLUDES_VALUES


class SARWorld:
    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.size = cfg.size
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.victims: list[Victim] = []
        self.charging: list[ChargingStation] = []
        self.dyn_obstacles: list[DynamicObstacle] = []
        self.reset(rng)

    # --------------------------------------------------------------------- #
    # generation                                                            #
    # --------------------------------------------------------------------- #
    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        r = self.rng
        self.grid[:] = CellType.FREE

        self._carve_roads(r)
        self._scatter(CellType.BUILDING, self.cfg.n_buildings, r, max_footprint=4)
        self._scatter(CellType.TREE, self.cfg.n_trees, r, max_footprint=1)
        self._scatter(CellType.RUBBLE, self.cfg.n_rubble, r, max_footprint=2)
        self._place_fire_and_smoke(r)
        self._place_no_fly(r)

        # charging stations on free/road cells
        self.charging = []
        for i in range(self.cfg.n_charging_stations):
            p = self._free_cell(r)
            self.grid[p[1], p[0]] = CellType.CHARGING
            self.charging.append(ChargingStation(i, np.array(p, dtype=float)))

        # victims (may lie in rubble — realistic)
        self.victims = []
        for i in range(self.cfg.n_victims):
            p = self._free_cell(r, allow=(CellType.FREE, CellType.ROAD, CellType.RUBBLE))
            self.victims.append(
                Victim(i, np.array(p, dtype=float), severity=float(r.uniform(0.3, 1.0)))
            )

        # dynamic obstacles (e.g. moving vehicles, other aircraft)
        self.dyn_obstacles = []
        for i in range(self.cfg.n_dynamic_obstacles):
            p = np.array(self._free_cell(r), dtype=float)
            v = r.uniform(-0.8, 0.8, size=2)
            self.dyn_obstacles.append(DynamicObstacle(i, p, v))

    def _carve_roads(self, r: np.random.Generator) -> None:
        s = self.size
        for _ in range(max(2, s // 24)):
            row = r.integers(2, s - 2)
            self.grid[row, :] = CellType.ROAD
            col = r.integers(2, s - 2)
            self.grid[:, col] = CellType.ROAD

    def _scatter(self, kind: CellType, n: int, r: np.random.Generator, max_footprint: int) -> None:
        s = self.size
        for _ in range(n):
            x = int(r.integers(1, s - 1)); y = int(r.integers(1, s - 1))
            fp = int(r.integers(1, max_footprint + 1))
            for dx in range(fp):
                for dy in range(fp):
                    xx, yy = min(x + dx, s - 1), min(y + dy, s - 1)
                    if self.grid[yy, xx] == CellType.FREE:
                        self.grid[yy, xx] = kind

    def _place_fire_and_smoke(self, r: np.random.Generator) -> None:
        s = self.size
        for _ in range(self.cfg.n_fire_zones):
            cx, cy = int(r.integers(3, s - 3)), int(r.integers(3, s - 3))
            self.grid[cy, cx] = CellType.FIRE
            rad = self.cfg.smoke_spread
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    xx, yy = cx + dx, cy + dy
                    if 0 <= xx < s and 0 <= yy < s:
                        d = (dx * dx + dy * dy) ** 0.5
                        if 0 < d <= rad * 2 and self.grid[yy, xx] in (CellType.FREE, CellType.ROAD):
                            if r.random() < 0.6:
                                self.grid[yy, xx] = CellType.SMOKE

    def _place_no_fly(self, r: np.random.Generator) -> None:
        s = self.size
        for _ in range(self.cfg.n_no_fly_zones):
            cx, cy = int(r.integers(4, s - 4)), int(r.integers(4, s - 4))
            w, h = int(r.integers(3, 7)), int(r.integers(3, 7))
            self.grid[max(0, cy - h // 2):cy + h // 2, max(0, cx - w // 2):cx + w // 2] = CellType.NO_FLY

    def _free_cell(self, r: np.random.Generator, allow=(CellType.FREE, CellType.ROAD)) -> tuple[int, int]:
        for _ in range(500):
            x = int(r.integers(1, self.size - 1)); y = int(r.integers(1, self.size - 1))
            if self.grid[y, x] in allow:
                return x, y
        return self.size // 2, self.size // 2

    # --------------------------------------------------------------------- #
    # queries                                                               #
    # --------------------------------------------------------------------- #
    def in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def cell(self, x: float, y: float) -> CellType:
        return CellType(int(self.grid[int(y), int(x)]))

    def is_flyable(self, x: float, y: float) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.grid[int(y), int(x)] not in _BLOCKS_FLIGHT

    def m_to_cells(self, metres: float) -> float:
        """Convert a metric distance to grid cells (single source of truth)."""
        return metres / max(1e-6, self.cfg.cell_size_m)

    def cells_to_m(self, cells: float) -> float:
        return cells * self.cfg.cell_size_m

    def clip_segment(self, start, end):
        """Furthest point along start→end reachable without entering a
        non-flyable cell.

        Walks the segment in ≤0.5-cell increments so a fast move cannot
        tunnel through an obstacle that lies strictly between its endpoints
        (endpoint-only checks let drones pass through one-cell buildings).
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            return end if self.is_flyable(end[0], end[1]) else start
        steps = max(1, int(np.ceil(dist / 0.5)))
        prev = start
        for k in range(1, steps + 1):
            p = start + delta * (k / steps)
            if not self.is_flyable(p[0], p[1]):
                return prev
            prev = p
        return end

    def occludes(self, x: int, y: int) -> bool:
        return self.grid[int(y), int(x)] in _OCCLUDES

    def line_of_sight(self, a, b) -> bool:
        """True if no occluding cell lies strictly between a and b."""
        if not self.in_bounds(a[0], a[1]) or not self.in_bounds(b[0], b[1]):
            return False
        cells = bresenham(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        for (x, y) in cells[1:-1]:
            if not self.in_bounds(x, y):
                return False
            if self.occludes(x, y):
                return False
        return True

    def nearest_charging(self, pos):
        """Return the closest charging station, or None if none exist."""
        if not self.charging:
            return None
        d = [np.linalg.norm(np.asarray(pos) - c.pos) for c in self.charging]
        return self.charging[int(np.argmin(d))]

    def step_dynamic(self) -> dict[str, int]:
        events = {
            "obstacle_moves": 0,
            "victim_moves": 0,
            "fire_spread": 0,
            "smoke_spread": 0,
            "smoke_decay": 0,
            "collapses": 0,
        }
        for o in self.dyn_obstacles:
            o.step(self.size, self.rng)
            events["obstacle_moves"] += 1
        if self.cfg.moving_victims:
            events["victim_moves"] = self._move_victims()
        if self.cfg.enable_fire_spread:
            fire, smoke, decay = self._spread_fire_and_smoke()
            events["fire_spread"] = fire
            events["smoke_spread"] = smoke
            events["smoke_decay"] = decay
        if self.cfg.enable_collapses:
            events["collapses"] = self._collapse_structures()
        return events

    def _move_victims(self) -> int:
        moved = 0
        candidates = [
            np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([0.0, -1.0]),
        ]
        for v in self.victims:
            if v.rescued or self.rng.random() >= self.cfg.victim_move_prob:
                continue
            order = self.rng.permutation(len(candidates))
            for idx in order:
                pos = np.clip(v.pos + candidates[int(idx)], 0, self.size - 1)
                cell = self.grid[int(pos[1]), int(pos[0])]
                if cell not in (CellType.BUILDING, CellType.NO_FLY, CellType.FIRE):
                    v.pos = pos.astype(float)
                    moved += 1
                    break
        return moved

    def _spread_fire_and_smoke(self) -> tuple[int, int, int]:
        fire_cells = np.argwhere(self.grid == CellType.FIRE)
        smoke_cells = np.argwhere(self.grid == CellType.SMOKE)
        new_fire: list[tuple[int, int]] = []
        new_smoke: list[tuple[int, int]] = []
        decayed: list[tuple[int, int]] = []
        for y, x in fire_cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = int(x + dx), int(y + dy)
                if not (0 <= xx < self.size and 0 <= yy < self.size):
                    continue
                cell = CellType(int(self.grid[yy, xx]))
                if cell in (CellType.FREE, CellType.ROAD, CellType.TREE, CellType.RUBBLE, CellType.SMOKE):
                    if self.rng.random() < self.cfg.fire_spread_prob:
                        new_fire.append((yy, xx))
                    elif cell in (CellType.FREE, CellType.ROAD) and self.rng.random() < self.cfg.smoke_spread_prob:
                        new_smoke.append((yy, xx))
        for y, x in smoke_cells:
            if self.rng.random() < self.cfg.smoke_decay_prob:
                decayed.append((int(y), int(x)))
        for y, x in decayed:
            if self.grid[y, x] == CellType.SMOKE:
                self.grid[y, x] = CellType.FREE
        for y, x in new_smoke:
            if self.grid[y, x] in (CellType.FREE, CellType.ROAD):
                self.grid[y, x] = CellType.SMOKE
        for y, x in new_fire:
            if self.grid[y, x] not in (CellType.CHARGING, CellType.NO_FLY):
                self.grid[y, x] = CellType.FIRE
        return len(new_fire), len(new_smoke), len(decayed)

    def _collapse_structures(self) -> int:
        collapsed = 0
        buildings = np.argwhere(self.grid == CellType.BUILDING)
        for y, x in buildings:
            if self.rng.random() < self.cfg.collapse_prob:
                self.grid[int(y), int(x)] = CellType.RUBBLE
                collapsed += 1
        return collapsed

    # --------------------------------------------------------------------- #
    # convenience masks for visualization                                    #
    # --------------------------------------------------------------------- #
    @property
    def flyable_mask(self) -> np.ndarray:
        m = np.ones_like(self.grid, dtype=bool)
        for value in _BLOCKS_FLIGHT:
            m &= self.grid != value
        return m
