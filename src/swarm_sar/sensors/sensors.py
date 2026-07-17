"""Noisy sensor suite: GPS (noise + drift + failure), IMU, camera (victim
detection under occlusion/smoke) and LiDAR (local occupancy).

Sensor noise is central to the research question (sim-to-real): all readings are
corrupted, and detection is probabilistic and occlusion-aware.
"""
from __future__ import annotations
import numpy as np
from swarm_sar.config import SensorConfig


class GPS:
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg; self.rng = rng
        self._drift = np.zeros(2)

    def read(self, true_pos, ok: bool) -> np.ndarray:
        if not ok:                                   # GPS failure -> stale/huge error
            return true_pos + self.rng.normal(0, 8.0, size=2)
        self._drift += self.rng.normal(0, self.cfg.gps_drift_m_per_s, size=2)
        return np.asarray(true_pos, float) + self.rng.normal(0, self.cfg.gps_noise_std_m, size=2) + self._drift


class IMU:
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg; self.rng = rng

    def read(self, vel, yaw) -> np.ndarray:
        acc = np.asarray(vel, float) + self.rng.normal(0, self.cfg.imu_noise_std, size=len(vel))
        return np.concatenate([acc, [yaw + self.rng.normal(0, self.cfg.imu_noise_std)]])


class Camera:
    """Down-facing camera used for victim detection."""
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg; self.rng = rng

    def detect(self, drone_pos, victim_pos, world, ok: bool, visibility: float = 1.0) -> bool:
        """Probabilistic detection; ``visibility`` (weather) shrinks the range."""
        if not ok:
            return False
        d = np.linalg.norm(np.asarray(drone_pos) - np.asarray(victim_pos))
        eff_range = max(1e-6, self.cfg.camera_range_m * float(visibility) / world.cfg.cell_size_m)
        if d > eff_range:
            return False
        if not world.line_of_sight(drone_pos, victim_pos):
            return False
        p = self.cfg.detection_base_prob * (1.0 - 0.5 * d / eff_range)
        # smoke over victim reduces detection
        from swarm_sar.environment.entities import CellType
        if world.cell(victim_pos[0], victim_pos[1]) in (CellType.SMOKE, CellType.FIRE):
            p *= self.cfg.smoke_occlusion
        return bool(self.rng.random() < p)


class Lidar:
    """Ranges to nearest occluders in 8 compass directions (local occupancy)."""
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg; self.rng = rng

    def scan(self, drone_pos, world) -> np.ndarray:
        rng_cells = int(self.cfg.lidar_range_m / world.cfg.cell_size_m)
        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        out = np.full(8, rng_cells, dtype=float)
        for i, (dx, dy) in enumerate(dirs):
            for r in range(1, rng_cells + 1):
                x, y = drone_pos[0] + dx * r, drone_pos[1] + dy * r
                if not world.in_bounds(x, y) or not world.is_flyable(x, y):
                    out[i] = r; break
        out += self.rng.normal(0, 0.2, size=8)       # ranging noise
        return np.clip(out, 0, rng_cells)


class SensorSuite:
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.gps = GPS(cfg, rng)
        self.imu = IMU(cfg, rng)
        self.camera = Camera(cfg, rng)
        self.lidar = Lidar(cfg, rng)
