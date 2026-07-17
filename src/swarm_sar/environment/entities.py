"""Static & dynamic entities that populate the disaster world."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np


class CellType(IntEnum):
    FREE = 0
    ROAD = 1
    BUILDING = 2
    TREE = 3
    RUBBLE = 4
    FIRE = 5
    SMOKE = 6
    NO_FLY = 7
    CHARGING = 8

    @property
    def blocks_flight(self) -> bool:
        return self in (CellType.BUILDING, CellType.NO_FLY)

    @property
    def occludes_vision(self) -> bool:
        return self in (CellType.BUILDING, CellType.TREE, CellType.SMOKE, CellType.FIRE)


@dataclass
class Victim:
    idx: int
    pos: np.ndarray                 # (x, y) in cell coords
    severity: float = 1.0
    detected: bool = False
    classified: bool = False
    detected_by: int = -1
    detected_step: int = -1
    rescued: bool = False
    rescued_by: int = -1
    rescued_step: int = -1
    assigned_drone: int = -1
    _dwell: int = 0                 # accumulated hover time by any drone
    _confirmations: int = 0         # positive detections before confirmation


@dataclass
class ChargingStation:
    idx: int
    pos: np.ndarray
    capacity: int = 2               # simultaneous drones
    occupied_by: list = field(default_factory=list)


@dataclass
class DynamicObstacle:
    idx: int
    pos: np.ndarray
    vel: np.ndarray
    radius: float = 1.0
    def step(self, size: int, rng: np.random.Generator) -> None:
        self.pos = self.pos + self.vel
        # bounce off world edges
        for d in (0, 1):
            if self.pos[d] < 1 or self.pos[d] > size - 2:
                self.vel[d] *= -1
                self.pos[d] = np.clip(self.pos[d], 1, size - 2)
        if rng.random() < 0.05:      # occasional heading change
            self.vel = rng.uniform(-0.8, 0.8, size=2)
