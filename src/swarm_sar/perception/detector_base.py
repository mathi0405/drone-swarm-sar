"""Detector abstractions used by simulation, YOLO and thermal perception."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    world_pos: tuple[float, float] | None = None
    source: str = "simulated"


@dataclass
class DetectorConfig:
    mode: str = "simulated"
    model_path: str | None = None
    confidence_threshold: float = 0.35
    thermal_threshold: float = 0.75
    fusion_threshold: float = 0.5


class BaseDetector(Protocol):
    def detect(self, frame: np.ndarray, **context) -> list[Detection]:
        """Return victim detections for an image-like frame."""


class SimulatedVictimDetector:
    """Detector that turns known simulator victim positions into detections."""

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()

    def detect(self, frame: np.ndarray, **context) -> list[Detection]:
        victims = context.get("victims", [])
        camera_pos = np.asarray(context.get("camera_pos", (0.0, 0.0)), dtype=float)
        max_range = float(context.get("range", np.inf))
        detections: list[Detection] = []
        for victim in victims:
            if getattr(victim, "rescued", False):
                continue
            pos = np.asarray(getattr(victim, "pos", victim), dtype=float)
            if np.linalg.norm(pos - camera_pos) <= max_range:
                conf = float(context.get("confidence", 0.9))
                detections.append(
                    Detection(
                        "victim",
                        conf,
                        (pos[0] - 0.5, pos[1] - 0.5, pos[0] + 0.5, pos[1] + 0.5),
                        world_pos=(float(pos[0]), float(pos[1])),
                        source="simulated",
                    )
                )
        return detections


def fuse_detections_into_belief(
    belief: np.ndarray,
    detections: Iterable[Detection],
    confidence_floor: float = 0.5,
) -> np.ndarray:
    """Fuse detector outputs into a victim-belief grid in-place and return it."""
    h, w = belief.shape
    for det in detections:
        if det.confidence < confidence_floor:
            continue
        if det.world_pos is not None:
            x, y = det.world_pos
        else:
            x = (det.bbox[0] + det.bbox[2]) * 0.5
            y = (det.bbox[1] + det.bbox[3]) * 0.5
        ix = int(np.clip(round(x), 0, w - 1))
        iy = int(np.clip(round(y), 0, h - 1))
        belief[iy, ix] = max(float(belief[iy, ix]), float(det.confidence))
    return belief
