"""Simple thermal hotspot detector for optional SAR perception demos."""
from __future__ import annotations

import numpy as np

from swarm_sar.perception.detector_base import Detection, DetectorConfig


class ThermalDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig(mode="thermal")

    def detect(self, frame: np.ndarray, **context) -> list[Detection]:
        if frame.size == 0:
            return []
        heat = np.asarray(frame, dtype=float)
        if heat.ndim == 3:
            heat = heat.mean(axis=-1)
        peak = float(heat.max())
        if peak <= 0:
            return []
        norm = heat / peak
        ys, xs = np.where(norm >= self.cfg.thermal_threshold)
        detections = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            conf = float(norm[y, x])
            detections.append(
                Detection(
                    "victim",
                    conf,
                    (x - 1.0, y - 1.0, x + 1.0, y + 1.0),
                    world_pos=context.get("pixel_to_world", lambda px, py: (px, py))(x, y),
                    source="thermal",
                )
            )
        return detections
