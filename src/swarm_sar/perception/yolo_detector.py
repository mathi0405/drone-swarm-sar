"""Optional Ultralytics YOLO victim detector."""
from __future__ import annotations

import numpy as np

from swarm_sar.perception.detector_base import Detection, DetectorConfig


class YOLODetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig(mode="yolo")
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Ultralytics is optional. Install perception extras with "
                "pip install -e '.[perception]' or pip install -r requirements-sim.txt"
            ) from exc
        if not self.cfg.model_path:
            raise ValueError("YOLODetector requires DetectorConfig.model_path")
        self.model = YOLO(self.cfg.model_path)

    def detect(self, frame: np.ndarray, **context) -> list[Detection]:
        results = self.model.predict(frame, verbose=False)
        detections: list[Detection] = []
        pixel_to_world = context.get("pixel_to_world")
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf.item())
                if conf < self.cfg.confidence_threshold:
                    continue
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, cls_id))
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                world_pos = pixel_to_world(*center) if pixel_to_world else None
                detections.append(
                    Detection(label, conf, (x1, y1, x2, y2), world_pos, source="yolo")
                )
        return detections
