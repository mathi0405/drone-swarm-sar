"""Optional victim-detection interfaces for simulated, thermal and YOLO inputs."""

from swarm_sar.perception.detector_base import (
    BaseDetector,
    Detection,
    DetectorConfig,
    SimulatedVictimDetector,
    fuse_detections_into_belief,
)
from swarm_sar.perception.thermal_detector import ThermalDetector

__all__ = [
    "BaseDetector",
    "Detection",
    "DetectorConfig",
    "SimulatedVictimDetector",
    "ThermalDetector",
    "fuse_detections_into_belief",
]
