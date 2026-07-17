import numpy as np

from swarm_sar.environment.entities import Victim
from swarm_sar.perception import (
    DetectorConfig,
    SimulatedVictimDetector,
    ThermalDetector,
    fuse_detections_into_belief,
)


def test_simulated_detector_returns_world_positions():
    detector = SimulatedVictimDetector()
    victims = [Victim(0, np.array([3.0, 4.0]))]

    detections = detector.detect(np.zeros((8, 8)), victims=victims, camera_pos=(3.0, 3.0), range=5.0)

    assert len(detections) == 1
    assert detections[0].world_pos == (3.0, 4.0)


def test_thermal_detector_finds_hotspot():
    frame = np.zeros((6, 6), dtype=float)
    frame[2, 4] = 10.0
    detector = ThermalDetector(DetectorConfig(thermal_threshold=0.9))

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].world_pos == (4, 2)


def test_detection_fusion_updates_belief_grid():
    detector = SimulatedVictimDetector()
    detections = detector.detect(
        np.zeros((8, 8)),
        victims=[Victim(0, np.array([2.0, 5.0]))],
        camera_pos=(0.0, 0.0),
        range=10.0,
        confidence=0.8,
    )
    belief = np.zeros((8, 8), dtype=np.float32)

    fuse_detections_into_belief(belief, detections, confidence_floor=0.5)

    assert belief[5, 2] == 0.8
