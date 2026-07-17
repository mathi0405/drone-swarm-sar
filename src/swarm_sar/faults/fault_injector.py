"""Randomly injects hardware/comm faults to evaluate swarm robustness.

Faults: full drone loss, GPS failure, camera failure, communication loss and
motor degradation. Each is sampled per-drone per-step from configurable rates.
The evaluator measures how gracefully the swarm degrades and recovers.
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

from swarm_sar.config import FaultConfig


class FaultInjector:
    def __init__(self, cfg: FaultConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.events: List[dict] = []

    def reset(self) -> None:
        self.events = []

    def step(self, drones, t: int) -> None:
        if not self.cfg.enable:
            return
        c = self.cfg
        for d in drones:
            s = d.faults
            if not d.mission.alive:
                continue
            if self.rng.random() < c.drone_failure_prob:
                d.mission.alive = False
                self._log(d.idx, "drone_failure", t)
                continue
            if s.gps_ok and self.rng.random() < c.gps_failure_prob:
                s.gps_ok = False; self._log(d.idx, "gps_failure", t)
            if s.camera_ok and self.rng.random() < c.camera_failure_prob:
                s.camera_ok = False; self._log(d.idx, "camera_failure", t)
            if s.comm_ok and self.rng.random() < c.comm_loss_prob:
                s.comm_ok = False; self._log(d.idx, "comm_loss", t)
            if not s.motor_degraded and self.rng.random() < c.motor_degradation_prob:
                s.motor_degraded = True; self._log(d.idx, "motor_degradation", t)
            # occasional recovery of transient faults (self-healing radios etc.)
            if not s.gps_ok and self.rng.random() < 0.05:
                s.gps_ok = True; self._log(d.idx, "gps_recovered", t)
            if not s.comm_ok and self.rng.random() < 0.05:
                s.comm_ok = True; self._log(d.idx, "comm_recovered", t)
            if not s.camera_ok and self.rng.random() < 0.05:
                s.camera_ok = True; self._log(d.idx, "camera_recovered", t)
            if s.motor_degraded and self.rng.random() < 0.05:
                s.motor_degraded = False; self._log(d.idx, "motor_recovered", t)

    def _log(self, drone: int, kind: str, t: int) -> None:
        self.events.append({"drone": drone, "type": kind, "step": t})

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.events:
            out[e["type"]] = out.get(e["type"], 0) + 1
        return out

    @staticmethod
    def fault_risk(drone) -> float:
        """Predictive health score from battery and degraded subsystems."""
        risk = 1.0 - float(drone.battery.soc)
        state = drone.faults
        risk += 0.2 if not state.gps_ok else 0.0
        risk += 0.2 if not state.camera_ok else 0.0
        risk += 0.25 if not state.comm_ok else 0.0
        risk += 0.15 if state.motor_degraded else 0.0
        return float(min(1.0, max(0.0, risk)))
