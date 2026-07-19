"""Per-drone state container and low-level command interface."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swarm_sar.battery.battery import Battery
from swarm_sar.config import EnvConfig
from swarm_sar.drone.dynamics import PointMassDynamics
from swarm_sar.sensors.sensors import SensorSuite

# Discrete action set (see docs/architecture.md)
ACTIONS = ["hover", "north", "south", "east", "west",
           "ascend", "descend", "rotate", "broadcast", "return_to_base"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
_MOVE = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}

@dataclass
class Kinematics:
    pos: np.ndarray                      # (x, y) cell coords (true)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    alt: float = 10.0                    # metres
    yaw: float = 0.0                     # radians

@dataclass
class FaultState:
    gps_ok: bool = True
    camera_ok: bool = True
    comm_ok: bool = True
    motor_degraded: bool = False

@dataclass
class MissionState:
    alive: bool = True
    charging: bool = False
    returning: bool = False


class _DroneStateView:
    """Backward-compatible view over a drone's kinematic and fault state.

    Earlier notebooks accessed ``drone.state.pos`` and ``drone.state.comm_ok``.
    The simulator now separates those concerns into dataclasses, but preserving
    this small proxy keeps existing analyses and user scripts working without
    duplicating state.
    """

    def __init__(self, drone: Drone):
        self._drone = drone

    @property
    def pos(self):
        return self._drone.kinematics.pos

    @pos.setter
    def pos(self, value):
        self._drone.kinematics.pos = np.asarray(value, dtype=float)

    @property
    def vel(self):
        return self._drone.kinematics.vel

    @vel.setter
    def vel(self, value):
        self._drone.kinematics.vel = np.asarray(value, dtype=float)

    @property
    def alt(self):
        return self._drone.kinematics.alt

    @alt.setter
    def alt(self, value):
        self._drone.kinematics.alt = float(value)

    @property
    def alive(self):
        return self._drone.mission.alive

    @alive.setter
    def alive(self, value):
        self._drone.mission.alive = bool(value)

    @property
    def comm_ok(self):
        return self._drone.faults.comm_ok

    @comm_ok.setter
    def comm_ok(self, value):
        self._drone.faults.comm_ok = bool(value)

    @property
    def gps_ok(self):
        return self._drone.faults.gps_ok

    @gps_ok.setter
    def gps_ok(self, value):
        self._drone.faults.gps_ok = bool(value)

    @property
    def camera_ok(self):
        return self._drone.faults.camera_ok

    @camera_ok.setter
    def camera_ok(self, value):
        self._drone.faults.camera_ok = bool(value)

class Drone:
    def __init__(self, idx: int, cfg: EnvConfig, rng: np.random.Generator, start_pos):
        self.idx = idx
        self.cfg = cfg
        self.rng = rng
        self.kinematics = Kinematics(pos=np.asarray(start_pos, dtype=float))
        self.faults = FaultState()
        self.mission = MissionState()
        self.battery = Battery(cfg.battery, dt=cfg.step_seconds)
        self.sensors = SensorSuite(cfg.sensors, rng,
                                   cell_size_m=cfg.world.cell_size_m)
        self.dyn = PointMassDynamics(cfg.max_speed_mps, cfg.motion_noise_std,
                                     cell_size_m=cfg.world.cell_size_m)
        self.last_action = 0
        self.idle_counter = 0
        self.comm_count = 0              # broadcasts this episode

    @property
    def state(self) -> _DroneStateView:
        """Compatibility alias for legacy notebooks and integrations."""
        return _DroneStateView(self)

    def gps_position(self) -> np.ndarray:
        """Position as *believed* by the drone (noisy / drifting / failed GPS)."""
        return self.sensors.gps.read(self.kinematics.pos, self.faults.gps_ok)

    def mark_broadcast(self, extra_energy: bool = False) -> None:
        """Record a broadcast, optionally charging only the radio energy."""
        self.comm_count += 1
        if extra_energy:
            self.battery.drain(self.cfg.battery.comm_draw_w * self.battery.dt_h)

    def apply(self, action: int, world, wind: np.ndarray, comm: bool = False) -> dict:
        """Apply a discrete action; return an info dict of what happened."""
        k = self.kinematics
        m = self.mission
        f = self.faults
        info = {"moved": False, "broadcast": False, "returned": False, "action": action}

        if not m.alive:
            return info

        name = ACTIONS[action] if 0 <= action < len(ACTIONS) else "hover"
        speed_scale = self.cfg.faults.motor_degradation_factor if f.motor_degraded else 1.0

        accel = np.zeros(2)
        if name in _MOVE:
            dx, dy = _MOVE[name]
            accel = np.array([dx, dy], dtype=float) * self.cfg.max_speed_mps
            info["moved"] = True
        elif name == "ascend":
            k.alt = min(self.cfg.world.max_altitude_m, k.alt + 2.0)
        elif name == "descend":
            k.alt = max(2.0, k.alt - 2.0)
        elif name == "rotate":
            k.yaw = (k.yaw + np.pi / 4) % (2 * np.pi)
        elif name == "broadcast":
            info["broadcast"] = True
        elif name == "return_to_base":
            m.returning = True
            info["returned"] = True

        if comm:
            info["broadcast"] = True

        if name in _MOVE:
            new_pos, new_vel = self.dyn.step(k.pos, k.vel, accel, wind, self.rng, speed_scale)
            # Enforce the SWEPT path, not just the endpoint: the move stops at
            # the last flyable point before any obstacle on the segment, so a
            # multi-cell step (or the RTB autopilot) can never tunnel through
            # a building.  Collision *penalties* are handled by the env.
            reached = world.clip_segment(k.pos, new_pos)
            if np.linalg.norm(reached - new_pos) > 1e-9:
                k.pos = reached
                k.vel = np.zeros(2)
                info["blocked"] = True
            else:
                k.pos, k.vel = new_pos, new_vel
        else:
            k.vel = np.zeros(2)

        # idle tracking
        self.idle_counter = self.idle_counter + 1 if name == "hover" else 0
        self.last_action = action
        if info["broadcast"]:
            self.mark_broadcast()

        # energy accounting
        self.battery.consume(name, k.alt, sensor_on=f.camera_ok, comm=info["broadcast"])
        return info
