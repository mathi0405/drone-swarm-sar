"""AirSim / Unreal Engine backend adapter.

The self-contained :class:`SARSwarmEnv` is used for fast, reproducible training
and CI. For high-fidelity sim-to-real evaluation we swap the physics/sensing
backend for Microsoft AirSim running inside Unreal Engine, *without changing the
learning code*: this adapter exposes the same ``reset`` / ``step`` surface and
translates the swarm's discrete actions into AirSim MultirotorClient commands and
AirSim sensor streams back into the observation vectors.

AirSim is an optional dependency (``pip install -r requirements-sim.txt``). If it
is not installed, constructing the adapter raises a clear, actionable error.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from swarm_sar.config import EnvConfig
from swarm_sar.drone.drone import ACTIONS

try:  # optional heavy dependency
    import airsim  # type: ignore
    _HAS_AIRSIM = True
except Exception:  # pragma: no cover - exercised only without AirSim
    airsim = None
    _HAS_AIRSIM = False


# Mapping from our discrete actions to (vx, vy, vz) velocity setpoints (m/s).
_VEL = {
    "hover": (0, 0, 0), "north": (3, 0, 0), "south": (-3, 0, 0),
    "east": (0, 3, 0), "west": (0, -3, 0),
    "ascend": (0, 0, -2), "descend": (0, 0, 2), "rotate": (0, 0, 0),
    "broadcast": (0, 0, 0), "return_to_base": (0, 0, 0),
}


class AirSimSwarmEnv:
    """Drop-in high-fidelity backend mirroring :class:`SARSwarmEnv`'s API."""

    def __init__(self, cfg: EnvConfig, ip: str = "127.0.0.1", duration: float = 1.0):
        if not _HAS_AIRSIM:
            raise ImportError(
                "AirSim is not installed. Install the simulation extras with\n"
                "  pip install -r requirements-sim.txt\n"
                "and launch an AirSim/Unreal environment (see docs/installation.md)."
            )
        self.cfg = cfg
        self.duration = duration
        self.vehicles = [f"Drone{i}" for i in range(cfg.num_drones)]
        self.agents = [f"drone_{i}" for i in range(cfg.num_drones)]
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()
        for v in self.vehicles:
            self.client.enableApiControl(True, v)
            self.client.armDisarm(True, v)
            
        from swarm_sar.environment.sar_env import SARSwarmEnv
        self.virtual_env = SARSwarmEnv(cfg)
        
        # Optionally init YOLO (missing ultralytics OR missing model_path both
        # simply disable the perception add-on rather than crashing).
        try:
            from swarm_sar.perception.yolo_detector import YOLODetector
            self.detector = YOLODetector()
        except (ImportError, ValueError):
            self.detector = None

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self.client.reset()
        self.virtual_env.reset()
        for i, v in enumerate(self.vehicles):
            self.client.enableApiControl(True, v)
            self.client.armDisarm(True, v)
            # Add reset/start positions for multiple drones to prevent overlap
            pose = airsim.Pose(airsim.Vector3r(i * 2.0, 0.0, -1.0), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(pose, True, vehicle_name=v)
            self.client.takeoffAsync(vehicle_name=v)
        return {a: self._obs(i) for i, a in enumerate(self.agents)}, {a: {} for a in self.agents}

    def step(self, actions: Dict[str, int]):
        futures = []
        for i, a in enumerate(self.agents):
            name = ACTIONS[int(actions.get(a, 0))]
            vx, vy, vz = _VEL[name]
            futures.append(
                self.client.moveByVelocityAsync(vx, vy, vz, self.duration,
                                                vehicle_name=self.vehicles[i])
            )
        for f in futures:
            f.join()

        _, rewards, term, trunc, infos = self.virtual_env.step(actions)

        # Re-anchor the virtual twin on AirSim's authoritative poses so the two
        # backends cannot drift apart over the episode (rewards/detection are
        # computed from virtual positions).
        cell = self.virtual_env.world.cfg.cell_size_m
        S = self.virtual_env.world.size
        for i, v in enumerate(self.vehicles):
            st = self.client.getMultirotorState(vehicle_name=v)
            p = st.kinematics_estimated.position
            vel = st.kinematics_estimated.linear_velocity
            drone = self.virtual_env.drones[i]
            drone.kinematics.pos = np.clip(
                np.array([p.x_val, p.y_val]) / cell, 0.0, S - 1.0)
            drone.kinematics.vel = np.array([vel.x_val, vel.y_val]) / cell
            drone.kinematics.alt = float(-p.z_val)

        obs = {a: self._obs(i) for i, a in enumerate(self.agents)}

        for i, a in enumerate(self.agents):
            has_collided = self.client.simGetCollisionInfo(self.vehicles[i]).has_collided
            infos[a]["collision"] = infos[a].get("collision", False) or has_collided
            if has_collided:
                rewards[a] += self.cfg.reward.collision

        return obs, rewards, term, trunc, infos

    def _obs(self, i: int) -> np.ndarray:
        v = self.vehicles[i]
        st = self.client.getMultirotorState(vehicle_name=v)
        p = st.kinematics_estimated.position
        vel = st.kinematics_estimated.linear_velocity
        
        # Match observation shape with SARSwarmEnv
        base_obs = self.virtual_env._obs(i).copy()
        base_obs[0] = p.x_val
        base_obs[1] = p.y_val
        base_obs[2] = -p.z_val
        base_obs[3] = vel.x_val
        base_obs[4] = vel.y_val
        
        return base_obs

    def close(self):
        for v in self.vehicles:
            self.client.armDisarm(False, v)
            self.client.enableApiControl(False, v)
