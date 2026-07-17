"""Simplified quadrotor point-mass dynamics with wind & motion noise.

This is deliberately lightweight (Euler integration of a velocity command) so we
can run many parallel episodes on CPU. The AirSim adapter replaces this with the
full rigid-body model when high-fidelity sim-to-real evaluation is required.
"""
from __future__ import annotations

import numpy as np


class PointMassDynamics:
    def __init__(self, max_speed: float, motion_noise_std: float, dt: float = 1.0):
        self.max_speed = max_speed
        self.motion_noise_std = motion_noise_std
        self.dt = dt

    def step(self, pos, vel, accel_cmd, wind, rng, speed_scale: float = 1.0):
        """Integrate one step. Returns (pos, vel)."""
        vel = np.asarray(vel, dtype=float) + np.asarray(accel_cmd, dtype=float) * self.dt
        speed = np.linalg.norm(vel)
        vmax = self.max_speed * speed_scale
        if speed > vmax and speed > 1e-9:
            vel = vel / speed * vmax
        noise = rng.normal(0, self.motion_noise_std, size=len(vel))
        pos = np.asarray(pos, dtype=float) + (vel + wind[: len(vel)] + noise) * self.dt
        return pos, vel
