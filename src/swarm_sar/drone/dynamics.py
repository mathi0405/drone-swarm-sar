"""Simplified quadrotor point-mass dynamics with wind & motion noise.

This is deliberately lightweight (Euler integration of a velocity command) so we
can run many parallel episodes on CPU. The AirSim adapter replaces this with the
full rigid-body model when high-fidelity sim-to-real evaluation is required.
"""
from __future__ import annotations

import numpy as np


class PointMassDynamics:
    """Positions live on the CELL grid; velocity, wind and noise are metric.

    ``dt`` is the kinematic sub-step (1 s of maneuvering per env step — the
    rest of ``step_seconds`` is sensing/decision time, which is why energy
    integrates over ``step_seconds`` while motion covers ~one cell).  The
    metre→cell conversion happens exactly once, here, when the metric
    displacement is applied to the grid position.
    """

    def __init__(self, max_speed: float, motion_noise_std: float, dt: float = 1.0,
                 cell_size_m: float = 1.0):
        self.max_speed = max_speed
        self.motion_noise_std = motion_noise_std
        self.dt = dt
        self.cell_size_m = max(1e-6, cell_size_m)

    def step(self, pos, vel, accel_cmd, wind, rng, speed_scale: float = 1.0):
        """Integrate one step. Returns (pos [cells], vel [m/s])."""
        vel = np.asarray(vel, dtype=float) + np.asarray(accel_cmd, dtype=float) * self.dt
        speed = np.linalg.norm(vel)
        vmax = self.max_speed * speed_scale
        if speed > vmax and speed > 1e-9:
            vel = vel / speed * vmax
        noise = rng.normal(0, self.motion_noise_std, size=len(vel))
        disp_m = (vel + wind[: len(vel)] + noise) * self.dt
        pos = np.asarray(pos, dtype=float) + disp_m / self.cell_size_m
        return pos, vel
