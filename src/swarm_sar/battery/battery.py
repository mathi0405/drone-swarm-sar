"""Energy model: consumption depends on speed/altitude/comm/sensor usage,
with charging, emergency-return thresholds and capacity degradation (sim-to-real)."""
from __future__ import annotations

from swarm_sar.config import BatteryConfig


class Battery:
    def __init__(self, cfg: BatteryConfig, dt: float = 1.0):
        self.cfg = cfg
        self.dt_h = dt / 3600.0                     # step = 1 s -> hours
        self.capacity = cfg.capacity_wh
        self.energy = cfg.capacity_wh               # Wh remaining
        self.cycles = 0.0
        # Cumulative energy actually drawn from the pack this episode.  The
        # mission energy metric must count consumption, not net depletion —
        # otherwise recharging launders energy use out of the score.
        self.drawn_wh = 0.0

    @property
    def soc(self) -> float:
        """State of charge in [0, 1]."""
        return max(0.0, self.energy / max(self.capacity, 1e-6))

    @property
    def depleted(self) -> bool:
        return self.energy <= 0.0

    @property
    def needs_return(self) -> bool:
        return self.soc <= self.cfg.low_battery_frac

    def return_energy_fraction(self, distance_m: float) -> float:
        """Minimum state-of-charge needed to reach a charger plus reserve."""
        needed_wh = max(0.0, distance_m) * self.cfg.wh_per_meter_return
        return min(1.0, self.cfg.reserve_frac + needed_wh / max(self.capacity, 1e-6))

    def needs_return_for_distance(self, distance_m: float) -> bool:
        """Distance-aware return trigger used by mission planning."""
        return self.soc <= max(self.cfg.low_battery_frac, self.return_energy_fraction(distance_m))

    def consume(self, action_name: str, altitude: float, sensor_on: bool, comm: bool) -> float:
        c = self.cfg
        draw = c.hover_draw_w
        if action_name in ("north", "south", "east", "west", "ascend", "descend"):
            draw = c.move_draw_w
        if action_name == "ascend":
            draw += c.ascend_extra_w
        draw += altitude * 0.4                       # higher = more work against wind
        if sensor_on:
            draw += c.sensor_draw_w
        if comm:
            draw += c.comm_draw_w
        return self.drain(draw * self.dt_h)

    def drain(self, wh: float) -> float:
        """Draw ``wh`` from the pack (bounded by remaining energy); returns the
        amount actually drawn and adds it to the cumulative ``drawn_wh``."""
        actual = min(max(0.0, wh), self.energy)
        self.energy -= actual
        self.drawn_wh += actual
        return actual

    def charge(self) -> None:
        c = self.cfg
        before = self.energy
        self.energy = min(self.capacity, self.energy + c.charge_rate_w * self.dt_h)
        if self.energy >= self.capacity - 1e-6 and before < self.capacity:
            self.cycles += 1.0
            # capacity fade with cycles (battery ageing); energy must be
            # re-clamped to the FADED capacity or SoC exceeds 1.0.
            self.capacity = max(0.0, self.cfg.capacity_wh * (1.0 - self.cfg.degradation_per_cycle * self.cycles))
            self.energy = min(self.energy, self.capacity)
