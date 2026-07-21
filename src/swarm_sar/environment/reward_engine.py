"""Reward computation for one agent transition.

Design principles (see docs/reward_design.md for the full rationale and audit):

1. **Every dense term is bounded per step.** Cell-based signals (exploration,
   team coverage, duplicate search) are normalized by the sensor footprint, so
   each contributes at most |weight| per drone per step regardless of camera
   radius. Before normalization the duplicate-search penalty alone reached
   -19.7/step and out-massed a rescue (+10) by two orders of magnitude — the
   gradient told policies to avoid explored ground *harder* than to rescue.
2. **Mission events dominate.** A rescue is worth more than an entire episode
   of dense shaping; mission completion tops everything except safety.
3. **Rescue is time-critical (triage).** With ``time_critical_rescue`` the
   rescue reward scales with victim severity and decays over the episode, so
   the swarm is paid for *fast* triage-ordered rescues, not eventual ones.
4. **Sparse mode keeps only events** (detection/rescue/completion/safety), so
   reward ablations compare shaping against a meaningful baseline.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RewardInputs:
    exploration_cells: int = 0
    frontier_cells: int = 0
    team_new_cells: int = 0
    overlap_cells: int = 0
    # Cells inside the sensor footprint this step; normalizes the three counts
    # above so every dense term is O(weight) per step.
    sensor_footprint: int = 1
    new_victim_detected: bool = False
    victim_classified: bool = False
    victim_rescued: bool = False
    # Severity- and time-weighted multiplier for the rescue event, computed by
    # the environment at rescue time (1.0 when triage weighting is disabled).
    rescue_urgency: float = 1.0
    all_victims_rescued: bool = False
    new_information_broadcast: bool = False
    # This transition rescued a victim the rescuer first heard about by radio.
    comm_assisted_rescue: bool = False
    # Collision fires on contact ONSET only (the env tracks contact pairs);
    # lingering proximity is handled by near_collision shaping.
    collision: bool = False
    near_collision: bool = False
    safely_separated: bool = False
    hovering: bool = False
    idle: bool = False
    # Hazard: full penalty on ENTERING fire/smoke, small dwell penalty per
    # step spent inside — bounded gradient instead of a -5/step bleed.
    hazard_entered: bool = False
    hazard_dwell: bool = False
    battery_depleted: bool = False
    excessive_comm: bool = False
    progress_dist: float = 0.0
    commitment_steps: int = 0


class RewardEngine:
    def __init__(self, config):
        self.r = config

    def compute(self, inputs: RewardInputs) -> float:
        """Return the configured reward for one agent transition."""
        r = self.r
        reward = 0.0

        # Mission and safety events: present in both sparse and shaped modes.
        if inputs.new_victim_detected:
            reward += r.victim_detected
        if inputs.victim_classified:
            reward += r.victim_classified
        if inputs.victim_rescued:
            urgency = inputs.rescue_urgency if r.time_critical_rescue else 1.0
            reward += r.victim_rescued * urgency
            # Reward acting on received information (0 unless the comm-reward
            # experiment enables it), so communication has a downstream payoff.
            if inputs.comm_assisted_rescue:
                reward += r.comm_assisted_rescue
        if inputs.all_victims_rescued:
            reward += r.mission_complete
        if inputs.collision:
            reward += r.collision
        if inputs.hazard_entered:
            reward += r.hazard_penalty
        if inputs.hazard_dwell:
            reward += r.hazard_dwell
        if inputs.battery_depleted:
            reward += r.battery_depleted

        if r.mode != "sparse":
            foot = max(1, inputs.sensor_footprint)
            reward += r.time_penalty
            reward += r.explore_new_cell * (inputs.exploration_cells / foot)
            reward += r.coverage_new_cell * (inputs.team_new_cells / foot)
            reward += r.duplicate_explore * (inputs.overlap_cells / foot)
            reward += r.frontier_cell * inputs.frontier_cells   # bounded: 0..4 cells
            if inputs.new_information_broadcast:
                reward += r.useful_broadcast
            if inputs.excessive_comm:
                reward += r.excessive_comm
            if inputs.near_collision:
                reward += r.near_collision
            elif inputs.safely_separated:
                reward += r.separation
            if inputs.idle:
                reward += r.idle
            if inputs.hovering:
                reward += r.hover_penalty

            # Bounded dense progress term avoids a large reward from a single
            # noisy position update while still teaching victim approach.
            reward += r.approach_victim * float(np.clip(inputs.progress_dist, -1.0, 1.0))
            if inputs.commitment_steps > 0 and inputs.progress_dist > 0:
                reward += r.rescue_dwell

        return float(np.clip(reward, -100.0, 100.0))
