from dataclasses import dataclass

import numpy as np


@dataclass
class RewardInputs:
    exploration_cells: int = 0
    frontier_cells: int = 0
    team_new_cells: int = 0
    new_victim_detected: bool = False
    victim_classified: bool = False
    victim_rescued: bool = False
    all_victims_rescued: bool = False
    overlap_cells: int = 0
    new_information_broadcast: bool = False
    collision: bool = False
    near_collision: bool = False
    safely_separated: bool = False
    hovering: bool = False
    idle: bool = False
    hazard: bool = False
    battery_depleted: bool = False
    excessive_comm: bool = False
    progress_dist: float = 0.0
    commitment_steps: int = 0


class RewardEngine:
    def __init__(self, config):
        self.r = config

    def compute(self, inputs: RewardInputs) -> float:
        """Return the configured reward for one agent transition.

        Detection, rescue, completion and safety events are present in both
        sparse and shaped modes.  Exploration, duplicate-search, idleness,
        time-pressure and progress terms are deliberately absent in sparse
        mode, making reward ablations meaningful.
        """
        r = self.r
        reward = 0.0

        # Mission events: retained in sparse rewards.
        if inputs.new_victim_detected:
            reward += r.victim_detected
        if inputs.victim_classified:
            reward += r.victim_classified
        if inputs.victim_rescued:
            reward += r.victim_rescued
        if inputs.all_victims_rescued:
            reward += r.mission_complete
        if inputs.collision:
            reward += r.collision
        if inputs.hazard:
            reward += r.hazard_penalty
        if inputs.battery_depleted:
            reward += r.battery_depleted

        if r.mode != "sparse":
            reward += r.time_penalty
            reward += r.explore_new_cell * inputs.exploration_cells
            reward += r.frontier_cell * inputs.frontier_cells
            reward += r.coverage_new_cell * inputs.team_new_cells
            reward += r.duplicate_explore * inputs.overlap_cells
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
