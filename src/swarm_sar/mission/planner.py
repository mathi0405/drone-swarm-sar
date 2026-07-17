"""Mission planner (per-drone FSM) and a heuristic swarm controller.

The :class:`HeuristicSwarmController` is a strong, non-learned baseline used for
(a) sanity-checking the environment, (b) generating the runnable demo and all
publication figures without a GPU, and (c) providing behaviour-cloning warm-starts
and a comparison point for the RL policies. It combines frontier-based exploration,
battery-aware return-to-charge, and cooperative victim assignment via the
pluggable :class:`~swarm_sar.mission.task_allocation.TaskAllocator`.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from swarm_sar.drone.drone import ACTION_TO_IDX, ACTIONS
from swarm_sar.mission.task_allocation import TaskAllocator

MOVE_ACTIONS = ("north", "south", "east", "west")


class MissionState(Enum):
    EXPLORE = "explore"
    RESCUE = "rescue"
    RETURN_CHARGE = "return_charge"
    CHARGING = "charging"


def _move_action_towards(src, dst) -> int:
    """Discrete move action that reduces distance to dst the most."""
    dx = dst[0] - src[0]; dy = dst[1] - src[1]
    if abs(dx) >= abs(dy):
        return ACTION_TO_IDX["east"] if dx > 0 else ACTION_TO_IDX["west"]
    return ACTION_TO_IDX["north"] if dy > 0 else ACTION_TO_IDX["south"]


class MissionPlanner:
    """Tiny finite-state machine deciding each drone's high-level intent."""

    def __init__(self):
        self.state = MissionState.EXPLORE

    def update(self, drone, has_target: bool, distance_to_charger: float | None = None) -> MissionState:
        s = drone.mission
        needs_return = drone.battery.needs_return
        if distance_to_charger is not None:
            needs_return = drone.battery.needs_return_for_distance(distance_to_charger)
        if needs_return or s.returning:
            self.state = (MissionState.CHARGING if s.charging else MissionState.RETURN_CHARGE)
        elif has_target:
            self.state = MissionState.RESCUE
        else:
            self.state = MissionState.EXPLORE
        return self.state


class HeuristicSwarmController:
    def __init__(self, env, strategy: str = "auction", seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.allocator = TaskAllocator(strategy, self.rng)
        self.planners = [MissionPlanner() for _ in range(env.n)]
        self._frontier_target: dict[int, np.ndarray] = {}
        self._broadcast_cooldown = np.zeros(env.n, dtype=int)

    # -- helpers ---------------------------------------------------------- #
    def _nearest_frontier(self, pos, explored) -> np.ndarray:
        S = self.env.world.size
        cx, cy = int(pos[0]), int(pos[1])
        flyable = self.env.world.flyable_mask
        valid = (~explored) & flyable
        ys, xs = np.nonzero(valid)
        if len(xs) > 0:
            dists = (xs - cx)**2 + (ys - cy)**2
            idx = int(np.argmin(dists))
            return np.array([xs[idx], ys[idx]], dtype=float)
        return np.array([self.rng.integers(S), self.rng.integers(S)], float)

    def _safe_action(self, i: int, preferred: int, target=None) -> int:
        mask = self.env.action_mask(i)
        if mask[preferred] > 0:
            return preferred
        candidates = [ACTION_TO_IDX[name] for name in MOVE_ACTIONS if mask[ACTION_TO_IDX[name]] > 0]
        if target is not None and candidates:
            best = min(candidates, key=lambda a: np.linalg.norm(
                self.env._predict_action_pos(i, ACTIONS[a]) - target
            ))
            return int(best)
        if mask[ACTION_TO_IDX["hover"]] > 0:
            return ACTION_TO_IDX["hover"]
        valid = np.flatnonzero(mask > 0)
        return int(valid[0]) if len(valid) else ACTION_TO_IDX["hover"]

    # -- main ------------------------------------------------------------- #
    def act(self) -> dict[str, int]:
        env = self.env
        actions: dict[str, int] = {}

        # cooperative victim assignment among alive drones
        victims = [v for v in env.world.victims if v.detected and not v.rescued]
        alive_idx = [i for i in range(env.n) if env.drones[i].mission.alive]
        assign: dict[int, int] = {}
        if victims and alive_idx:
            dpos = np.array([env.drones[i].kinematics.pos for i in alive_idx])
            vpos = np.array([v.pos for v in victims])
            batt = np.array([env.drones[i].battery.soc for i in alive_idx])
            chargers = np.array([c.pos for c in env.world.charging]) if env.world.charging else None
            raw = self.allocator.assign(dpos, vpos, battery=batt, charger_pos=chargers)
            for local_i, vlocal in raw.items():
                if vlocal is not None and 0 <= vlocal < len(victims):
                    assign[alive_idx[local_i]] = victims[vlocal].idx

        for i in range(env.n):
            d = env.drones[i]
            a = "hover"
            comm = None
            target_point = None
            if not d.mission.alive:
                actions[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue

            target_vid = assign.get(i, -1)
            has_target = target_vid >= 0
            st = env.world.nearest_charging(d.kinematics.pos)
            dist_to_charger = float(np.linalg.norm(d.kinematics.pos - st.pos)) if st is not None else None
            state = self.planners[i].update(d, has_target, dist_to_charger)

            if state in (MissionState.RETURN_CHARGE, MissionState.CHARGING):
                st = env.world.nearest_charging(d.kinematics.pos)
                if st is None:
                    actions[f"drone_{i}"] = ACTION_TO_IDX["hover"]
                    continue
                target_point = st.pos
                a = "hover" if np.linalg.norm(d.kinematics.pos - st.pos) <= 1.5 else \
                    ACTIONS[_move_action_towards(d.kinematics.pos, st.pos)]
            elif state == MissionState.RESCUE and has_target:
                vpos = env.world.victims[target_vid].pos
                target_point = vpos
                if np.linalg.norm(d.kinematics.pos - vpos) <= env.cfg.collision_radius_m + 1.0:
                    a = "hover"                       # dwell to rescue
                else:
                    a = ACTIONS[_move_action_towards(d.kinematics.pos, vpos)]
            else:  # EXPLORE
                tgt = self._frontier_target.get(i)
                if tgt is None or self.env.explored[i][int(tgt[1]), int(tgt[0])] or \
                        np.linalg.norm(d.kinematics.pos - tgt) < 2:
                    tgt = self._nearest_frontier(d.kinematics.pos, self.env.explored[i])
                    self._frontier_target[i] = tgt
                target_point = tgt
                a = ACTIONS[_move_action_towards(d.kinematics.pos, tgt)]

            # decide whether to broadcast (just discovered a victim, or periodic)
            just_found = any(v.detected_by == i and v.detected_step == env.t
                             for v in env.world.victims)
            self._broadcast_cooldown[i] = max(0, self._broadcast_cooldown[i] - 1)
            if (just_found or (victims and self._broadcast_cooldown[i] == 0)) and d.faults.comm_ok:
                comm = "victim"
                self._broadcast_cooldown[i] = 6

            move = ACTION_TO_IDX[a]
            move = self._safe_action(i, move, target_point)
            actions[f"drone_{i}"] = {"move": move, "comm": comm} if comm else move
        return actions
