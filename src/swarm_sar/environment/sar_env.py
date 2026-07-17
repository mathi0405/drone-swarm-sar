"""SARSwarmEnv - a decentralized multi-agent search-and-rescue environment.

API mirrors PettingZoo's *parallel* interface (``reset`` / ``step`` returning
per-agent dicts) so it plugs directly into RLlib's multi-agent stack, while
remaining runnable with only NumPy for lightweight experimentation and CI.

Observation (per drone, decentralized / partial): an 8-frame temporal stack of
[own state, victim-belief summary, locally sensed fire/smoke hazards, k-nearest
in-range peers with intent, comm-link summary] (see
:class:`~swarm_sar.environment.observation.ObservationEngine`). Actions are the
10 discrete commands in ``swarm_sar.drone.ACTIONS``. Rewards implement the full
shaped objective from the research design (exploration, detection, rescue,
communication, safety, energy).
"""
from __future__ import annotations

import numpy as np

from swarm_sar.communication.comms import CommNetwork, Message
from swarm_sar.config import EnvConfig, load_config
from swarm_sar.drone.drone import ACTION_TO_IDX, ACTIONS, Drone
from swarm_sar.environment.entities import CellType
from swarm_sar.environment.observation import HISTORY_LEN, ObservationEngine
from swarm_sar.environment.reward_engine import RewardEngine, RewardInputs
from swarm_sar.environment.world import (
    BLOCKS_FLIGHT_VALUES,
    OCCLUDES_VALUES,
    SARWorld,
)
from swarm_sar.faults.fault_injector import FaultInjector
from swarm_sar.mission.planner import _move_action_towards
from swarm_sar.mission.task_allocation import TaskAllocator
from swarm_sar.utils.geometry import disk_visibility_offsets
from swarm_sar.utils.seeding import make_rng


class SARSwarmEnv:
    metadata = {"name": "swarm_sar_v0", "render_modes": ["state"]}

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None):
        self.cfg = cfg or EnvConfig()
        self.reward_engine = RewardEngine(self.cfg.reward)
        self.rng = make_rng(seed)
        self.n = self.cfg.num_drones
        self.agents = [f"drone_{i}" for i in range(self.n)]
        self.possible_agents = list(self.agents)
        self.n_actions = len(ACTIONS)
        self._patch = self.cfg.obs_patch
        self.obs_dim = HISTORY_LEN * self._compute_obs_dim()
        self._built = False
        self.obs_engine = ObservationEngine(self.cfg, self.n)
        self.allocator = TaskAllocator(self.cfg.task_allocation_strategy, self.rng)

    _MOVE_DELTAS = {
        "north": np.array([0.0, 1.0]),
        "south": np.array([0.0, -1.0]),
        "east": np.array([1.0, 0.0]),
        "west": np.array([-1.0, 0.0]),
    }

    # ------------------------------------------------------------------ #
    # spaces                                                             #
    # ------------------------------------------------------------------ #
    def _compute_obs_dim(self) -> int:
        # Match ObservationEngine frame layout: own state (8) + victim info (9)
        # + hazards (4) + egocentric map patch (3 channels x P x P) + LiDAR (8)
        # + peers (6 * k) + comms (4).
        p = self.cfg.obs_map_patch
        return 8 + 9 + 4 + (3 * p * p) + 8 + (6 * self.cfg.k_nearest_drones) + 4

    @property
    def action_space_n(self) -> int:
        return self.n_actions

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.total_explore = 0.0
        self.total_progress = 0.0
        self.total_commitment = 0.0
        self.total_collision = 0.0
        self.total_hazard = 0.0
        self.total_idle = 0.0
        self.total_overlap = 0.0

        if seed is not None:
            self.rng = make_rng(seed)
        if self.cfg.world.randomize_each_episode or not self._built:
            self.world = SARWorld(self.cfg.world, self.rng)
        else:
            # Reset entity states without regenerating the map
            for v in self.world.victims:
                v.detected = False; v.rescued = False; v._dwell = 0
                v._confirmations = 0; v.detected_by = -1; v.rescued_by = -1
                v.classified = False
                v.assigned_drone = -1
                v.detected_step = -1; v.rescued_step = -1
                if hasattr(v, '_dwell_by'):
                    v._dwell_by.clear()
            for o in self.world.dyn_obstacles:
                o.pos = np.array(self.world._free_cell(self.rng), dtype=float)
                o.vel = self.rng.uniform(-0.8, 0.8, size=2)
        self._built = True

        # spawn drones near a (random) charging station
        base = self.world.charging[0].pos
        self.drones: list[Drone] = []
        starts: list[np.ndarray] = []
        phase = self.rng.uniform(0, 2 * np.pi)
        for i in range(self.n):
            start = self._safe_spawn_position(base, starts, i, phase)
            starts.append(start)
            self.drones.append(Drone(i, self.cfg, self.rng, start))

        self.comm = CommNetwork(self.cfg.comm, self.n, self.rng)
        self.comm.reset()
        self.faults = FaultInjector(self.cfg.faults, self.rng)
        self.faults.reset()
        self.obs_engine.reset()

        S = self.world.size
        self.global_explored = np.zeros((S, S), dtype=bool)
        self.explored = [np.zeros((S, S), dtype=bool) for _ in range(self.n)]
        self.victim_belief = [np.zeros((S, S), dtype=np.float32) for _ in range(self.n)]
        self.wind = self.rng.uniform(-1, 1, size=2) * self.cfg.wind_max_mps
        self.visibility = 1.0
        self.gps_noise_scale = 1.0
        self.environment_events: list[dict] = []

        self.t = 0
        self.agents = list(self.possible_agents)
        self.history: list[dict] = []
        self.collisions: list[dict] = []
        self.comm_events = 0
        self._prev_found = 0
        self._last_auto_comm = np.full(self.n, -10_000, dtype=int)
        self._last_comm_explored = np.zeros(self.n, dtype=int)
        self._hover_counts = np.zeros(self.n, dtype=int)
        # Per-drone chronological list of newly explored (y, x) cells since the
        # drone's last broadcast, and the victim ids it has already shared.
        self._new_cells_since_comm: list[list[list]] = [[] for _ in range(self.n)]
        self._victims_shared: list[set] = [set() for _ in range(self.n)]
        self.allocator = TaskAllocator(self.cfg.task_allocation_strategy, self.rng)
        # Event-based safety tracking: contact pairs currently touching (so
        # collisions penalize the ONSET only) and who was in a hazard cell
        # last step (so hazard entry and dwell are separate signals).
        self._contacts: set = set()
        self._in_hazard = np.zeros(self.n, dtype=bool)

        self._refresh_world_masks()
        self._sense_all()
        obs, _ = self.obs_engine.get_obs(self)
        self._record_frame({a: 0 for a in self.agents})
        infos = {a: {} for a in self.agents}
        infos["global_state"] = self.global_state()
        return obs, infos

    def _safe_spawn_position(self, base: np.ndarray, starts: list[np.ndarray],
                             idx: int, phase: float) -> np.ndarray:
        sep = max(self.cfg.min_spawn_separation_m, self.cfg.collision_radius_m * 2.0)
        S = self.world.size
        if not starts:
            center = np.clip(np.asarray(base, float), sep, S - 1 - sep)
            if self.world.is_flyable(center[0], center[1]):
                return center
        base = np.asarray(base, float)
        radius0 = max(sep, sep * self.n / (2 * np.pi))
        for ring in range(1, 8):
            radius = radius0 + ring * sep * 0.5
            for offset in range(max(8, self.n * 3)):
                angle = phase + 2 * np.pi * ((idx + offset / max(8, self.n * 3)) / max(1, self.n))
                cand = base + radius * np.array([np.cos(angle), np.sin(angle)])
                cand = np.clip(cand, 1, S - 2)
                if not self.world.is_flyable(cand[0], cand[1]):
                    continue
                if all(np.linalg.norm(cand - p) >= sep for p in starts):
                    return cand.astype(float)
        for _ in range(1000):
            cand = self.rng.uniform(1, S - 2, size=2)
            if self.world.is_flyable(cand[0], cand[1]) and \
                    all(np.linalg.norm(cand - p) >= sep for p in starts):
                return cand.astype(float)
        return np.clip(base + self.rng.uniform(-sep, sep, size=2), 1, S - 2)

    def _action_index(self, action) -> int:
        if isinstance(action, str):
            return ACTION_TO_IDX.get(action, ACTION_TO_IDX["hover"])
        return int(action)

    def _split_action(self, action) -> tuple[int, object]:
        """Support old integer actions and new movement+communication actions."""
        comm = None
        move = action
        if isinstance(action, dict):
            move = action.get("move", action.get("action", ACTION_TO_IDX["hover"]))
            comm = action.get("comm", action.get("broadcast", None))
        elif isinstance(action, (tuple, list)) and len(action) == 2:
            move, comm = action

        move_idx = self._action_index(move)
        name = ACTIONS[move_idx] if 0 <= move_idx < len(ACTIONS) else "hover"
        if name == "broadcast":
            move_idx = ACTION_TO_IDX["hover"]
            comm = "victim"
        return move_idx, comm

    @staticmethod
    def _comm_requested(comm) -> bool:
        if comm is None or comm is False:
            return False
        if isinstance(comm, str) and comm.lower() in ("", "none", "false", "0"):
            return False
        return bool(comm)

    def action_mask(self, i: int) -> np.ndarray:
        """Mask actions that would immediately enter obstacles or another drone."""
        allowed = set(self.cfg.allowed_actions)
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for name in allowed:
            if name in ACTION_TO_IDX:
                mask[ACTION_TO_IDX[name]] = 1.0
        d = self.drones[i]
        if not d.mission.alive:
            mask[:] = 0.0
            mask[ACTION_TO_IDX["hover"]] = 1.0
            return mask
        if d.mission.returning:
            # Return-to-base is a committed macro-action: the autopilot flies
            # the drone home, so the ONLY action the policy may sample is
            # ``return_to_base``.  Without this, PPO stored sampled actions and
            # log-probs for moves the environment overrode — off-policy
            # garbage in every rollout containing a returning drone.
            mask[:] = 0.0
            mask[ACTION_TO_IDX["return_to_base"]] = 1.0
            return mask

        has_peer_in_range = False
        for j, other in enumerate(self.drones):
            if i != j and other.mission.alive and other.faults.comm_ok:
                if np.linalg.norm(d.kinematics.pos - other.kinematics.pos) <= self.cfg.comm.range_m:
                    has_peer_in_range = True
                    break

        if not has_peer_in_range and "broadcast" in allowed:
            mask[ACTION_TO_IDX["broadcast"]] = 0.0

        for name in self._MOVE_DELTAS:
            idx = ACTION_TO_IDX[name]
            if mask[idx] <= 0:
                continue
            cand = self._predict_action_pos(i, name)
            if not self._segment_is_flyable(d.kinematics.pos, cand):
                mask[idx] = 0.0
                continue
            for j, other in enumerate(self.drones):
                if j == i or not other.mission.alive:
                    continue
                dist = self._point_segment_distance(other.kinematics.pos, d.kinematics.pos, cand)
                if dist < max(self.cfg.collision_radius_m, self.cfg.safety_margin_m):
                    mask[idx] = 0.0
                    break

        # If momentum or wind makes all actions illegal, fallback to hover to prevent NaN probabilities
        if mask.sum() == 0:
            mask[ACTION_TO_IDX["hover"]] = 1.0
        return mask

    def action_masks(self) -> np.ndarray:
        return np.stack([self.action_mask(i) for i in range(self.n)], axis=0)

    def _predict_action_pos(self, i: int, action_name: str) -> np.ndarray:
        d = self.drones[i]
        if action_name not in self._MOVE_DELTAS:
            return d.kinematics.pos.copy()
        accel = self._MOVE_DELTAS[action_name] * self.cfg.max_speed_mps
        vel = d.kinematics.vel + accel
        speed = np.linalg.norm(vel)
        if speed > self.cfg.max_speed_mps and speed > 1e-9:
            vel = vel / speed * self.cfg.max_speed_mps
        return d.kinematics.pos + vel + self.wind[:2]

    def _segment_is_flyable(self, start: np.ndarray, end: np.ndarray) -> bool:
        steps = max(1, int(np.ceil(np.linalg.norm(end - start))))
        for k in range(1, steps + 1):
            p = start + (end - start) * (k / steps)
            if not self.world.is_flyable(p[0], p[1]):
                return False
        return True

    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        seg = end - start
        denom = float(np.dot(seg, seg))
        if denom <= 1e-9:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, seg) / denom, 0.0, 1.0))
        closest = start + t * seg
        return float(np.linalg.norm(point - closest))

    def step(self, actions: dict[str, int]):
        self.t += 1
        self.comm.edges_this_step = []          # collect this step's comm links
        rewards = {a: 0.0 for a in self.agents}
        self.reward_inputs = {a: RewardInputs() for a in self.agents}
        infos = {a: {} for a in self.agents}
        r = self.cfg.reward
        shaped = r.mode != "sparse"   # sparse = event rewards only (reward-ablation)
        prev_pos = [d.kinematics.pos.copy() for d in self.drones]
        parsed_actions: dict[int, tuple[int, object]] = {}

        # gust of wind varies over time (sim-to-real)
        self.wind += self.rng.normal(0, 0.2, size=2)
        self.wind = np.clip(self.wind, -self.cfg.wind_max_mps, self.cfg.wind_max_mps)
        self._apply_weather_profile()

        self._update_victim_assignments()

        # 1) apply actions & dynamics
        for i, a in enumerate(self.agents):
            d = self.drones[i]
            if not d.mission.alive:
                continue
            act, comm_req = self._split_action(actions.get(a, 0))
            # Return-to-base autopilot overrides the policy action so a
            # returning drone moves exactly once per step (previously the
            # autopilot applied a *second* move on top of the policy's).
            if d.mission.returning and not d.mission.charging:
                st = self.world.nearest_charging(d.kinematics.pos)
                if st is not None and np.linalg.norm(d.kinematics.pos - st.pos) > 1.5:
                    act = _move_action_towards(d.kinematics.pos, st.pos)
            parsed_actions[i] = (act, comm_req)
            d.apply(act, self.world, self.wind, comm=self._comm_requested(comm_req))
            if d.battery.depleted and d.mission.alive:
                d.mission.alive = False
                self.reward_inputs[a].battery_depleted = True

        # 1b) hover penalties
        for i, a in enumerate(self.agents):
            if not self.drones[i].mission.alive: continue
            action_name = ACTIONS[parsed_actions.get(i, (0, None))[0]]
            if action_name == "hover":
                self._hover_counts[i] += 1
                if self._hover_counts[i] > 5:
                    self.reward_inputs[a].hovering = True
            else:
                self._hover_counts[i] = 0

        # 2) advance world & comms
        env_events = self.world.step_dynamic()
        if any(env_events.values()):
            self.environment_events.append({"step": self.t, **env_events})
        self._refresh_world_masks()      # fire/smoke spread changes occlusion
        self.faults.step(self.drones, self.t)
        self.comm.step(self.t)

        # 3) sensing / exploration / detection rewards
        sense_rad = max(1, int(self.cfg.sensors.camera_range_m * self.visibility
                               / self.world.cfg.cell_size_m))
        footprint = len(disk_visibility_offsets(sense_rad)[0])
        for i, a in enumerate(self.agents):
            new_cells, new_team_cells, dup, frontiers = self._sense_one(i)
            self.reward_inputs[a].sensor_footprint = footprint
            if shaped:
                self.reward_inputs[a].exploration_cells += new_cells
                self.reward_inputs[a].team_new_cells += new_team_cells
                self.reward_inputs[a].overlap_cells += dup
                self.reward_inputs[a].frontier_cells += frontiers

        # 4) detection & rescue
        for i, a in enumerate(self.agents):
            det, res, dwell = self._detect_and_rescue(i)
            self.reward_inputs[a].new_victim_detected = det > 0
            self.reward_inputs[a].victim_rescued = res > 0
            if shaped:
                self.reward_inputs[a].progress_dist = self._rescue_progress_delta(i, prev_pos[i])
                self.reward_inputs[a].commitment_steps = dwell
                if self.drones[i].idle_counter > 3 and not self._near_detected_victim(i):
                    self.reward_inputs[a].idle = True

            x, y = np.asarray(self.drones[i].kinematics.pos, dtype=int)
            x = int(np.clip(x, 0, self.world.size - 1))
            y = int(np.clip(y, 0, self.world.size - 1))
            in_hazard = self.world.grid[y, x] in (CellType.FIRE, CellType.SMOKE)
            self.reward_inputs[a].hazard_entered = in_hazard and not self._in_hazard[i]
            self.reward_inputs[a].hazard_dwell = in_hazard and self._in_hazard[i]
            self._in_hazard[i] = in_hazard

        # Reallocate immediately after a new detection.  Assignment is
        # battery-aware and one-to-one, avoiding multiple drones converging on
        # a single victim while other victims remain unattended.
        self._update_victim_assignments()

        # 4b) sensor false positives + belief decay (precision < 1)
        self._inject_false_positives()

        # 5) communication after sensing, so messages include new detections
        for i, a in enumerate(self.agents):
            explicit = self._comm_requested(parsed_actions.get(i, (0, None))[1])
            auto = self.cfg.comm.auto_broadcast and self._should_auto_broadcast(i)
            if explicit or auto:
                if auto and not explicit:
                    self.drones[i].mark_broadcast(extra_energy=True)
                # Rewarded only for EXPLICIT broadcasts whose payload contains
                # NEW information (fresh cells or newly shared victims).
                # Automatic heartbeat broadcasts still deliver the data but
                # earn nothing — otherwise the policy learns the broadcast
                # action is redundant and abandons it.
                if self._handle_broadcast(i) and shaped and explicit:
                    self.reward_inputs[a].new_information_broadcast = True
                self._last_auto_comm[i] = self.t
                self._last_comm_explored[i] = int(self.explored[i].sum())
                if self.drones[i].comm_count > self.cfg.comm.excessive_comm_threshold:
                    self.reward_inputs[a].excessive_comm = True

        # 6) charging
        self._handle_charging()

        # 7) collisions and separation
        self._handle_collisions()
        if shaped:
            self._handle_separation_reward()

        self._prev_found = sum(v.detected for v in self.world.victims)

        # 9) termination
        all_rescued = all(v.rescued for v in self.world.victims)
        alive_any = any(d.mission.alive for d in self.drones)
        truncated = self.t >= self.cfg.max_steps
        terminated = all_rescued or not alive_any
        if all_rescued:
            for a in self.agents:
                self.reward_inputs[a].all_victims_rescued = True

        for a in self.agents:
            inputs = self.reward_inputs[a]
            rewards[a] = self.reward_engine.compute(inputs)

            # Track components for debugging (mirrors RewardEngine normalization)
            if shaped:
                foot = max(1, inputs.sensor_footprint)
                self.total_explore += (r.explore_new_cell * inputs.exploration_cells / foot
                                       + r.frontier_cell * inputs.frontier_cells
                                       + r.coverage_new_cell * inputs.team_new_cells / foot)
                self.total_overlap += r.duplicate_explore * inputs.overlap_cells / foot
                if inputs.idle:
                    self.total_idle += r.idle
                self.total_progress += r.approach_victim * float(np.clip(inputs.progress_dist, -1.0, 1.0))
                if inputs.commitment_steps > 0 and inputs.progress_dist > 0:
                    self.total_commitment += r.rescue_dwell
            if inputs.collision:
                self.total_collision += r.collision
            if inputs.hazard_entered:
                self.total_hazard += r.hazard_penalty
            if inputs.hazard_dwell:
                self.total_hazard += r.hazard_dwell

        if (terminated or truncated) and self.cfg.log_reward_breakdown:
            print("\nReward Breakdown")
            print(f"Exploration : {self.total_explore:.2f}")
            print(f"Progress    : {self.total_progress:.2f}")
            print(f"Commitment  : {self.total_commitment:.2f}")
            print(f"Collision   : {self.total_collision:.2f}")
            print(f"Hazard      : {self.total_hazard:.2f}")
            print(f"Idle        : {self.total_idle:.2f}")
            print(f"Overlap     : {self.total_overlap:.2f}")

        # Merge inboxes for all agents deterministically before computing observations
        for i in range(self.n):
            self._merge_inbox(i)
        obs, _ = self.obs_engine.get_obs(self)

        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        frame_actions = {a: parsed_actions.get(i, (0, None))[0]
                         for i, a in enumerate(self.possible_agents)}
        self._record_frame(frame_actions)
        # One shared (read-only) info dict — the content is identical per agent.
        shared_info = self._agent_info(frame_actions)
        infos = {a: shared_info for a in self.agents}
        # Compact privileged state for the centralized critic (CTDE).
        infos["global_state"] = self.global_state()
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def global_state(self) -> np.ndarray:
        """Compact PRIVILEGED state for the centralized critic (training only).

        Under CTDE the critic may see ground truth the actors cannot: true
        victim positions with detected/rescued flags, per-drone pose, battery
        and mission state, plus global mission progress and the episode clock.
        ~50-80 dims instead of the concatenated observation stacks (thousands
        of dims), which the critic had to re-learn perception from.  Victim
        slots are padded to ``global_state_max_victims`` so the dimension is
        constant across curriculum stages.
        """
        S = float(self.world.size)
        max_v = self.cfg.global_state_max_victims or self.cfg.world.n_victims
        feats: list[float] = []
        for v in self.world.victims[:max_v]:
            feats += [v.pos[0] / S, v.pos[1] / S, float(v.detected), float(v.rescued)]
        feats += [0.0] * (4 * max(0, max_v - len(self.world.victims)))
        for d in self.drones:
            feats += [d.kinematics.pos[0] / S, d.kinematics.pos[1] / S,
                      d.battery.soc, float(d.mission.alive), float(d.mission.returning)]
        n_v = max(1, len(self.world.victims))
        feats += [float(self.global_explored.mean()),
                  self.t / max(1, self.cfg.max_steps),
                  sum(v.detected for v in self.world.victims) / n_v,
                  sum(v.rescued for v in self.world.victims) / n_v]
        return np.asarray(feats, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # sensing / detection                                               #
    # ------------------------------------------------------------------ #
    def _sense_all(self):
        for i in range(self.n):
            self._sense_one(i) # updated for frontiers


    def _update_victim_assignments(self):
        """Allocate detected victims using distance, battery and return risk."""
        victims = [v for v in self.world.victims if v.detected and not v.rescued]
        alive_indices = [i for i, d in enumerate(self.drones) if d.mission.alive]
        for victim in victims:
            victim.assigned_drone = -1
        if not victims or not alive_indices:
            return

        drone_pos = np.asarray([self.drones[i].kinematics.pos for i in alive_indices], dtype=float)
        victim_pos = np.asarray([v.pos for v in victims], dtype=float)
        battery = np.asarray([self.drones[i].battery.soc for i in alive_indices], dtype=float)
        chargers = (np.asarray([station.pos for station in self.world.charging], dtype=float)
                    if self.world.charging else None)
        assignment = self.allocator.assign(drone_pos, victim_pos, battery=battery, charger_pos=chargers)
        for local_drone, local_victim in assignment.items():
            if local_victim >= 0:
                victims[local_victim].assigned_drone = alive_indices[local_drone]

    def _refresh_world_masks(self) -> None:
        """Cache per-step boolean views of the grid used by sensing and obs."""
        self._occ_mask = np.isin(self.world.grid, OCCLUDES_VALUES)
        self._blocked_mask = np.isin(self.world.grid, BLOCKS_FLIGHT_VALUES)

    def _sense_one(self, i: int) -> tuple[int, int, int, int]:
        """Vectorized sensing sweep with exact Bresenham line-of-sight.

        Semantics are identical to the per-cell ``world.line_of_sight`` loop,
        but the lines are precomputed per radius (translation-invariant) so the
        sweep is a handful of NumPy fancy-indexing operations.
        """
        d = self.drones[i]
        if not d.mission.alive:
            return 0, 0, 0, 0
        rad = int(
            self.cfg.sensors.camera_range_m
            * getattr(self, "visibility", 1.0)
            / self.world.cfg.cell_size_m
        )
        rad = max(1, rad)
        cx, cy = int(d.kinematics.pos[0]), int(d.kinematics.pos[1])
        S = self.world.size

        offs, line_y, line_x, line_valid = disk_visibility_offsets(rad)
        ys = cy + offs[:, 0]
        xs = cx + offs[:, 1]
        in_bounds = (xs >= 0) & (xs < S) & (ys >= 0) & (ys < S)
        # Intermediate line cells lie inside the bounding box of the endpoints,
        # so for in-bounds targets they are in bounds too; the clip only guards
        # indexing for targets that in_bounds already discards.
        ly = np.clip(cy + line_y, 0, S - 1)
        lx = np.clip(cx + line_x, 0, S - 1)
        blocked = (self._occ_mask[ly, lx] & line_valid).any(axis=1)
        visible = in_bounds & ~blocked

        vy, vx = ys[visible], xs[visible]
        already = self.explored[i][vy, vx]
        ny, nx = vy[~already], vx[~already]
        dup = int(already.sum())
        new_cells = int(ny.size)
        if new_cells:
            self.explored[i][ny, nx] = True
            team_new = ~self.global_explored[ny, nx]
            self.global_explored[ny, nx] = True
            new_team_cells = int(team_new.sum())
            self._new_cells_since_comm[i].extend(
                [int(y), int(x)] for y, x in zip(ny, nx))
        else:
            new_team_cells = 0

        frontiers = 0
        if new_cells > 0:
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                fx, fy = cx+dx, cy+dy
                if 0 <= fx < S and 0 <= fy < S and not self.explored[i][fy, fx]:
                    frontiers += 1
        return new_cells, new_team_cells, dup, frontiers

    def _detect_and_rescue(self, i: int) -> tuple[int, int, int]:
        d = self.drones[i]
        if not d.mission.alive:
            return 0, 0, 0
        det = res = dwell = 0
        rescue_radius = self.cfg.collision_radius_m + 1.0
        for v in self.world.victims:
            if v.rescued:
                continue
            dist = np.linalg.norm(d.kinematics.pos - v.pos)
            visible = d.sensors.camera.detect(d.kinematics.pos, v.pos, self.world,
                                              d.faults.camera_ok, visibility=self.visibility)
            if visible:
                v._confirmations += 1
                self.victim_belief[i][int(v.pos[1]), int(v.pos[0])] = min(
                    1.0, 0.5 + 0.25 * v._confirmations
                )
                if not v.detected and v._confirmations >= self.cfg.confirmations_required:
                    v.detected = True; v.detected_by = i; v.detected_step = self.t
                    v.assigned_drone = i
                    det += 1
                if v.detected and v._confirmations >= self.cfg.confirmations_required + 2 and not v.classified:
                    v.classified = True
                    self.reward_inputs[f"drone_{i}"].victim_classified = True
            if v.detected and dist <= rescue_radius:
                # The assigned drone has priority, but a victim must never be
                # blocked outright: any drone may rescue when the assignee is
                # absent (dead, unassigned or not at the victim).
                assigned = getattr(v, "assigned_drone", -1)
                if assigned is not None and assigned >= 0 and assigned != i:
                    ad = self.drones[assigned]
                    if ad.mission.alive and \
                            np.linalg.norm(ad.kinematics.pos - v.pos) <= rescue_radius:
                        continue
                if not hasattr(v, '_dwell_by'):
                    v._dwell_by = {}
                v._dwell_by[i] = v._dwell_by.get(i, 0) + 1
                v._dwell = max(v._dwell_by.values())
                dwell += 1
                if v._dwell >= self.cfg.dwell_steps_to_rescue:
                    v.rescued = True
                    v.rescued_by = max(v._dwell_by, key=v._dwell_by.get)
                    v.rescued_step = self.t
                    res += 1
                    # Triage urgency: severe victims rescued early pay full
                    # value; late or low-severity rescues pay less (never
                    # below the configured floor).
                    floor = self.cfg.reward.rescue_urgency_floor
                    decay = max(0.0, 1.0 - self.t / max(1, self.cfg.max_steps))
                    self.reward_inputs[f"drone_{i}"].rescue_urgency = (
                        (floor + (1.0 - floor) * decay)
                        * (0.5 + 0.5 * float(v.severity))
                    )
            elif v.detected and not v.rescued:
                # Reset per-drone dwell if this drone moved away (continuous presence required)
                if hasattr(v, '_dwell_by') and i in v._dwell_by:
                    del v._dwell_by[i]
                    v._dwell = max(v._dwell_by.values()) if v._dwell_by else 0
        return det, res, dwell

    def _near_detected_victim(self, i: int) -> bool:
        d = self.drones[i]
        if not d.mission.alive:
            return False
        radius = self.cfg.collision_radius_m + 1.0
        return any(v.detected and not v.rescued and np.linalg.norm(d.kinematics.pos - v.pos) <= radius
                   for v in self.world.victims)

    def _rescue_progress_delta(self, i: int, prev_pos: np.ndarray) -> float:
        """Bounded distance improvement toward a known victim for reward shaping."""
        d = self.drones[i]
        targets = [v for v in self.world.victims if v.detected and not v.rescued]
        if not d.mission.alive or not targets:
            return 0.0
        victim = min(targets, key=lambda v: np.linalg.norm(d.kinematics.pos - v.pos))
        before = np.linalg.norm(prev_pos - victim.pos)
        after = np.linalg.norm(d.kinematics.pos - victim.pos)
        return float(np.clip(before - after, -1.0, 1.0))

    def _inject_false_positives(self):
        """Camera precision < 1: occasional phantom detections, and belief decay so
        stale information fades (uncertainty grows when a region is left)."""
        fp = self.cfg.sensors.detection_false_positive
        S = self.world.size
        for i, d in enumerate(self.drones):
            self.victim_belief[i] *= 0.98
            if d.mission.alive and d.faults.camera_ok and self.rng.random() < fp:
                cx = int(np.clip(d.kinematics.pos[0] + self.rng.integers(-4, 5), 0, S - 1))
                cy = int(np.clip(d.kinematics.pos[1] + self.rng.integers(-4, 5), 0, S - 1))
                self.victim_belief[i][cy, cx] = max(self.victim_belief[i][cy, cx], 0.6)

    def _should_auto_broadcast(self, i: int) -> bool:
        d = self.drones[i]
        if not d.mission.alive or not d.faults.comm_ok:
            return False
        if self.t - self._last_auto_comm[i] < max(1, self.cfg.comm.heartbeat_interval_steps):
            just_found = any(v.detected_by == i and v.detected_step == self.t
                             for v in self.world.victims)
            if not just_found:
                return False
        just_found = any(v.detected_by == i and v.detected_step == self.t
                         for v in self.world.victims)
        coverage_delta = int(self.explored[i].sum()) - int(self._last_comm_explored[i])
        has_target = any(v.detected and not v.rescued for v in self.world.victims)
        return just_found or has_target or coverage_delta >= self.cfg.comm.coverage_delta_threshold

    # ------------------------------------------------------------------ #
    # comms                                                              #
    # ------------------------------------------------------------------ #
    def _handle_broadcast(self, i: int) -> bool:
        """Broadcast this drone's new knowledge; True iff the payload was new.

        ``_new_cells_since_comm`` is an explicit chronological log of cells the
        drone explored since its last broadcast (``np.argwhere`` over the
        explored mask is row-major spatial order, NOT discovery order, so
        slicing it by a running count sent the wrong cells).
        """
        d = self.drones[i]
        positions = {j: self.drones[j].kinematics.pos for j in range(self.n)}
        alive = {j: self.drones[j].mission.alive for j in range(self.n)}
        ok = {j: self.drones[j].faults.comm_ok for j in range(self.n)}
        known = [(v.idx, v.pos.tolist()) for v in self.world.victims if v.detected and not v.rescued]
        delta_cells = self._new_cells_since_comm[i]
        new_victim_ids = {idx for idx, _ in known} - self._victims_shared[i]
        payload = {"victims": known, "explored": int(self.explored[i].sum()),
                   "explored_cells": delta_cells[-100:],
                   "battery": d.battery.soc, "pos": d.kinematics.pos.tolist()}
        self.comm.broadcast(i, positions, Message(i, "victim", payload, self.t), alive, ok)
        self.comm_events += 1
        is_new_information = bool(delta_cells) or bool(new_victim_ids)
        self._new_cells_since_comm[i] = []
        self._victims_shared[i].update(idx for idx, _ in known)
        return is_new_information

    def _merge_inbox(self, i: int):
        """Fuse received victim/coverage information into local beliefs."""
        for m in self.comm.inbox(i):
            for (_vidx, pos) in m.payload.get("victims", []):
                x, y = int(pos[0]), int(pos[1])
                self.victim_belief[i][y, x] = max(self.victim_belief[i][y, x], 0.9)
            for y, x in m.payload.get("explored_cells", []):
                if 0 <= y < self.world.size and 0 <= x < self.world.size:
                    self.explored[i][y, x] = True

    def _nearest_known_victim_rel(self, i: int) -> np.ndarray:
        """Relative direction to the nearest persistent victim belief."""
        cells = np.argwhere(self.victim_belief[i] > 0.5)
        if cells.size == 0:
            return np.zeros(2, dtype=np.float32)
        positions = cells[:, [1, 0]].astype(float)
        pos = self.drones[i].kinematics.pos
        dists = np.linalg.norm(positions - pos, axis=1)
        target = positions[int(np.argmin(dists))]
        return ((target - pos) / max(1, self.world.size)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # charging / collisions                                             #
    # ------------------------------------------------------------------ #
    def _handle_charging(self):
        # Clear occupancy each step
        for st in self.world.charging:
            st.occupied_by.clear()
        for d in self.drones:
            if not d.mission.alive:
                continue
            st = self.world.nearest_charging(d.kinematics.pos)
            if st is None:
                continue
            if np.linalg.norm(d.kinematics.pos - st.pos) <= 1.5:
                if len(st.occupied_by) < st.capacity:
                    st.occupied_by.append(d.idx)
                    d.mission.charging = True
                    d.battery.charge()
                    if d.battery.soc > 0.95:
                        d.mission.charging = False
                        d.mission.returning = False
                else:
                    d.mission.charging = False  # station full

    def _handle_collisions(self):
        """Penalize collisions on contact ONSET only.

        A pair that stays overlapped (velocities are zeroed, positions may
        coincide for many steps) previously bled -10 per drone per step,
        dominating the return with a penalty the policy could no longer act
        on.  Now the event fires once per new contact; lingering proximity is
        handled by the near-miss shaping in :meth:`_handle_separation_reward`.
        """
        cr = self.cfg.collision_radius_m
        alive = [d for d in self.drones if d.mission.alive]
        current: set = set()
        for a_i in range(len(alive)):
            for b_i in range(a_i + 1, len(alive)):
                da, db = alive[a_i], alive[b_i]
                if np.linalg.norm(da.kinematics.pos - db.kinematics.pos) < cr:
                    key = ("d", min(da.idx, db.idx), max(da.idx, db.idx))
                    current.add(key)
                    for d in (da, db):
                        d.kinematics.vel = np.zeros(2)
                    if key not in self._contacts:
                        for d in (da, db):
                            self.reward_inputs[f"drone_{d.idx}"].collision = True
                        self.collisions.append({"step": self.t, "pos": da.kinematics.pos.tolist(),
                                                "drones": [da.idx, db.idx]})
        # drone vs dynamic obstacle
        for d in alive:
            for o in self.world.dyn_obstacles:
                if np.linalg.norm(d.kinematics.pos - o.pos) < cr:
                    key = ("o", d.idx, o.idx)
                    current.add(key)
                    if key not in self._contacts:
                        self.reward_inputs[f"drone_{d.idx}"].collision = True
                        self.collisions.append({"step": self.t, "pos": d.kinematics.pos.tolist(),
                                                "drones": [d.idx], "obstacle": o.idx})
        self._contacts = current

    def _handle_separation_reward(self):
        """Shaped near-miss penalty / safe-separation bonus between drones.

        Applies within 2.5 collision radii INCLUDING ongoing contact, so a
        stuck pair still feels steady (bounded) pressure to separate after
        the one-time collision event has fired.
        """
        cr = self.cfg.collision_radius_m
        for i in range(self.n):
            di = self.drones[i]
            if not di.mission.alive:
                continue
            nearest = float("inf")
            for j in range(self.n):
                if i == j or not self.drones[j].mission.alive:
                    continue
                nearest = min(nearest, float(np.linalg.norm(di.kinematics.pos - self.drones[j].kinematics.pos)))
            if nearest < cr * 2.5:
                self.reward_inputs[f"drone_{i}"].near_collision = True
            elif np.isfinite(nearest):
                self.reward_inputs[f"drone_{i}"].safely_separated = True

    # ------------------------------------------------------------------ #
    # observation                                                        #
    # ------------------------------------------------------------------ #
    def _obs(self, i: int) -> np.ndarray:
        """Return a compact single-frame observation for legacy callers.

        The public reset/step API returns the flattened eight-frame observation
        from :class:`ObservationEngine`.  This helper preserves the historical
        private method used by notebooks while maintaining decentralized peer
        visibility and local-map sensing.
        """
        d = self.drones[i]
        S = float(self.world.size)
        own = np.array([
            d.gps_position()[0] / S,
            d.gps_position()[1] / S,
            d.kinematics.vel[0] / max(1e-6, self.cfg.max_speed_mps),
            d.kinematics.vel[1] / max(1e-6, self.cfg.max_speed_mps),
            d.kinematics.alt / max(1e-6, self.cfg.world.max_altitude_m),
            d.battery.soc,
            float(d.mission.returning),
            float(d.faults.camera_ok),
            float(d.faults.comm_ok),
        ], dtype=np.float32)
        peer_features = []
        visible = [
            j for j, other in enumerate(self.drones)
            if j != i and other.mission.alive and other.faults.comm_ok
            and np.linalg.norm(other.kinematics.pos - d.kinematics.pos) <= self.cfg.comm.range_m
        ]
        visible.sort(key=lambda j: np.linalg.norm(self.drones[j].kinematics.pos - d.kinematics.pos))
        for j in visible[: self.cfg.k_nearest_drones]:
            other = self.drones[j]
            rel = (other.kinematics.pos - d.kinematics.pos) / S
            peer_features.extend([rel[0], rel[1], other.kinematics.vel[0], other.kinematics.vel[1]])
        peer_features.extend([0.0] * (4 * self.cfg.k_nearest_drones - len(peer_features)))
        x, y = int(d.kinematics.pos[0]), int(d.kinematics.pos[1])
        local_map = self._local_patch(self.world.grid.astype(np.float32), x, y)
        known_victim = float((self.victim_belief[i] > 0.5).any())
        return np.concatenate([own, np.asarray(peer_features, dtype=np.float32), local_map,
                               np.array([known_victim, len(self.comm.inbox(i))], dtype=np.float32)])

    def _local_patch(self, grid, cx, cy) -> np.ndarray:
        p = self._patch; h = p // 2; S = self.world.size
        out = np.zeros((p, p), dtype=np.float32)
        for dx in range(-h, h + 1):
            for dy in range(-h, h + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < S and 0 <= y < S:
                    out[dy + h, dx + h] = grid[y, x]
        return out.flatten()

    # ------------------------------------------------------------------ #
    # logging / rendering                                                #
    # ------------------------------------------------------------------ #
    def _record_frame(self, actions: dict[str, int]):
        self.history.append({
            "t": self.t,
            "pos": [d.kinematics.pos.copy() for d in self.drones],
            "alt": [d.kinematics.alt for d in self.drones],
            "soc": [d.battery.soc for d in self.drones],
            "alive": [d.mission.alive for d in self.drones],
            "coverage": float(self.global_explored.mean()),
            "found": int(sum(v.detected for v in self.world.victims)),
            "rescued": int(sum(v.rescued for v in self.world.victims)),
            "comm_edges": list(self.comm.edges_this_step),
            "environment": self.environment_events[-1] if self.environment_events else {},
            "actions": [int(actions.get(a, 0)) for a in self.possible_agents],
            "faults": [{"drone": d.idx, "gps": d.faults.gps_ok, "cam": d.faults.camera_ok,
                        "comm": d.faults.comm_ok, "motor": not d.faults.motor_degraded}
                       for d in self.drones],
        })
        # running metrics snapshot (TensorBoard/logging)
        self.metrics = self.episode_summary()

    def _agent_info(self, actions: dict[str, int]) -> dict:
        """Per-agent info dict returned from :meth:`step`."""
        vic = self.world.victims
        rescued_count = int(sum(v.rescued for v in vic))
        total_victims = len(vic)
        return {
            "alive": [d.mission.alive for d in self.drones],
            "coverage": float(self.global_explored.mean()),
            "coverage_percent": float(self.global_explored.mean()) * 100.0,
            "found": int(sum(v.detected for v in vic)),
            "rescued": rescued_count,
            "rescue_rate": float(rescued_count / total_victims) if total_victims > 0 else 0.0,
            "comm_edges": list(self.comm.edges_this_step),
            "environment": self.environment_events[-1] if self.environment_events else {},
            "actions": [int(actions.get(a, 0)) for a in self.possible_agents],
        }

    def render_state(self) -> dict:
        return self.history[-1] if self.history else {}

    def episode_summary(self) -> dict:
        vic = self.world.victims
        rescued_count = int(sum(v.rescued for v in vic))
        total_victims = len(vic)
        return {
            "steps": self.t,
            "coverage": float(self.global_explored.mean()),
            "coverage_percent": float(self.global_explored.mean()) * 100.0,
            "victims_total": total_victims,
            "victims_detected": int(sum(v.detected for v in vic)),
            "victims_rescued": rescued_count,
            "rescue_rate": float(rescued_count / total_victims) if total_victims > 0 else 0.0,
            "collisions": len(self.collisions),
            "comm_sent": self.comm.n_sent,
            "comm_delivered": self.comm.n_delivered,
            "delivery_ratio": self.comm.delivery_ratio,
            "faults": self.faults.summary(),
            "environment_events": list(self.environment_events),
            "energy_wh": float(sum(d.battery.capacity - d.battery.energy for d in self.drones)),
            "energy_budget_wh": float(sum(d.battery.cfg.capacity_wh for d in self.drones)),
            "success": bool(total_victims > 0 and rescued_count == total_victims),
        }

    def _apply_weather_profile(self) -> None:
        if not self.cfg.world.weather_enabled:
            self.visibility = 1.0
            self.gps_noise_scale = 1.0
            return
        if self.rng.random() < self.cfg.world.wind_gust_prob:
            gust = self.rng.uniform(-1, 1, size=2) * self.cfg.world.wind_gust_mps
            self.wind = np.clip(self.wind + gust, -self.cfg.wind_max_mps * 2, self.cfg.wind_max_mps * 2)
        wind_mag = float(np.linalg.norm(self.wind))
        max_wind = max(self.cfg.wind_max_mps * 2, 1e-6)
        severity = np.clip(wind_mag / max_wind, 0.0, 1.0)
        self.visibility = float(1.0 - severity * (1.0 - self.cfg.world.min_visibility))
        self.gps_noise_scale = float(1.0 + severity)


def make_env(config_path: str | None = None, seed: int | None = None, **overrides) -> SARSwarmEnv:
    """Factory that builds an env from a YAML config (+ overrides)."""
    cfg = load_config(config_path, **({"env": overrides} if overrides else {}))
    return SARSwarmEnv(cfg.env, seed=seed)
