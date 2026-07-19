from collections import deque

import numpy as np

from swarm_sar.environment.entities import CellType

# Single source of truth for the temporal stack depth. The Transformer encoder
# reshapes flattened observations back into HISTORY_LEN frames, so both sides
# must agree on this value.
HISTORY_LEN = 8


def frame_layout(k_nearest: int, map_patch: int) -> dict[str, int]:
    """Ordered per-frame segment sizes (name -> width).

    Single source of truth for the frame structure, shared by
    ``ObservationEngine.get_obs`` / ``SARSwarmEnv._compute_obs_dim`` (which
    must sum to the same total) and the entity-tokenizing Transformer (which
    slices a frame back into (self, victims, map, peers, messages) tokens).
    The ``peers`` segment is ``k_nearest`` blocks of 6 features each.
    """
    p = map_patch
    return {
        "own_state": 8,
        "victim_info": 9,
        "hazards": 4,
        "map_patch": 3 * p * p,
        "lidar": 8,
        "peers": 6 * k_nearest,
        "comms": 4,
    }


class ObservationEngine:
    def __init__(self, cfg, num_drones):
        self.cfg = cfg
        self.num_drones = num_drones
        self.history_len = HISTORY_LEN
        self.history = [deque(maxlen=self.history_len) for _ in range(num_drones)]

    def reset(self):
        for h in self.history:
            h.clear()

    def get_obs(self, env):
        # Per-drone frame layout (see SARSwarmEnv._compute_obs_dim):
        #   own state (8) + victim info (9) + hazards (4)
        #   + egocentric map patch (3 x P x P) + LiDAR (8)
        #   + peers (6 * k) + comms (4)
        k = env.cfg.k_nearest_drones
        S = env.world.size

        obs_dict = {}
        frame_dim = 0

        for i in range(self.num_drones):
            d = env.drones[i]
            s = d.kinematics

            # Own state.  The clock (t/max_steps) is essential: with per-step
            # penalties and hard truncation, V(s) depends strongly on time
            # remaining, which the network otherwise cannot observe.  The
            # `returning` flag tells the policy it is committed to the
            # return-to-base macro-action.
            gps = d.gps_position() / S
            own_state = [
                gps[0], gps[1], s.alt / env.cfg.world.max_altitude_m,
                s.vel[0] / env.cfg.max_speed_mps, s.vel[1] / env.cfg.max_speed_mps,
                env.t / max(1, env.cfg.max_steps),
                d.battery.soc,
                float(d.mission.returning),
            ]

            # Victim info
            inbox = env.comm.inbox(i)
            vs = [vv for m in inbox for vv in m.payload.get("victims", [])]
            victim_positions = []
            for vv in vs:
                pv = np.array(vv[1]) if isinstance(vv, (list, tuple)) else np.array(vv)
                victim_positions.append(pv)

            if (env.victim_belief[i] > 0.5).any():
                ys, xs = np.where(env.victim_belief[i] > 0.5)
                for x, y in zip(xs, ys):
                    victim_positions.append(np.array([x, y]))

            # Remove duplicates roughly
            unique_v = []
            for vp in victim_positions:
                if not any(np.linalg.norm(vp - u) < 2.0 for u in unique_v):
                    unique_v.append(vp)

            victim_count = min(1.0, len(unique_v) / 10.0)
            victim_density = len(unique_v) / max(1, len(env.world.victims))

            dists = []
            for vp in unique_v:
                rel = (vp - s.pos) / S
                dist = float(np.linalg.norm(rel))
                dists.append((dist, rel))
            dists.sort(key=lambda x: x[0])

            nearest_dist = 1.0; nearest_rel = [0.0, 0.0]
            sec_nearest_dist = 1.0; sec_nearest_rel = [0.0, 0.0]
            confidence = 0.0

            if len(dists) > 0:
                nearest_dist, nearest_rel = dists[0]
                confidence = 1.0
            if len(dists) > 1:
                sec_nearest_dist, sec_nearest_rel = dists[1]

            victim_info = [
                victim_count, victim_density,
                nearest_dist, nearest_rel[0], nearest_rel[1], confidence,
                sec_nearest_dist, sec_nearest_rel[0], sec_nearest_rel[1]
            ]

            # Hazards, sensed only within the drone's (visibility-scaled) camera
            # radius so partial observability is preserved.  A failed camera is
            # blind: it reports "no hazards seen" rather than ground truth.
            cx, cy = int(s.pos[0]), int(s.pos[1])
            if not d.faults.camera_ok:
                hazards = [1.0, 1.0, 0.0, 0.0]
            else:
                rad = max(1, int(env.cfg.sensors.camera_range_m
                                 * getattr(env, "visibility", 1.0)
                                 / env.world.cfg.cell_size_m))
                x0, x1 = max(0, cx - rad), min(S, cx + rad + 1)
                y0, y1 = max(0, cy - rad), min(S, cy + rad + 1)
                patch = env.world.grid[y0:y1, x0:x1]
                fire_mask = patch == CellType.FIRE
                smoke_mask = patch == CellType.SMOKE
                fire_density = float(fire_mask.mean()) if patch.size else 0.0
                smoke_density = float(smoke_mask.mean()) if patch.size else 0.0
                fire_dist = smoke_dist = 1.0
                if fire_mask.any():
                    ys, xs = np.nonzero(fire_mask)
                    dd = np.sqrt((xs + x0 - s.pos[0]) ** 2 + (ys + y0 - s.pos[1]) ** 2)
                    fire_dist = float(np.clip(dd.min() / rad, 0.0, 1.0))
                if smoke_mask.any():
                    ys, xs = np.nonzero(smoke_mask)
                    dd = np.sqrt((xs + x0 - s.pos[0]) ** 2 + (ys + y0 - s.pos[1]) ** 2)
                    smoke_dist = float(np.clip(dd.min() / rad, 0.0, 1.0))
                hazards = [fire_dist, smoke_dist, fire_density, smoke_density]

            # Egocentric map patch: a P x P grid of sample points spanning the
            # camera footprint, three channels — occupancy (blocked cells; the
            # world outside the map reads as blocked), the drone's OWN explored
            # mask, and its OWN victim belief.  This is what lets the policy
            # see where it has not searched yet; without it, exploration
            # rewards are unlearnable from the observation.
            P = env.cfg.obs_map_patch
            map_rad = env.cfg.sensors.camera_range_m / env.world.cfg.cell_size_m
            steps_1d = np.round(np.linspace(-map_rad, map_rad, P)).astype(int)
            gy_raw = cy + steps_1d[:, None]          # (P, 1)
            gx_raw = cx + steps_1d[None, :]          # (1, P)
            in_map = ((gy_raw >= 0) & (gy_raw < S)) & ((gx_raw >= 0) & (gx_raw < S))
            gy = np.clip(gy_raw, 0, S - 1)
            gx = np.clip(gx_raw, 0, S - 1)
            explored_patch = np.where(in_map, env.explored[i][gy, gx], False)
            # Occupancy is fresh visual sensing; with a failed camera only
            # cells this drone previously explored are readable (memory of
            # mostly-static structure) — the rest read as unknown/blocked.
            occ_visible = in_map if d.faults.camera_ok else (in_map & explored_patch)
            occupancy = np.where(occ_visible, env._blocked_mask[gy, gx], True)
            belief_patch = np.where(in_map, env.victim_belief[i][gy, gx], 0.0)
            map_patch = np.concatenate([
                occupancy.astype(np.float32).ravel(),
                explored_patch.astype(np.float32).ravel(),
                belief_patch.astype(np.float32).ravel(),
            ])

            # LiDAR: normalized ranges to the nearest occluder in 8 directions.
            lidar_cells = max(1, int(env.cfg.sensors.lidar_range_m
                                     / env.world.cfg.cell_size_m))
            lidar = (d.sensors.lidar.scan(s.pos, env.world, env._blocked_mask)
                     / lidar_cells).astype(np.float32)

            # Team info (peers within comm range, nearest first).  Relative
            # POSITION is physical proximity sensing (needed for separation);
            # intent (target/assignment/mode) comes ONLY from messages this
            # drone actually received — reading it from world state teleported
            # every assignment past packet loss, latency and the broadcast
            # action, making learned communication nearly worthless.
            cs = max(1e-6, env.world.cfg.cell_size_m)
            crange = env.cfg.comm.range_m / cs
            others = sorted([j for j in range(env.n)
                             if j != i and env.drones[j].mission.alive and env.drones[j].faults.comm_ok
                             and np.linalg.norm(env.drones[j].kinematics.pos - s.pos) <= crange],
                            key=lambda j: np.linalg.norm(env.drones[j].kinematics.pos - s.pos))
            peer_intent = getattr(env, "_peer_intent", None)
            peers = []
            for j in others[:k]:
                noise = env.rng.normal(0, env.cfg.sensors.gps_noise_std_m / cs, size=2)
                rel = (env.drones[j].kinematics.pos + noise - s.pos) / S

                target_x = target_y = 0.0
                has_assignment = 0.0
                mode = 0.0                               # 0 = explore (default)
                intent = peer_intent[i].get(j) if peer_intent is not None else None
                if intent is not None:
                    if intent.get("target") is not None:
                        t_rel = (np.asarray(intent["target"], dtype=float) - s.pos) / S
                        target_x, target_y = float(t_rel[0]), float(t_rel[1])
                    has_assignment = float(intent.get("has_assignment", 0.0))
                    mode = float(intent.get("mode", 0.0))

                peers += [rel[0], rel[1], target_x, target_y, has_assignment, mode]
            peers += [0.0] * (6 * k - len(peers))

            # Comm
            msg_received = len(inbox)
            msg_age = 0.0
            if inbox:
                msg_age = min([env.t - m.step for m in inbox]) / max(1, env.t)

            if others:
                nearest_peer = np.linalg.norm(
                    env.drones[others[0]].kinematics.pos - s.pos)
                signal_strength = float(np.clip(1.0 - nearest_peer / max(crange, 1e-6), 0.0, 1.0))
            else:
                signal_strength = 0.0
            packet_loss = env.cfg.comm.packet_loss
            comms = [float(msg_received), msg_age, signal_strength, packet_loss]

            frame = np.concatenate([
                np.asarray(own_state, np.float32),
                np.asarray(victim_info, np.float32),
                np.asarray(hazards, np.float32),
                map_patch,
                lidar,
                np.asarray(peers, np.float32),
                np.asarray(comms, np.float32)
            ])
            frame_dim = frame.shape[0]

            # temporal history
            if len(self.history[i]) == 0:
                for _ in range(self.history_len):
                    self.history[i].append(frame)
            else:
                self.history[i].append(frame)

            # Policies receive one fixed-size vector per agent.  Keeping the
            # temporal stack flattened makes MLP/GNN policies work alongside
            # the Transformer, which reshapes this vector back into time tokens
            # internally.  Returning a 2-D stack here made the non-transformer
            # policies emit one action *per history frame*.
            obs_dict[env.agents[i] if i < len(env.agents) else f"drone_{i}"] = (
                np.stack(self.history[i]).reshape(-1).astype(np.float32, copy=False)
            )

        return obs_dict, frame_dim
