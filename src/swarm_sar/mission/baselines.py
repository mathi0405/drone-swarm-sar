"""Non-learned baselines that the multi-robot community expects to see.

These give a spectrum of coordination sophistication so that "does learning help?"
can be answered against respected classical methods rather than only the internal
heuristic:

* ``LawnmowerController``  - systematic boustrophedon sweep (pure coverage).
* ``FrontierController``   - greedy frontier exploration + nearest-drone rescue.
* ``GreedyTSPController``  - nearest-neighbour tour over assigned victims.
* ``LloydController``      - Voronoi/Lloyd coverage control (Cortes-Bullo).
* ``OracleController``     - privileged full-state Hungarian assignment (upper bound).

Each exposes ``act() -> {agent: action}`` and is wrapped by :class:`ControllerActor`.
"""
from __future__ import annotations

import numpy as np

from swarm_sar.drone.drone import ACTION_TO_IDX
from swarm_sar.mission.planner import _move_action_towards
from swarm_sar.mission.task_allocation import TaskAllocator


def _act_to(src, dst) -> int:
    return _move_action_towards(src, dst)   # already an action index


class ControllerActor:
    """Adapt any baseline controller to the run_episode actor interface."""
    def __init__(self, controller):
        self.c = controller

    def __call__(self, obs):
        return self.c.act()


class LawnmowerController:
    """Divide the map into horizontal bands (one per drone) and sweep them."""
    def __init__(self, env, **_):
        self.env = env
        self.dir = [1] * env.n

    def act(self) -> dict[str, int]:
        env = self.env; S = env.world.size; acts = {}
        band = S / max(1, env.n)
        for i, d in enumerate(env.drones):
            if not d.mission.alive:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
            x, y = d.kinematics.pos; target_y = (i + 0.5) * band
            if abs(y - target_y) > 1.5:
                a = "north" if target_y > y else "south"
            else:
                if x >= S - 2:
                    self.dir[i] = -1
                elif x <= 1:
                    self.dir[i] = 1
                a = "east" if self.dir[i] > 0 else "west"
            acts[f"drone_{i}"] = ACTION_TO_IDX[a]
        return acts


class FrontierController:
    """Greedy nearest-frontier exploration; nearest drone services a known victim."""
    def __init__(self, env, seed=0, **_):
        self.env = env; self.rng = np.random.default_rng(seed)

    def act(self) -> dict[str, int]:
        env = self.env; acts = {}
        victims = [v for v in env.world.victims if v.detected and not v.rescued]
        for i, d in enumerate(env.drones):
            if not d.mission.alive:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
            if d.battery.needs_return:
                st = env.world.nearest_charging(d.kinematics.pos)
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"] if np.linalg.norm(d.kinematics.pos - st.pos) <= 1.5 \
                    else _act_to(d.kinematics.pos, st.pos); continue
            if victims:  # go to nearest known victim
                v = min(victims, key=lambda v: np.linalg.norm(d.kinematics.pos - v.pos))
                if np.linalg.norm(d.kinematics.pos - v.pos) <= env.cfg.collision_radius_m + 1.0:
                    acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
                acts[f"drone_{i}"] = _act_to(d.kinematics.pos, v.pos); continue
            # else frontier: nearest unexplored flyable cell (subsampled ring)
            tgt = self._frontier(d.kinematics.pos, env.explored[i])
            acts[f"drone_{i}"] = _act_to(d.kinematics.pos, tgt)
        return acts

    def _frontier(self, pos, explored):
        S = self.env.world.size; cx, cy = int(pos[0]), int(pos[1])
        for rad in range(3, S, 3):
            for x in range(max(0, cx - rad), min(S, cx + rad), 2):
                for y in range(max(0, cy - rad), min(S, cy + rad), 2):
                    if not explored[y, x] and self.env.world.is_flyable(x, y):
                        return np.array([x, y], float)
        return np.array([self.rng.integers(S), self.rng.integers(S)], float)


class GreedyTSPController(FrontierController):
    """Assign victims by nearest and visit each drone's set in NN-tour order."""
    def act(self) -> dict[str, int]:
        env = self.env; acts = {}
        victims = [v for v in env.world.victims if v.detected and not v.rescued]
        alive = [i for i in range(env.n) if env.drones[i].mission.alive]
        assign = {i: [] for i in alive}
        for v in victims:                    # greedy nearest-drone
            i = min(alive, key=lambda i: np.linalg.norm(env.drones[i].kinematics.pos - v.pos))
            assign[i].append(v)
        for i in range(env.n):
            d = env.drones[i]
            if not d.mission.alive:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
            mine = assign.get(i, [])
            if mine:                          # nearest-neighbour next hop
                v = min(mine, key=lambda v: np.linalg.norm(d.kinematics.pos - v.pos))
                if np.linalg.norm(d.kinematics.pos - v.pos) <= env.cfg.collision_radius_m + 1.0:
                    acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
                acts[f"drone_{i}"] = _act_to(d.kinematics.pos, v.pos)
            else:
                acts[f"drone_{i}"] = _act_to(d.kinematics.pos, self._frontier(d.kinematics.pos, env.explored[i]))
        return acts


class LloydController:
    """Voronoi/Lloyd coverage control: move toward the centroid of the unexplored
    mass in this drone's Voronoi cell (classic multi-robot coverage, Cortes 2004)."""
    def __init__(self, env, **_):
        self.env = env

    def act(self) -> dict[str, int]:
        env = self.env; S = env.world.size; acts = {}
        pts = np.array([d.kinematics.pos for d in env.drones])
        # subsample unexplored, flyable cells as the density to cover
        gx, gy = np.meshgrid(np.arange(0, S, 3), np.arange(0, S, 3))
        cells = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
        unexplored = np.array([not env.global_explored[int(y), int(x)] and env.world.is_flyable(x, y)
                               for x, y in cells])
        cells = cells[unexplored]
        for i, d in enumerate(env.drones):
            if not d.mission.alive or len(cells) == 0:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
            dists = np.linalg.norm(cells[:, None, :] - pts[None, :, :], axis=-1)
            owned = cells[np.argmin(dists, axis=1) == i]
            centroid = owned.mean(0) if len(owned) else d.kinematics.pos
            if np.linalg.norm(centroid - d.kinematics.pos) < 1.0:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]
            else:
                acts[f"drone_{i}"] = _act_to(d.kinematics.pos, centroid)
        return acts


class OracleController:
    """Privileged upper bound: knows every victim from t=0, Hungarian-assigns and
    flies straight to targets (ignores sensing/comms). Not a fair method - a ceiling."""
    def __init__(self, env, **_):
        self.env = env; self.alloc = TaskAllocator("hungarian")

    def act(self) -> dict[str, int]:
        env = self.env; acts = {}
        remaining = [v for v in env.world.victims if not v.rescued]
        alive = [i for i in range(env.n) if env.drones[i].mission.alive]
        assign = {}
        if remaining and alive:
            dp = np.array([env.drones[i].kinematics.pos for i in alive])
            vp = np.array([v.pos for v in remaining])
            raw = self.alloc.assign(dp, vp)
            for li, vl in raw.items():
                if 0 <= vl < len(remaining):
                    assign[alive[li]] = remaining[vl]
        for i in range(env.n):
            d = env.drones[i]
            if not d.mission.alive:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
            v = assign.get(i)
            if v is None:
                acts[f"drone_{i}"] = ACTION_TO_IDX["hover"]; continue
            # oracle "reveals" the victim so the env can rescue it on arrival
            v.detected = True; v._confirmations = max(v._confirmations, env.cfg.confirmations_required)
            acts[f"drone_{i}"] = ACTION_TO_IDX["hover"] if \
                np.linalg.norm(d.kinematics.pos - v.pos) <= env.cfg.collision_radius_m + 1.0 \
                else _act_to(d.kinematics.pos, v.pos)
        return acts


BASELINES = {
    "lawnmower": LawnmowerController,
    "frontier": FrontierController,
    "greedy_tsp": GreedyTSPController,
    "lloyd_coverage": LloydController,
    "oracle": OracleController,
}
