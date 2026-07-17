import numpy as np

from swarm_sar.config import EnvConfig
from swarm_sar.drone.drone import ACTION_TO_IDX
from swarm_sar.environment.sar_env import SARSwarmEnv


def test_reset_obs_shape():
    env = SARSwarmEnv(EnvConfig(num_drones=3, max_steps=20), seed=0)
    obs, info = env.reset(seed=0)
    assert set(obs) == set(env.agents)
    for a in env.agents:
        assert obs[a].shape == (env.obs_dim,)
        assert np.isfinite(obs[a]).all()


def test_step_runs_and_terminates():
    env = SARSwarmEnv(EnvConfig(num_drones=3, max_steps=15), seed=1)
    env.reset(seed=1)
    steps = 0
    rng = np.random.default_rng(0)
    while env.agents and steps < 100:
        obs, rew, term, trunc, info = env.step({a: int(rng.integers(env.n_actions)) for a in env.agents})
        steps += 1
    assert steps <= env.cfg.max_steps
    s = env.episode_summary()
    assert 0.0 <= s["coverage"] <= 1.0
    assert s["victims_rescued"] <= s["victims_total"]


def test_determinism_same_seed():
    def run(seed):
        env = SARSwarmEnv(EnvConfig(num_drones=3, max_steps=20), seed=seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(0)
        tot = 0.0
        while env.agents:
            _, r, *_ = env.step({a: int(rng.integers(env.n_actions)) for a in env.agents})
            tot += sum(r.values())
        return tot, env.episode_summary()["coverage"]
    assert run(7) == run(7)                  # fully reproducible


def test_scales_to_ten_drones():
    env = SARSwarmEnv(EnvConfig(num_drones=10, max_steps=5), seed=0)
    obs, _ = env.reset(seed=0)
    assert len(obs) == 10


def test_reset_spawns_drones_with_safe_separation():
    cfg = EnvConfig(num_drones=3, max_steps=3, min_spawn_separation_m=4.0)
    for seed in range(10):
        env = SARSwarmEnv(cfg, seed=seed)
        env.reset(seed=seed)
        for i in range(env.n):
            for j in range(i + 1, env.n):
                dist = np.linalg.norm(env.drones[i].state.pos - env.drones[j].state.pos)
                assert dist >= cfg.min_spawn_separation_m


def test_separate_comm_action_allows_move_and_broadcast():
    env = SARSwarmEnv(EnvConfig(num_drones=2, max_steps=3), seed=0)
    env.reset(seed=0)
    start = env.drones[0].state.pos.copy()

    env.step({"drone_0": {"move": "east", "comm": "coverage"}, "drone_1": 0})

    assert env.drones[0].comm_count == 1
    assert env.drones[0].state.pos[0] > start[0]


def test_action_mask_blocks_moves_toward_nearby_drone():
    env = SARSwarmEnv(EnvConfig(num_drones=2, max_steps=3), seed=0)
    env.reset(seed=0)
    env.drones[0].state.pos = env.drones[1].state.pos + [-1.0, 0.0]

    mask = env.action_mask(0)

    assert mask[3] == 0.0  # east would move drone_0 into drone_1
    assert mask[0] == 1.0  # hover remains available


def test_action_mask_can_disable_non_mission_actions():
    cfg = EnvConfig(
        num_drones=2,
        max_steps=3,
        allowed_actions=["hover", "north", "south", "east", "west"],
    )
    env = SARSwarmEnv(cfg, seed=0)
    env.reset(seed=0)

    mask = env.action_mask(0)

    assert mask[ACTION_TO_IDX["rotate"]] == 0.0
    assert mask[ACTION_TO_IDX["descend"]] == 0.0
    assert mask[ACTION_TO_IDX["return_to_base"]] == 0.0


def test_action_mask_predicts_full_speed_collision():
    env = SARSwarmEnv(EnvConfig(num_drones=2, max_steps=3, wind_max_mps=0.0), seed=0)
    env.reset(seed=0)
    env.drones[0].state.pos = np.array([10.0, 10.0])
    env.drones[0].state.vel = np.zeros(2)
    env.drones[1].state.pos = np.array([16.0, 10.0])
    env.drones[1].state.vel = np.zeros(2)

    mask = env.action_mask(0)

    assert mask[ACTION_TO_IDX["east"]] == 0.0


def test_known_victim_relative_persists_after_inbox_clears():
    env = SARSwarmEnv(EnvConfig(num_drones=1, max_steps=3), seed=0)
    env.reset(seed=0)
    env.drones[0].state.pos = np.array([10.0, 10.0])
    env.victim_belief[0][20, 30] = 0.9

    rel = env._nearest_known_victim_rel(0)

    assert np.allclose(rel, np.array([20.0, 10.0]) / env.world.size)
