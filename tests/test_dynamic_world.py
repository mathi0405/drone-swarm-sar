import numpy as np

from swarm_sar.config import EnvConfig, WorldConfig
from swarm_sar.environment.entities import CellType
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.environment.world import SARWorld


def test_fire_spreads_deterministically_with_seed():
    cfg = WorldConfig(
        size=12,
        n_buildings=0,
        n_trees=0,
        n_rubble=0,
        n_fire_zones=0,
        n_no_fly_zones=0,
        n_charging_stations=0,
        n_victims=0,
        n_dynamic_obstacles=0,
        enable_fire_spread=True,
        fire_spread_prob=1.0,
        smoke_spread_prob=0.0,
    )
    w1 = SARWorld(cfg, np.random.default_rng(3))
    w2 = SARWorld(cfg, np.random.default_rng(3))
    for w in (w1, w2):
        w.grid[:] = CellType.FREE
        w.grid[6, 6] = CellType.FIRE

    e1 = w1.step_dynamic()
    e2 = w2.step_dynamic()

    assert e1["fire_spread"] == e2["fire_spread"] == 4
    assert np.array_equal(w1.grid, w2.grid)


def test_collapse_changes_occupancy():
    cfg = WorldConfig(
        size=8,
        n_buildings=0,
        n_trees=0,
        n_rubble=0,
        n_fire_zones=0,
        n_no_fly_zones=0,
        n_charging_stations=0,
        n_victims=0,
        n_dynamic_obstacles=0,
        enable_collapses=True,
        collapse_prob=1.0,
    )
    world = SARWorld(cfg, np.random.default_rng(0))
    world.grid[:] = CellType.FREE
    world.grid[3, 3] = CellType.BUILDING

    events = world.step_dynamic()

    assert events["collapses"] == 1
    assert world.grid[3, 3] == CellType.RUBBLE


def test_moving_victims_stay_in_bounds():
    cfg = WorldConfig(
        size=8,
        n_buildings=0,
        n_trees=0,
        n_rubble=0,
        n_fire_zones=0,
        n_no_fly_zones=0,
        n_charging_stations=0,
        n_victims=1,
        n_dynamic_obstacles=0,
        moving_victims=True,
        victim_move_prob=1.0,
    )
    world = SARWorld(cfg, np.random.default_rng(1))

    for _ in range(20):
        world.step_dynamic()
        x, y = world.victims[0].pos
        assert 0 <= x < world.size
        assert 0 <= y < world.size


def test_observation_updates_when_known_environment_changes():
    cfg = EnvConfig(
        num_drones=1,
        max_steps=2,
        world=WorldConfig(size=12, n_victims=0, n_buildings=0, n_fire_zones=0, n_dynamic_obstacles=0),
    )
    env = SARSwarmEnv(cfg, seed=0)
    env.reset(seed=0)
    env.drones[0].state.pos = np.array([6.0, 6.0])
    env.explored[0][6, 6] = True
    before = env._obs(0).copy()
    env.world.grid[6, 6] = CellType.FIRE
    after = env._obs(0)

    assert not np.array_equal(before, after)
