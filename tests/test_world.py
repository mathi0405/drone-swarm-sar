import numpy as np
from swarm_sar.config import WorldConfig
from swarm_sar.environment.world import SARWorld
from swarm_sar.environment.entities import CellType


def test_world_generates_entities():
    w = SARWorld(WorldConfig(size=48, n_victims=6, n_charging_stations=2), np.random.default_rng(0))
    assert len(w.victims) == 6
    assert len(w.charging) == 2
    assert w.grid.shape == (48, 48)


def test_world_randomizes_per_seed():
    a = SARWorld(WorldConfig(size=48), np.random.default_rng(1)).grid.copy()
    b = SARWorld(WorldConfig(size=48), np.random.default_rng(2)).grid.copy()
    assert not np.array_equal(a, b)          # different maps -> anti-memorization


def test_flyable_and_line_of_sight():
    w = SARWorld(WorldConfig(size=40), np.random.default_rng(3))
    assert 0.0 < w.flyable_mask.mean() <= 1.0
    # a cell is always visible to itself
    assert w.line_of_sight((5, 5), (5, 5))


def test_nearest_charging():
    w = SARWorld(WorldConfig(size=40, n_charging_stations=3), np.random.default_rng(4))
    st = w.nearest_charging(np.array([0.0, 0.0]))
    assert st in w.charging
