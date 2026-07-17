import numpy as np
from swarm_sar.evaluation.metrics import (
    swarm_intelligence_score, aggregate, exploration_entropy, episode_diagnostics,
)


def test_sis_bounds():
    assert swarm_intelligence_score({"coverage": 1, "rescue": 1, "energy": 1,
                                     "communication": 1, "safety": 1}) == 100.0
    assert 0 <= swarm_intelligence_score({"coverage": 0, "rescue": 0, "energy": 0,
                                          "communication": 0, "safety": 0}) < 5


def test_sis_geometric_penalizes_weak_dimension():
    balanced = swarm_intelligence_score({"coverage": .6, "rescue": .6, "energy": .6,
                                         "communication": .6, "safety": .6})
    lopsided = swarm_intelligence_score({"coverage": 1, "rescue": 1, "energy": 1,
                                         "communication": 1, "safety": .01})
    assert lopsided < balanced     # one terrible dimension tanks the score


def test_aggregate_mean_std():
    data = [{"x": 1.0}, {"x": 3.0}]
    agg = aggregate(data)
    assert agg["x"]["mean"] == 2.0
    assert abs(agg["x"]["std"] - 1.0) < 1e-9


def test_exploration_entropy_range():
    frames = [{"pos": [np.array([i % 16, (i * 3) % 16])]} for i in range(50)]
    h = exploration_entropy(frames, grid_size=16, bins=8)
    assert 0.0 <= h <= 1.0


def test_episode_diagnostics_counts_actions_and_comms():
    class Log:
        frames = [
            {"actions": [0, 8], "comm_edges": [(0, 1)]},
            {"actions": [1, 1], "comm_edges": []},
        ]
        summary = {"comm_sent": 1, "comm_delivered": 1, "collisions": 0}

    diag = episode_diagnostics(Log())

    assert diag["action_histogram"]["hover"] == 1
    assert diag["action_histogram"]["broadcast"] == 1
    assert diag["action_histogram"]["north"] == 2
    assert diag["comm_edges"] == 1
