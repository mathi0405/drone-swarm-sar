import numpy as np

from swarm_sar.mission.coordination import elect_leader, line_formation, reassign_failed_tasks
from swarm_sar.mission.region_partition import voronoi_partition


def test_regions_cover_flyable_map_without_overlap():
    regions = voronoi_partition(6, np.array([[1.0, 1.0], [4.0, 4.0]]), np.ones((6, 6), dtype=bool))

    assert regions.shape == (6, 6)
    assert np.all(regions >= 0)
    assert set(np.unique(regions)) == {0, 1}


def test_leader_election_chooses_alive_reachable_drone():
    leader = elect_leader(
        np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]]),
        alive=[False, True, True],
        comm_ok=[True, True, True],
        battery=[1.0, 0.9, 0.8],
        comm_range=10.0,
    )

    assert leader == 1


def test_failed_drone_task_is_reassigned():
    new_assign = reassign_failed_tasks(
        {0: 0, 1: 1},
        alive=[False, True],
        positions=np.array([[0.0, 0.0], [5.0, 0.0]]),
        target_pos=np.array([[1.0, 0.0], [10.0, 0.0]]),
    )

    assert new_assign[1] in (0, 1)


def test_line_formation_returns_one_target_per_drone():
    pts = line_formation(np.array([10.0, 10.0]), 3, spacing=2.0)

    assert pts.shape == (3, 2)
    assert np.allclose(pts[:, 1], 10.0)
