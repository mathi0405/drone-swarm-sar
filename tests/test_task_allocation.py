import numpy as np
from swarm_sar.mission.task_allocation import (
    TaskAllocator, hungarian, auction, cost_matrix, _hungarian_numpy, ALLOCATORS,
)


def test_hungarian_optimal_on_identity():
    # drones and victims paired diagonally -> optimal assignment is identity
    dp = np.array([[0, 0.], [10, 0], [0, 10]])
    vp = np.array([[0, 0.], [10, 0], [0, 10]])
    C = cost_matrix(dp, vp)
    assert hungarian(C) == [0, 1, 2]


def test_numpy_hungarian_matches_known():
    C = np.array([[4., 2., 8.], [4., 3., 7.], [3., 1., 6.]])
    assign = _hungarian_numpy(C)
    # optimal assignment for this matrix has total cost 12 (verified by brute force)
    total = sum(C[i, assign[i]] for i in range(3) if assign[i] >= 0)
    assert total == 12.0
    assert sorted(assign) == [0, 1, 2]        # valid permutation


def test_all_strategies_return_valid():
    dp = np.random.default_rng(0).uniform(0, 30, (3, 2))
    vp = np.random.default_rng(1).uniform(0, 30, (3, 2))
    for s in ALLOCATORS:
        if s == "rl":
            continue
        a = TaskAllocator(s, np.random.default_rng(0)).assign(dp, vp)
        assert len(a) == 3
        assigned = [v for v in a.values() if v >= 0]
        assert len(assigned) == len(set(assigned))    # no victim double-booked


def test_auction_near_optimal():
    rng = np.random.default_rng(5)
    dp = rng.uniform(0, 40, (4, 2)); vp = rng.uniform(0, 40, (4, 2))
    C = cost_matrix(dp, vp)
    ha = hungarian(C); aa = auction(C)
    cost_h = sum(C[i, ha[i]] for i in range(4))
    cost_a = sum(C[i, aa[i]] for i in range(4) if aa[i] >= 0)
    assert cost_a <= cost_h * 1.5          # auction within 50% of optimal
