import numpy as np

from swarm_sar.battery.battery import Battery
from swarm_sar.config import BatteryConfig
from swarm_sar.mission.task_allocation import TaskAllocator


def test_consumption_decreases_soc():
    b = Battery(BatteryConfig())
    s0 = b.soc
    for _ in range(30):
        b.consume("north", altitude=15, sensor_on=True, comm=False)
    assert b.soc < s0


def test_charge_increases_soc():
    b = Battery(BatteryConfig())
    for _ in range(100):
        b.consume("east", 15, True, True)
    low = b.soc
    for _ in range(50):
        b.charge()
    assert b.soc > low


def test_low_battery_flag_and_depletion():
    b = Battery(BatteryConfig(capacity_wh=1.0))
    for _ in range(200):
        b.consume("ascend", 30, True, True)
    assert b.depleted and b.soc == 0.0


def test_move_costs_more_than_hover():
    b1 = Battery(BatteryConfig()); b2 = Battery(BatteryConfig())
    b1.consume("hover", 10, False, False)
    b2.consume("north", 10, False, False)
    assert b2.energy < b1.energy


def test_distance_aware_return_threshold():
    b = Battery(BatteryConfig(capacity_wh=10.0, low_battery_frac=0.1, wh_per_meter_return=0.1))
    b.energy = 3.0

    assert b.needs_return_for_distance(30.0)


def test_low_battery_drone_not_assigned_distant_victim():
    allocator = TaskAllocator("hungarian", np.random.default_rng(0))
    assignment = allocator.assign(
        np.array([[0.0, 0.0], [90.0, 0.0]]),
        np.array([[100.0, 0.0]]),
        battery=np.array([0.08, 0.9]),
        charger_pos=np.array([[0.0, 0.0]]),
        min_battery_for_task=0.18,
    )

    assert assignment[1] == 0
