import numpy as np

from swarm_sar.communication.comms import CommNetwork
from swarm_sar.config import CommConfig, EnvConfig
from swarm_sar.environment.sar_env import SARSwarmEnv
from swarm_sar.faults.fault_injector import FaultInjector
from swarm_sar.mission.coordination import reassign_failed_tasks


def test_fault_risk_increases_with_battery_and_sensor_degradation():
    env = SARSwarmEnv(EnvConfig(num_drones=1, max_steps=1), seed=0)
    env.reset(seed=0)
    drone = env.drones[0]
    base = FaultInjector.fault_risk(drone)
    drone.battery.energy = 0.1 * drone.battery.capacity
    drone.state.comm_ok = False

    assert FaultInjector.fault_risk(drone) > base


def test_comm_components_degrade_gracefully():
    cn = CommNetwork(CommConfig(range_m=10.0), 3, np.random.default_rng(0))
    comps = cn.connected_components(
        {0: np.array([0.0, 0.0]), 1: np.array([5.0, 0.0]), 2: np.array([50.0, 0.0])},
        alive={0: True, 1: True, 2: True},
        comm_ok={0: True, 1: True, 2: False},
    )

    assert comps == [[0, 1]]


def test_killing_drone_does_not_abandon_task_when_peer_alive():
    assignments = reassign_failed_tasks(
        {0: 0},
        alive=[False, True],
        positions=np.array([[0.0, 0.0], [2.0, 0.0]]),
        target_pos=np.array([[3.0, 0.0]]),
    )

    assert assignments == {1: 0}
