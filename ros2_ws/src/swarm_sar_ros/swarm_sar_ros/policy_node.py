"""ROS2 node that runs one drone's decentralized policy.

Each drone runs an identical node (decentralized execution): it subscribes to its
own odometry/battery/received-messages, builds the observation vector, queries the
trained policy, and publishes a velocity command + broadcast messages. This is the
deployment counterpart to :class:`SARSwarmEnv` and works against either AirSim
(via airsim_ros_pkgs) or real PX4/MAVROS drones.
"""
from __future__ import annotations

import json

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState
    from std_msgs.msg import String
    _HAS_ROS = True
except Exception:            # allow import without a ROS2 install
    _HAS_ROS = False
    Node = object            # type: ignore

# Discrete policy actions -> (vx, vy, vz) body-frame velocity setpoints (m/s).
# Must stay aligned with swarm_sar.drone.ACTIONS.
_ACTION_TO_VEL = {
    0: (0.0, 0.0, 0.0),    # hover
    1: (0.0, 3.0, 0.0),    # north
    2: (0.0, -3.0, 0.0),   # south
    3: (3.0, 0.0, 0.0),    # east
    4: (-3.0, 0.0, 0.0),   # west
    5: (0.0, 0.0, 2.0),    # ascend
    6: (0.0, 0.0, -2.0),   # descend
    7: (0.0, 0.0, 0.0),    # rotate (yaw handled by autopilot)
    8: (0.0, 0.0, 0.0),    # broadcast
    9: (0.0, 0.0, 0.0),    # return_to_base (delegated to mission layer)
}
_BROADCAST_ACTION = 8


class PolicyNode(Node):      # pragma: no cover - requires a ROS2 runtime
    def __init__(self, drone_id: int = 0, checkpoint: str | None = None):
        super().__init__(f"swarm_sar_policy_{drone_id}")
        self.drone_id = drone_id
        self.policy = self._load_policy(checkpoint)
        self._obs_dim = getattr(getattr(self.policy, "encoder", None), "frame_dim", None)
        self._latest_odom: Odometry | None = None
        self._latest_battery: BatteryState | None = None
        self._inbox: list[dict] = []

        self.cmd_pub = self.create_publisher(Twist, f"/drone_{drone_id}/cmd_vel", 10)
        self.msg_pub = self.create_publisher(String, "/swarm/messages", 10)
        self.create_subscription(Odometry, f"/drone_{drone_id}/odom", self._on_odom, 10)
        self.create_subscription(BatteryState, f"/drone_{drone_id}/battery", self._on_battery, 10)
        self.create_subscription(String, "/swarm/messages", self._on_msg, 10)
        self.create_timer(0.1, self._control_step)   # 10 Hz

    def _load_policy(self, checkpoint):
        from swarm_sar.policies import load_policy
        return load_policy(checkpoint) if checkpoint else None

    def _on_odom(self, msg):
        self._latest_odom = msg

    def _on_battery(self, msg):
        self._latest_battery = msg

    def _on_msg(self, msg):
        try:
            data = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if data.get("sender") != self.drone_id:
            self._inbox.append(data)
            del self._inbox[:-32]                 # bounded inbox

    def _build_observation(self):
        """Assemble the flat observation vector from cached sensor topics.

        Deployment note: field integrations should mirror
        ObservationEngine's frame layout exactly; here we populate the own-state
        block and zero-pad the rest, which is sufficient for bench testing the
        policy in the loop.
        """
        import numpy as np
        if self._latest_odom is None or self.policy is None:
            return None
        frame_dim = self._obs_dim or 32
        history_len = getattr(getattr(self.policy, "encoder", None), "history_len", 8)
        p = self._latest_odom.pose.pose.position
        v = self._latest_odom.twist.twist.linear
        soc = self._latest_battery.percentage if self._latest_battery else 1.0
        frame = np.zeros(frame_dim, dtype=np.float32)
        frame[:7] = [p.x, p.y, p.z, v.x, v.y, 0.0, soc]
        return np.tile(frame, history_len)

    def _control_step(self):
        obs = self._build_observation()
        twist = Twist()
        action = 0
        if obs is not None:
            import torch
            with torch.no_grad():
                batch = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                act, _, _ = self.policy.act(batch, deterministic=True)
            action = int(act.item())
        vx, vy, vz = _ACTION_TO_VEL.get(action, (0.0, 0.0, 0.0))
        twist.linear.x, twist.linear.y, twist.linear.z = vx, vy, vz
        self.cmd_pub.publish(twist)
        if action == _BROADCAST_ACTION and self._latest_odom is not None:
            p = self._latest_odom.pose.pose.position
            out = String()
            out.data = json.dumps({"sender": self.drone_id, "pos": [p.x, p.y]})
            self.msg_pub.publish(out)


def main():
    if not _HAS_ROS:
        raise SystemExit("ROS2 (rclpy) not found. Source your ROS2 Humble setup first.")
    rclpy.init()
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
