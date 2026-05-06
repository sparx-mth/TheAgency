import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json


class TwistToXtendBridge(Node):
    def __init__(self):
        super().__init__('twist_to_drone_bridge')

        # Mapping constants
        self.max_thrust = 1000  # Max value for self.send_command['axes']
        self.pulse_duration = 2000  # Fixed duration in ms for each command pulse
        self.threshold = 0.05

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            10
        )

        self.publisher_ = self.create_publisher(String, '/drone/cmd_nav', 10)
        self.get_logger().info("Twist Bridge started: Mapping intensity to thrust and duration.")

    def twist_callback(self, msg):
        """
        Translates Twist to: {"action": str, "value": int, "duration": float}
        'value' is the axes thrust (0-1000).
        'duration' is the time in ms.
        """
        # Linear X: Forward/Backward
        if abs(msg.linear.x) > self.threshold:
            action = "forward" if msg.linear.x > 0 else "backward"
            self.publish_cmd(action, abs(msg.linear.x) * self.max_thrust)

        # Linear Y: Left/Right
        if abs(msg.linear.y) > self.threshold:
            action = "left" if msg.linear.y > 0 else "right"
            self.publish_cmd(action, abs(msg.linear.y) * self.max_thrust)

        # Linear Z: Up/Down
        if abs(msg.linear.z) > self.threshold:
            action = "up" if msg.linear.z > 0 else "down"
            self.publish_cmd(action, abs(msg.linear.z) * self.max_thrust)

        # Angular Z: Rotation
        if abs(msg.angular.z) > self.threshold:
            action = "rotate_left" if msg.angular.z > 0 else "rotate_right"
            self.publish_cmd(action, abs(msg.angular.z) * self.max_thrust)

    def publish_cmd(self, action, thrust):
        cmd = {
            "action": action,
            "value": int(thrust),
            "duration": float(self.pulse_duration)
        }
        msg = String()
        msg.data = json.dumps(cmd)
        self.publisher_.publish(msg)


def main():
    rclpy.init()
    node = TwistToXtendBridge()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()