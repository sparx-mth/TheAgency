#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class GCSKeepAlive(Node):
    def __init__(self, drone_id="R2"):
        super().__init__("gcs_keep_alive_node")
        self.pub = self.create_publisher(Bool, f"/{drone_id}/gcs_keep_alive", 10)
        self.timer = self.create_timer(1.0, self.tick)

    def tick(self):
        msg = Bool()
        msg.data = True
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GCSKeepAlive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

