import rclpy
from rclpy.node import Node

def main():
    rclpy.init()
    node = Node('pycharm_test_node')
    print("--- ROS 2 humble is fully functional in PyCharm! ---")
    node.get_logger().info("Hello from the Jetson container!")
    rclpy.shutdown()
    # print("Hello from the Jetson container!")

if __name__ == '__main__':
    main()
