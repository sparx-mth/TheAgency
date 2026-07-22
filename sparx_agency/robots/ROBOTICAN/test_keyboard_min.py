import rclpy
from rclpy.node import Node

from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive

print("before init")
rclpy.init()
print("after init")

node = Node("test_keyboard_node")
print("after node")

pub1 = node.create_publisher(ManualControl, "/R1/manual_control", 10)
print("after manual publisher")

pub2 = node.create_publisher(KeepAlive, "/R1/keep_alive", 10)
print("after keepalive publisher")

pub3 = node.create_publisher(Bool, "/R1/gcs_keep_alive", 10)
print("after gcs publisher")

cli = node.create_client(SetBool, "/R1/fcu/command/force_arm")
print("after arm client")

node.destroy_node()
rclpy.shutdown()
print("done")