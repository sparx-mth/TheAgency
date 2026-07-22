#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose 

class OdomAndGtToTum(Node):
    def __init__(self):
        super().__init__('odom_to_tum_node')
        
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Parameter.Type.BOOL, True)])
        
        self.declare_parameter("est_topic", "/odometry/filtered")
        self.declare_parameter("est_output_file", "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/tum_results/est_tum_kalman.txt")
        
        self.declare_parameter("gt_topic", "/simple_drone/gt_pose")
        self.declare_parameter("gt_output_file", "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/tum_results/gt_tum.txt")
        
        est_topic = self.get_parameter("est_topic").get_parameter_value().string_value
        self.est_filename = self.get_parameter("est_output_file").get_parameter_value().string_value
        
        gt_topic = self.get_parameter("gt_topic").get_parameter_value().string_value
        self.gt_filename = self.get_parameter("gt_output_file").get_parameter_value().string_value
        
        self.est_file = open(self.est_filename, 'w')
        self.gt_file = open(self.gt_filename, 'w')
        
        self.get_logger().info(f"Saving ESTIMATION from {est_topic} to {self.est_filename}")
        self.get_logger().info(f"Saving GROUND TRUTH from {gt_topic} to {self.gt_filename}")
        
        self.est_sub = self.create_subscription(
            Odometry,
            est_topic,
            self.est_callback,
            10)
            
        self.gt_sub = self.create_subscription(
            Pose,
            gt_topic,
            self.gt_callback,
            10)

    def est_callback(self, msg):
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        timestamp = sec + (nanosec / 1e9)
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        line = f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
        self.est_file.write(line)

    def gt_callback(self, msg):
        now = self.get_clock().now()
        sec, nanosec = now.seconds_nanoseconds()
        timestamp = sec + (nanosec / 1e9)
        
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        
        line = f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
        self.gt_file.write(line)

    def destroy_node(self):
        self.est_file.close()
        self.gt_file.close()
        self.get_logger().info("Files saved and closed successfully.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OdomAndGtToTum()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok(): 
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()