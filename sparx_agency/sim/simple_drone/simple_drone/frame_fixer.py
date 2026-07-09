import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

class FrameFixer(Node):
    def __init__(self):
        super().__init__('frame_fixer')

        # Subscriber for camera info
        self.sub_cam_info = self.create_subscription(Image, '/camera/depth/camera_info', self.cam_info_callback, 10)
        self.pub_cam_info = self.create_publisher(Image, '/camera/depth/camera_info_fixed', 10)
        
        # Subscriber for Image
        self.sub_img = self.create_subscription(Image, '/camera/depth/image', self.img_callback, 10)
        self.pub_img = self.create_publisher(Image, '/camera/depth/image_fixed', 10)
        
        # Subscriber for PointCloud
        self.sub_pc = self.create_subscription(PointCloud2, '/camera/depth/points', self.pc_callback, 10)
        self.pub_pc = self.create_publisher(PointCloud2, '/camera/depth/points_fixed', 10)

    def cam_info_callback(self, msg):
        msg.header.frame_id = 'camera_link'
        self.pub_cam_info.publish(msg)

    def img_callback(self, msg):
        msg.header.frame_id = 'camera_link'
        self.pub_img.publish(msg)

    def pc_callback(self, msg):
        msg.header.frame_id = 'camera_link'
        self.pub_pc.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FrameFixer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()