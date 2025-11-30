import rclpy
from video_handler_interfaces.srv import SetVideoMode
from std_msgs.msg import Bool
from rclpy.node import Node
from rclpy.client import Client
import subprocess

'''
This code demonstrate video stream control
show video:
gst-launch-1.0 udpsrc port=5001 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false
'''
class VideoExample(Node):
    def __init__(self):
        super().__init__('video_example')
        self.id = "R2" # Rooster ID
        self.set_video_mode_srv : rclpy.client.Client = self.create_client(srv_type=SetVideoMode,
                                                          srv_name=f"/{self.id}/video_handler/set_video_mode")
        # Publish  gcs_keep_alive periodically to avoid video off failsafe
        self.gcs_keep_alive_publisher = self.create_publisher(Bool, f"/{self.id}/gcs_keep_alive", 10)
        self.gcs_keep_alive_timer = self.create_timer(1, self.gcs_keep_alive_timer_callback)
        self.video_on_timer = self.create_timer(3, self.video_on_timer_callback)

        pipeline = [
            "gst-launch-1.0",
            "udpsrc", "port=5001",
            "caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96",
            "!", "rtph264depay",
            "!", "decodebin",
            "!", "videoconvert",
            "!", "autovideosink", "sync=false"
        ]

        self.g_process = subprocess.Popen(pipeline)

    def gcs_keep_alive_timer_callback(self):
        self.get_logger().info(f"{self.id}: publish keep alive")
        msg = Bool()
        msg.data = True
        self.gcs_keep_alive_publisher.publish(msg)

    def video_on_timer_callback(self):
        self.video_on_timer.cancel()
        set_video_mode_request = SetVideoMode.Request()
        set_video_mode_request.camera_id = 0 # Main camera
        set_video_mode_request.playing = True # turn video on
        set_video_mode_request.port = 5001 # video destination port
        set_video_mode_request.host = "192.168.131.5" # video destination host
        HIGH_RESO = 640
        set_video_mode_request.resolution_width = HIGH_RESO # High resolution
        set_video_mode_request.resolution_height = int(HIGH_RESO * 9 / 16)
        set_video_mode_request.recording = False # turn off video recording on drone
        set_video_mode_request.bitrate = SetVideoMode.Request.BITRATE_1500000
        set_video_mode_request.fps = 0 # default fps

        self.get_logger().info(f"send {self.set_video_mode_srv.srv_name} [{set_video_mode_request}]")

        future = self.set_video_mode_srv.call_async(set_video_mode_request)
        def set_video_mode_cb(future: rclpy.client.Future):
            result = future.result()
            self.get_logger().info(f"{self.id}: set_video_mode_cb: {result.success}")

        future.add_done_callback(set_video_mode_cb)

def main(args=None):
    rclpy.init(args=args)
    node = VideoExample()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
