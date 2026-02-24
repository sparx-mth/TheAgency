import rclpy
from video_handler_interfaces.srv import SetVideoMode
from std_msgs.msg import Bool
from rclpy.node import Node
from rclpy.client import Client
import signal
import subprocess
import datetime
import os
import sys

class VideoSaver(Node):
    def __init__(self):
        super().__init__('video_saver')
        self.id = "R2" 
        
        # 1. Generate a unique filename with a timestamp (.mkv – VLC-compatible)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"drone_video_{self.id}_{timestamp}.mkv"
        self.get_logger().info(f"Saving video to: {self.filename}")

        self.set_video_mode_srv = self.create_client(SetVideoMode, f"/{self.id}/video_handler/set_video_mode")
        self.gcs_keep_alive_publisher = self.create_publisher(Bool, f"/{self.id}/gcs_keep_alive", 10)
        self.gcs_keep_alive_timer = self.create_timer(1, self.gcs_keep_alive_timer_callback)
        self.video_on_timer = self.create_timer(3, self.video_on_timer_callback)

        # 2. GStreamer pipeline – save to MKV (Matroska) so VLC can open it.
        #    matroskamux writes index/headers incrementally, so VLC does NOT
        #    need to wait for the file to be finalised (unlike mp4mux which
        #    requires a moov-atom at the end and produces a black screen if the
        #    process exits without proper EOS).
        pipeline = [
            "gst-launch-1.0",
            "-e",       # Send EOS on SIGINT so matroskamux can close cleanly
            "udpsrc", "port=5001",
            "caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96",
            "!", "rtph264depay",
            "!", "h264parse",
            "!", "matroskamux",   # MKV – works in VLC, ffplay, and all major players
            "!", "filesink", f"location={self.filename}"
        ]

        # start_new_session=True puts GStreamer in its own process group so that
        # Ctrl+C from the terminal reaches only Python; Python then sends ONE
        # clean SIGINT to GStreamer for proper EOS/index writing.
        self.g_process = subprocess.Popen(pipeline, start_new_session=True)

    def gcs_keep_alive_timer_callback(self):
        msg = Bool()
        msg.data = True
        self.gcs_keep_alive_publisher.publish(msg)

    def video_on_timer_callback(self):
        self.video_on_timer.cancel()
        set_video_mode_request = SetVideoMode.Request()
        set_video_mode_request.camera_id = 0 
        set_video_mode_request.playing = True 
        set_video_mode_request.port = 5001 
        set_video_mode_request.host = "192.168.131.20" 
        
        HIGH_RESO = 640
        set_video_mode_request.resolution_width = HIGH_RESO 
        set_video_mode_request.resolution_height = int(HIGH_RESO * 9 / 16)
        set_video_mode_request.recording = False 
        set_video_mode_request.bitrate = SetVideoMode.Request.BITRATE_1500000
        set_video_mode_request.fps = 0 

        self.get_logger().info(f"Requesting video stream to port 5001...")
        future = self.set_video_mode_srv.call_async(set_video_mode_request)
        
        def set_video_mode_cb(future):
            result = future.result()
            self.get_logger().info(f"Stream started: {result.success}")

        future.add_done_callback(set_video_mode_cb)

def main(args=None):
    rclpy.init(args=args)
    node = VideoSaver()

    # Catch SIGINT/SIGTERM: send EOS to GStreamer first, wait for it to
    # finalise the MKV index, then clean up ROS.
    def _shutdown(signum, frame):
        node.get_logger().info("Shutdown: sending EOS to GStreamer, waiting for MKV finalisation...")
        try:
            node.g_process.send_signal(signal.SIGINT)   # triggers EOS in gst-launch
            node.g_process.wait(timeout=15)             # wait for matroskamux to close
        except subprocess.TimeoutExpired:
            node.get_logger().warn("GStreamer EOS timed out – killing process.")
            node.g_process.kill()
        except Exception as e:
            node.get_logger().warn(f"GStreamer shutdown error: {e}")
            node.g_process.kill()

        if os.path.exists(node.filename) and os.path.getsize(node.filename) == 0:
            os.remove(node.filename)
            node.get_logger().info(f"Removed empty file: {node.filename}")
        else:
            node.get_logger().info(f"Video saved to: {node.filename} ({os.path.getsize(node.filename)//1024} KB)")

        node.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    rclpy.spin(node)


if __name__ == "__main__":
    main()