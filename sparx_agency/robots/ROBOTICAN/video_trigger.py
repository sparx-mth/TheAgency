#!/usr/bin/env python3
"""video_trigger.py

Minimal video-stream trigger: calls SetVideoMode (playing=True) on a repeating
interval and keeps the gcs_keep_alive heartbeat running so the drone stack
doesn't cut the stream.

Run this INSIDE the 'it' container (needs Foxy + video_handler_interfaces):
  ./run_video_trigger.sh --drone-id R1 --host-ip 127.0.0.1 --port 5001

Re-triggers every ~retrigger_interval_sec instead of firing once: confirmed
live that a Sphera/R1 restart silently drops the stream (video_handler comes
back up but nothing is telling it to play again) with no error anywhere
downstream -- frames just stop, and previously this needed a human to notice
and manually re-run this script. Calling SetVideoMode(playing=True) again on
an already-playing stream is a harmless no-op, so periodic re-triggering is a
simple, stateless way to recover automatically without needing to detect
"is the stream actually still flowing" (this node has no visibility into
that -- it doesn't know the frame output directory rooster_frame_dir_publisher
writes to).

Single responsibility: trigger video + keep-alive. No cmd_nav routing, no
FCU commands, no status publishing.
"""
from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from video_handler_interfaces.srv import SetVideoMode


class VideoTriggerNode(Node):
    def __init__(self, drone_id: str, host_ip: str, port: int, width: int, height: int,
                 retrigger_interval_sec: float = 5.0):
        super().__init__("video_trigger")
        self._call_in_flight = False
        self._n_triggers = 0

        self._keep_alive_pub = self.create_publisher(
            Bool, f"/{drone_id}/gcs_keep_alive", 10
        )
        self.create_timer(1.0, self._on_keep_alive)

        self._svc = self.create_client(
            SetVideoMode, f"/{drone_id}/video_handler/set_video_mode"
        )
        self._host_ip = host_ip
        self._port = port
        self._width = width
        self._height = height
        self.create_timer(retrigger_interval_sec, self._trigger_tick)

        self.get_logger().info(
            f"video_trigger ready — waiting for {self._svc.srv_name}, "
            f"re-triggering every {retrigger_interval_sec:.1f}s"
        )

    def _on_keep_alive(self):
        msg = Bool()
        msg.data = True
        self._keep_alive_pub.publish(msg)

    def _trigger_tick(self):
        if self._call_in_flight:
            return
        if not self._svc.service_is_ready():
            self.get_logger().warn(f"{self._svc.srv_name} not ready yet...")
            return

        req = SetVideoMode.Request()
        req.camera_id = 0
        req.playing = True
        req.host = self._host_ip
        req.port = self._port
        req.resolution_width = self._width
        req.resolution_height = self._height
        req.recording = False
        req.bitrate = SetVideoMode.Request.BITRATE_1500000
        req.fps = 0

        self._call_in_flight = True
        future = self._svc.call_async(req)
        future.add_done_callback(self._on_result)
        self._n_triggers += 1
        # First trigger (or one after a gap) is the interesting one; log the
        # rest at debug so a long-running session doesn't spam INFO forever.
        log = self.get_logger().info if self._n_triggers == 1 else self.get_logger().debug
        log(f"SetVideoMode sent → {self._host_ip}:{self._port} "
            f"{self._width}x{self._height} (trigger #{self._n_triggers})")

    def _on_result(self, fut):
        self._call_in_flight = False
        try:
            r = fut.result()
            log = self.get_logger().info if self._n_triggers == 1 else self.get_logger().debug
            log(f"SetVideoMode result: success={r.success}, msg='{r.message}'")
        except Exception as e:
            self.get_logger().error(f"SetVideoMode call failed: {e}")


def main():
    p = argparse.ArgumentParser(
        description="Trigger Rooster video stream and maintain keep-alive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--drone-id", default="R1")
    p.add_argument("--host-ip", required=True,
                   help="IP of this machine on the drone network (where UDP stream is sent)")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--retrigger-interval-sec", type=float, default=5.0,
                   help="Re-send SetVideoMode on this interval so a dropped "
                        "stream (e.g. after a Sphera/R1 restart) recovers "
                        "without needing to manually re-run this script.")
    args = p.parse_args()

    rclpy.init()
    node = VideoTriggerNode(args.drone_id, args.host_ip, args.port, args.width, args.height,
                            retrigger_interval_sec=args.retrigger_interval_sec)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
