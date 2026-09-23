#!/usr/bin/env python3
"""rooster_imu_video_sync.py

Sole owner of the FCU/MAVLink link (no other node reads it directly).
Continuously estimates the host<->FCU clock offset from HIGHRES_IMU's
time_usec (smoothed running estimate, not the MAVLink TIMESYNC protocol --
not worth it over this wired USB link). For each frame announced on
rgb_frame_sync, interpolates IMU to that frame's exact capture instant and
publishes the pair on frame_imu_sync.
"""
from __future__ import annotations

import argparse
import bisect
import json
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pymavlink import mavutil

FCU_DEVICE = "/dev/FCU"
FCU_BAUD = 115200
HEARTBEAT_TIMEOUT_SEC = 10.0
OFFSET_EMA_ALPHA = 0.1
OFFSET_OUTLIER_REJECT_SEC = 0.05
MAX_LOOKUP_GAP_SEC = 0.05


class FCULink:
    """Background MAVLink reader: HIGHRES_IMU -> a host-clock-aligned ring buffer."""

    def __init__(self, device: str, baud: int, buffer_sec: float):
        self._conn = mavutil.mavlink_connection(device, baud=baud)
        # Some FCUs only start streaming once they've seen a GCS heartbeat.
        self._conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        hb = self._conn.recv_match(type="HEARTBEAT", timeout=HEARTBEAT_TIMEOUT_SEC)
        if hb is None:
            raise RuntimeError(f"No HEARTBEAT from FCU on {device} within {HEARTBEAT_TIMEOUT_SEC}s")

        self._conn.mav.request_data_stream_send(
            self._conn.target_system, self._conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS, 50, 1)

        self._buffer_sec = buffer_sec
        self._lock = threading.Lock()
        self._times: deque[float] = deque()  # host-clock-aligned monotonic seconds
        self._samples: deque[tuple] = deque()  # (ax, ay, az, gx, gy, gz)
        self._offset = None  # host_monotonic - fcu_time_usec/1e6

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self):
        while not self._stop.is_set():
            msg = self._conn.recv_match(type="HIGHRES_IMU", blocking=True, timeout=0.2)
            if msg is None:
                continue
            host_now = time.monotonic()
            fcu_sec = msg.time_usec / 1e6
            raw_offset = host_now - fcu_sec

            with self._lock:
                if self._offset is None:
                    self._offset = raw_offset
                elif abs(raw_offset - self._offset) <= OFFSET_OUTLIER_REJECT_SEC:
                    self._offset += OFFSET_EMA_ALPHA * (raw_offset - self._offset)

                self._times.append(fcu_sec + self._offset)
                self._samples.append((msg.xacc, msg.yacc, msg.zacc, msg.xgyro, msg.ygyro, msg.zgyro))

                cutoff = host_now - self._buffer_sec
                while self._times and self._times[0] < cutoff:
                    self._times.popleft()
                    self._samples.popleft()

    def lookup(self, target_time: float):
        """Interpolated (ax,ay,az,gx,gy,gz) at target_time, or None if nothing
        is within MAX_LOOKUP_GAP_SEC of it."""
        with self._lock:
            times = list(self._times)
            samples = list(self._samples)

        if not times:
            return None

        idx = bisect.bisect_left(times, target_time)
        if idx == 0:
            return samples[0] if times[0] - target_time <= MAX_LOOKUP_GAP_SEC else None
        if idx == len(times):
            return samples[-1] if target_time - times[-1] <= MAX_LOOKUP_GAP_SEC else None

        t0, t1 = times[idx - 1], times[idx]
        if t1 - t0 <= 0:
            return samples[idx]
        frac = (target_time - t0) / (t1 - t0)
        s0, s1 = samples[idx - 1], samples[idx]
        return tuple(a + frac * (b - a) for a, b in zip(s0, s1))


class RoosterImuVideoSync(Node):
    def __init__(self, args):
        super().__init__("rooster_imu_video_sync")

        self.fcu = FCULink(args.device, args.baud, args.imu_buffer_sec)
        self.fcu.start()

        self.output_pub = self.create_publisher(String, args.output_topic, 10)
        self.create_subscription(String, args.sync_topic, self._on_frame_sync, 10)

        self.get_logger().info(
            f"rooster_imu_video_sync ready\n"
            f"  fcu device:  {args.device}\n"
            f"  frame in:    {args.sync_topic}\n"
            f"  synced out:  {args.output_topic}"
        )

    def _on_frame_sync(self, msg: String):
        frame = json.loads(msg.data)
        imu = self.fcu.lookup(frame["capture_monotonic"])
        if imu is None:
            self.get_logger().warn(
                f"no IMU sample within range for frame_id={frame['frame_id']} -- skipped")
            return

        ax, ay, az, gx, gy, gz = imu
        out = String()
        out.data = json.dumps({
            "path": frame["path"],
            "frame_id": frame["frame_id"],
            "capture_monotonic": frame["capture_monotonic"],
            "accel": [ax, ay, az],
            "gyro": [gx, gy, gz],
        })
        self.output_pub.publish(out)

    def destroy_node(self):
        self.fcu.stop()
        super().destroy_node()


def parse_args():
    p = argparse.ArgumentParser(
        description="Sync HIGHRES_IMU (direct MAVLink) to camera frames announced on rgb_frame_sync.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rooster-id", default="R1", help="Used to build default --sync-topic/--output-topic.")
    p.add_argument("--device", default=FCU_DEVICE)
    p.add_argument("--baud", type=int, default=FCU_BAUD)
    p.add_argument("--sync-topic", default="", help="Defaults to /<rooster-id>/rgb_frame_sync.")
    p.add_argument("--output-topic", default="", help="Defaults to /<rooster-id>/frame_imu_sync.")
    p.add_argument("--imu-buffer-sec", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.sync_topic:
        args.sync_topic = f"/{args.rooster_id}/rgb_frame_sync"
    if not args.output_topic:
        args.output_topic = f"/{args.rooster_id}/frame_imu_sync"

    rclpy.init()
    node = RoosterImuVideoSync(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
