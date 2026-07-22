#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo


@dataclass
class StampRecord:
    topic: str
    stamp_sec: float
    arrival_sec: float
    seq: int


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class XtendTimestampComparator(Node):
    def __init__(
        self,
        rgb_topic: str,
        camera_info_topic: str,
        depth_topic: str,
        reliability: str,
        print_every: int,
    ):
        super().__init__("xtend_timestamp_comparator")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if reliability == "best_effort"
                else ReliabilityPolicy.RELIABLE
            ),
        )

        self.print_every = max(1, int(print_every))
        self.seq = {
            "rgb": 0,
            "camera_info": 0,
            "depth": 0,
        }

        self.latest: dict[str, Optional[StampRecord]] = {
            "rgb": None,
            "camera_info": None,
            "depth": None,
        }

        self.create_subscription(Image, rgb_topic, self.rgb_cb, qos)
        # self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_cb, qos)
        self.create_subscription(Image, depth_topic, self.depth_cb, qos)

        self.get_logger().info(f"RGB topic:         {rgb_topic}")
        self.get_logger().info(f"CameraInfo topic:  {camera_info_topic}")
        self.get_logger().info(f"Depth topic:       {depth_topic}")
        self.get_logger().info(f"QoS reliability:   {reliability}")

    def now_sec(self) -> float:
        now = self.get_clock().now().to_msg()
        return stamp_to_sec(now)

    def update(self, key: str, topic: str, header) -> None:
        self.seq[key] += 1

        rec = StampRecord(
            topic=topic,
            stamp_sec=stamp_to_sec(header.stamp),
            arrival_sec=self.now_sec(),
            seq=self.seq[key],
        )
        self.latest[key] = rec

        if rec.seq % self.print_every == 0:
            self.print_report(key, rec)

    def fmt_delta(self, a: Optional[StampRecord], b: Optional[StampRecord]) -> str:
        if a is None or b is None:
            return "NA"
        return f"{(a.stamp_sec - b.stamp_sec) * 1000.0:+.3f} ms"

    def fmt_age(self, rec: Optional[StampRecord]) -> str:
        if rec is None:
            return "NA"
        return f"{(rec.arrival_sec - rec.stamp_sec) * 1000.0:+.3f} ms"

    def print_report(self, changed_key: str, rec: StampRecord) -> None:
        rgb = self.latest["rgb"]
        cam = self.latest["camera_info"]
        dep = self.latest["depth"]

        print("\n--- timestamp sample ---")
        print(
            f"changed={changed_key:11s} "
            f"seq={rec.seq:06d} "
            f"stamp={rec.stamp_sec:.9f} "
            f"arrival={rec.arrival_sec:.9f} "
            f"age={self.fmt_age(rec)}"
        )

        if rgb is not None:
            print(
                f"rgb          seq={rgb.seq:06d} "
                f"stamp={rgb.stamp_sec:.9f} "
                f"age={self.fmt_age(rgb)}"
            )
        else:
            print("rgb          NA")

        if cam is not None:
            print(
                f"camera_info  seq={cam.seq:06d} "
                f"stamp={cam.stamp_sec:.9f} "
                f"age={self.fmt_age(cam)} "
                f"cam-rgb={self.fmt_delta(cam, rgb)}"
            )
        else:
            print("camera_info  NA")

        if dep is not None:
            print(
                f"depth        seq={dep.seq:06d} "
                f"stamp={dep.stamp_sec:.9f} "
                f"age={self.fmt_age(dep)} "
                f"depth-rgb={self.fmt_delta(dep, rgb)} "
                f"depth-cam={self.fmt_delta(dep, cam)}"
            )
        else:
            print("depth        NA")

    def rgb_cb(self, msg: Image) -> None:
        self.update("rgb", "/xtend/rgb", msg.header)

    # def camera_info_cb(self, msg: CameraInfo) -> None:
    #     self.update("camera_info", "/xtend/camera_info", msg.header)

    def depth_cb(self, msg: Image) -> None:
        self.update("depth", "/xtend/depth_m", msg.header)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-topic", default="/xtend/rgb")
    parser.add_argument("--camera-info-topic", default="/xtend/camera_info")
    parser.add_argument("--depth-topic", default="/xtend/depth_m")
    parser.add_argument(
        "--reliability",
        choices=["reliable", "best_effort"],
        default="best_effort",
    )
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = XtendTimestampComparator(
        rgb_topic=args.rgb_topic,
        camera_info_topic=args.camera_info_topic,
        depth_topic=args.depth_topic,
        reliability=args.reliability,
        print_every=args.print_every,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()