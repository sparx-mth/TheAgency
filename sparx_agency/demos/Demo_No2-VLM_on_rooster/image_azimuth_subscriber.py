#!/usr/bin/env python3
import os
import time

import argparse
from typing import Optional


import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data

from concurrent.futures import ThreadPoolExecutor

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PointStamped
from message_filters import Subscriber, TimeSynchronizer

from fcu_driver_interfaces.msg import UAVState
from sparx_agency.robots.common.txt_and_image_utils import _update_sidecar_json


def _remap_path(local_path: str, src_root: Optional[str], dst_root: Optional[str]) -> str:
    if not src_root or not dst_root:
        return os.path.abspath(local_path)
    local_path = os.path.abspath(local_path)
    src_root = os.path.abspath(src_root)
    try:
        rel = os.path.relpath(local_path, src_root)
    except ValueError:
        return local_path
    return os.path.join(dst_root, rel)



def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class ImageStateBuffer(Node):
    """
    Node layout:

    - When idle (default): no subscription to image/state -> no image traffic
      from the other machine.

    """

    def __init__(self, args):
        super().__init__("image_state_buffer")

        self.out_dir = None
        self.drone_id = args.drone_id
        self.pose_mode = getattr(args, "pose_mode", "state")
        self.base_dir = os.path.abspath(args.out_dir)

        self.bridge = CvBridge()
        self.reentrant_cb_group = ReentrantCallbackGroup()

        # Topics
        self.image_topic_name =  f"/{self.drone_id}/selected_frame"

        self.state_topic_name = f"/{self.drone_id}/fcu/state"

        # Latest messages
        self.last_img_msg: Optional[Image] = None
        self.last_state_msg: Optional[UAVState] = None
        self._last_img_stamp = None  # tuple(sec, nanosec)
        self._same_frame_hits = 0
        self._freeze_warn_every = 30  # log every N repeats


        # Approx sync tolerance (seconds)
        self.sync_slop = 0.10  # 100 ms

        # Subscriptions (created/destroyed on demand)
        self.image_sub = self.create_subscription(
            Image, self.image_topic_name, self.image_cb, qos_profile_sensor_data)
        self.state_sub = self.create_subscription(
            UAVState, self.state_topic_name, self.state_cb, qos_profile_sensor_data)
        self.get_logger().info(
            f"ImageStateBuffer started for {self.drone_id}\n"
            f"  image topic: {self.image_topic_name}\n"
            f"  state topic: {self.state_topic_name}\n"
            f"  base_dir:     {self.base_dir}\n"
            f"  capture:     ON (saving images only, no VLM here)"
        )

    # ------------------------------------------------------------------
    # Sub callbacks: keep latest messages in memory
    # ------------------------------------------------------------------
    def state_cb(self, state_msg: UAVState):
        # Only keep the latest message
        self.last_state_msg = state_msg
        # self.get_logger().info("State msg received (buffered)") # Removed verbose logging


    # ------------------------------------------------------------------
    # Helper: extract pose dict from UAVState
    # ------------------------------------------------------------------
    @staticmethod
    def extract_pose_from_state(state_msg: UAVState) -> dict:
        """
        Best-effort extraction of {x,y,z,yaw} from UAVState.

        Adjust this mapping to match your exact UAVState definition.
        """
        pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}

        # Position-like fields
        if hasattr(state_msg, "position"):
            p = state_msg.position
            pose["x"] = p.x  # float(getattr(p, "x", 0.0))
            pose["y"] = p.y  # float(getattr(p, "y", 0.0))
            pose["z"] = p.z  # float(getattr(p, "z", 0.0))

        # Heading-like field
        if hasattr(state_msg, "azimuth"):
            pose["yaw"] = float(state_msg.azimuth)

        for k, v in pose.items():
            pose[k] = round(float(v), 5)

        return pose

    @staticmethod
    def parse_frame_id( frame_id: str):
        """
        Expected format:
          "<out_dir>_____ <azimuth>"
        Example:
          "2025_12_21___15_30_49_____95.18695746993248"
        Returns: (out_dir: str, azimuth: float) or (frame_id, None) if not parseable
        """
        if not frame_id:
            return "", None

        sep = "_____"
        if sep not in frame_id:
            return frame_id, None

        left, right = frame_id.split(sep, 1)
        out_dir = left.strip()
        az_str = right.strip()

        try:
            az = float(az_str)
        except ValueError:
            az = None

        return out_dir, az

    # ------------------------------------------------------------------
    # Capture service: save one JPG+JSON  from latest messages
    # ------------------------------------------------------------------
    def image_cb(self, img_msg: Image):
        try:

            t_img = _stamp_to_sec(img_msg.header.stamp)
            stamp_tup = (img_msg.header.stamp.sec, img_msg.header.stamp.nanosec)

            if stamp_tup == self._last_img_stamp:
                self._same_frame_hits += 1
                if (self._same_frame_hits % self._freeze_warn_every) == 0:
                    self.get_logger().warn(
                        f"Frozen/repeated frame detected (same stamp) hits={self._same_frame_hits}. Skipping."
                    )
                return

            # pose from state if available + within slop
            pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
            if self.last_state_msg is not None:
                t_state = _stamp_to_sec(self.last_state_msg.header.stamp)
                dt = abs(t_img - t_state)
                if dt <= self.sync_slop:
                    pose = ImageStateBuffer.extract_pose_from_state(self.last_state_msg)

            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            msg = f"Failed to convert image: {e}"
            self.get_logger().error(msg)
            return

        unique_out_dir, azimuth = ImageStateBuffer.parse_frame_id(img_msg.header.frame_id)
        # Optionally override yaw with AprilTag azimuth if available
        if azimuth is not None:
            pose["yaw"] = round(azimuth, 5)

        # Use image ROS time for filename
        stamp = img_msg.header.stamp
        t_sec = _stamp_to_sec(stamp)

        # 1. Get the 0.1s digit (the first decimal place)
        # We use % 1 to get the decimal part, then * 10 to make it a whole number
        decisec = int(round((t_sec % 1), 1) * 10) % 10

        # 2. Format the time string and append the decisecond
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_sec))
        base_name = f"{self.drone_id}_{ts_str}"  # Result: R2_20251224_142736_1

        # 3. Define paths
        self.out_dir = os.path.join(self.base_dir, str(unique_out_dir))
        jpg_path = os.path.join(self.out_dir, base_name + ".jpg")
        json_path = os.path.join(self.out_dir, base_name + ".json")
        img_basename = os.path.basename(jpg_path)

        os.makedirs(self.out_dir, exist_ok=True)
        # Update 'latest' symbolic link
        latest_link = os.path.join(self.base_dir, "latest")
        latest_ann_link = os.path.join(self.base_dir, "latest_ann")
        try:
            if os.path.lexists(latest_ann_link):
                os.remove(latest_ann_link)
            if os.path.lexists(latest_link):
                os.remove(latest_link)
            os.symlink(self.out_dir, latest_link)
        except Exception as e:
            self.get_logger().error(f"Failed to create symlink {latest_link}: {e}")


        self.save_image_to_dir(cv_img, jpg_path)

        # First create/update JSON with pose and image name
        _update_sidecar_json(json_path, pose, img_basename, vlm_text=None)
        self._last_img_stamp = stamp_tup
        self._same_frame_hits = 0

    def save_image_to_dir(self, cv_img, jpg_path: str):
        import cv2
        try:
            # txt = f"t={stamp.sec}.{stamp.nanosec:09d}"
            # cv2.putText(cv_img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(jpg_path, cv_img)
            self.get_logger().info(f"Saved: {jpg_path}")
        except Exception as e:
            msg = f"Failed to save image {jpg_path}: {e}"
            self.get_logger().error(msg)

    # ------------------------------------------------------------------
    # Cleanup (Subscriptions are destroyed in destroy_node, which is fine)
    # ------------------------------------------------------------------
    def destroy_node(self):
        if self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
        if self.state_sub is not None:
            self.destroy_subscription(self.state_sub)
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(
        description="On-demand capture: /<id>/capture_frame using continuous subscriptions."
    )
    parser.add_argument(
        "--drone-id",
        default="R2",
        help="Drone ID prefix (topics: /<id>/camera/image_raw, /<id>/fcu/state)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        default="/home/user/jetson-containers/data/R2",
        help="Output directory for jpg+json pairs",
    )
    parser.add_argument(
        "--pose-mode",
        choices=["state", "april", "both"],
        default="state",
        help=(
            "How to compute pose for JSON: "
            "'state' = from /fcu/state (current behavior), "
            "'april' = from AprilTag-based function, "
            "'both' = merge AprilTag pose into state pose."
        ),
    )

    args = parser.parse_args()

    rclpy.init()
    node = ImageStateBuffer(args)

    # Use MultiThreadedExecutor, although SingleThreadedExecutor would also work now.
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()