#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

from sparx_agency.core.common.types import PoseSE3
from sparx_agency.core.common.types.perception import Observation
from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, MappingPipelineConfig
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config

from sparx_agency.robots.common.spatial_math import euler_to_rot_zyx, intrinsics_from_fov
from sparx_agency.robots.common.state_converter import costmap_to_occupancygrid

from sparx_agency.robots.XTEND.adapters.xtend_robot_adapter import XtendRobotAdapter
from sparx_agency.robots.XTEND.adapters.xtend_video_adapter import XtendVideoAdapter
from sparx_agency.robots.XTEND.get_xtend_probe import RtspProbe


def normalize_angle(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def angle_diff(a: float, b: float) -> float:
    return normalize_angle(a - b)


def pose_from_yaw(yaw_rad: float) -> PoseSE3:
    R = euler_to_rot_zyx(roll=0.0, pitch=0.0, yaw=float(yaw_rad))
    t = np.zeros(3, dtype=np.float32)
    return PoseSE3(R=R, t=t)


def build_pipeline(args) -> MappingPipeline:
    costmap = ProbabilisticGridCostmap(
        ProbabilisticGridConfig(
            resolution_m=args.resolution_m,
            width=args.grid_width,
            height=args.grid_height,
            origin_x=args.origin_x,
            origin_y=args.origin_y,
        )
    )

    depth_model = DepthAnythingV2DepthModel(
        DepthAnythingV2Config(model_id=args.depth_model_id, device=args.device)
    )

    return MappingPipeline(
        costmap=costmap,
        depth_model=depth_model,
        cfg=MappingPipelineConfig(
            stride=args.stride,
            z_min=args.z_min,
            z_max=args.z_max,
            range_min=args.range_min,
            range_max=args.range_max,
            debug=args.debug,
        ),
    )


class XtendMapRoomNode(Node):
    """
    Exposes the map in two ways:
      1) In memory: pipeline.costmap
      2) Published: /xtend/costmap/occupancy
    """
    def __init__(self, pipeline: MappingPipeline, frame_id: str = "map"):
        super().__init__("xtend_map_room")
        self.pipeline = pipeline
        self.frame_id = frame_id
        self.pub_occ = self.create_publisher(OccupancyGrid, "/xtend/costmap/occupancy", 10)

    def publish_costmap(self) -> None:
        stamp = self.get_clock().now().to_msg()
        msg = costmap_to_occupancygrid(self.pipeline.costmap, stamp=stamp, frame_id=self.frame_id)
        self.pub_occ.publish(msg)


async def mapping_loop(robot: XtendRobotAdapter, video: XtendVideoAdapter, pipeline: MappingPipeline, process_hz: float):
    period = 1.0 / max(process_hz, 1e-6)
    while True:
        yaw = robot.telemetry.last.yaw_rad if robot.telemetry.last else None
        pose = pose_from_yaw(yaw) if yaw is not None else None

        obs = video.get_latest_observation(pose_map_base=pose)
        if obs is not None:
            pipeline.step(obs)

        await asyncio.sleep(period)


async def publish_loop(node: XtendMapRoomNode, publish_hz: float):
    period = 1.0 / max(publish_hz, 1e-6)
    while True:
        node.publish_costmap()
        await asyncio.sleep(period)


async def ros_spin_loop(node: Node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.0)
        await asyncio.sleep(0.01)


async def keepalive_loop(robot: XtendRobotAdapter):
    """
    If WS reconnects, server sometimes expects immediate traffic.
    Hover re-push is a cheap heartbeat at the control layer.
    """
    while True:
        await robot.control.hover()
        await asyncio.sleep(1.0)


async def motion_scenario(robot: XtendRobotAdapter, takeoff_sec: float, yaw_cmd: int):
    """
    Up -> rotate 360 -> down
    """
    await robot.control.hover()
    await asyncio.sleep(0.5)

    await robot.control.disarm()
    await asyncio.sleep(0.5)

    await robot.control.arm()
    await asyncio.sleep(0.5)

    # Takeoff time-based (tune takeoff_sec to reach ~1–1.5m)
    await robot.control.takeoff(seconds=takeoff_sec)
    await asyncio.sleep(0.5)

    # Rotate ~360 using telemetry yaw integration
    start_yaw = robot.telemetry.last.yaw_rad if robot.telemetry.last else 0.0
    last = float(start_yaw)
    acc = 0.0

    try:
        await robot.control.set_xy_yaw_trigger(x=0, y=0, trigger=0, yaw=int(yaw_cmd))
        while acc < 2.0 * math.pi:
            await asyncio.sleep(0.02)
            if not robot.telemetry.last:
                continue
            now = float(robot.telemetry.last.yaw_rad)
            step = abs(angle_diff(now, last))
            if step < 0.5:  # reject glitch jumps
                acc += step
            last = now
    finally:
        await robot.control.set_xy_yaw_trigger(x=0, y=0, trigger=0, yaw=0)

    await asyncio.sleep(0.5)

    await robot.control.land(seconds=3.1)
    await asyncio.sleep(0.5)

    await robot.control.disarm()


async def main_async(args):
    pipeline = build_pipeline(args)

    # ROS publisher
    rclpy.init()
    node = XtendMapRoomNode(pipeline=pipeline, frame_id="map")

    # WS robot
    robot = XtendRobotAdapter(
        host=args.host,
        port=args.port,
        robot_uid=args.robot_uid,
        frequency_hz=args.ws_frequency_hz,
    )

    # RTSP video
    intr = intrinsics_from_fov(args.width, args.height, args.hfov_deg, args.vfov_deg)
    rtsp = RtspProbe(uri=args.rtsp_uri, latency_ms=args.rtsp_latency_ms)
    rtsp.start()
    video = XtendVideoAdapter(rtsp_probe=rtsp, intrinsics=intr, frame_id="xtend_camera")

    await robot.start()

    tasks = [
        asyncio.create_task(ros_spin_loop(node)),
        asyncio.create_task(keepalive_loop(robot)),
        asyncio.create_task(mapping_loop(robot, video, pipeline, process_hz=args.process_hz)),
        asyncio.create_task(publish_loop(node, publish_hz=args.publish_hz)),
    ]

    try:
        await motion_scenario(robot, takeoff_sec=args.takeoff_sec, yaw_cmd=args.yaw_cmd)
        await asyncio.sleep(1.0)  # publish a bit after landing
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        await robot.stop()
        rtsp.stop()
        node.destroy_node()
        rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser("XTEND map a room: takeoff -> 360 -> land + publish costmap")

    # WS
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--robot-uid", required=True)
    p.add_argument("--ws-frequency-hz", type=float, default=30.0)

    # RTSP
    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8556/osd_snapshot")
    p.add_argument("--rtsp-latency-ms", type=int, default=0)

    # Rates
    p.add_argument("--process-hz", type=float, default=2.0)
    p.add_argument("--publish-hz", type=float, default=2.0)

    # Motion tuning
    p.add_argument("--takeoff-sec", type=float, default=3.1)
    p.add_argument("--yaw-cmd", type=int, default=1000)

    # Intrinsics fallback
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hfov-deg", type=float, default=130.0)
    p.add_argument("--vfov-deg", type=float, default=90.0)

    # DepthAnything + mapping
    p.add_argument("--depth-model-id", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--z-min", type=float, default=-1.5)
    p.add_argument("--z-max", type=float, default=1.0)
    p.add_argument("--range-min", type=float, default=0.5)
    p.add_argument("--range-max", type=float, default=15.0)
    p.add_argument("--debug", action="store_true")

    # Costmap
    p.add_argument("--resolution-m", type=float, default=0.3)
    p.add_argument("--grid-width", type=int, default=400)
    p.add_argument("--grid-height", type=int, default=400)
    p.add_argument("--origin-x", type=float, default=-50.0)
    p.add_argument("--origin-y", type=float, default=-50.0)

    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
