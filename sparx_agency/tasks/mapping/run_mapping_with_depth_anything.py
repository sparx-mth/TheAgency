#!/usr/bin/env python3
"""
Generic mapping runner.

- Builds MappingPipeline once.
- Uses source plugins (xtend / gazebo / ros2_trigger / ...).
- Runs either sync (ROS spin) or async sources transparently.
"""

from __future__ import annotations
import argparse
import asyncio
import inspect
from typing import Callable, Protocol, Any

import rclpy

from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, MappingPipelineConfig
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.core.mapping.depth.depth_anything_v2 import DepthAnythingV2DepthModel, DepthAnythingV2Config


# ---------------------------
# Plugin protocol
# ---------------------------

RunnerFn = Callable[[], Any]  # may be coroutine or normal function


class SourcePlugin(Protocol):
    name: str

    def add_cli(self, p: argparse.ArgumentParser) -> None: ...
    def build_runner(self, args: argparse.Namespace, pipeline: MappingPipeline, costmap: ProbabilisticGridCostmap) -> RunnerFn: ...


# ---------------------------
# Pipeline builder (shared)
# ---------------------------

def build_pipeline(args: argparse.Namespace) -> tuple[MappingPipeline, ProbabilisticGridCostmap]:
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

    pipeline = MappingPipeline(
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
    return pipeline, costmap


# ---------------------------
# Source plugins
# ---------------------------

class XtendPlugin:
    name = "xtend"

    def add_cli(self, p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("XTEND")
        g.add_argument("--host", default="192.0.0.15")
        g.add_argument("--port", type=int, default=8000)
        g.add_argument("--robot-uid", required=True)
        g.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8556/osd_snapshot")
        g.add_argument("--rtsp-latency-ms", type=int, default=0)
        g.add_argument("--ws-frequency-hz", type=float, default=10.0)
        g.add_argument("--process-hz", type=float, default=2.0)

        # intrinsics fallback (XTEND currently has no CameraInfo topic)
        g.add_argument("--width", type=int, default=1280)
        g.add_argument("--height", type=int, default=720)
        g.add_argument("--hfov-deg", type=float, default=130.0)
        g.add_argument("--vfov-deg", type=float, default=90.0)

    def build_runner(self, args: argparse.Namespace, pipeline: MappingPipeline, costmap: ProbabilisticGridCostmap) -> RunnerFn:
        async def _run():
            from sparx_agency.core.common.spatial_math import intrinsics_from_fov
            from sparx_agency.robots.XTEND.adapters.xtend_robot_adapter import XtendRobotAdapter

            intr = intrinsics_from_fov(
                width=args.width,
                height=args.height,
                hfov_deg=args.hfov_deg,
                vfov_deg=args.vfov_deg,
            )

            robot = XtendRobotAdapter(
                host=args.host,
                port=args.port,
                robot_uid=args.robot_uid,
                frequency_hz=args.ws_frequency_hz,
                rtsp_uri=args.rtsp_uri,
                rtsp_latency_ms=args.rtsp_latency_ms,
                intrinsics=intr,
                frame_id="xtend_camera",
            )

            await robot.start()
            try:
                period = 1.0 / max(args.process_hz, 1e-6)
                while True:
                    obs = robot.getLatestObservation()
                    if obs is not None:
                        pipeline.step(obs)
                    await asyncio.sleep(period)
            finally:
                await robot.stop()

        return _run


class GazeboPlugin:
    name = "gazebo"

    def add_cli(self, p: argparse.ArgumentParser) -> None:
        # No extra args for now
        return

    def build_runner(self, args: argparse.Namespace, pipeline: MappingPipeline, costmap: ProbabilisticGridCostmap) -> RunnerFn:
        def _run():
            from sparx_agency.robots.SJTU.adapters.gazebo_ros2_ingest import GazeboRos2Ingestor

            rclpy.init()
            node = GazeboRos2Ingestor(pipeline=pipeline, costmap=costmap)
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            finally:
                node.destroy_node()
                rclpy.shutdown()

        return _run


class RoosterPlugin:
    name = "rooster_trigger"

    def add_cli(self, p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("Rooster Trigger")
        g.add_argument("--drone-id", default="R1")
        g.add_argument("--process-hz", type=float, default=2.0)

        # intrinsics fallback if no CameraInfo is used
        g.add_argument("--width", type=int, default=1280)
        g.add_argument("--height", type=int, default=720)
        g.add_argument("--hfov-deg", type=float, default=130.0)
        g.add_argument("--vfov-deg", type=float, default=90.0)

    def build_runner(self, args: argparse.Namespace, pipeline: MappingPipeline, costmap: ProbabilisticGridCostmap) -> RunnerFn:
        def _run():
            from sparx_agency.core.common.spatial_math import intrinsics_from_fov
            from sparx_agency.robots.ROBOTICAN.adapters.rooster_ingestor import RoosterIngestor

            intr = intrinsics_from_fov(
                width=args.width,
                height=args.height,
                hfov_deg=args.hfov_deg,
                vfov_deg=args.vfov_deg,
            )

            rclpy.init()
            node = RoosterIngestor(
                pipeline=pipeline,
                drone_id=args.drone_id,
                intrinsics=intr,
                process_hz=args.process_hz,
            )
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            finally:
                node.destroy_node()
                rclpy.shutdown()

        return _run


PLUGINS: dict[str, SourcePlugin] = {
    "xtend": XtendPlugin(),
    "gazebo": GazeboPlugin(),
    "ros2_trigger": RoosterPlugin(),
}


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--source", choices=sorted(PLUGINS.keys()), required=True)

    # Shared pipeline args
    p.add_argument("--depth-model-id", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--z-min", type=float, default=-1.5)
    p.add_argument("--z-max", type=float, default=1.0)
    p.add_argument("--range-min", type=float, default=0.5)
    p.add_argument("--range-max", type=float, default=15.0)
    p.add_argument("--debug", action="store_true")

    # Costmap base config (shared)
    p.add_argument("--resolution-m", type=float, default=0.3)
    p.add_argument("--grid-width", type=int, default=400)
    p.add_argument("--grid-height", type=int, default=400)
    p.add_argument("--origin-x", type=float, default=-50.0)
    p.add_argument("--origin-y", type=float, default=-50.0)

    # Add plugin args (all of them; harmless for others)
    for plug in PLUGINS.values():
        plug.add_cli(p)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline, costmap = build_pipeline(args)

    plugin = PLUGINS[args.source]
    runner = plugin.build_runner(args, pipeline, costmap)

    res = runner()
    if inspect.isawaitable(res):
        asyncio.run(res)
    else:
        # blocking (ROS spin)
        return


if __name__ == "__main__":
    main()
