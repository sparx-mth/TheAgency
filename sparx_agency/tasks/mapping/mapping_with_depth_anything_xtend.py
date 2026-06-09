#!/usr/bin/env python3
import argparse
import asyncio
import time
from pathlib import Path

from sparx_agency.core.mapping.pipeline.mapping_pipeline import MappingPipeline, MappingPipelineConfig  # :contentReference[oaicite:2]{index=2}
from sparx_agency.core.mapping.costmap.probabilistic_grid import ProbabilisticGridCostmap
from sparx_agency.core.mapping.costmap.probabilistic_grid_config import ProbabilisticGridConfig
from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.core.common.spatial_math import intrinsics_from_fov
from sparx_agency.robots.XTEND.adapters.xtend_robot_adapter import XtendRobotAdapter


async def run_xtend_mapping(args: argparse.Namespace) -> None:
    # 1) Intrinsics (XTEND stream is 1280x720 in your probe; make configurable)
    intr = intrinsics_from_fov(
        width=args.width,
        height=args.height,
        hfov_deg=args.hfov_deg,
        vfov_deg=args.vfov_deg,
    )

    # 2) Robot
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

    # 3) Pipeline
    costmap = ProbabilisticGridCostmap(ProbabilisticGridConfig())
    depth_model = DA3TensorRTModel(
        engine_path=args.engine_path,
        yaml_path=args.config_yaml,
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

    await robot.start()
    try:
        period = 1.0 / max(args.process_hz, 1e-6)
        last_print = time.time()

        while True:
            obs = robot.getLatestObservation()
            if obs is not None and obs.rgb is not None and obs.intrinsics is not None:
                pipeline.step(obs)

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                # Minimal health print
                print(f"[mapping] frames={pipeline.frame_count} cloudN={pipeline.last_cloud_global.shape[0]}")

            await asyncio.sleep(period)

    finally:
        await robot.stop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="192.0.0.15")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--robot-uid", required=True)

    p.add_argument("--rtsp-uri", default="rtsp://192.0.0.15:8510/active_drone_fpv")
    p.add_argument("--rtsp-latency-ms", type=int, default=0)

    p.add_argument("--ws-frequency-hz", type=float, default=10.0)
    p.add_argument("--process-hz", type=float, default=2.0)

    # Camera assumptions (tune later)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hfov-deg", type=float, default=130.0)
    p.add_argument("--vfov-deg", type=float, default=90.0)

    # DepthAnything V3 TensorRT
    p.add_argument(
        "--engine-path",
        default=str(
            Path.home()
            / "depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine"
        ),
        help="Path to the DA3 TensorRT .engine file.",
    )
    p.add_argument(
        "--config-yaml",
        default=str(
            Path.home()
            / "depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_example.yaml"
        ),
        help="Path to camera intrinsics YAML used by DA3TensorRTModel.",
    )

    # Pipeline filtering
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--z-min", type=float, default=-1.5)
    p.add_argument("--z-max", type=float, default=1.0)
    p.add_argument("--range-min", type=float, default=0.5)
    p.add_argument("--range-max", type=float, default=15.0)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_xtend_mapping(args))


if __name__ == "__main__":
    main()
