#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse
import asyncio
import csv
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
from sparx_agency.robots.XTEND.get_xtend_probe import RtspProbe, XtendProbe


def extract_bearing(robot_status: Optional[dict[str, Any]]) -> float:
    if not robot_status:
        return float("nan")

    telemetry = robot_status.get("telemetry", {})
    if not isinstance(telemetry, dict):
        return float("nan")

    details = telemetry.get("details", {})
    if not isinstance(details, dict):
        return float("nan")

    bearing = details.get("bearing")
    if bearing is None:
        return float("nan")

    try:
        return float(bearing)
    except (TypeError, ValueError):
        return float("nan")


def colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth_clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_clipped = np.clip(depth_clean, 0.0, max_depth_m)
    depth_norm = (depth_clipped / max_depth_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)


async def capture_xtend_da3_frames(args: argparse.Namespace, stop_event: asyncio.Event) -> None:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"xtend_da3_take_{run_name}"
    rgb_dir = run_dir / "rgb"
    depth_npy_dir = run_dir / "depth_npy"
    depth_vis_dir = run_dir / "depth_vis"

    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "metadata.csv"

    print(f"[take] Output directory: {run_dir}")
    print(f"[take] Loading DA3 engine: {args.engine_path}")
    print(f"[take] Loading camera YAML: {args.config_yaml}")

    depth_model = DA3TensorRTModel(
        engine_path=args.engine_path,
        yaml_path=args.config_yaml,
    )

    rtsp = RtspProbe(uri=args.rtsp_uri, latency_ms=args.rtsp_latency_ms)

    ws_uri = f"ws://{args.xtend_host}:{args.xtend_port}"
    xtend_probe = XtendProbe(
        ws_uri=ws_uri,
        robot_uid=args.robot_uid,
        mode=args.xtend_mode,
        frequency_hz=args.xtend_frequency_hz,
        raw_dump_seconds=args.xtend_raw_dump_seconds,
        print_robot_status_only=True,
    )

    async def telemetry_task() -> None:
        try:
            await xtend_probe.run()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(f"[take] XTEND telemetry error: {exc}")

    telemetry_runner = asyncio.create_task(telemetry_task())

    with open(metadata_path, "w", newline="") as metadata_fp:
        writer = csv.writer(metadata_fp)
        writer.writerow([
            "frame_idx",
            "stamp_sec",
            "bearing",
            "bearing_status_age_sec",
            "rgb_path",
            "depth_npy_path",
            "depth_vis_path",
            "rgb_height",
            "rgb_width",
            "depth_height",
            "depth_width",
            "depth_min_m",
            "depth_max_m",
            "depth_mean_m",
        ])

        rtsp.start()
        print("[take] RTSP started. Capturing without movement.")
        print("[take] XTEND telemetry started. Capturing bearing from ROBOT_STATUS.")
        print("[take] Press Ctrl+C to stop safely.")

        try:
            period = 1.0 / max(args.capture_hz, 1e-6)
            frame_idx = 0
            start_t = time.time()
            last_capture_t = 0.0

            while not stop_event.is_set():
                now = time.time()

                if args.duration_sec > 0.0 and (now - start_t) >= args.duration_sec:
                    print("[take] Duration reached. Stopping.")
                    break

                if args.max_frames > 0 and frame_idx >= args.max_frames:
                    print("[take] Max frames reached. Stopping.")
                    break

                if now - last_capture_t < period:
                    await asyncio.sleep(0.001)
                    continue

                latest = rtsp.get_latest()
                if latest is None:
                    await asyncio.sleep(0.01)
                    continue

                bgr, stamp_sec = latest

                bearing = extract_bearing(xtend_probe.last_robot_status)
                if xtend_probe.last_robot_status_t > 0.0:
                    bearing_status_age_sec = now - xtend_probe.last_robot_status_t
                else:
                    bearing_status_age_sec = float("nan")

                depth_m = depth_model.infer_depth(bgr).astype(np.float32)
                depth_vis = colorize_depth(depth_m, max_depth_m=args.max_depth_m)

                frame_name = f"frame_{frame_idx:06d}"
                rgb_path = rgb_dir / f"{frame_name}.jpg"
                depth_npy_path = depth_npy_dir / f"{frame_name}.npy"
                depth_vis_path = depth_vis_dir / f"{frame_name}.png"

                cv2.imwrite(str(rgb_path), bgr)
                np.save(str(depth_npy_path), depth_m)
                cv2.imwrite(str(depth_vis_path), depth_vis)

                finite_depth = depth_m[np.isfinite(depth_m)]
                if finite_depth.size > 0:
                    depth_min = float(np.min(finite_depth))
                    depth_max = float(np.max(finite_depth))
                    depth_mean = float(np.mean(finite_depth))
                else:
                    depth_min = float("nan")
                    depth_max = float("nan")
                    depth_mean = float("nan")

                writer.writerow([
                    frame_idx,
                    stamp_sec,
                    bearing,
                    bearing_status_age_sec,
                    str(rgb_path),
                    str(depth_npy_path),
                    str(depth_vis_path),
                    int(bgr.shape[0]),
                    int(bgr.shape[1]),
                    int(depth_m.shape[0]),
                    int(depth_m.shape[1]),
                    depth_min,
                    depth_max,
                    depth_mean,
                ])
                metadata_fp.flush()

                print(
                    f"[take] saved frame={frame_idx} "
                    f"rgb={bgr.shape[1]}x{bgr.shape[0]} "
                    f"depth={depth_m.shape[1]}x{depth_m.shape[0]} "
                    f"mean_depth={depth_mean:.3f}m "
                    f"bearing={bearing:.6f}"
                )

                frame_idx += 1
                last_capture_t = now

        finally:
            print("[take] Shutdown requested. Stopping RTSP...")
            rtsp.stop()
            telemetry_runner.cancel()
            await asyncio.gather(telemetry_runner, return_exceptions=True)
            cv2.destroyAllWindows()
            print("[take] RTSP stopped.")
            print(f"[take] Saved metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture XTEND RGB frames and DA3 depth frames without movement."
    )

    p.add_argument(
        "--rtsp-uri",
        default="rtsp://192.0.0.15:8510/active_drone_fpv",
        help="XTEND RTSP stream URI.",
    )
    p.add_argument(
        "--rtsp-latency-ms",
        type=int,
        default=0,
        help="RTSP latency in milliseconds.",
    )

    p.add_argument(
        "--xtend-host",
        default="192.0.0.15",
        help="XTEND websocket host.",
    )
    p.add_argument(
        "--xtend-port",
        type=int,
        default=8000,
        help="XTEND websocket port.",
    )
    p.add_argument(
        "--robot-uid",
        default="drnb177ede2",
        help="XTEND robot UID used to select telemetry.",
    )
    p.add_argument(
        "--xtend-mode",
        choices=["send", "listen", "both"],
        default="both",
        help="XTEND telemetry websocket mode.",
    )
    p.add_argument(
        "--xtend-frequency-hz",
        type=float,
        default=10.0,
        help="Frequency for neutral VIRTUAL_CONTROLLER heartbeat messages.",
    )
    p.add_argument(
        "--xtend-raw-dump-seconds",
        type=float,
        default=0.0,
        help="Print raw XTEND telemetry JSON for the first N seconds. Use 0 to disable.",
    )

    p.add_argument(
        "--engine-path",
        default=str(
            Path.home()
            / "depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3METRIC-LARGE/DA3METRIC-LARGE_v1.engine"
        ),
        help="Path to DA3 TensorRT .engine file.",
    )
    p.add_argument(
        "--config-yaml",
        default=str(Path.home() / "depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_example.yaml"),
        help="Path to camera intrinsics YAML."
    ),

    p.add_argument(
        "--output-dir",
        default=str(Path.home() / "Documents" / "xtend_da3_takes"),
        help="Directory where captures will be saved.",
    )
    p.add_argument(
        "--capture-hz",
        type=float,
        default=1.0,
        help="Capture rate in Hz.",
    )
    p.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Capture duration. Use 0 for unlimited until Ctrl+C or max-frames.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to capture. Use 0 for unlimited.",
    )
    p.add_argument(
        "--max-depth-m",
        type=float,
        default=15.0,
        help="Max depth used only for visualization color scaling.",
    )

    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        if not stop_event.is_set():
            print("[take] Stop signal received.")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: request_stop())

    await capture_xtend_da3_frames(args, stop_event)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()