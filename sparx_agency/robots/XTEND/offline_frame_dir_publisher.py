#!/usr/bin/env python3
"""
Offline frame replay publisher — faithful mock of the online XTEND bridge.

Reads a take session (produced by take_xtend_da3_frames.py) and replays
RGB frames + bearing + optional depth through the same ROS topics used online.

--session-dir  root with metadata.csv, rgb/, depth_npy/
--input-dir    legacy: plain JPEG directory (no metadata, no bearing)

Topics: /xtend/rgb_frame_path, /xtend/bearing, /xtend/depth_frame_path
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, String

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_RELIABLE = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)


@dataclass
class FrameMeta:
    stamp_sec: float
    bearing: float
    rgb_path: Path
    depth_npy_path: Optional[Path]


def _load_metadata(session_dir: Path) -> list[FrameMeta]:
    csv_path = session_dir / "metadata.csv"
    if not csv_path.exists():
        return []
    rows: list[FrameMeta] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                bearing = float(row["bearing"])
            except (KeyError, ValueError):
                bearing = float("nan")
            def _rel(csv_val: str, subdir: str) -> Path:
                p = Path(csv_val)
                # Relative (new format) → join with session_dir.
                # Absolute (legacy Jetson) → filename only under subdir.
                return session_dir / p if not p.is_absolute() else session_dir / subdir / p.name

            depth_csv = row.get("depth_npy_path", "").strip()
            rows.append(FrameMeta(
                stamp_sec=float(row["stamp_sec"]),
                bearing=bearing,
                rgb_path=_rel(row["rgb_path"], "rgb"),
                depth_npy_path=_rel(depth_csv, "depth_npy") if depth_csv else None,
            ))
    return rows


def _collect_frames(input_dir: Path) -> list[Path]:
    frames = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found in: {input_dir}")
    return frames


def _split_stamp(stamp_sec: float) -> tuple[int, int]:
    sec     = int(stamp_sec)
    nanosec = int((stamp_sec % 1) * 1e9)
    return sec, nanosec


class OfflineFrameDirPublisher(Node):
    def __init__(self, args):
        super().__init__("offline_frame_dir_publisher")

        # ── Input source ──────────────────────────────────────────────────────
        meta_list: list[FrameMeta] = []
        plain_frames: list[Path]  = []

        if args.session_dir:
            session = Path(args.session_dir).expanduser().resolve()
            if not session.exists():
                raise FileNotFoundError(f"session-dir not found: {session}")
            meta_list = _load_metadata(session)
            if not meta_list:
                self.get_logger().warn("No metadata.csv found — falling back to rgb/ glob")
                plain_frames = _collect_frames(session / "rgb")
        else:
            inp = Path(args.input_dir).expanduser().resolve()
            if not inp.exists():
                raise FileNotFoundError(f"input-dir not found: {inp}")
            plain_frames = _collect_frames(inp)

        self._frames: list[tuple[Path, Optional[FrameMeta]]] = (
            [(m.rgb_path, m) for m in meta_list] if meta_list
            else [(p, None) for p in plain_frames]
        )

        # ── Output directory ──────────────────────────────────────────────────
        self._out = Path(args.out_dir).expanduser().resolve()
        self._out.mkdir(parents=True, exist_ok=True)
        if not args.no_clear_on_start:
            for f in self._out.glob("frame_*"):
                if f.suffix in (".jpg", ".tmp"):
                    f.unlink(missing_ok=True)

        # ── Settings ──────────────────────────────────────────────────────────
        self._loop       = bool(args.loop)
        self._max_kept   = max(0, int(args.max_frames_kept))
        self._depth_mode = args.depth_mode
        self._use_orig   = bool(args.use_original_timing)
        self._next_t: Optional[float] = None  # monotonic deadline for original timing

        # ── Depth setup (infer mode) ──────────────────────────────────────────
        self._depth_model = None
        self._depth_out: Optional[Path] = None
        if self._depth_mode == "infer":
            import cv2 as _cv2
            self._cv2 = _cv2
            import numpy as _np
            self._np = _np
            from sparx_agency.core.mapping.depth.depth_anything_v3 import DA3TensorRTModel
            self._depth_out = Path(args.depth_out_dir).expanduser().resolve()
            self._depth_out.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Loading DA3 engine: {args.engine_path}")
            self._depth_model = DA3TensorRTModel(
                engine_path=args.engine_path,
                yaml_path=args.config_yaml,
            )

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_rgb   = self.create_publisher(String,  args.path_topic,         _RELIABLE)
        self._pub_bear  = self.create_publisher(Float32, args.bearing_topic,      10)
        self._pub_depth = (
            self.create_publisher(String, args.depth_path_topic, _RELIABLE)
            if self._depth_mode != "none" else None
        )

        # ── Timer ─────────────────────────────────────────────────────────────
        tick_hz = 100.0 if self._use_orig else max(0.01, float(args.frequency))
        self._timer = self.create_timer(1.0 / tick_hz, self._tick)
        self._index = 0
        self._seq   = 0
        self._done  = False

        self.get_logger().info(
            f"Offline replay: {len(self._frames)} frames  "
            f"depth_mode={self._depth_mode}  "
            f"{'original-timing' if self._use_orig else f'{args.frequency:.1f} Hz'}  "
            f"loop={self._loop}"
        )

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._done:
            return

        if self._index >= len(self._frames):
            if self._loop:
                self._index = 0
                self._next_t = None
                self.get_logger().info("Looping back to first frame")
            else:
                self.get_logger().info("All frames published. Stopping.")
                self._done = True
                return

        src, meta = self._frames[self._index]

        # Original-timing pacing: fire only when wall clock reaches this frame's deadline.
        if self._use_orig and meta is not None:
            now = time.monotonic()
            if self._next_t is None:
                self._next_t = now
            if now < self._next_t:
                return
            next_meta = (
                self._frames[self._index + 1][1]
                if self._index + 1 < len(self._frames) else None
            )
            gap = max(next_meta.stamp_sec - meta.stamp_sec, 0.001) if next_meta else 0.05
            self._next_t += gap

        self._index += 1
        self._seq   += 1

        # ── Timestamp ─────────────────────────────────────────────────────
        if meta is not None:
            sec, nsec = _split_stamp(meta.stamp_sec)
        else:
            t = self.get_clock().now().to_msg()
            sec, nsec = t.sec, t.nanosec

        # ── Copy RGB → out_dir (atomic) ───────────────────────────────────
        final = self._out / f"frame_{self._seq:08d}.jpg"
        tmp   = final.with_suffix(".tmp")
        try:
            shutil.copy2(str(src), str(tmp))
            tmp.rename(final)
        except Exception as e:
            self.get_logger().error(f"Copy failed for {src.name}: {e}")
            return

        if self._max_kept > 0:
            existing = sorted(self._out.glob("frame_*.jpg"))
            for old in existing[: max(0, len(existing) - self._max_kept)]:
                old.unlink(missing_ok=True)

        # ── Publish RGB path ──────────────────────────────────────────────
        self._pub_rgb.publish(String(data=f"{final} {sec} {nsec}"))

        # ── Publish bearing ───────────────────────────────────────────────
        if meta is not None and not math.isnan(meta.bearing):
            self._pub_bear.publish(Float32(data=float(meta.bearing)))

        # ── Publish depth path ────────────────────────────────────────────
        if self._pub_depth is not None:
            dp = self._resolve_depth(src, meta)
            if dp:
                self._pub_depth.publish(String(data=f"{dp} {sec} {nsec}"))

        self.get_logger().info(
            f"[{self._index}/{len(self._frames)}] {src.name}"
            + (f"  bearing={meta.bearing:.1f}" if meta and not math.isnan(meta.bearing) else ""),
            throttle_duration_sec=2.0,
        )

    def _resolve_depth(self, rgb_path: Path, meta: Optional[FrameMeta]) -> Optional[str]:
        if self._depth_mode == "npy":
            if meta and meta.depth_npy_path and meta.depth_npy_path.exists():
                return str(meta.depth_npy_path)
            self.get_logger().warn(
                f"Depth .npy missing for {rgb_path.name}", throttle_duration_sec=5.0
            )
            return None

        if self._depth_mode == "infer":
            bgr = self._cv2.imread(str(rgb_path))
            if bgr is None:
                self.get_logger().error(f"Cannot read RGB for inference: {rgb_path}")
                return None
            depth_m = self._depth_model.infer_depth(bgr).astype(self._np.float32)
            out = self._depth_out / f"frame_{self._seq:08d}.npy"
            self._np.save(str(out), depth_m)
            return str(out)

        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a take session through the XTEND ROS topics.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--session-dir", help="Root of a take session (metadata.csv, rgb/, depth_npy/)")
    src.add_argument("--input-dir",   help="Legacy: plain frame directory (no metadata)")
    p.add_argument("--out-dir",             default="/tmp/xtend_frames")
    p.add_argument("--path-topic",          default="/xtend/rgb_frame_path")
    p.add_argument("--bearing-topic",       default="/xtend/bearing")
    p.add_argument("--depth-path-topic",    default="/xtend/depth_frame_path")
    p.add_argument("--depth-mode",          default="none", choices=["none", "npy", "infer"])
    p.add_argument("--depth-out-dir",       default="/tmp/xtend_offline_depth")
    p.add_argument("--engine-path",         default="")
    p.add_argument("--config-yaml",         default="")
    p.add_argument("--frequency",           type=float, default=10.0)
    p.add_argument("--use-original-timing", action="store_true",
                   help="Pace replay from original inter-frame timestamps in metadata.csv")
    p.add_argument("--loop",                action="store_true")
    p.add_argument("--max-frames-kept",     type=int, default=30)
    p.add_argument("--no-clear-on-start",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = None
    try:
        node = OfflineFrameDirPublisher(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()