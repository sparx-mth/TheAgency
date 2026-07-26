"""Export a simulated flight into the on-disk layout ``datasets/recording.py`` reads.

Counterpart to :mod:`bag_extract` for simulator-sourced flights (Isaac Sim /
Pegasus Simulator, see ``sparx_agency/robots/PEGASUS/``) instead of a real rosbag.
This module has **no simulator dependency** — it only writes numpy arrays and
JSON, so it can run and be unit-tested outside Isaac Sim. The sim-specific part
(driving the vehicle, pulling RGB/depth frames and ground-truth pose out of the
running simulation) lives in ``robots/PEGASUS/adapters/`` and is handed to
:func:`export_flight` as a plain iterable of :class:`SimFrame`.

Unlike a rosbag extraction, a simulated flight has **exact ground-truth pose**
for every frame, so this is the first extractor that writes the full
``recording.py`` schema including ``poses.npy`` (real recordings do not have
this yet — see the ``_todo_hardware`` note in :mod:`bag_extract`).

Output (the full ``recording.py`` schema)::

    <out_dir>/
      rgb/000000.jpg         colour frame (optional, only if a frame supplies one)
      depth/000000.npy       (H, W) float32 metres
      intrinsics.json        {width,height,fx,fy,cx,cy}
      meta.json              {rate_hz, camera_height_m, pitch_deg, frames}
      poses.npy              (N, 4) float32 rows [t, x, y, yaw], world frame FLU
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from sparx_agency.core.common.types import Intrinsics


@dataclass(frozen=True)
class SimFrame:
    """One captured instant of a simulated flight.

    Attributes:
        depth: (H, W) float32 metres.
        pose: (x, y, yaw) in the world frame, FLU convention.
        rgb: Optional (H, W, 3) uint8 RGB frame.
    """

    depth: np.ndarray
    pose: tuple  # (x, y, yaw)
    rgb: Optional[np.ndarray] = None


MAX_DEPTH_M = 20.0
"""Range beyond which simulated depth is treated as a no-return.

Isaac's depth sensor reports ``inf`` for a ray that never hits anything -- out
a window, through a doorway, past the far plane. In an office recording that
was 1186 of 1516 frames, up to 54% of the pixels in the worst one. Real depth
sources the rest of the stack consumes (the XTEND's metric-depth engine,
rosbag extractions) are always finite, so nothing downstream guards against it
and the non-finite values propagate into every normalisation and label.
"""


def clamp_depth(depth: np.ndarray, max_depth_m: float = MAX_DEPTH_M) -> np.ndarray:
    """Replace non-finite depth with ``max_depth_m`` and clamp the rest to it.

    Args:
        depth: Raw ``(H, W)`` depth in metres, possibly containing ``inf``/``nan``.
        max_depth_m: Saturation range, metres.

    Returns:
        A finite ``(H, W)`` float32 array in ``[0, max_depth_m]``.
    """
    finite = np.nan_to_num(
        depth.astype(np.float32), nan=max_depth_m, posinf=max_depth_m, neginf=0.0,
    )
    return np.clip(finite, 0.0, max_depth_m)


def export_flight(
    frames: Iterable[SimFrame],
    out_dir: Path,
    intrinsics: Intrinsics,
    rate_hz: float,
    camera_height_m: float,
    pitch_deg: float,
    max_depth_m: float = MAX_DEPTH_M,
) -> dict:
    """Write ``frames`` into ``out_dir`` using the ``recording.py`` schema.

    Args:
        frames: The captured flight, in temporal order.
        out_dir: Destination recording directory (created if missing).
        intrinsics: The sim camera's pinhole intrinsics.
        rate_hz: Capture rate the frames were sampled at.
        camera_height_m: Camera height above the vehicle's ground contact point.
        pitch_deg: Fixed camera pitch, degrees (0 = level, positive = down).
        max_depth_m: Depth saturation range, see :func:`clamp_depth`.

    Returns:
        The stats dict also written to ``meta.json``.
    """
    out_dir = Path(out_dir)
    depth_dir = out_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    poses = []
    saved = 0
    have_rgb = False
    for t, frame in enumerate(frames):
        np.save(depth_dir / f"{saved:06d}.npy", clamp_depth(frame.depth, max_depth_m))
        if frame.rgb is not None:
            if not have_rgb:
                (out_dir / "rgb").mkdir(parents=True, exist_ok=True)
                have_rgb = True
            bgr = frame.rgb[:, :, ::-1]
            cv2.imwrite(str(out_dir / "rgb" / f"{saved:06d}.jpg"), bgr)
        x, y, yaw = frame.pose
        poses.append([t / rate_hz, x, y, yaw])
        saved += 1

    if saved == 0:
        raise ValueError("export_flight: no frames were provided")

    poses_arr = np.array(poses, dtype=np.float32)
    np.save(out_dir / "poses.npy", poses_arr)

    (out_dir / "intrinsics.json").write_text(json.dumps({
        "width": intrinsics.width, "height": intrinsics.height,
        "fx": intrinsics.fx, "fy": intrinsics.fy,
        "cx": intrinsics.cx, "cy": intrinsics.cy,
    }, indent=2))

    stats = {
        "source": "sim:PEGASUS", "frames": saved, "rate_hz": rate_hz,
        "camera_height_m": camera_height_m, "pitch_deg": pitch_deg,
        "has_rgb": have_rgb,
        "width": intrinsics.width, "height": intrinsics.height,
        "max_depth_m": max_depth_m,
    }
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2))
    return stats
