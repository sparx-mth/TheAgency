"""Flight-recording schema + reader (numpy, no torch).

A fine-tune recording is an on-disk directory produced from a synchronized drone
flight. This module defines the expected layout, loads it, and provides the
body-frame transforms the label generator needs (future path, goal). It is
deliberately format-simple so a rosbag exporter can target it.

Layout::

    recording_dir/
      intrinsics.json     {"width","height","fx","fy","cx","cy"}
      meta.json           {"rate_hz","camera_height_m","pitch_deg", "frames": N}
      depth/000000.npy    (H, W) float32 meters, one per frame
      rgb/000000.jpg      (H, W, 3) uint8, co-registered (optional for label-gen)
      poses.npy           (N, 4) float32 rows [t, x, y, yaw]  world frame

The goal is supplied per-fine-tune (a body point for NavDP, an image for FlowNav);
for auto-labels the goal defaults to the pose ``lookahead`` frames ahead.

**No usable recording exists in the repo yet** (only a depth-only AprilTag test
bag); :func:`synthesize_recording` builds a tiny synthetic one for tests and to
document the schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Intrinsics


@dataclass(frozen=True)
class FlightRecording:
    """A loaded flight recording (lazy per-frame depth/rgb access)."""

    root: Path
    intrinsics: Intrinsics
    poses: np.ndarray          # (N, 4) [t, x, y, yaw]
    rate_hz: float
    camera_height_m: float
    pitch_deg: float

    @property
    def num_frames(self) -> int:
        return int(self.poses.shape[0])

    def depth(self, i: int) -> np.ndarray:
        return np.load(self.root / "depth" / f"{i:06d}.npy").astype(np.float32)

    def rgb(self, i: int) -> Optional[np.ndarray]:
        p = self.root / "rgb" / f"{i:06d}.jpg"
        if not p.exists():
            return None
        import cv2
        bgr = cv2.imread(str(p))
        return None if bgr is None else bgr[:, :, ::-1].copy()

    def pose(self, i: int) -> np.ndarray:
        return self.poses[i, 1:4]  # (x, y, yaw)

    def future_path_body(self, i: int, horizon: int, stride: int = 1) -> np.ndarray:
        """The flown-future world path expressed in frame ``i``'s body FLU frame.

        Returns ``(K, 2)`` ``[fwd, left]`` starting at the origin (the robot). Used
        as the seed the PF/ESDF corrector refines into the label.
        """
        x0, y0, yaw0 = self.pose(i)
        idx = range(i, min(i + horizon * stride + 1, self.num_frames), stride)
        pts = self.poses[list(idx), 1:3] - np.array([x0, y0])
        # rotate world offsets into the body FLU frame (fwd = +x, left = +y)
        fwd = pts[:, 0] * np.cos(yaw0) + pts[:, 1] * np.sin(yaw0)
        left = -pts[:, 0] * np.sin(yaw0) + pts[:, 1] * np.cos(yaw0)
        out = np.stack([fwd, left], axis=1).astype(np.float32)
        return out if out.shape[0] >= 2 else np.array([[0.0, 0.0], [0.1, 0.0]], np.float32)

    def goal_body(self, i: int, lookahead: int) -> Tuple[float, float]:
        """Auto-goal: the pose ``lookahead`` frames ahead, in body FLU (fwd, left)."""
        j = min(i + lookahead, self.num_frames - 1)
        x0, y0, yaw0 = self.pose(i)
        dx, dy = self.poses[j, 1] - x0, self.poses[j, 2] - y0
        fwd = dx * np.cos(yaw0) + dy * np.sin(yaw0)
        left = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
        return float(fwd), float(left)


def load_recording(root) -> FlightRecording:
    """Load a recording directory into a :class:`FlightRecording`."""
    root = Path(root)
    intr = json.loads((root / "intrinsics.json").read_text())
    meta = json.loads((root / "meta.json").read_text())
    poses = np.load(root / "poses.npy").astype(np.float32)
    return FlightRecording(
        root=root,
        intrinsics=Intrinsics(width=intr["width"], height=intr["height"],
                              fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"]),
        poses=poses,
        rate_hz=float(meta.get("rate_hz", 10.0)),
        camera_height_m=float(meta.get("camera_height_m", 1.0)),
        pitch_deg=float(meta.get("pitch_deg", 0.0)),
    )


def synthesize_recording(root, num_frames: int = 12, obstacle: bool = True) -> FlightRecording:
    """Write a tiny synthetic recording (for tests / schema documentation)."""
    root = Path(root)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    h, w = 120, 160
    intr = {"width": w, "height": h, "fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 40.0}
    (root / "intrinsics.json").write_text(json.dumps(intr))
    (root / "meta.json").write_text(json.dumps(
        {"rate_hz": 10.0, "camera_height_m": 1.0, "pitch_deg": 0.0, "frames": num_frames}))
    # straight forward flight along +x
    t = np.arange(num_frames, dtype=np.float32)
    poses = np.stack([t * 0.1, t * 0.25, np.zeros_like(t), np.zeros_like(t)], axis=1)
    np.save(root / "poses.npy", poses.astype(np.float32))
    for i in range(num_frames):
        depth = np.full((h, w), 8.0, np.float32)
        if obstacle:
            depth[:, : w // 2] = 2.5   # near wall on the left half
        np.save(root / "depth" / f"{i:06d}.npy", depth)
    return load_recording(root)
