"""Read paired RGB + depth frames from a single recording session."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class FrameRecord:
    frame_idx: int
    rgb_path: Path
    depth_path: Path

    def load_rgb(self) -> np.ndarray:
        img = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read RGB: {self.rgb_path}")
        return img

    def load_depth(self) -> np.ndarray:
        arr = np.load(str(self.depth_path))
        arr = arr[..., 0] if arr.ndim == 3 else arr
        return arr.astype(np.float32)


def iter_frames(data_dir: Path, stride: int = 1) -> Iterator[FrameRecord]:
    """
    Yield paired FrameRecords from rgb_1/ + depth_npy_1/, sorted by frame index.

    stride=N returns every Nth matched pair (use to reduce density).
    Raises FileNotFoundError if the expected subdirectories are absent.
    """
    data_dir = Path(data_dir)
    rgb_dir = data_dir / "rgb_1"
    depth_dir = data_dir / "depth_npy_1"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb_1/ not found under {data_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth_npy_1/ not found under {data_dir}")

    depth_index = {p.stem: p for p in depth_dir.glob("*.npy")}

    rgb_paths = sorted(
        rgb_dir.glob("*.jpg"),
        key=lambda p: int(p.stem.replace("frame_", "")),
    )

    n = 0
    for rgb_path in rgb_paths:
        if rgb_path.stem not in depth_index:
            continue
        if n % stride == 0:
            yield FrameRecord(
                frame_idx=int(rgb_path.stem.replace("frame_", "")),
                rgb_path=rgb_path,
                depth_path=depth_index[rgb_path.stem],
            )
        n += 1