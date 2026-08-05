"""Read paired RGB + depth frames from a single recording session."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class FrameRecord:
    frame_idx: int
    rgb_path: Path
    depth_path: Path
    pose: Optional[dict] = field(default=None)  # {x, y, z, yaw} from JSON sidecar

    def load_rgb(self) -> np.ndarray:
        img = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read RGB: {self.rgb_path}")
        return img

    def load_depth(self) -> np.ndarray:
        arr = np.load(str(self.depth_path))
        arr = arr[..., 0] if arr.ndim == 3 else arr
        return arr.astype(np.float32)


def _load_sidecar_pose(rgb_path: Path) -> Optional[dict]:
    """Load {x, y, z, yaw} from a JSON sidecar alongside the RGB file, if present."""
    j = rgb_path.with_suffix(".json")
    if not j.exists():
        return None
    try:
        data = json.loads(j.read_text())
        p = data.get("pose")
        if p and all(k in p for k in ("x", "y", "z", "yaw")):
            return p
    except Exception:
        pass
    return None


def iter_frames(
    data_dir: Path,
    stride: int = 1,
    rgb_subdir: str = "rgb_1",
    depth_subdir: str = "depth_npy_1",
) -> Iterator[FrameRecord]:
    """
    Yield paired FrameRecords from rgb_subdir/ + depth_subdir/, sorted by filename.

    rgb_subdir="." means JPGs are directly inside data_dir (flat layout).
    depth_subdir may be an absolute path (e.g. /tmp/xtend_depth).
    stride=N returns every Nth matched pair (use to reduce density).
    Raises FileNotFoundError if the expected directories are absent.
    """
    data_dir = Path(data_dir)
    # "." resolves to data_dir itself; absolute depth_subdir overrides data_dir prefix
    rgb_dir   = (data_dir / rgb_subdir).resolve()
    depth_dir = (data_dir / depth_subdir).resolve()

    if not rgb_dir.exists():
        raise FileNotFoundError(f"{rgb_subdir}/ not found under {data_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth dir not found: {depth_dir}")

    depth_index = {p.stem: p for p in depth_dir.glob("*.npy")}

    rgb_paths = sorted(rgb_dir.glob("*.jpg"), key=lambda p: p.stem)
    n_matched = sum(1 for p in rgb_paths if p.stem in depth_index)
    if n_matched < len(rgb_paths):
        print(f"[frame_reader] {len(rgb_paths)} RGB frames found, "
              f"{n_matched} have a matching depth .npy — "
              f"{len(rgb_paths) - n_matched} skipped (no depth pair)")

    n = 0
    for i, rgb_path in enumerate(rgb_paths):
        if rgb_path.stem not in depth_index:
            continue
        if n % stride == 0:
            yield FrameRecord(
                frame_idx=i,
                rgb_path=rgb_path,
                depth_path=depth_index[rgb_path.stem],
                pose=_load_sidecar_pose(rgb_path),
            )
        n += 1
