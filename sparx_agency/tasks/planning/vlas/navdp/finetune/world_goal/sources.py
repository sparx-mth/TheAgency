"""Which recorded frames are worth training on, and why the rest are not.

Every flight recording in this repo shares one on-disk schema (see
``vlas/common/finetune/datasets/recording.py``), so an Isaac A-to-B episode, a
FALCON exploration run and a real XTEND rosbag all load the same way. They are
*not* equally usable as training data, and the filters here are the difference
between a dataset and a pile of frames:

* **Outcome.** Only a clean flight is an expert demonstration. A ``crashed``
  episode ends with the aircraft on its side, and the last seconds of a
  ``stalled`` one are a stationary drone staring at a wall.
* **Altitude.** The surveyed map is a 60 cm slab around cruise height. A frame
  taken at 0.4 m during the climb is looking at a completely different building
  from the one the label is planned on, so it would be supervised against
  geometry that is not in front of it.
* **Attitude.** A frame taken at 40 degrees of bank has a horizon that NavDP's
  ground-plane assumption does not model, and the camera is pointing at the
  ceiling or the floor.
* **Where the aircraft is.** If the map says the aircraft is inside a wall, then
  either the map or the pose is wrong; either way nothing downstream is valid.

Exploration runs are accepted only on request. They carry no ``outcome`` in the
sense an A-to-B episode does (FALCON decides where to go, so there is no plan to
succeed at), and they were recorded through a *different* camera calibration --
which does not matter here, because the goal comes from the map rather than from
back-projecting a pixel, but does mean the imagery is not interchangeable with
the real drone's.

Pure numpy. Loading a source neither reads an image nor touches torch.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from math import acos, degrees
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
    FlightRecording, load_recording,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import Scene


@dataclass(frozen=True)
class SourceConfig:
    """Frame-admission rules.

    Because the label is planned on the map rather than copied from the flown
    path, **the flight's outcome barely matters**. A crashed episode still
    contains a hundred perfectly good posed RGB-D observations before it hit
    anything, and the label taught at each of them is the safe route a planner
    would have flown, not the one that ended in a wall. What has to be true is
    that the *pose* is valid: right height, near-level, inside the map, not
    inside geometry. So the defaults admit everything and let geometry filter.
    ``require_clean_outcome`` restores the strict reading for an ablation.

    Attributes:
        outcomes: Outcomes counted as clean, when ``require_clean_outcome``.
        require_clean_outcome: Admit only A-to-B episodes that ended in
            :data:`outcomes`. Off by default -- see above.
        include_exploration: Accept FALCON exploration runs (recognised by
            having no ``goal_xy``). They roughly double the corpus and add a lot
            of viewpoint diversity. They were recorded through a different, wider
            camera, which is diversity rather than error here: nothing in this
            pipeline back-projects a pixel, so intrinsics never enter a label.
        altitude_tolerance_m: Keep frames within this of the map's altitude.
        max_tilt_deg: Discard frames banked or pitched beyond this.
        min_pose_clearance_m: Discard frames where the map says the aircraft is
            this close to (or inside) geometry.
        drop_tail_s: Seconds discarded from the end of a recording that did not
            end in a clean landing.
        frame_stride: Keep every N-th admissible frame. At 10 Hz and 1.2 m/s,
            consecutive frames are 12 cm apart and almost the same picture;
            striding buys diversity per unit of label-generation cost.
        min_frames: Skip a recording that has fewer usable frames than this.
    """

    outcomes: Tuple[str, ...] = ("landed",)
    require_clean_outcome: bool = False
    include_exploration: bool = True
    altitude_tolerance_m: float = 0.35
    max_tilt_deg: float = 25.0
    min_pose_clearance_m: float = 0.25
    drop_tail_s: float = 3.0
    frame_stride: int = 3
    min_frames: int = 20


@dataclass
class RecordingSource:
    """One admitted recording plus the frame indices worth sampling."""

    index: int
    path: Path
    scene: str
    recording: FlightRecording
    frames: np.ndarray
    meta: Dict
    rejected: Dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name if self.path.name != "recording" else self.path.parent.name

    def summary(self) -> Dict:
        """JSON-safe provenance for the dataset index."""
        intr = self.recording.intrinsics
        return {
            "index": self.index,
            "name": self.name,
            "path": str(self.path),
            "scene": self.scene,
            "total_frames": int(self.recording.num_frames),
            "used_frames": int(self.frames.size),
            "rate_hz": float(self.recording.rate_hz),
            "outcome": self.meta.get("outcome", "exploration"),
            "source": self.meta.get("source", "unknown"),
            "intrinsics": {"width": intr.width, "height": intr.height,
                           "fx": intr.fx, "fy": intr.fy,
                           "cx": intr.cx, "cy": intr.cy},
            "rejected": dict(self.rejected),
        }


def tilt_deg(poses: np.ndarray) -> np.ndarray:
    """Per-frame tilt from vertical, degrees, from the attitude quaternion.

    Columns 5..8 are ``(qx, qy, qz, qw)``, scalar last, body FLU -> world ENU.
    The angle between body ``+z`` and world ``+z`` is ``acos(R[2][2])`` and
    ``R[2][2] = 1 - 2 (qx^2 + qy^2)``.

    Args:
        poses: ``(N, >=9)`` pose array. Narrower arrays return zeros -- a legacy
            four-column recording carries no attitude, so nothing can be judged.
    """
    if poses.ndim != 2 or poses.shape[1] < 9:
        return np.zeros(poses.shape[0], dtype=np.float64)
    qx, qy = poses[:, 5].astype(np.float64), poses[:, 6].astype(np.float64)
    return np.degrees(np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))


def admissible_frames(recording: FlightRecording, scene: Scene, meta: Dict,
                      config: SourceConfig) -> Tuple[np.ndarray, Dict[str, int]]:
    """Frame indices that pass every admission rule, plus a rejection tally."""
    poses = recording.poses
    count = poses.shape[0]
    keep = np.ones(count, dtype=bool)
    rejected: Dict[str, int] = {}

    def drop(mask: np.ndarray, reason: str) -> None:
        removed = int(np.count_nonzero(keep & mask))
        if removed:
            rejected[reason] = rejected.get(reason, 0) + removed
        keep[mask] = False

    outcome = meta.get("outcome")
    if outcome not in config.outcomes and config.drop_tail_s > 0:
        tail = int(round(config.drop_tail_s * max(recording.rate_hz, 1e-3)))
        mask = np.zeros(count, dtype=bool)
        mask[max(0, count - tail):] = True
        drop(mask, "tail_of_failed_flight")

    if poses.shape[1] > 4:
        altitude = poses[:, 4].astype(np.float64)
        drop(np.abs(altitude - scene.config.altitude_m) > config.altitude_tolerance_m,
             "off_altitude")
        drop(tilt_deg(poses) > config.max_tilt_deg, "tilted")

    x_min, y_min, x_max, y_max = scene.bounds
    xs, ys = poses[:, 1].astype(np.float64), poses[:, 2].astype(np.float64)
    drop((xs < x_min) | (xs > x_max) | (ys < y_min) | (ys > y_max), "outside_map")
    drop(scene.clearance(xs, ys) < config.min_pose_clearance_m, "pose_in_geometry")

    indices = np.flatnonzero(keep)
    stride = max(1, int(config.frame_stride))
    return indices[::stride], rejected


def load_source(path, scene: Scene, config: SourceConfig,
                index: int = 0) -> Tuple[Optional[RecordingSource], str]:
    """Load and filter one recording directory.

    Returns:
        ``(source, "")`` when admitted, else ``(None, reason)``.
    """
    path = Path(path)
    if not (path / "poses.npy").exists():
        return None, "no poses.npy (needs a recording with ground-truth pose)"
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}

    if meta.get("scene") not in (None, scene.name):
        return None, f"recorded in scene {meta.get('scene')!r}, not {scene.name!r}"
    if not meta.get("has_rgb", True):
        return None, "no RGB stream (NavDP needs colour)"

    # An A-to-B episode records where it was sent; an exploration run has no goal
    # because FALCON decides that for itself. That, not the outcome field, is
    # what tells the two apart -- both carry an outcome.
    exploration = "goal_xy" not in meta
    if exploration and not config.include_exploration:
        return None, "exploration run (excluded by --no-exploration)"
    if config.require_clean_outcome:
        if exploration:
            return None, "exploration run (excluded by --strict-outcomes)"
        if meta.get("outcome") not in config.outcomes:
            return None, f"outcome {meta.get('outcome')!r} not in {config.outcomes}"

    recording = load_recording(path)
    frames, rejected = admissible_frames(recording, scene, meta, config)
    if frames.size < config.min_frames:
        return None, (f"only {frames.size} admissible frames "
                      f"(rejected: {rejected or 'none'})")
    return RecordingSource(index=index, path=path, scene=scene.name,
                           recording=recording, frames=frames, meta=meta,
                           rejected=rejected), ""


MAX_DISCOVERY_DEPTH = 4
"""How far below a given root a recording directory may sit.

Two is what an unattended campaign already needs: ``campaign_supervisor.py``
gives every worker launch its own directory so a relaunched worker cannot
overwrite its own earlier flights, which puts recordings at
``<root>/w0_c001/office_w0_e000``. Four leaves room to point at a parent of
several campaigns without having to name each one.
"""


def _walk(directory: Path, depth: int, found: List[Path]) -> None:
    """Collect recording directories at or below ``directory``.

    A directory holding ``poses.npy`` *is* the recording, so the walk stops
    there rather than descending into its ``rgb/`` and ``depth/`` — which on a
    large campaign is the difference between reading a few hundred directory
    entries and a million.
    """
    if (directory / "poses.npy").exists():
        found.append(directory)
        return
    if (directory / "recording" / "poses.npy").exists():
        found.append(directory / "recording")
        return
    if depth <= 0:
        return
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_dir():
            _walk(child, depth - 1, found)


def discover(roots: Sequence[str]) -> List[Path]:
    """Expand recording roots into concrete recording directories.

    Accepts a recording directory, a campaign directory holding several, a
    directory of campaign directories, or a glob. A directory is a recording if
    it has ``poses.npy``; a FALCON run keeps its recording in a ``recording/``
    subdirectory.

    A pattern is expanded against the filesystem after ``~`` is resolved, not
    against the process's working directory — an absolute pattern is the normal
    case here and ``Path().glob`` rejects one outright.
    """
    found: List[Path] = []
    for root in roots:
        pattern = str(Path(root).expanduser())
        candidates = ([Path(match) for match in sorted(glob.glob(pattern))]
                      if any(char in pattern for char in "*?[") else [Path(pattern)])
        for candidate in candidates:
            if candidate.is_dir():
                _walk(candidate, MAX_DISCOVERY_DEPTH, found)
    seen, unique = set(), []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique
