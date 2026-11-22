"""The frames of a recorded episode, for a replay: a frame log's when the run kept one, else one per recorded action.

A replay of a recording through the frontier-view selector
(``frontier_view_replay``) needs, for every action, the frames a decision
taken then could have been shown: their names, steps and capture poses, and
the camera. There are two places to get them from, and the replay should
not care which:

* **the run's frame log** (``OBJNAV_FRAME_LOG_DIR``,
  ``core/planning/object_search/frame_log``): the index is read with the
  log's own reader, the camera from its header, and the images are there to
  be drawn. The log is checked against the recording before it is believed:
  a frame whose capture pose is not the pose the recording has for its step
  belongs to another run, and frames of another run chosen over this run's
  map would look like a result;
* **the recording alone**: a frame log holds exactly one frame per decided
  action at the pose of that action, so ``steps.jsonl`` says which frames
  there would have been. Every recording with a map journal can be
  replayed this way; only pictures of the frames cannot be drawn.

Python 3.9 syntax.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.object_search.frame_log.disk_store import read_frame_log
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameRecord, frame_name
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_extent_replay import benchmark_camera, camera_for_scene

#: Where an episode's frames came from (``EpisodeFrames.source``).
FROM_LOG, FROM_POSES = "frame log", "recorded poses (no frame log)"
#: A frame's capture pose and the recording's pose of its step may differ by no more than this, in any field.
POSE_TOLERANCE = 1e-6


@dataclass
class EpisodeFrames:
    """The frames of one episode: their records in step order, the camera, and where the images are (if anywhere)."""

    records: List[FrameRecord]
    camera: CameraSpec
    source: str
    directory: Optional[Path] = None
    _steps: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.records = sorted(self.records, key=lambda record: record.step)
        self._steps = [record.step for record in self.records]

    def until(self, step: int) -> List[FrameRecord]:
        """Every frame captured at or before ``step``: what a decision at ``step`` could be shown."""
        return self.records[:bisect.bisect_right(self._steps, int(step))]

    def at(self, step: int) -> List[FrameRecord]:
        """The frame of ``step`` itself, as a list (empty when that step logged none)."""
        index = bisect.bisect_left(self._steps, int(step))
        return self.records[index:index + 1] if index < len(self._steps) and self._steps[index] == int(step) else []


def pose_of(row: Dict) -> AgentPose:
    """The pose of a ``steps.jsonl`` row."""
    pose = row["pose"]
    return AgentPose(float(pose["x"]), float(pose["y"]), float(pose.get("z", 0.0) or 0.0), float(pose["yaw"]),
                     float(pose.get("camera_pitch", 0.0) or 0.0))


def camera_from_header(header: Dict) -> CameraSpec:
    """The camera a frame log's header records."""
    camera = header["camera"]
    far = camera.get("max_depth_m")
    return CameraSpec(Intrinsics(int(camera["width"]), int(camera["height"]), float(camera["fx"]), float(camera["fy"]),
                                 float(camera["cx"]), float(camera["cy"])),
                      float(camera["height_m"]), float(camera["min_depth_m"]), math.inf if far is None else float(far))


def frame_folder(frames_root, key: str) -> Optional[Path]:
    """An episode's frame folder under ``frames_root``: the root itself, or its ``<slug>-<key>`` child; None when there is none.

    ``key`` is the recording directory's name: the twelve hex digits of the
    episode id's hash, which the frame log's folder name ends with too.
    """
    if frames_root is None:
        return None
    root = Path(frames_root)
    if (root / "frames.jsonl").is_file():
        return root
    found = sorted(root.glob("*-%s" % key))
    return found[0] if found else None


def load_frames(directory: Path, rows: Sequence[Dict], episode: Dict, frames_root=None, camera=None) -> EpisodeFrames:
    """The episode's frames: a frame log's when one is given for the recording, else one per recorded action.

    Args:
        directory: The recording directory; its name is the episode's key.
        rows: The recording's ``steps.jsonl`` rows.
        episode: The recording's ``episode.json``.
        frames_root: A frame log root, or one episode's folder; None reads no log.
        camera: ``hm3d`` or ``mp3d`` to override the camera; by default the
            log's own, else the benchmark's by the scene id.

    Raises:
        SystemExit: If a frame log was asked for and none is found for this
            recording, or the log's poses are not the recording's.
    """
    folder = frame_folder(frames_root, directory.name)
    if frames_root is not None and folder is None:
        raise SystemExit("no frame log for recording %s under %s (looked for *-%s)" % (directory, frames_root, directory.name))
    if folder is None:
        spec = benchmark_camera(camera or camera_for_scene(episode.get("scene")))
        records = [FrameRecord(frame_name(int(row["step"]), ".jpg"), int(row["step"]), pose_of(row)) for row in rows]
        return EpisodeFrames(records, spec, FROM_POSES)
    header, by_name = read_frame_log(folder)
    recorded = {int(row["step"]): pose_of(row) for row in rows}
    for record in by_name.values():
        pose = recorded.get(record.step)
        if pose is not None and max(abs(a - b) for a, b in zip(
                (pose.x, pose.y, pose.z, pose.yaw, pose.camera_pitch),
                (record.pose.x, record.pose.y, record.pose.z, record.pose.yaw, record.pose.camera_pitch))) > POSE_TOLERANCE:
            raise SystemExit("%s: the frame of step %d was captured at another pose than the recording's step %d -- "
                             "this frame log is not of this recording" % (folder, record.step, record.step))
    spec = camera_from_header(header) if camera is None else benchmark_camera(camera)
    return EpisodeFrames(list(by_name.values()), spec, FROM_LOG, folder)
