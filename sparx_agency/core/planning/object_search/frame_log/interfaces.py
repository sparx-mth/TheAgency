"""The frame log's contract: what one logged frame is, where its image is kept, and how the log fails.

The frame log (:mod:`~sparx_agency.core.planning.object_search.frame_log.episode_frame_log`)
keeps every RGB frame of an episode under the exact pose it was captured
from. This module is its seam, kept apart from both of its sides:

* :class:`FrameRecord` -- one logged frame: its name, the step it was
  captured at and its capture pose. The log's lookup dictionary maps a
  frame's name to its record, and the same record is one line of an
  episode's index on disk.
* :class:`FrameStore` -- where the encoded images are kept. A folder on disk
  (``disk_store.py``) and a dictionary in RAM (``memory_store.py``) both
  satisfy it. A store is handed bytes under a record and asked for the
  bytes of a name; it never sees a pixel, so the log above it cannot tell
  which store it was given, and the two serve identical images.
* :class:`FrameLogError` -- how the log fails. Loudly, and as
  infrastructure: see the class.

**A frame's name is its file name.** ``step_000137.jpg`` is the frame the
decision of step 137 was taken on (:func:`frame_name`). The name is the key
of the lookup dictionary whatever the store, so a consumer written against
frames on disk reads frames in RAM unchanged.

Python 3.8 syntax, standard library only beyond the pose type.
"""
from __future__ import annotations

import dataclasses
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: Version of what a store is handed: the keys of an episode's header and of a record's row.
SCHEMA = "objnav_frame_log/1"

#: A frame is named after the step it was captured at, zero-padded so that names sort in capture order.
FRAME_NAME = "step_%06d"
#: Every name :func:`frame_name` makes, whatever its extension: what a store clears when an episode is logged again.
FRAME_GLOB = "step_*"

#: The pose fields of a record's row, in :class:`AgentPose` order.
POSE_FIELDS = tuple(field.name for field in dataclasses.fields(AgentPose))

#: What the pose fields mean, written into every episode's header so a folder of frames explains itself.
POSE_CONVENTION = ("x, y, z: metres in world ENU (z up); z is the height of the floor under the agent, not of the "
                   "camera. yaw: radians, counter-clockwise from world +x. camera_pitch: radians, positive looks down.")


class FrameLogError(ObjNavInternalError):
    """The frame log could not keep, find or return a frame.

    An :class:`~sparx_agency.core.planning.objnav.errors.ObjNavInternalError`
    on purpose. The log is fed from inside a policy's decision, and a
    benchmark run that records agent errors would otherwise score a full
    disk as a failed episode of the method. The harness never records this
    family: it stops the run. A log that silently dropped a frame would be
    worse still -- whatever reads the log later would reason over a memory
    with a hole in it.
    """


def frame_name(step: int, extension: str) -> str:
    """The name of the frame captured at ``step``: ``step_000137`` plus the codec's extension.

    Args:
        step: The observation's step index.
        extension: The image format's file extension, dot included (``".jpg"``).

    Returns:
        The frame's name, which is also its file name in a disk store.
    """
    return (FRAME_NAME % step) + extension


def check_frame_name(name: Any) -> str:
    """``name``, once it is known to be one file name and nothing more.

    Raises:
        FrameLogError: If ``name`` is not a non-empty string, or names a
            directory level (a separator, ``.`` or ``..``): a store writes it
            inside the episode's own folder and nowhere else.
    """
    if (not isinstance(name, str) or not name or name in (".", "..")
            or any(mark in name for mark in ("/", "\\", "\x00"))):
        raise FrameLogError("a frame's name is one file name with no directory in it, got %r" % (name,))
    return name


@dataclass(frozen=True)
class FrameRecord:
    """One logged frame: which image it is and the exact pose it was captured from.

    Attributes:
        name: The frame's name, unique within its episode -- the key of the
            log's lookup dictionary and the image's file name on disk.
        step: The step the frame was captured at: 0 for the frame an episode
            starts with, one more per executed action.
        pose: The capture pose, as the observation carried it: position
            ``(x, y, z)`` in world ENU metres, heading ``yaw`` and the
            camera's ``camera_pitch`` in radians (:data:`POSE_CONVENTION`).

    Raises:
        FrameLogError: On a name that is not one file name, a step that is
            not a non-negative integer, or a pose that is not an
            :class:`~sparx_agency.core.planning.objnav.types.pose.AgentPose`.
    """

    name: str
    step: int
    pose: AgentPose

    def __post_init__(self) -> None:
        check_frame_name(self.name)
        if not isinstance(self.step, numbers.Integral) or isinstance(self.step, bool) or self.step < 0:
            raise FrameLogError("a frame's step is a non-negative integer, got %r" % (self.step,))
        if not isinstance(self.pose, AgentPose):
            raise FrameLogError("a frame's pose is an AgentPose, got %r" % (self.pose,))

    def to_row(self) -> Dict[str, Any]:
        """The record as plain JSON types: ``{"name", "step", "pose": {x, y, z, yaw, camera_pitch}}``.

        Every pose value is written as a Python float, which JSON round-trips
        exactly: the pose read back is the pose that was captured.
        """
        return {"name": self.name, "step": int(self.step),
                "pose": {field: float(getattr(self.pose, field)) for field in POSE_FIELDS}}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FrameRecord":
        """The record a :meth:`to_row` row describes.

        Raises:
            FrameLogError: If the row lacks a key or holds a value no record
                accepts (a non-finite pose included).
        """
        try:
            pose = row["pose"]
            return cls(row["name"], row["step"], AgentPose(**{field: pose[field] for field in POSE_FIELDS}))
        except FrameLogError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameLogError("not a frame record: %r (%s: %s)" % (row, type(exc).__name__, exc)) from exc


class FrameStore(Protocol):
    """Where one episode's encoded frames are kept, each under its record's name.

    One episode at a time: :meth:`begin` starts an empty one, and the frames
    of the episode before it are no longer served. Nothing is put or got
    before the first :meth:`begin`.
    """

    def begin(self, episode_id: str, header: Mapping[str, Any]) -> None:
        """Start an episode with no frames in it.

        Args:
            episode_id: The episode's identifier; a store that keeps
                episodes apart keys them by it.
            header: What the log says of the episode, in plain JSON types
                (its identity, the image format, the camera, the pose
                convention). A store that outlives the process keeps it
                beside the frames.
        """

    def put(self, record: FrameRecord, image: bytes) -> None:
        """Keep one frame's encoded image under ``record.name``, and the record with it where the store persists."""

    def get(self, name: str) -> bytes:
        """The encoded image kept under ``name`` in the episode in force.

        Raises:
            FrameLogError: If the episode holds no frame of that name.
        """

    def describe(self) -> Dict[str, Any]:
        """What a run's record says of the store, in plain JSON types: its kind, and where it keeps the frames."""
