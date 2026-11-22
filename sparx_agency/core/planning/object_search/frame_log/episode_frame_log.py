"""The episode's frame log: every RGB frame a policy is shown, kept under the exact pose it was captured from.

The memory a later decision reads frames back from. Whoever is shown the
frames feeds it -- :meth:`EpisodeFrameLog.begin` once per episode, then
:meth:`EpisodeFrameLog.add` for every observation, in step order -- and
whoever needs a past frame asks it two things:

* **which frames exist, and where each was taken** --
  :attr:`EpisodeFrameLog.records`, the lookup dictionary from a frame's name
  to its :class:`~sparx_agency.core.planning.object_search.frame_log.interfaces.FrameRecord`
  (step and capture pose), in capture order;
* **what a frame shows** -- :meth:`EpisodeFrameLog.rgb`, the image of a name.

Where the images live is the store's business
(:class:`~sparx_agency.core.planning.object_search.frame_log.interfaces.FrameStore`):
a folder on disk or a dictionary in RAM. The lookup dictionary is in memory
either way; a disk store also writes it beside the images, one line per
frame, so a logged episode can be read back without the process that wrote
it (``disk_store.read_frame_log``).

The quiet failures designed out:

* **A frame filed under the wrong pose.** The record is built from the
  observation itself -- its own step, its own pose -- in the same call that
  stores its image; there is no second place to get them from.
* **One image overwriting another.** A frame is named after its step, and a
  step is accepted once, in increasing order. The same step offered twice,
  or an older one after a newer, raises: it would replace an image under a
  pose that may differ.
* **A memory with a hole in it.** Nothing is skipped on failure. A frame
  that cannot be encoded or stored raises ``FrameLogError``, which ends a
  benchmark run instead of being scored as a failed episode.
* **Frames of two episodes mixed.** :meth:`EpisodeFrameLog.begin` empties
  the lookup, and the store starts the episode without frames.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import math
import os
import types
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from sparx_agency.core.planning.object_search.frame_log.codec import FrameCodec
from sparx_agency.core.planning.object_search.frame_log.disk_store import DiskFrameStore
from sparx_agency.core.planning.object_search.frame_log.interfaces import (
    POSE_CONVENTION, SCHEMA, FrameLogError, FrameRecord, FrameStore, frame_name)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation

#: Environment variable of :meth:`EpisodeFrameLog.from_env`: the directory a run's frames are kept under.
FRAME_LOG_DIR_ENV = "OBJNAV_FRAME_LOG_DIR"


class EpisodeFrameLog:
    """The frames of the episode in force, by name, with the pose each was captured from.

    Args:
        store: Where the encoded images are kept (a ``DiskFrameStore``, a
            ``MemoryFrameStore``).
        codec: How a frame is encoded; JPEG at quality 95 by default.

    Attributes:
        store: That store.
        codec: That codec.
        episode_id: The episode in force; None before the first :meth:`begin`.
    """

    def __init__(self, store: FrameStore, codec: Optional[FrameCodec] = None) -> None:
        self.store = store
        self.codec = FrameCodec() if codec is None else codec
        self.episode_id = None  # type: Optional[str]
        self._records = {}  # type: Dict[str, FrameRecord]
        self._last_step = None  # type: Optional[int]

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> Optional["EpisodeFrameLog"]:
        """The log ``OBJNAV_FRAME_LOG_DIR`` asks for: frames on disk under that directory; None when it is unset or blank."""
        env = os.environ if environ is None else environ
        root = env.get(FRAME_LOG_DIR_ENV, "").strip()
        return cls(DiskFrameStore(root)) if root else None

    def begin(self, episode: ObjNavEpisode) -> None:
        """Start an episode: an empty lookup, and a store with none of its frames yet.

        Only the episode's typed public fields are read -- its identity, its
        target and its camera -- never its free-form ``metadata``.

        Args:
            episode: The episode about to run.

        Raises:
            FrameLogError: If ``episode`` is not an ``ObjNavEpisode``, or the
                store cannot start it (an unwritable directory, say).
        """
        self.episode_id, self._records, self._last_step = None, {}, None
        if not isinstance(episode, ObjNavEpisode):
            raise FrameLogError("begin() needs the ObjNavEpisode about to run, got %r" % (episode,))
        header = {"schema": SCHEMA, "episode_id": episode.episode_id, "scene_id": episode.scene_id,
                  "benchmark": episode.benchmark, "split": episode.split, "target_category": episode.target_category,
                  "image": self.codec.describe(), "camera": _camera_row(episode.camera), "pose": POSE_CONVENTION}
        self._stored("begin episode %r" % (episode.episode_id,), lambda: self.store.begin(episode.episode_id, header))
        self.episode_id = episode.episode_id

    def add(self, observation: ObjNavObservation) -> FrameRecord:
        """Keep one observation's RGB frame under its own step and pose.

        Args:
            observation: The observation; its ``rgb`` is encoded and stored,
                its ``step`` names the frame and its ``pose`` is recorded
                as it is. The arrays are not written into.

        Returns:
            The frame's record, now in :attr:`records` under ``record.name``.

        Raises:
            FrameLogError: Before :meth:`begin`; on anything but an
                ``ObjNavObservation``; on a step that does not exceed the
                last one logged; or when the frame cannot be encoded or
                stored.
        """
        if self.episode_id is None:
            raise FrameLogError("add() before begin(): the frame log has no episode to file a frame under")
        if not isinstance(observation, ObjNavObservation):
            raise FrameLogError("add() needs an ObjNavObservation, got %r" % (observation,))
        step = int(observation.step)
        if self._last_step is not None and step <= self._last_step:
            raise FrameLogError("the frame of step %d was offered after step %d in episode %r; a step is logged "
                                "once, in increasing order" % (step, self._last_step, self.episode_id))
        record = FrameRecord(frame_name(step, self.codec.extension), step, observation.pose)
        self._stored("keep frame %s of episode %r" % (record.name, self.episode_id),
                     lambda: self.store.put(record, self.codec.encode(observation.rgb)))
        self._records[record.name] = record
        self._last_step = step
        return record

    @property
    def records(self) -> Mapping[str, FrameRecord]:
        """The lookup dictionary of the episode in force: frame name -> record, in capture order (read-only).

        ``records[name].pose`` is the exact pose the frame was captured
        from, ``records[name].step`` the step. Ask again after
        :meth:`begin`: a view taken before it shows the episode before.
        """
        return types.MappingProxyType(self._records)

    def rgb(self, name: str) -> np.ndarray:
        """The frame logged under ``name``.

        Args:
            name: A key of :attr:`records`.

        Returns:
            ``(H, W, 3)`` uint8 in RGB order, a fresh array -- the same
            pixels whichever store keeps the frame.

        Raises:
            FrameLogError: If the episode in force has no frame of that
                name, or its image cannot be read back.
        """
        if not isinstance(name, str) or name not in self._records:
            raise FrameLogError("episode %r has no frame named %r (%d logged)" % (self.episode_id, name, len(self)))
        return self._stored("read frame %s of episode %r back" % (name, self.episode_id),
                            lambda: self.codec.decode(self.store.get(name)))

    def __len__(self) -> int:
        """Frames logged in the episode in force."""
        return len(self._records)

    def describe(self) -> Dict[str, Any]:
        """What a run's record says of the log, in plain JSON types: the schema, the store and the image format."""
        return {"schema": SCHEMA, "store": self.store.describe(), "image": self.codec.describe()}

    @staticmethod
    def _stored(what: str, action: Callable[[], Any]) -> Any:
        """``action()``, with whatever the store or the codec raised restated as the log's own failure."""
        try:
            return action()
        except FrameLogError:
            raise
        except Exception as exc:
            raise FrameLogError("the frame log cannot %s: %s: %s" % (what, type(exc).__name__, exc)) from exc


def _camera_row(camera: CameraSpec) -> Dict[str, Any]:
    """The camera every frame of the episode was captured with, in plain JSON types.

    What turns a frame's pose into what the frame sees: the image size and
    pinhole intrinsics, the mount height above ``pose.z`` and the depth
    range. A sensor without a range limit (``max_depth_m`` infinite) is
    written as null, which strict JSON can hold.
    """
    k = camera.intrinsics
    return {"width": int(k.width), "height": int(k.height), "fx": float(k.fx), "fy": float(k.fy),
            "cx": float(k.cx), "cy": float(k.cy), "height_m": float(camera.height_m),
            "min_depth_m": float(camera.min_depth_m),
            "max_depth_m": None if math.isinf(camera.max_depth_m) else float(camera.max_depth_m)}
