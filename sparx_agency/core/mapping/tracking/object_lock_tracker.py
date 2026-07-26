"""Abstract contract for a single-target image tracker (detect-once / track-many).

The mission closes on an object through one of two strategies, and both must look
identical to the servo / FSM / recovery stack that consumes their output:

  * :class:`~...target_tracker.TargetTracker` — a detector seeds an optical-flow
    box tracker that propagates the box every camera frame *between* detections;
  * :class:`~...detection_only_tracker.DetectionOnlyTracker` — the detector alone
    drives closure; the box is simply the detector's most recent output.

This ABC is the seam that lets a task pick between them at runtime (see
:func:`~...factory.make_lock_tracker`) without the pipeline or the ROS node
knowing which is in play. It declares exactly the surface those callers use and
nothing about ROS or optical flow. Every method takes a monotonic ``stamp_s`` and
emits :class:`~sparx_agency.core.common.types.Track2D`, so the two strategies are
drop-in interchangeable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D, Track2D


class ObjectLockTracker(ABC):
    """A single-target tracker driven by (occasional) detections + camera frames."""

    @abstractmethod
    def reset(self) -> None:
        """Forget the target entirely (no lock, no history)."""
        raise NotImplementedError

    @abstractmethod
    def on_detection(self, image: np.ndarray, detection: Detection2D,
                     stamp_s: float) -> bool:
        """(Re)seed / update the lock from a fresh detection; True on success."""
        raise NotImplementedError

    @abstractmethod
    def on_frame(self, image: np.ndarray, stamp_s: float) -> Track2D:
        """Advance one camera frame and return the current track (check ``.valid``)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self) -> str:
        """Class label of the currently locked target ("" before acquisition)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def has_target(self) -> bool:
        """True once a detection has seeded the lock (even if now lost)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def propagates(self) -> bool:
        """True if the tracker carries the box *between* detections on its own.

        A propagating tracker (optical flow + motion model) can be seeded once and
        coast; a non-propagating one (detector box only) has no state to advance and
        must be fed every fresh detection or it goes stale. Callers use this so a
        "seed once, don't re-seed" option only applies where propagation makes it
        safe — a non-propagating tracker is always re-fed the latest detection.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_locked(self) -> bool:
        """True while the tracker currently holds a (fresh) box on the target."""
        raise NotImplementedError

    @property
    @abstractmethod
    def last_track(self) -> Optional[Track2D]:
        """The most recent track emitted, or None before the first one."""
        raise NotImplementedError

    @abstractmethod
    def time_since_valid(self, stamp_s: float) -> Optional[float]:
        """Seconds since the last valid (measured) box, or None if never valid."""
        raise NotImplementedError
