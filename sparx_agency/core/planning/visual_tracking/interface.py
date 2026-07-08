"""Visual box-tracker interface (ROS-free).

A ``BoxTracker`` is the *classic*, per-object, detect-once/track-many half of the
tracking stack: given a seed bounding box on one frame, it propagates that box to
the next frame from image content alone (no detector, no neural net in the loop).
The Lucas-Kanade implementation lives in
:mod:`sparx_agency.core.planning.visual_tracking.lk_box_tracker`.

The tracker operates on **grayscale** frames (``HxW`` uint8) so the RGB->gray
conversion happens once per frame in the caller
(:class:`~sparx_agency.core.planning.visual_tracking.target_tracker.TargetTracker`),
not once per tracker. It reports only what it can measure from pixels — the raw
box and how many features survived — leaving velocity/prediction/labels to the
composing :class:`TargetTracker`, which fuses in a motion model and detector
re-seeds and emits the richer :class:`~sparx_agency.core.common.types.Track2D`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxObservation:
    """One frame's raw measurement from a :class:`BoxTracker`.

    Attributes:
        bbox_xyxy: Measured box ``(x1, y1, x2, y2)`` in pixels, or ``None`` if the
            track was lost this frame.
        n_matches: Number of tracked features that survived to produce the box.
        valid: True iff the tracker still holds lock (``bbox_xyxy`` is set).
    """

    bbox_xyxy: Optional[BBox]
    n_matches: int
    valid: bool


class BoxTracker(ABC):
    """Classic single-object bounding-box tracker (grayscale, ROS-free)."""

    name: str = "box_tracker"

    @abstractmethod
    def reset(self) -> None:
        """Drop all state; the tracker becomes invalid until re-seeded."""
        raise NotImplementedError

    @abstractmethod
    def seed(self, gray: np.ndarray, bbox_xyxy: BBox) -> bool:
        """(Re)initialise the tracker from a bounding box on ``gray``.

        Args:
            gray: ``HxW`` uint8 grayscale image the box refers to.
            bbox_xyxy: Seed box in pixels.

        Returns:
            True on success (enough features found), else False (tracker invalid).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, gray: np.ndarray) -> BoxObservation:
        """Propagate the box to the next grayscale frame.

        Returns a :class:`BoxObservation`; on loss ``valid`` is False and the
        tracker must be re-seeded before it will track again.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_valid(self) -> bool:
        """True while the tracker holds lock."""
        raise NotImplementedError
