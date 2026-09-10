"""Object-landmark map: per-class radius dedupe with observation confirmation.

Pure Python, ROS-free, Python-3.8-safe. Ports the landmark bookkeeping of the
SJTU ``semantic_mapper/object_mapper_node.py`` (``_add`` / ``_confirmed`` /
``_color_for``): a detection observed within ``dedupe_radius_m`` of an
existing landmark **of the same class** merges into it (running-average
centroid, observation count up), anything else opens a new landmark, and a
landmark is only *confirmed* — trusted for publication/planning — once it has
been observed ``min_observations`` times (false-positive guard).

Landmark XY is **world ENU** (the frame of ``Pose2D`` and the BEV grid), e.g.
from :func:`sparx_agency.core.mapping.objects.geometry.backproject_bbox_to_world`.
"""
from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

_HUE_BUCKETS = 997
"""Prime hue-bucket count (ported constant): spreads class hues over [0, 1)."""


@dataclass
class ObjectLandmark:
    """One deduplicated object on the map.

    Attributes:
        id: Stable landmark id, assigned in discovery order.
        class_name: Detector class (e.g. ``"chair"``).
        xy: Running-average centroid ``(x, y)`` in world ENU meters.
        count: Number of observations merged into this landmark.
    """

    id: int
    class_name: str
    xy: Tuple[float, float]
    count: int


def class_color(class_name: str) -> Tuple[float, float, float]:
    """Deterministic RGB color for a class name, each channel in ``[0, 1]``.

    The hue is derived from the **md5** of the class name, not the builtin
    ``hash()`` the old node used: ``hash(str)`` is salted per process by
    ``PYTHONHASHSEED``, so the old colors changed on every restart. md5 makes
    every chair the same green in every run, log and RViz session.

    Args:
        class_name: Detector class name.

    Returns:
        ``(r, g, b)`` floats in ``[0, 1]`` (HSV with s=0.80, v=1.0, as before).
    """
    digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    hue = (int(digest, 16) % _HUE_BUCKETS) / float(_HUE_BUCKETS)
    return colorsys.hsv_to_rgb(hue, 0.80, 1.0)


class ObjectLandmarkMap:
    """Per-class radius-deduplicated landmark store with a confirm threshold.

    Args:
        dedupe_radius_m: An observation within this distance of a same-class
            landmark's running centroid merges into it. Note the radius is
            measured against the *current* centroid, so a slowly re-observed
            object can walk the centroid (ported semantics).
        min_observations: Observations required before a landmark appears in
            :meth:`confirmed`.

    Raises:
        ValueError: If ``dedupe_radius_m`` is not positive or
            ``min_observations`` is less than 1.
    """

    def __init__(self, dedupe_radius_m: float = 0.70,
                 min_observations: int = 2) -> None:
        if float(dedupe_radius_m) <= 0.0:
            raise ValueError("dedupe_radius_m must be positive, got %r"
                             % (dedupe_radius_m,))
        if int(min_observations) < 1:
            raise ValueError("min_observations must be >= 1, got %r"
                             % (min_observations,))
        self._radius_sq = float(dedupe_radius_m) ** 2
        self._min_obs = int(min_observations)
        self._landmarks = {}  # type: Dict[int, ObjectLandmark]
        self._next_id = 0

    def observe(self, class_name: str,
                xy: Tuple[float, float]) -> ObjectLandmark:
        """Fold one world-ENU observation in; return the landmark it landed on.

        The first same-class landmark (in discovery order) whose running
        centroid is within the dedupe radius absorbs the observation via a
        running average; otherwise a new landmark is opened with ``count=1``.

        Args:
            class_name: Detector class of the observation.
            xy: Observation ``(x, y)`` in world ENU meters.

        Returns:
            The merged-into or newly created landmark (live object — its
            ``xy``/``count`` keep updating on later observations).
        """
        wx, wy = float(xy[0]), float(xy[1])
        for landmark in self._landmarks.values():
            if landmark.class_name != class_name:
                continue
            ox, oy = landmark.xy
            if (ox - wx) ** 2 + (oy - wy) ** 2 <= self._radius_sq:
                n = landmark.count
                landmark.xy = ((ox * n + wx) / (n + 1),
                               (oy * n + wy) / (n + 1))
                landmark.count = n + 1
                return landmark
        landmark = ObjectLandmark(id=self._next_id, class_name=class_name,
                                  xy=(wx, wy), count=1)
        self._landmarks[self._next_id] = landmark
        self._next_id += 1
        return landmark

    def confirmed(self) -> List[ObjectLandmark]:
        """Landmarks observed at least ``min_observations`` times, id order."""
        return [lm for lm in self._landmarks.values()
                if lm.count >= self._min_obs]

    def all_landmarks(self) -> List[ObjectLandmark]:
        """Every landmark, confirmed or not, in id order (for diagnostics)."""
        return list(self._landmarks.values())

    def __len__(self) -> int:
        """Total landmark count, confirmed or not."""
        return len(self._landmarks)
