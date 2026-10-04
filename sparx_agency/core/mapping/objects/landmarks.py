"""Object-landmark map: per-class radius dedupe with observation confirmation.

Pure Python, ROS-free, Python-3.8-safe. Ports the landmark bookkeeping of the
SJTU ``semantic_mapper/object_mapper_node.py`` (``_add`` / ``_confirmed`` /
``_color_for``): a detection observed within ``dedupe_radius_m`` of an
existing landmark **of the same class** merges into it (running-average
centroid, observation count up), anything else opens a new landmark, and a
landmark is only *confirmed* — trusted for publication/planning — once it has
been observed ``min_observations`` times (false-positive guard).

**Class voting** (``class_votes=True``) changes what "the same object" means:
an observation that lands on an existing landmark's footprint -- centroid
within the dedupe radius, or footprint discs overlapping by at least
``footprint_iou`` -- is folded into that landmark *whatever its class*, as a
vote. The landmark's class is the plurality of its votes, so a bed seen five
times and called a sofa once stays a bed and the sofa never reaches the map;
a landmark whose votes change majority is relabelled (and the change is
recorded). Confirmation then needs a clear plurality: the leading class must
hold ``min_observations`` votes and strictly more than the runner-up.

Landmark XY is **world ENU** (the frame of ``Pose2D`` and the BEV grid), e.g.
from :func:`sparx_agency.core.mapping.objects.geometry.backproject_bbox_to_world`.
"""
from __future__ import annotations

import colorsys
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

_HUE_BUCKETS = 997
"""Prime hue-bucket count (ported constant): spreads class hues over [0, 1)."""


@dataclass
class ObjectLandmark:
    """One deduplicated object on the map.

    Attributes:
        id: Stable landmark id, assigned in discovery order.
        class_name: Detector class (e.g. ``"chair"``). Under class voting,
            the plurality class of ``votes``.
        xy: Running-average centroid ``(x, y)`` in world ENU meters.
        count: Number of observations merged into this landmark.
        votes: Observations per detector class. Empty on a map without
            class voting (``count`` is then the only tally).
        radius_m: Running-average footprint half-extent, when the observer
            measured one; None otherwise.
    """

    id: int
    class_name: str
    xy: Tuple[float, float]
    count: int
    votes: Dict[str, int] = field(default_factory=dict)
    radius_m: Optional[float] = None

    def plurality(self) -> Tuple[Optional[str], int, int]:
        """``(leading class, its votes, runner-up votes)``; the leader is None without votes."""
        if not self.votes:
            return None, 0, 0
        ranked = sorted(self.votes.items(), key=lambda item: (-item[1], item[0]))
        second = ranked[1][1] if len(ranked) > 1 else 0
        return ranked[0][0], ranked[0][1], second


def disc_iou(center_a: Tuple[float, float], radius_a: float,
             center_b: Tuple[float, float], radius_b: float) -> float:
    """Intersection over union of two discs in the plane (the lens formula).

    The footprint model behind class voting: a detection's world footprint
    is a disc around its centroid, half as wide as its box at its depth, so
    a bed seen from two sides overlaps itself even when the two centroids
    sit a metre apart, while two cups 30 cm apart do not.
    """
    ra, rb = max(0.0, float(radius_a)), max(0.0, float(radius_b))
    if ra <= 0.0 or rb <= 0.0:
        return 0.0
    d = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    if d >= ra + rb:
        return 0.0
    area_a, area_b = math.pi * ra * ra, math.pi * rb * rb
    if d <= abs(ra - rb):
        inter = min(area_a, area_b)
    else:
        alpha = math.acos(max(-1.0, min(1.0, (d * d + ra * ra - rb * rb) / (2.0 * d * ra))))
        beta = math.acos(max(-1.0, min(1.0, (d * d + rb * rb - ra * ra) / (2.0 * d * rb))))
        inter = (ra * ra * (alpha - math.sin(2.0 * alpha) / 2.0)
                 + rb * rb * (beta - math.sin(2.0 * beta) / 2.0))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


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
        nearest_match: Try the nearest landmark first rather than the first
            in discovery order.
        class_votes: Associate by position alone and let the classes vote
            (see the module docstring). Off, the ported same-class rule.
        footprint_iou: Under class voting, two footprints whose discs
            overlap by at least this much are the same object even when
            their centroids sit further apart than the dedupe radius. The
            default 0.15 lets two equal discs merge up to about 1.25 radii
            apart -- a 1 m bed seen from both sides, not two cups.

    Raises:
        ValueError: If ``dedupe_radius_m`` is not positive, ``min_observations``
            is less than 1, or ``footprint_iou`` is outside ``(0, 1]``.
    """

    def __init__(self, dedupe_radius_m: float = 0.70,
                 min_observations: int = 2, nearest_match: bool = False,
                 class_votes: bool = False, footprint_iou: float = 0.15) -> None:
        if float(dedupe_radius_m) <= 0.0:
            raise ValueError("dedupe_radius_m must be positive, got %r"
                             % (dedupe_radius_m,))
        if int(min_observations) < 1:
            raise ValueError("min_observations must be >= 1, got %r"
                             % (min_observations,))
        if not 0.0 < float(footprint_iou) <= 1.0:
            raise ValueError("footprint_iou must lie in (0, 1], got %r" % (footprint_iou,))
        self._radius_sq = float(dedupe_radius_m) ** 2
        self._min_obs = int(min_observations)
        self._nearest_match = bool(nearest_match)
        self._class_votes = bool(class_votes)
        self._footprint_iou = float(footprint_iou)
        self._frames = {}  # type: Dict[int, int]
        self._landmarks = {}  # type: Dict[int, ObjectLandmark]
        self._next_id = 0
        #: ``(landmark id, old class, new class, count at the change)`` per relabel.
        self.relabels = []  # type: List[Tuple[int, str, str, int]]

    @property
    def class_votes(self) -> bool:
        """Whether this map lets the classes of co-located observations vote."""
        return self._class_votes

    def _associated(self, landmark: ObjectLandmark, wx: float, wy: float,
                    radius_m: Optional[float]) -> bool:
        ox, oy = landmark.xy
        if (ox - wx) ** 2 + (oy - wy) ** 2 <= self._radius_sq:
            return True
        if (self._class_votes and radius_m is not None and landmark.radius_m is not None
                and disc_iou((wx, wy), float(radius_m), landmark.xy, landmark.radius_m) >= self._footprint_iou):
            return True
        return False

    def matches(self, xy: Tuple[float, float], radius_m: Optional[float] = None,
                class_name: Optional[str] = None,
                exclude: Iterable[int] = ()) -> List[ObjectLandmark]:
        """Every landmark an observation at ``xy`` could belong to, best first.

        Under class voting any class qualifies; otherwise only ``class_name``
        (required then). ``exclude`` drops landmark ids already fed by the
        same frame. The order is by centroid distance when ``nearest_match``,
        else discovery order -- the same order :meth:`observe` folds in.
        """
        wx, wy = float(xy[0]), float(xy[1])
        excluded = set(int(i) for i in exclude)
        candidates = [lm for lm in self._landmarks.values() if lm.id not in excluded]
        if not self._class_votes:
            if class_name is None:
                raise ValueError("class_name is required without class voting")
            candidates = [lm for lm in candidates if lm.class_name == class_name]
        if self._nearest_match:
            candidates.sort(key=lambda lm: (lm.xy[0] - wx) ** 2 + (lm.xy[1] - wy) ** 2)
        return [lm for lm in candidates if self._associated(lm, wx, wy, radius_m)]

    def match(self, xy: Tuple[float, float], radius_m: Optional[float] = None,
              class_name: Optional[str] = None, exclude: Iterable[int] = ()) -> Optional[ObjectLandmark]:
        """The landmark :meth:`observe` would fold an observation at ``xy`` into, or None."""
        found = self.matches(xy, radius_m, class_name, exclude)
        return found[0] if found else None

    def observe(self, class_name: str,
                xy: Tuple[float, float], frame_id: Optional[int] = None,
                radius_m: Optional[float] = None,
                landmark: Optional[ObjectLandmark] = None) -> ObjectLandmark:
        """Fold one world-ENU observation in; return the landmark it landed on.

        The first associated landmark (see :meth:`matches`) absorbs the
        observation via a running average; otherwise a new landmark is
        opened with ``count=1``. A frame that already fed a landmark does
        not count twice on it.

        Args:
            class_name: Detector class of the observation.
            xy: Observation ``(x, y)`` in world ENU meters.
            frame_id: The frame the observation came from, for the once-per-frame rule.
            radius_m: The observation's footprint half-extent, if measured.
            landmark: The landmark the caller already chose (its own
                association, e.g. with a height check this 2-D map cannot
                make); None lets the map choose.

        Returns:
            The merged-into or newly created landmark (live object — its
            ``xy``/``count``/``votes`` keep updating on later observations).
        """
        wx, wy = float(xy[0]), float(xy[1])
        if landmark is None:
            landmark = self.match((wx, wy), radius_m, class_name)
        if landmark is not None:
            if frame_id is not None and self._frames.get(landmark.id) == frame_id:
                return landmark
            self._fold(landmark, class_name, wx, wy, radius_m)
            if frame_id is not None:
                self._frames[landmark.id] = frame_id
            return landmark
        landmark = ObjectLandmark(id=self._next_id, class_name=class_name,
                                  xy=(wx, wy), count=1,
                                  votes={class_name: 1} if self._class_votes else {},
                                  radius_m=None if radius_m is None else float(radius_m))
        self._landmarks[self._next_id] = landmark
        if frame_id is not None:
            self._frames[landmark.id] = frame_id
        self._next_id += 1
        return landmark

    def _fold(self, landmark: ObjectLandmark, class_name: str, wx: float, wy: float,
              radius_m: Optional[float]) -> None:
        n = landmark.count
        ox, oy = landmark.xy
        landmark.xy = ((ox * n + wx) / (n + 1), (oy * n + wy) / (n + 1))
        if radius_m is not None:
            landmark.radius_m = (float(radius_m) if landmark.radius_m is None
                                 else (landmark.radius_m * n + float(radius_m)) / (n + 1))
        landmark.count = n + 1
        if not self._class_votes:
            return
        landmark.votes[class_name] = landmark.votes.get(class_name, 0) + 1
        leader, top, _ = landmark.plurality()
        # A tie keeps the current class: the evidence has to beat it, not match it.
        if leader is not None and leader != landmark.class_name and top > landmark.votes.get(landmark.class_name, 0):
            self.relabels.append((landmark.id, landmark.class_name, leader, landmark.count))
            landmark.class_name = leader

    def is_confirmed(self, landmark: ObjectLandmark) -> bool:
        """Whether one landmark passes the confirmation rule of this map."""
        if not self._class_votes:
            return landmark.count >= self._min_obs
        _, top, second = landmark.plurality()
        return top >= self._min_obs and top > second

    def confirmed(self) -> List[ObjectLandmark]:
        """Landmarks observed at least ``min_observations`` times, id order.

        Under class voting: whose leading class holds that many votes and
        strictly more than the runner-up -- a bed/sofa tie is not an object
        the search may act on yet.
        """
        return [lm for lm in self._landmarks.values() if self.is_confirmed(lm)]

    def all_landmarks(self) -> List[ObjectLandmark]:
        """Every landmark, confirmed or not, in id order (for diagnostics)."""
        return list(self._landmarks.values())

    def __len__(self) -> int:
        """Total landmark count, confirmed or not."""
        return len(self._landmarks)
