"""Train / validation / test as three *places*, not three random subsets.

A navigation policy that has seen a corridor will fly that corridor. Splitting
frames at random would put the same corridor -- often the same metre of it, at
10 Hz -- in both the training and the test set, and the resulting numbers would
measure memorisation. So the split is **spatial**: each split owns a disjoint
region of the building, and a sample counts for a split only if the aircraft is
inside that region *and the whole expert route it is being taught stays there*.

What may cross a boundary is the **goal**. A goal is two numbers handed to the
policy as a direction; it reveals no imagery and no geometry, and far goals
pointing out of the current region are exactly the samples worth keeping (they
are what teaches "head for the door"). What must not cross is anything the
network sees or is scored on.

A ``buffer_m`` strip between regions keeps sample anchors off the exact
boundary line so a metre of estimator noise cannot move a sample between splits.

When more than one building has been surveyed, name a whole scene as a split
(``scene_split:``) -- an unseen *building* is a far stronger generalisation test
than an unseen wing, and this file supports both at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

SPLITS: Tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    """An axis-aligned world region, metres."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def contains(self, xs: np.ndarray, ys: np.ndarray, shrink: float = 0.0) -> np.ndarray:
        """Boolean mask of which points lie inside, optionally eroded."""
        return ((xs >= self.x_min + shrink) & (xs <= self.x_max - shrink)
                & (ys >= self.y_min + shrink) & (ys <= self.y_max - shrink))


@dataclass(frozen=True)
class SceneZones:
    """The three regions of one scene, plus the strip between them."""

    boxes: Dict[str, Tuple[Box, ...]]
    buffer_m: float = 1.5

    def _mask(self, split: str, xs: np.ndarray, ys: np.ndarray,
              shrink: float) -> np.ndarray:
        boxes = self.boxes.get(split, ())
        if not boxes:
            return np.zeros(np.shape(xs), dtype=bool)
        inside = np.zeros(np.shape(xs), dtype=bool)
        for box in boxes:
            inside |= box.contains(xs, ys, shrink)
        return inside

    def assign(self, x: float, y: float) -> Optional[str]:
        """Which split a sample *anchor* at ``(x, y)`` belongs to.

        Returns ``None`` inside the buffer strip or outside every region, which
        is the caller's signal to drop the sample rather than guess.
        """
        xs = np.asarray([x], dtype=np.float64)
        ys = np.asarray([y], dtype=np.float64)
        hits = [s for s in SPLITS if self._mask(s, xs, ys, self.buffer_m)[0]]
        return hits[0] if len(hits) == 1 else None

    def route_inside(self, split: str, points: np.ndarray) -> bool:
        """True if every point of an ``(N, 2)`` route stays in ``split``'s region.

        The region is *not* eroded here: the buffer strip belongs to whichever
        side reaches into it, since no other split's anchors are there.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return bool(self._mask(split, points[:, 0], points[:, 1], 0.0).all())


@dataclass(frozen=True)
class SplitPlan:
    """The whole split definition: per-scene zones plus whole-scene assignments."""

    zones: Dict[str, SceneZones]
    scene_split: Dict[str, str]

    def assign(self, scene: str, x: float, y: float) -> Optional[str]:
        """Split for an anchor, whole-scene assignment taking precedence."""
        forced = self.scene_split.get(scene)
        if forced is not None:
            return forced
        zones = self.zones.get(scene)
        return None if zones is None else zones.assign(x, y)

    def route_inside(self, scene: str, split: str, points: np.ndarray) -> bool:
        """True if a route is allowed to be taught in ``split``."""
        if self.scene_split.get(scene) is not None:
            return True                      # a whole scene has no internal boundary
        zones = self.zones.get(scene)
        return False if zones is None else zones.route_inside(split, points)

    def describe(self) -> List[str]:
        """Human-readable lines, for the run log and the report."""
        lines: List[str] = []
        for scene, split in sorted(self.scene_split.items()):
            lines.append(f"{scene}: entire scene -> {split}")
        for scene, zones in sorted(self.zones.items()):
            for split in SPLITS:
                for box in zones.boxes.get(split, ()):
                    lines.append(
                        f"{scene}/{split}: x [{box.x_min:.1f}, {box.x_max:.1f}] "
                        f"y [{box.y_min:.1f}, {box.y_max:.1f}]")
            lines.append(f"{scene}: buffer {zones.buffer_m:.1f} m between regions")
        return lines


def load_split_plan(path) -> SplitPlan:
    """Read a split YAML.

    Expected shape::

        buffer_m: 1.5
        scenes:
          office:
            train: [[x_min, y_min, x_max, y_max], ...]
            val:   [...]
            test:  [...]
        scene_split:            # optional; a whole surveyed building per split
          warehouse: test

    Raises:
        ValueError: If a split names no region in any scene -- an empty test set
            is a silent way to report a meaningless result.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    default_buffer = float(raw.get("buffer_m", 1.5))

    zones: Dict[str, SceneZones] = {}
    for scene, per_split in (raw.get("scenes") or {}).items():
        boxes: Dict[str, Tuple[Box, ...]] = {}
        for split in SPLITS:
            entries = (per_split or {}).get(split) or []
            boxes[split] = tuple(Box(*[float(v) for v in entry]) for entry in entries)
        zones[scene] = SceneZones(boxes=boxes,
                                  buffer_m=float((per_split or {}).get("buffer_m",
                                                                       default_buffer)))

    scene_split = {str(k): str(v) for k, v in (raw.get("scene_split") or {}).items()}
    for split in scene_split.values():
        if split not in SPLITS:
            raise ValueError(f"scene_split names unknown split {split!r}")

    covered = set(scene_split.values())
    for zone in zones.values():
        covered |= {s for s in SPLITS if zone.boxes.get(s)}
    missing = [s for s in SPLITS if s not in covered]
    if missing:
        raise ValueError(
            f"split plan {path} defines no region for {missing} -- every split needs "
            f"one, or the run reports a metric over an empty set")
    return SplitPlan(zones=zones, scene_split=scene_split)


def split_counts(assignments: Iterable[Optional[str]]) -> Dict[str, int]:
    """Tally split assignments, counting ``None`` as ``dropped``."""
    counts = {split: 0 for split in SPLITS}
    counts["dropped"] = 0
    for value in assignments:
        counts[value if value in counts else "dropped"] += 1
    return counts


def format_counts(counts: Dict[str, int]) -> str:
    """One-line summary with percentages."""
    total = max(sum(counts.values()), 1)
    return "  ".join(f"{name}={counts[name]} ({100.0 * counts[name] / total:.1f}%)"
                     for name in ("train", "val", "test", "dropped"))
