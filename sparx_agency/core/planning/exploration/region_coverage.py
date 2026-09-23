"""Which rooms have been looked into, and how thoroughly -- the survey scoreboard.

:class:`~sparx_agency.core.planning.exploration.visibility_coverage.VisibilityCoverage`
answers "how much of the building" as one number. A supervisor deciding where to
send the aircraft next needs that number broken out per room and per corridor,
because "explore the hospital" is really "clear a checklist" and the checklist is
the rooms.

It is a *view* over the coverage mask rather than a second accumulator: the mask
is the one record of what the camera has seen, and two things counting the same
cells independently is how two numbers that must agree stop agreeing.

**Scanned is a threshold, not a certainty.** A room is called scanned when
enough of its floor has been seen, and "enough" is well under all of it -- the
far corner behind a bed is occluded from every pose the aircraft can reach, so a
supervisor waiting for 100 % would never leave the first room. The threshold is
the caller's, and it is the single number that trades thoroughness for progress.

ROS-free, numpy-only, Python 3.8 syntax.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.core.planning.exploration.region_map import Region, RegionMap


@dataclass(frozen=True)
class RegionProgress:
    """How far through one region the survey is.

    Attributes:
        region: The region itself.
        cells_total: Countable cells in it.
        cells_seen: How many of those the camera has reached.
        entered: Whether the aircraft has ever been inside it.
    """

    region: Region
    cells_total: int
    cells_seen: int
    entered: bool

    @property
    def fraction(self) -> float:
        """Share of this region's floor seen, 0..1."""
        return (self.cells_seen / float(self.cells_total)) if self.cells_total else 0.0


class RegionCoverage:
    """Per-region progress, read off a shared seen-mask.

    Args:
        region_map: The building's decomposition.
        countable: ``(H, W)`` boolean of the cells that count towards coverage
            -- normally ``VisibilityCoverage.countable_mask``, so the two agree
            on the denominator by construction.
        scanned_fraction: How much of a region's floor must be seen before it
            counts as scanned.

    Raises:
        ValueError: ``countable`` does not match the region grid.
    """

    def __init__(self, region_map, countable, scanned_fraction=0.60):
        # type: (RegionMap, np.ndarray, float) -> None
        countable = np.asarray(countable, dtype=bool)
        if countable.shape != region_map.labels.shape:
            raise ValueError("countable %r does not match the region grid %r"
                             % (countable.shape, region_map.labels.shape))
        self.region_map = region_map
        self.scanned_fraction = float(scanned_fraction)
        self._countable = countable
        # One flat pass now so every later query is a bincount rather than a
        # mask-and-sum per region: the supervisor asks for this several times a
        # second and there are twenty-seven regions.
        self._cell_region = np.where(countable, region_map.labels, 0).ravel()
        self._n = int(region_map.labels.max()) + 1
        self._totals = np.bincount(self._cell_region, minlength=self._n)
        self._entered = set()  # type: set

    @property
    def countable(self):
        # type: () -> np.ndarray
        """``(H, W)`` boolean of the cells that count, shared with the tracker."""
        return self._countable

    # -- updates ----------------------------------------------------------

    def note_pose(self, x, y):
        # type: (float, float) -> Optional[Region]
        """Record that the aircraft is here; returns the region, or None.

        "Entered" is deliberately separate from "seen". A room can be seen
        thoroughly from its doorway without the aircraft ever crossing the
        threshold, and an instruction that says to go *into* every room is
        entitled to know the difference.
        """
        region = self.region_map.region_at(x, y)
        if region is not None:
            self._entered.add(region.id)
        return region

    # -- queries ----------------------------------------------------------

    def progress(self, seen_mask):
        # type: (np.ndarray) -> Dict[int, RegionProgress]
        """Progress for every region, against the current seen mask."""
        seen = np.asarray(seen_mask, dtype=bool).ravel()
        counts = np.bincount(self._cell_region[seen], minlength=self._n)
        out = {}
        for rid, region in self.region_map.regions.items():
            out[rid] = RegionProgress(
                region=region, cells_total=int(self._totals[rid]),
                cells_seen=int(counts[rid]) if rid < counts.size else 0,
                entered=rid in self._entered)
        return out

    def fraction_of(self, seen_mask, region_id):
        # type: (np.ndarray, int) -> float
        """Share of one region's floor seen."""
        return self.progress(seen_mask).get(int(region_id),
                                            _EMPTY).fraction

    def is_scanned(self, progress, region_id):
        # type: (Dict[int, RegionProgress], int) -> bool
        """Has this region been seen thoroughly enough to tick off?"""
        entry = progress.get(int(region_id))
        return bool(entry) and entry.fraction >= self.scanned_fraction

    def unscanned_rooms(self, progress):
        # type: (Dict[int, RegionProgress]) -> List[RegionProgress]
        """Rooms still to do, emptiest first.

        Emptiest first rather than nearest: the supervisor applies distance and
        reachability itself, and a list ordered by what is *least known* is the
        honest starting point for that.
        """
        out = [p for p in progress.values()
               if p.region.is_room and p.fraction < self.scanned_fraction]
        return sorted(out, key=lambda p: (p.fraction, p.region.id))

    def summary(self, progress):
        # type: (Dict[int, RegionProgress]) -> Tuple[int, int, float]
        """``(rooms scanned, rooms total, share of the whole floor seen)``."""
        rooms = [p for p in progress.values() if p.region.is_room]
        done = sum(1 for p in rooms if p.fraction >= self.scanned_fraction)
        total = sum(p.cells_total for p in progress.values())
        seen = sum(p.cells_seen for p in progress.values())
        return (done, len(rooms), (seen / float(total)) if total else 0.0)


_EMPTY = RegionProgress(region=Region(id=0, name="", kind="room", area_m2=0.0,
                                      centre=(0.0, 0.0)),
                        cells_total=0, cells_seen=0, entered=False)
