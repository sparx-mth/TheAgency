"""A building as rooms, corridors and the openings between them.

The vocabulary an exploration supervisor needs and an occupancy grid does not
provide. A grid answers "is this cell free"; a mission needs "which room am I
in", "which rooms have I not been in", and "where is the door from here to
there". This is that layer: a per-cell label image co-registered with the map,
plus a small graph over it.

**It is ground truth, computed once and read back.** The decomposition
(:mod:`region_decomposition`) runs offline against a surveyed world and the
result is committed beside the map, for the same reason the map itself is: it
never changes unless the building does, it is worth a human's eye before a
flight rather than after one, and a supervisor that re-derived it at start-up
would be re-deciding what a room is on every run.

Two files, mirroring ``robots/SJTU/maps/hospital.{yaml,npz}``:

* the **YAML** is the part a person reads and may correct -- names, kinds,
  centres, areas, and the portal list;
* the **NPZ** carries the exact ``(H, W)`` int32 label grid, because a polygon
  approximation of a room drawn from a raster is a second answer to a question
  the raster already answered.

ROS-free, numpy-only, no scipy, Python 3.8 syntax.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from sparx_agency.core.common.types.geometry import Pose2D
from sparx_agency.core.common.types.semantics import Portal2D

#: Label value for a cell that belongs to no region (outside the building).
NO_REGION = 0


@dataclass(frozen=True)
class Region:
    """One room or one stretch of corridor.

    Deliberately not
    :class:`~sparx_agency.core.common.types.semantics.Region2D`, which is the
    abstract shape of the same idea: string ids, a ``boundary`` of ``Any``, and
    everything else in a free-form ``tags`` dict. Nothing consumes it today, and
    adopting it here would mean stringly-typed ids that have to index an integer
    label grid on every pose lookup, with the name, kind and area a supervisor
    reads on every tick buried in ``tags``. The portal type below *is* the
    shared one, because there the fields line up exactly.

    Attributes:
        id: Label value in the grid. Never :data:`NO_REGION`.
        name: Human name, used verbatim in the instruction given to the policy
            -- so it has to read naturally after "you are in the ".
        kind: ``"room"`` or ``"corridor"``. The mission generator treats them
            differently: a room is somewhere to go into and come out of, a
            corridor is somewhere to travel along looking for doors.
        area_m2: Floor area.
        centre: ``(x, y)`` world centroid, metres. Snapped to a cell that is
            actually inside the region, so it is somewhere the aircraft could
            be, not the centre of mass of a horseshoe.
        portals: Ids of the portals on this region's boundary (strings, as
            :class:`Portal2D` numbers them).
    """

    id: int
    name: str
    kind: str
    area_m2: float
    centre: Tuple[float, float]
    portals: Tuple[str, ...] = ()

    @property
    def is_room(self) -> bool:
        return self.kind == "room"


@dataclass(frozen=True)
class Portal(Portal2D):
    """An opening between two regions -- a doorway, or a corridor junction.

    A :class:`~sparx_agency.core.common.types.semantics.Portal2D` with the one
    thing that type has no room for: which two regions it joins. Everything else
    -- the centre pose, the outward normal, the clear width -- is inherited, so
    a portal found here is directly usable by
    :class:`~sparx_agency.core.planning.behaviors.algorithmic.enter_portal.EnterPortalBehavior`
    and by anything else already speaking that vocabulary, rather than being a
    second description of the same doorway.

    ``normal_yaw`` points from the lower-numbered region towards the higher, and
    it is worth having for more than tidiness: the approach-then-cross geometry
    that behaviour uses is built from it.

    Attributes:
        between: The two region ids it joins, ascending. Defaulted only because
            a dataclass cannot put a required field after inherited optional
            ones; a portal without it is not a portal this module made.
    """

    between: Tuple[int, int] = (0, 0)

    @property
    def centre(self) -> Tuple[float, float]:
        """``(x, y)`` of the opening, metres. The spelling this package uses."""
        return (self.center.x, self.center.y)

    def other(self, region_id: int) -> int:
        """The region on the far side from ``region_id``."""
        first, second = self.between
        if region_id == first:
            return second
        if region_id == second:
            return first
        raise ValueError("portal %s does not touch region %d" % (self.id, region_id))


class RegionMap:
    """Regions, portals and the label grid that places them.

    Args:
        labels: ``(H, W)`` int array, row 0 at minimum y, holding region ids
            and :data:`NO_REGION` outside the building.
        resolution: Metres per cell.
        origin_x: World x of column 0's left edge.
        origin_y: World y of row 0's bottom edge.
        regions: The regions, which must cover every label present.
        portals: The openings.

    Raises:
        ValueError: A label in the grid has no region, or a portal names a
            region that does not exist. Both mean the two files have drifted
            apart, which is worse than either being missing.
    """

    def __init__(self, labels, resolution, origin_x, origin_y, regions, portals):
        # type: (np.ndarray, float, float, float, Sequence[Region], Sequence[Portal]) -> None
        self.labels = np.asarray(labels)
        if self.labels.ndim != 2:
            raise ValueError("labels must be 2D, got %r" % (self.labels.shape,))
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.height, self.width = self.labels.shape
        self.regions = {r.id: r for r in regions}  # type: Dict[int, Region]
        self.portals = {p.id: p for p in portals}  # type: Dict[str, Portal]

        present = set(int(v) for v in np.unique(self.labels)) - {NO_REGION}
        missing = present - set(self.regions)
        if missing:
            raise ValueError("labels contain regions with no entry: %s"
                             % (sorted(missing),))
        for portal in self.portals.values():
            for rid in portal.between:
                if rid not in self.regions:
                    raise ValueError("portal %d names region %d, which does not exist"
                                     % (portal.id, rid))

    # -- lookup -----------------------------------------------------------

    #: How far to look for a label when the cell under the aircraft has none,
    #: metres. A drone cruising at 1.20 m spends a good deal of its time over
    #: furniture -- the flight-band map counts a 0.75 m desk as occupied because
    #: it is something the aircraft *could* hit, and an unlabelled cell there
    #: means "above a desk", not "outside the building". One metre covers the
    #: widest piece of furniture in this world without ever reaching across a
    #: wall into the next room.
    SEARCH_M = 1.0

    def cell_of(self, x, y):
        # type: (float, float) -> Tuple[int, int]
        """``(col, row)`` for a world point -- may be off the grid.

        Deliberately not clamped: a caller needs to be able to tell "outside
        the map" from "on its edge", and silently returning the nearest legal
        cell is how an off-map query starts reading a wall on the far side of
        the building.
        """
        return (int(np.floor((x - self.origin_x) / self.resolution)),
                int(np.floor((y - self.origin_y) / self.resolution)))

    def region_at(self, x, y, search_m=None):
        # type: (float, float, Optional[float]) -> Optional[Region]
        """Which region contains a world point, or None outside the building.

        Falls back to the nearest labelled cell within ``search_m`` when the
        cell itself has no label. Without that the aircraft loses its place
        every time it crosses a bed, and a supervisor with no place has no
        mission -- measured in simulation as the survey going silent for the
        rest of the flight the first time the aircraft passed over a desk.
        """
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        gx, gy = self.cell_of(x, y)
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return None
        value = int(self.labels[gy, gx])
        if value != NO_REGION:
            return self.regions.get(value)
        radius = self.SEARCH_M if search_m is None else float(search_m)
        cells = int(radius / self.resolution)
        if cells <= 0:
            return None
        lo_y, hi_y = max(0, gy - cells), min(self.height, gy + cells + 1)
        lo_x, hi_x = max(0, gx - cells), min(self.width, gx + cells + 1)
        window = self.labels[lo_y:hi_y, lo_x:hi_x]
        rows, cols = np.nonzero(window != NO_REGION)
        if not rows.size:
            return None
        # A distance-weighted vote, not the single nearest cell. Hovering over
        # a desk against a room's wall, the nearest labelled cell is as likely
        # to be the corridor on the other side as the room the aircraft is
        # plainly in -- a tie that `argmin` breaks by scan order, which is to
        # say by accident. Weighting every labelled cell in the window by
        # 1/(1+d^2) lets the region the aircraft is surrounded by win, which is
        # the question being asked.
        d2 = ((rows + lo_y - gy) ** 2 + (cols + lo_x - gx) ** 2).astype(float)
        votes = np.bincount(window[rows, cols].astype(np.int64),
                            weights=1.0 / (1.0 + d2))
        return self.regions.get(int(np.argmax(votes)))

    def mask_of(self, region_id):
        # type: (int) -> np.ndarray
        """``(H, W)`` boolean of the cells belonging to one region."""
        return self.labels == int(region_id)

    def rooms(self):
        # type: () -> List[Region]
        """Every room, in id order. The checklist a full survey has to clear."""
        return [r for _, r in sorted(self.regions.items()) if r.is_room]

    def corridors(self):
        # type: () -> List[Region]
        """Every corridor stretch, in id order."""
        return [r for _, r in sorted(self.regions.items()) if not r.is_room]

    def portals_of(self, region_id):
        # type: (int) -> List[Portal]
        """The openings on one region's boundary, widest first.

        Widest first because that is the order a mission generator should try
        them in: the airframe is 0.63 m wide and eighteen of this building's
        twenty-six doorways are 0.93 m clear, so the widest way out of a room
        is the one most likely to be flyable.
        """
        region = self.regions.get(int(region_id))
        if region is None:
            return []
        found = [self.portals[p] for p in region.portals if p in self.portals]
        return sorted(found, key=lambda p: -p.width_m)

    def neighbours(self, region_id):
        # type: (int) -> List[Tuple[Region, Portal]]
        """Regions reachable directly from this one, with the portal to each."""
        out = []
        for portal in self.portals_of(region_id):
            other = self.regions.get(portal.other(int(region_id)))
            if other is not None:
                out.append((other, portal))
        return out

    def portal_between(self, a, b):
        # type: (int, int) -> Optional[Portal]
        """The widest opening directly joining two regions, if any."""
        pair = (min(int(a), int(b)), max(int(a), int(b)))
        found = [p for p in self.portals.values() if p.between == pair]
        return max(found, key=lambda p: p.width_m) if found else None

    # -- persistence ------------------------------------------------------

    def to_yaml_dict(self):
        # type: () -> Dict
        """The human-readable half, ready for ``yaml.safe_dump``."""
        return {
            "labels": os.path.basename(self._labels_name),
            "resolution": self.resolution,
            "origin": [self.origin_x, self.origin_y],
            "regions": [
                {"id": r.id, "name": r.name, "kind": r.kind,
                 "area_m2": round(r.area_m2, 2),
                 "centre": [round(r.centre[0], 2), round(r.centre[1], 2)],
                 "portals": [str(p) for p in r.portals]}
                for _, r in sorted(self.regions.items())
            ],
            "portals": [
                {"id": p.id, "between": list(p.between),
                 "centre": [round(p.centre[0], 2), round(p.centre[1], 2)],
                 "normal_yaw": round(float(p.normal_yaw or 0.0), 4),
                 "width_m": round(p.width_m, 2)}
                for _, p in sorted(self.portals.items(), key=lambda kv: int(kv[0]))
            ],
        }

    _labels_name = "regions.npz"

    def save(self, yaml_path):
        # type: (str) -> Tuple[str, str]
        """Write the YAML and its sibling NPZ. Returns both paths."""
        yaml_path = os.path.abspath(yaml_path)
        stem = os.path.splitext(yaml_path)[0]
        npz_path = stem + ".npz"
        self._labels_name = os.path.basename(npz_path)
        np.savez_compressed(npz_path, labels=self.labels.astype(np.int32),
                            resolution=self.resolution,
                            origin=np.array([self.origin_x, self.origin_y]))
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, "w") as handle:
            yaml.safe_dump(self.to_yaml_dict(), handle, sort_keys=False,
                           default_flow_style=None)
        return yaml_path, npz_path

    @classmethod
    def load(cls, yaml_path):
        # type: (str) -> "RegionMap"
        """Read a region map back. ``labels:`` resolves relative to the YAML.

        Raises:
            FileNotFoundError: The YAML or the NPZ it names is missing.
            ValueError: The YAML is not a region map, or has drifted from the
                grid it names.
        """
        yaml_path = os.path.abspath(yaml_path)
        with open(yaml_path, "r") as handle:
            meta = yaml.safe_load(handle) or {}
        for key in ("labels", "resolution", "origin", "regions"):
            if key not in meta:
                raise ValueError("%s is not a region map: no %r" % (yaml_path, key))
        npz_path = meta["labels"]
        if not os.path.isabs(npz_path):
            npz_path = os.path.join(os.path.dirname(yaml_path), npz_path)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError("region labels %s (named by %s)"
                                    % (npz_path, yaml_path))
        data = np.load(npz_path)
        regions = [Region(id=int(r["id"]), name=str(r["name"]), kind=str(r["kind"]),
                          area_m2=float(r["area_m2"]),
                          centre=(float(r["centre"][0]), float(r["centre"][1])),
                          portals=tuple(str(p) for p in r.get("portals", ())))
                   for r in meta["regions"]]
        portals = [Portal(id=str(p["id"]),
                          center=Pose2D(x=float(p["centre"][0]),
                                        y=float(p["centre"][1]),
                                        yaw=float(p.get("normal_yaw", 0.0))),
                          normal_yaw=float(p.get("normal_yaw", 0.0)),
                          width_m=float(p["width_m"]),
                          between=(int(p["between"][0]), int(p["between"][1])))
                   for p in meta.get("portals", [])]
        out = cls(data["labels"], float(meta["resolution"]),
                  float(meta["origin"][0]), float(meta["origin"][1]),
                  regions, portals)
        out._labels_name = os.path.basename(npz_path)
        return out


def load_region_map(path, logger=None):
    # type: (Optional[str], object) -> Optional[RegionMap]
    """Load a region map if one is configured and present, else ``None``.

    Missing is a legitimate state -- the supervisor simply cannot run and says
    so once -- but a map that is configured and *broken* is not, and raises.
    """
    if not path:
        return None
    if not os.path.isfile(path):
        if logger is not None:
            logger.warn("region map %s not found; no exploration supervisor" % path)
        return None
    region_map = RegionMap.load(path)
    if logger is not None:
        logger.info("region map %s: %d rooms, %d corridor stretches, %d portals"
                    % (path, len(region_map.rooms()), len(region_map.corridors()),
                       len(region_map.portals)))
    return region_map
