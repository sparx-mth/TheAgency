#!/usr/bin/env python3
"""Decompose a surveyed world into rooms, corridors and doorways.

The second half of :mod:`build_map`. That one turns a world into an occupancy
grid; this one turns *two* of those grids -- the band the aircraft flies in and
a band up under the ceiling -- into the topological map an exploration
supervisor plans over.

Why two bands is explained where the algorithm lives
(``core.planning.exploration.region_decomposition``): at flight height every
doorway stands open and the whole floor is one blob, while at lintel height the
openings are closed and the rooms fall apart into connected components by
themselves. Nothing is hand-drawn.

The corridors are the exception and always will be: they have no doors between
them, so no slice separates them, and where one stretch of corridor ends and the
next begins is a judgement rather than a measurement. Pass ``--corridor-band``
once per stretch; they are recorded in the output for a person to read and
argue with.

Usage, from the repo root, for the SJTU hospital::

    # 1. the wall band, if you have not built it (build_map writes it beside the
    #    flight-band map; it is an input here and is not itself committed)
    .venv/bin/python -m sparx_agency.tasks.mapping.gazebo_world_occupancy.build_map \\
        --world $SJTU_PROJECT_DIR/aws-robomaker-hospital-world/worlds/hospital.world \\
        --search-path $SJTU_PROJECT_DIR/aws-robomaker-hospital-world/models \\
        --search-path $SJTU_PROJECT_DIR/aws-robomaker-hospital-world/fuel_models \\
        --output-dir /tmp/walls --name hospital_walls \\
        --resolution 0.05 --z-min 2.10 --z-max 2.40

    # 2. the decomposition
    .venv/bin/python -m sparx_agency.tasks.mapping.gazebo_world_occupancy.build_regions \\
        --flight-map sparx_agency/robots/SJTU/maps/hospital.yaml \\
        --wall-map   /tmp/walls/hospital_walls.yaml \\
        --output     sparx_agency/robots/SJTU/maps/hospital_regions.yaml \\
        --corridor-seed -0.09 11.84 \\
        --corridor-band "the south hall:-40:-32.2" \\
        --corridor-band "the south corridor:-32.2:-26.2" \\
        --corridor-band "the southern spine:-26.2:-16.2" \\
        --corridor-band "the middle spine:-16.2:-4.9" \\
        --corridor-band "the reception:-4.9:3.9" \\
        --corridor-band "the atrium:3.9:14.8" \\
        --corridor-band "the north corridor:14.8:40" \\
        --preview /tmp/hospital_regions.png

**The names in the output are meant to be edited.** They are generated from
where each region sits in the building, which is honest and readable but knows
nothing about what a room is for. They are handed verbatim to the policy inside
"you are in the ...", so improving them improves the flight; the YAML is the
place to do it and re-running this would overwrite it, so re-run first and
rename after.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.planning.exploration.region_decomposition import (
    decompose_regions,
    split_region_by_bands,
)
from sparx_agency.core.planning.exploration.region_map import NO_REGION, RegionMap
from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
    load_map_backdrop,
)


def spatial_names(region_map, kind="room"):
    # type: (RegionMap, str) -> Dict[int, str]
    """A readable name per region from where it sits in the building.

    Compass-and-index, e.g. ``"the south-east room"`` and, where a quadrant
    holds several, ``"the south-east room (2)"``. It is a placeholder that
    reads correctly rather than a claim about what the room is for.
    """
    targets = [r for r in region_map.regions.values() if r.kind == kind]
    if not targets:
        return {}
    xs = [r.centre[0] for r in targets]
    ys = [r.centre[1] for r in targets]
    mid_x = 0.5 * (min(xs) + max(xs))
    mid_y = 0.5 * (min(ys) + max(ys))
    span_x = max(1e-6, max(xs) - min(xs))
    span_y = max(1e-6, max(ys) - min(ys))

    def compass(region):
        x, y = region.centre
        parts = []
        if abs(y - mid_y) > 0.12 * span_y:
            parts.append("north" if y > mid_y else "south")
        if abs(x - mid_x) > 0.12 * span_x:
            parts.append("east" if x > mid_x else "west")
        return "-".join(parts) if parts else "central"

    grouped = {}  # type: Dict[str, List[int]]
    for region in sorted(targets, key=lambda r: (-r.area_m2, r.id)):
        grouped.setdefault(compass(region), []).append(region.id)
    out = {}
    for where, ids in grouped.items():
        for index, rid in enumerate(ids, start=1):
            suffix = "" if len(ids) == 1 else " (%d)" % index
            out[rid] = "the %s %s%s" % (where, kind, suffix)
    return out


def _band(text):
    # type: (str) -> Tuple[str, float, float]
    """``"name:ymin:ymax"`` -> a band. The name may not contain a colon."""
    parts = text.rsplit(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "expected 'name:ymin:ymax', got %r" % (text,))
    try:
        return (parts[0].strip(), float(parts[1]), float(parts[2]))
    except ValueError:
        raise argparse.ArgumentTypeError("band bounds must be numbers: %r" % (text,))


def _preview(region_map, path):
    # type: (RegionMap, str) -> None
    """Write a coloured PNG of the decomposition, for the eye that must check it."""
    import cv2

    labels = region_map.labels
    rng = np.random.RandomState(7)
    image = np.full(labels.shape + (3,), 24, np.uint8)
    for region in region_map.regions.values():
        base = rng.randint(70, 235, 3)
        if region.kind == "corridor":
            base = (base * 0.45 + 60).astype(int)   # corridors muted, rooms bright
        image[labels == region.id] = tuple(int(c) for c in base)
    for portal in region_map.portals.values():
        col = int((portal.centre[0] - region_map.origin_x) / region_map.resolution)
        row = int((portal.centre[1] - region_map.origin_y) / region_map.resolution)
        cv2.circle(image, (col, row), 3, (255, 255, 255), -1, cv2.LINE_AA)
    for region in region_map.regions.values():
        col = int((region.centre[0] - region_map.origin_x) / region_map.resolution)
        row = int((region.centre[1] - region_map.origin_y) / region_map.resolution)
        cv2.putText(image, str(region.id), (col - 8, row + 5), 0, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, str(region.id), (col - 8, row + 5), 0, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, np.flipud(image))       # back to picture order, y down


def main(argv=None):
    # type: (Optional[Sequence[str]]) -> int
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flight-map", required=True,
                        help="nav2 YAML of the band the aircraft flies in")
    parser.add_argument("--wall-map", required=True,
                        help="nav2 YAML of a band where doorways are closed")
    parser.add_argument("--output", required=True,
                        help="region map YAML to write (an NPZ lands beside it)")
    parser.add_argument("--corridor-seed", nargs=2, type=float, metavar=("X", "Y"),
                        help="a point inside the circulation space; defaults to "
                             "the largest region")
    parser.add_argument("--corridor-band", action="append", type=_band, default=[],
                        metavar="NAME:YMIN:YMAX",
                        help="split the circulation into named y slices; repeatable")
    parser.add_argument("--min-room-m2", type=float, default=3.0)
    parser.add_argument("--min-portal-m", type=float, default=0.30)
    parser.add_argument("--preview", default=None, help="write a PNG to check by eye")
    args = parser.parse_args(argv)

    flight = load_map_backdrop(args.flight_map)
    walls = load_map_backdrop(args.wall_map)
    if flight is None or walls is None:
        print("[regions] a map is missing; nothing to decompose", file=sys.stderr)
        return 2
    if (flight.resolution != walls.resolution
            or flight.origin_x != walls.origin_x
            or flight.origin_y != walls.origin_y):
        print("[regions] the two bands are not co-registered -- rebuild them with "
              "the same --resolution and world", file=sys.stderr)
        return 2

    region_map = decompose_regions(
        flight.occupied_mask, walls.occupied_mask, flight.resolution,
        flight.origin_x, flight.origin_y,
        min_room_m2=args.min_room_m2, min_portal_m=args.min_portal_m,
        corridor_seed=tuple(args.corridor_seed) if args.corridor_seed else None)

    if args.corridor_band:
        corridors = region_map.corridors()
        if len(corridors) != 1:
            print("[regions] expected one circulation region to split, found %d"
                  % len(corridors), file=sys.stderr)
            return 2
        region_map = split_region_by_bands(region_map, corridors[0].id,
                                           args.corridor_band)

    names = spatial_names(region_map, "room")
    names.update({r.id: r.name for r in region_map.corridors()})
    region_map = _renamed(region_map, names)

    yaml_path, npz_path = region_map.save(args.output)
    rooms = region_map.rooms()
    print("wrote:        %s" % yaml_path)
    print("wrote:        %s" % npz_path)
    print("rooms:        %d, %.0f m2" % (len(rooms), sum(r.area_m2 for r in rooms)))
    print("corridors:    %d, %.0f m2"
          % (len(region_map.corridors()),
             sum(r.area_m2 for r in region_map.corridors())))
    print("portals:      %d  (%d wider than 0.90 m)"
          % (len(region_map.portals),
             sum(1 for p in region_map.portals.values() if p.width_m > 0.90)))
    unreachable = [r.name for r in rooms if not r.portals]
    if unreachable:
        print("SEALED (no opening at all): %s" % ", ".join(unreachable))
    if args.preview:
        _preview(region_map, args.preview)
        print("preview:      %s" % args.preview)
    return 0


def _renamed(region_map, names):
    # type: (RegionMap, Dict[int, str]) -> RegionMap
    """The same map with new names, everything else untouched."""
    from dataclasses import replace
    regions = [replace(r, name=names.get(r.id, r.name))
               for r in region_map.regions.values()]
    return RegionMap(region_map.labels, region_map.resolution, region_map.origin_x,
                     region_map.origin_y, regions, list(region_map.portals.values()))


if __name__ == "__main__":
    sys.exit(main())
