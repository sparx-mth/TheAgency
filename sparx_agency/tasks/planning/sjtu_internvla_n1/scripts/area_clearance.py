#!/usr/bin/env python3
"""Measure -- or propose -- start areas against the hospital occupancy map.

Two jobs, one distance transform:

* ``--check`` re-measures the clearance of every area in
  ``config/hospital_areas.yaml``, so the numbers recorded there stay true after
  the map is rebuilt at a different altitude band or resolution.
* ``--propose`` finds the roomiest reachable spot in each latitude band of the
  building, which is how the shipped areas were chosen in the first place. A
  start pose is only useful if the aircraft can sit in it without a wall in
  the first frame, and "roomiest" is a defensible way to pick one.

ROS-free: it reads the map off disk. Run it in the plain ``.venv``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))
_AREAS = os.path.join(_HERE, os.pardir, "config", "hospital_areas.yaml")
_MAP = os.path.join(_REPO_ROOT, "sparx_agency", "robots", "SJTU", "maps", "hospital.yaml")

# The building split into bands, north to south, with the name each one's best
# spot is published under. Bands rather than a global maximum because a global
# maximum picks the atrium five times.
_BANDS = (
    (14.0, 22.0, "north_wing", 90.0),
    (4.0, 14.0, "atrium", -90.0),
    (-6.0, 4.0, "reception", 90.0),
    (-24.0, -6.0, "east_wards", 180.0),
    (-36.0, -24.0, "south_hall", 90.0),
)


class ClearanceMap(object):
    """A map plus the distance from every cell to the nearest occupied one."""

    def __init__(self, yaml_path):
        # type: (str) -> None
        with open(yaml_path, "r") as handle:
            meta = yaml.safe_load(handle)
        image = meta["image"]
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)
        grid = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if grid is None:
            raise FileNotFoundError("could not read map image %s" % image)
        self.resolution = float(meta["resolution"])
        self.origin_x, self.origin_y = float(meta["origin"][0]), float(meta["origin"][1])
        self.height, self.width = grid.shape[:2]
        occupied = (grid <= 60).astype(np.uint8)
        self.distance = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, 5) * self.resolution

    def _cell(self, x, y):
        col = int(round((x - self.origin_x) / self.resolution))
        row = self.height - 1 - int(round((y - self.origin_y) / self.resolution))
        return row, col

    def clearance(self, x, y):
        # type: (float, float) -> float
        """Metres from ``(x, y)`` to the nearest occupied cell; -1 outside the map."""
        row, col = self._cell(x, y)
        if not (0 <= row < self.height and 0 <= col < self.width):
            return -1.0
        return float(self.distance[row, col])

    def roomiest_in_band(self, y_lo, y_hi):
        # type: (float, float) -> tuple
        """``(x, y, clearance)`` of the most open cell between two latitudes."""
        row_hi = self.height - 1 - int(round((y_lo - self.origin_y) / self.resolution))
        row_lo = self.height - 1 - int(round((y_hi - self.origin_y) / self.resolution))
        row_lo, row_hi = max(0, row_lo), min(self.height, row_hi)
        if row_lo >= row_hi:
            raise ValueError("band %.1f..%.1f is outside the map" % (y_lo, y_hi))
        band = self.distance[row_lo:row_hi]
        local = np.unravel_index(int(np.argmax(band)), band.shape)
        row, col = row_lo + int(local[0]), int(local[1])
        return (self.origin_x + col * self.resolution,
                self.origin_y + (self.height - 1 - row) * self.resolution,
                float(self.distance[row, col]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default=_MAP)
    ap.add_argument("--areas", default=_AREAS)
    ap.add_argument("--propose", action="store_true",
                    help="print the roomiest spot per band, as YAML")
    args = ap.parse_args(argv)

    cmap = ClearanceMap(args.map)
    if args.propose:
        print("areas:")
        for y_lo, y_hi, name, yaw in _BANDS:
            x, y, clear = cmap.roomiest_in_band(y_lo, y_hi)
            print("  %s:\n    x: %.2f\n    y: %.2f\n    yaw_deg: %.1f\n    clearance_m: %.2f"
                  % (name, x, y, yaw, clear))
        return 0

    with open(args.areas, "r") as handle:
        areas = (yaml.safe_load(handle) or {}).get("areas") or {}
    worst = 1e9
    for name, spec in sorted(areas.items()):
        measured = cmap.clearance(spec["x"], spec["y"])
        recorded = spec.get("clearance_m")
        flag = ""
        if recorded is not None and abs(measured - float(recorded)) > 0.15:
            flag = "  <-- recorded %.2f, STALE" % float(recorded)
        print("%-14s (%7.2f, %7.2f)  clearance %.2f m%s" % (name, spec["x"], spec["y"],
                                                            measured, flag))
        worst = min(worst, measured)
    print("worst clearance: %.2f m" % worst)
    return 0 if worst >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
