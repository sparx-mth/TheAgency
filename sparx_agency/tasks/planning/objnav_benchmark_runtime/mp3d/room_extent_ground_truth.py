"""The room extents against Matterport's room annotations: a 2-D simulation of the benchmark camera on the true floor plans.

Every MP3D scene ships a ``.house`` file with one floor polygon per room
(region) and the positions of the panoramas captured in it. Nothing else in
the benchmarks annotates rooms, so this is where the mechanism's sizes can
be checked against true ones: the polygons are rasterised onto the method's
0.1 m grid as free floor walled by their outlines (a doorway opened on every
wall two rooms share), the benchmark's depth camera -- 79 degrees wide,
0.5-5 m, 0.88 m high, level -- is cast over that plan from the panorama
positions as the mapping would see it (floor beyond the blind radius known
free, the first wall known occupied, everything else unknown), the region
of each known free cell stands in for the partition, and
:func:`~room_extent.room_extents` reads the result.

Two readings per room: ONE frame from the panorama nearest its boundary,
looking in (the room as first seen), and a full rotation at every panorama
inside it (the room as scanned). The report says, for each, how often the
size was called FINAL, how far the final sizes are from the true area and
the true dimensions, and whether a size called a LOWER BOUND ever exceeded
the true one. No simulator, no GPU; a scene takes seconds.

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.room_extent_ground_truth [--scenes ...]

Python 3.8 syntax; numpy, scipy, skimage.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import label as connected_components
from skimage.draw import line as draw_line, polygon as draw_polygon

from sparx_agency.core.mapping.topology.room_extent import ExtentParams, room_extents
from sparx_agency.core.planning.environment import OccupancyValues
from sparx_agency.core.planning.object_search.world_model.sightlines import floor_band

RESOLUTION_M = 0.1
VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
#: The MP3D region category codes, for the report.
REGION_LABELS = {"a": "bathroom", "b": "bedroom", "c": "closet", "d": "dining room", "e": "entryway", "f": "family room",
                 "g": "garage", "h": "hallway", "i": "library", "j": "laundry", "k": "kitchen", "l": "living room",
                 "m": "meeting room", "n": "lounge", "o": "office", "p": "porch", "r": "rec room", "s": "stairs",
                 "t": "toilet", "u": "utility", "v": "tv room", "w": "workout", "x": "outdoor", "y": "balcony",
                 "z": "other", "B": "bar", "C": "classroom", "D": "dining booth", "S": "spa", "Z": "junk", "-": "none"}
#: Width of the doorway opened on a shared wall, and the shortest shared wall that gets one.
DOOR_WIDTH_M = 0.9
DOOR_MIN_WALL_M = 1.4


def scenes_dir() -> Path:
    return Path(os.environ.get("MP3D_SCENES_DIR", os.path.expanduser("~/datasets/scene_datasets/mp3d")))


def house_path(scene: str) -> Path:
    return scenes_dir() / scene / ("%s.house" % scene)


def parse_house(path):
    """``(regions, panoramas)`` of a ``.house`` file: ``{index: {level, label, polygon}}`` and ``[(region, x, y)]``.

    Matterport's house frame is z-up; the floor polygon of each region is
    its one ``S`` surface's ``V`` vertices, in order.
    """
    regions = {}  # type: Dict[int, Dict[str, object]]
    surface_region = {}  # type: Dict[int, int]
    vertices = {}  # type: Dict[int, List[Tuple[float, float]]]
    panoramas = []  # type: List[Tuple[int, float, float]]
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            if kind == "R":
                regions[int(parts[1])] = {"level": int(parts[2]), "label": parts[5], "polygon": []}
            elif kind == "S":
                surface_region[int(parts[1])] = int(parts[2])
            elif kind == "V":
                vertices.setdefault(int(parts[2]), []).append((float(parts[4]), float(parts[5])))
            elif kind == "P" and len(parts) >= 8 and not parts[1].isdigit():
                # A panorama line: name, panorama index, region index, 0, px py pz.
                try:
                    panoramas.append((int(parts[3]), float(parts[5]), float(parts[6])))
                except ValueError:
                    continue
    for surface, region in surface_region.items():
        if region in regions and surface in vertices and len(vertices[surface]) >= 3:
            regions[region]["polygon"] = vertices[surface]
    return regions, panoramas


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def min_area_rectangle(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    """``(length, width, angle)`` of the smallest rectangle around ``points``, by 1-degree sweep."""
    pts = np.asarray(points, dtype=float)
    best = None
    for deg in range(0, 90):
        th = math.radians(deg)
        u = pts @ np.array([math.cos(th), math.sin(th)])
        v = pts @ np.array([-math.sin(th), math.cos(th)])
        lu, lv = float(u.max() - u.min()), float(v.max() - v.min())
        if best is None or lu * lv < best[0]:
            best = (lu * lv, max(lu, lv), min(lu, lv), th)
    return best[1], best[2], best[3]


class FloorPlan:
    """A level of a house on the method's grid: true occupancy, true region per cell, and the panoramas in it."""

    def __init__(self, regions, panoramas, level, resolution=RESOLUTION_M, margin_m=1.0):
        self.resolution = float(resolution)
        self.regions = {k: r for k, r in regions.items() if r["level"] == level and r["polygon"]}
        xs = [p[0] for r in self.regions.values() for p in r["polygon"]]
        ys = [p[1] for r in self.regions.values() for p in r["polygon"]]
        if not xs:
            raise ValueError("level %d has no region polygons" % level)
        self.origin = (min(xs) - margin_m, min(ys) - margin_m)
        w = int(math.ceil((max(xs) + margin_m - self.origin[0]) / self.resolution)) + 1
        h = int(math.ceil((max(ys) + margin_m - self.origin[1]) / self.resolution)) + 1
        self.shape = (h, w)
        self.truth = np.full(self.shape, VALUES.unknown, dtype=np.int8)    # the true plan: free or occupied
        self.region = np.zeros(self.shape, dtype=np.int32)                 # region index + 1 on true floor
        outline = np.zeros(self.shape, dtype=bool)
        for index, room in self.regions.items():
            rr, cc = self._cells(room["polygon"])
            inside = np.zeros(self.shape, dtype=bool)
            inside[rr, cc] = True
            self.region[inside & (self.region == 0)] = index + 1
            self.truth[inside] = VALUES.free
            poly = room["polygon"]
            for a, b in zip(poly, poly[1:] + poly[:1]):
                ra, ca = self.cell(*a)
                rb, cb = self.cell(*b)
                lr, lc = draw_line(ra, ca, rb, cb)
                keep = (lr >= 0) & (lr < h) & (lc >= 0) & (lc < w)
                outline[lr[keep], lc[keep]] = True
        self.truth[outline] = VALUES.occupied
        self.region[outline] = 0
        self.truth[(self.truth == VALUES.free) & (self.region == 0)] = VALUES.unknown
        self._open_doorways(outline)
        self.panoramas = [(r, x, y) for r, x, y in panoramas if r in self.regions and self.truth_at(x, y) == VALUES.free]

    def _cells(self, polygon):
        rr = [(p[1] - self.origin[1]) / self.resolution for p in polygon]
        cc = [(p[0] - self.origin[0]) / self.resolution for p in polygon]
        return draw_polygon(np.array(rr), np.array(cc), shape=self.shape)

    def cell(self, x, y):
        return (int(math.floor((y - self.origin[1]) / self.resolution)), int(math.floor((x - self.origin[0]) / self.resolution)))

    def truth_at(self, x, y):
        r, c = self.cell(x, y)
        if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
            return int(self.truth[r, c])
        return VALUES.unknown

    def _open_doorways(self, outline):
        """A ``DOOR_WIDTH_M`` opening in the middle of every wall two rooms share that is at least ``DOOR_MIN_WALL_M`` long."""
        h, w = self.shape
        pairs = {}  # type: Dict[Tuple[int, int], np.ndarray]
        ys, xs = np.nonzero(outline)
        for y, x in zip(ys, xs):
            y0, y1, x0, x1 = max(0, y - 2), min(h, y + 3), max(0, x - 2), min(w, x + 3)
            around = set(np.unique(self.region[y0:y1, x0:x1]).tolist()) - {0}
            if len(around) == 2:
                key = tuple(sorted(around))
                if key not in pairs:
                    pairs[key] = np.zeros(self.shape, dtype=bool)
                pairs[key][y, x] = True
        door_cells = max(1, int(round(DOOR_WIDTH_M / self.resolution)))
        min_cells = int(round(DOOR_MIN_WALL_M / self.resolution))
        for (a, b), shared in pairs.items():
            comp, n = connected_components(shared, structure=np.ones((3, 3), dtype=bool))
            for k in range(1, n + 1):
                cy, cx = np.nonzero(comp == k)
                if len(cy) < min_cells:
                    continue
                order = np.argsort(cx if cx.ptp() >= cy.ptp() else cy)
                middle = order[len(order) // 2 - door_cells // 2: len(order) // 2 - door_cells // 2 + door_cells]
                for y, x in zip(cy[middle], cx[middle]):
                    self.truth[y, x] = VALUES.free
                    self.region[y, x] = a                      # the door's cells go to one of the two rooms

    def region_area_m2(self, index):
        return polygon_area(self.regions[index]["polygon"])

    def truth_bounds_known(self, known, index, share=0.95):
        """Whether at least ``share`` of the wall cells around the region's true floor are known occupied in ``known``: the room really is all seen."""
        floor = self.region == index + 1
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(floor, structure=np.ones((3, 3), dtype=bool)) & ~floor & (self.truth == VALUES.occupied)
        if not ring.any():
            return False
        return bool((known[ring] == VALUES.occupied).mean() >= share)


class SimulatedMapping:
    """The method's map of a :class:`FloorPlan` from a sequence of camera poses: known free, known occupied, unknown; and the observed cells."""

    def __init__(self, plan, camera, half_fov_deg=39.5, detect_range_m=4.0):
        self.plan = plan
        self.camera = camera
        self.known = np.full(plan.shape, VALUES.unknown, dtype=np.int8)
        self.observed = np.zeros(plan.shape, dtype=bool)
        self.half_fov = math.radians(half_fov_deg)
        self.detect_range_m = float(detect_range_m)
        k = camera.intrinsics
        half_vfov = math.atan(0.5 * float(k.height) / float(k.fy))
        band = floor_band(float(camera.height_m), 0.0, half_vfov, float(camera.min_depth_m), float(camera.max_depth_m))
        self.blind_m, self.far_m = band[0], band[1]

    def frame(self, x, y, yaw):
        """One level frame: rays through the TRUE plan; floor between the blind radius and the depth range becomes known free, the wall hit known occupied, and the free cells within detection range observed."""
        plan, res = self.plan, self.plan.resolution
        h, w = plan.shape
        rays = int(math.ceil(2.0 * self.half_fov * (self.far_m / res) / 0.5)) + 1
        bearings = np.linspace(yaw - self.half_fov, yaw + self.half_fov, max(3, rays))
        step = 0.5 * res
        ranges = np.arange(step, self.far_m + step, step)
        px = x + np.outer(np.cos(bearings), ranges)
        py = y + np.outer(np.sin(bearings), ranges)
        gc = np.floor((px - plan.origin[0]) / res).astype(np.int64)
        gr = np.floor((py - plan.origin[1]) / res).astype(np.int64)
        inside = (gc >= 0) & (gc < w) & (gr >= 0) & (gr < h)
        gc_c, gr_c = np.clip(gc, 0, w - 1), np.clip(gr, 0, h - 1)
        truth = plan.truth[gr_c, gc_c]
        wall = (truth == VALUES.occupied) | ~inside
        any_wall = wall.any(axis=1)
        first = np.where(any_wall, wall.argmax(axis=1), wall.shape[1])
        idx = np.arange(wall.shape[1])[None, :]
        before = idx < first[:, None]
        # As the integrator maps it: a ray that hits a wall within the depth range clears every cell on its way
        # there (the hit is at wall height, not on the floor); a ray that hits nothing within range clears the
        # floor it sampled, which starts beyond the camera's blind radius.
        hits_within = any_wall & (ranges[np.minimum(first, len(ranges) - 1)] <= float(self.camera.max_depth_m))
        floor = before & (truth == VALUES.free) & ((ranges[None, :] >= self.blind_m) | hits_within[:, None])
        self.known[gr_c[floor], gc_c[floor]] = VALUES.free
        hit = any_wall & inside[np.arange(len(first)), np.minimum(first, wall.shape[1] - 1)]
        rows = np.arange(len(first))[hit]
        self.known[gr_c[rows, first[hit]], gc_c[rows, first[hit]]] = VALUES.occupied
        seen = before & (truth == VALUES.free) & (ranges[None, :] <= self.detect_range_m)
        self.observed[gr_c[seen], gc_c[seen]] = True
        # The agent's own footprint is known free, as the mapping writes it.
        r0, c0 = plan.cell(x, y)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if 0 <= r0 + dr < h and 0 <= c0 + dc < w and plan.truth[r0 + dr, c0 + dc] == VALUES.free:
                    self.known[r0 + dr, c0 + dc] = VALUES.free
                    self.observed[r0 + dr, c0 + dc] = True

    def rotation(self, x, y, turns=12):
        """A full look-around at ``(x, y)``, standing on floor the agent walked onto (the blind disc under it known)."""
        for k in range(turns):
            self.frame(x, y, k * 2.0 * math.pi / turns)
        self.clear_around(x, y)

    def clear_around(self, x, y):
        """The floor within the blind radius, with a clear line to the pose, is known free: the agent walked onto it."""
        plan, res = self.plan, self.plan.resolution
        h, w = plan.shape
        bearings = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        ranges = np.arange(0.5 * res, self.blind_m + res, 0.5 * res)
        px = x + np.outer(np.cos(bearings), ranges)
        py = y + np.outer(np.sin(bearings), ranges)
        gc = np.clip(np.floor((px - plan.origin[0]) / res).astype(np.int64), 0, w - 1)
        gr = np.clip(np.floor((py - plan.origin[1]) / res).astype(np.int64), 0, h - 1)
        truth = plan.truth[gr, gc]
        wall = truth != VALUES.free
        first = np.where(wall.any(axis=1), wall.argmax(axis=1), wall.shape[1])
        before = np.arange(wall.shape[1])[None, :] < first[:, None]
        self.known[gr[before], gc[before]] = VALUES.free
        self.observed[gr[before], gc[before]] = True

    def labels(self):
        """The partition: each known free cell labelled with its TRUE region (what a perfect watershed would give)."""
        out = np.where(self.known == VALUES.free, self.plan.region, 0).astype(np.int32)
        return out

    def extents(self, here_xy=None, params=None):
        return room_extents(self.known, self.labels(), self.plan.resolution, self.plan.origin, observed=self.observed,
                            here_xy=here_xy, params=params or ExtentParams())


def nearest_boundary_panorama(plan, index, panoramas):
    """The panorama of the room nearest its outline (the room as seen from its door), and the heading toward its centroid."""
    poly = plan.regions[index]["polygon"]
    cx, cy = float(np.mean([p[0] for p in poly])), float(np.mean([p[1] for p in poly]))
    best = None
    for _, x, y in panoramas:
        edge = min(_segment_distance((x, y), a, b) for a, b in zip(poly, poly[1:] + poly[:1]))
        if best is None or edge < best[0]:
            best = (edge, x, y)
    if best is None:
        return None
    _, x, y = best
    return x, y, math.atan2(cy - y, cx - x)


def _segment_distance(p, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def evaluate_scene(scene, camera, level=None, min_area_m2=3.0, params=None, verbose=False):
    """Per-room records for ``scene``: the truth, the first-frame reading and the scanned reading."""
    regions, panoramas = parse_house(house_path(scene))
    levels = sorted({r["level"] for r in regions.values() if r["polygon"]})
    if level is None:
        level = max(levels, key=lambda lv: sum(1 for r in regions.values() if r["level"] == lv and r["polygon"]))
    plan = FloorPlan(regions, panoramas, level)
    records = []
    for index, room in sorted(plan.regions.items()):
        truth_area = plan.region_area_m2(index)
        inside = [(r, x, y) for r, x, y in plan.panoramas if r == index]
        if truth_area < min_area_m2 or not inside:
            continue
        length, width, _ = min_area_rectangle(room["polygon"])
        record = {"scene": scene, "level": level, "region": index, "label": REGION_LABELS.get(room["label"], room["label"]),
                  "truth_area_m2": round(truth_area, 2), "truth_length_m": round(length, 2), "truth_width_m": round(width, 2),
                  "panoramas": len(inside)}
        # The room as first seen: one frame from the panorama nearest its outline, looking in.
        first = SimulatedMapping(plan, camera)
        door = nearest_boundary_panorama(plan, index, inside)
        first.frame(door[0], door[1], door[2])
        record["first"] = _summary(first.extents(here_xy=(door[0], door[1]), params=params).get(index), first, plan, index)
        # The room as scanned: a full rotation at every panorama inside it.
        scanned = SimulatedMapping(plan, camera)
        for _, x, y in inside:
            scanned.rotation(x, y)
        record["scanned"] = _summary(scanned.extents(here_xy=(inside[0][1], inside[0][2]), params=params).get(index),
                                     scanned, plan, index)
        records.append(record)
        if verbose:
            print(format_record(record))
    return records


def _summary(extent, mapping, plan, index):
    if extent is None:
        return None
    return {"area_m2": round(extent.area_m2, 2), "length_m": round(extent.length_m, 2), "width_m": round(extent.width_m, 2),
            "bounded": extent.bounded, "walls_seen": extent.walls_seen, "observed_fraction": round(extent.observed_fraction, 3),
            "truly_all_seen": plan.truth_bounds_known(mapping.known, index)}


def format_record(record):
    def part(name, s):
        if s is None:
            return "%s: not mapped" % name
        return "%s: %s%.1fm2 %.1fx%.1fm walls %d/4 obs %.0f%%%s" % (
            name, "" if s["bounded"] else ">=", s["area_m2"], s["length_m"], s["width_m"], s["walls_seen"],
            100 * s["observed_fraction"], "" if s["truly_all_seen"] == s["bounded"] else " [verdict differs from truth]")
    return "%s L%d R%-2d %-12s truth %5.1fm2 %4.1fx%4.1fm | %s | %s" % (
        record["scene"], record["level"], record["region"], record["label"][:12], record["truth_area_m2"],
        record["truth_length_m"], record["truth_width_m"], part("first", record["first"]), part("scanned", record["scanned"]))


def accuracy(records):
    """The headline numbers over ``records``."""
    out = {"rooms": len(records)}
    for stage in ("first", "scanned"):
        rows = [r for r in records if r[stage] is not None]
        final = [r for r in rows if r[stage]["bounded"]]
        lower = [r for r in rows if not r[stage]["bounded"]]
        area_err = [abs(r[stage]["area_m2"] - r["truth_area_m2"]) / r["truth_area_m2"] for r in final]
        dims_err = [max(abs(r[stage]["length_m"] - r["truth_length_m"]), abs(r[stage]["width_m"] - r["truth_width_m"])) for r in final]
        overstated = [r for r in lower if r[stage]["area_m2"] > 1.05 * r["truth_area_m2"]]
        agree = [r for r in rows if r[stage]["bounded"] == r[stage]["truly_all_seen"]]
        out[stage] = {"mapped": len(rows), "final": len(final), "lower_bound": len(lower),
                      "final_area_error_median": round(float(np.median(area_err)), 3) if area_err else None,
                      "final_area_error_p90": round(float(np.percentile(area_err, 90)), 3) if area_err else None,
                      "final_dims_error_median_m": round(float(np.median(dims_err)), 2) if dims_err else None,
                      "final_dims_error_p90_m": round(float(np.percentile(dims_err, 90)), 2) if dims_err else None,
                      "lower_bounds_overstated": len(overstated),
                      "verdict_agrees_with_truth": round(len(agree) / float(len(rows)), 3) if rows else None,
                      "lower_bound_area_share_median": round(float(np.median([r[stage]["area_m2"] / r["truth_area_m2"] for r in lower])), 3) if lower else None}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="*", default=None, help="scene ids (default: every scene with a .house file)")
    parser.add_argument("--limit", type=int, default=12, help="at most this many scenes")
    parser.add_argument("--min-area", type=float, default=3.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PROTOCOLS
    camera = next(iter(PROTOCOLS.values())).camera()
    scenes = args.scenes or sorted(p.name for p in scenes_dir().iterdir() if (p / ("%s.house" % p.name)).is_file())
    records = []
    for scene in scenes[:args.limit]:
        try:
            records += evaluate_scene(scene, camera, min_area_m2=args.min_area, verbose=args.verbose)
        except (ValueError, OSError) as exc:
            print("%s: skipped (%s)" % (scene, exc))
    import json
    print(json.dumps(accuracy(records), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
