"""Replay a recording's maps and poses through the room extents, offline, with no simulator.

What a recording holds is enough to read every room's walls, size status and
coverage at every action after the fact: ``map_journal.npz`` has the
occupancy grid, the settled unknown and the room partition of every action
as deltas, and ``steps.jsonl`` has the pose -- position, heading, camera
pitch -- of every action. This module rebuilds the per-cell observation
record from those poses with the benchmark's camera
(:func:`~sightlines.cone_observed`, the same cast the policy makes online),
reads :func:`~room_extent.room_extents` on the map of every action, and
reports what the mechanism would have told the oracle -- and whether it
kept its word: a size declared FINAL should not grow afterwards, a LOWER
BOUND may; a room renumbered, merged or split by the watershed should keep
the coverage of its floor.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.room_extent_replay <recording dir> [...]
        [--camera hm3d|mp3d] [--every N] [--json out.json]

Reads only; writes nothing into the recording. Python 3.8 syntax.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from sparx_agency.core.mapping.topology.room_extent import ExtentParams, room_extents
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.object_search.world_model.sightlines import (
    SightSettings, cone_observed, floor_band, object_band)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording import MapAtStep, load_json

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def benchmark_camera(name):
    """The RGB-D camera of a benchmark, from its protocol (HM3D and MP3D share one: 640x480, 79 degrees, 0.88 m, 0.5-5 m)."""
    if name == "hm3d":
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V1
        return HM3D_V1.camera()
    if name == "mp3d":
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PROTOCOLS
        return next(iter(PROTOCOLS.values())).camera()
    raise ValueError("unknown camera %r: one of hm3d, mp3d" % (name,))


def camera_for_scene(scene):
    """``mp3d`` for an 11-character Matterport id, else ``hm3d`` (the two protocols' cameras are one and the same)."""
    text = str(scene or "")
    if len(text) == 11 and text.isalnum():
        return "mp3d"
    return "hm3d"


class ObservedReplay:
    """The per-storey observation record rebuilt from recorded poses, as :class:`~sightlines.SightLedger` keeps it online."""

    def __init__(self, camera, settings=None):
        self.camera = camera
        self.settings = settings or SightSettings()
        k = camera.intrinsics
        self.half_vfov = math.atan(0.5 * float(k.height) / float(k.fy))
        self.half_fov = math.atan(0.5 * float(k.width) / float(k.fx))
        self.floors = {}

    def observe(self, world, floor_id, x, y, yaw, pitch):
        s = self.settings
        observed = self.floors.get(int(floor_id))
        if observed is None or observed.shape != world.grid.shape:
            observed = np.zeros(world.grid.shape, dtype=bool)
            self.floors[int(floor_id)] = observed
        band = object_band(float(self.camera.height_m), float(pitch), self.half_vfov, 0.0,
                           float(self.camera.max_depth_m), float(s.object_band_m[0]), float(s.object_band_m[1]))
        if band is None:
            return 0
        near, far = band[0], min(band[1], float(s.detect_range_m), float(self.camera.max_depth_m))
        if far <= near:
            return 0
        floor = floor_band(float(self.camera.height_m), float(pitch), self.half_vfov, float(self.camera.min_depth_m),
                           float(self.camera.max_depth_m))
        blind = far if floor is None else floor[0] + s.near_margin_m
        seen = cone_observed(world, (x, y), float(yaw), self.half_fov, near, far, blind_m=blind)
        fresh = int((seen & ~observed).sum())
        observed |= seen
        return fresh

    def mask(self, world, floor_id):
        observed = self.floors.get(int(floor_id))
        if observed is None or observed.shape != world.grid.shape:
            return np.zeros(world.grid.shape, dtype=bool)
        return observed


def load_rows(directory):
    rows = []
    with (Path(directory) / "steps.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def replay(directory, camera=None, every=1, params=None, verbose=False):
    """Read the room extents at every ``every``-th recorded action of ``directory``.

    Returns:
        A dict with ``scene``, ``episode``, ``steps`` (one record per action read: step, floor, the extents by
        pid, the recorded scan verdicts) and ``checks`` (:func:`checks`).
    """
    directory = Path(directory)
    episode = load_json(directory, "episode.json") or {}
    maps = MapAtStep(directory)
    if not maps.exact:
        raise SystemExit("%s has no map_journal.npz: the map of each action cannot be replayed" % directory)
    camera_name = camera or camera_for_scene(episode.get("scene"))
    spec = benchmark_camera(camera_name)
    ledger = ObservedReplay(spec)
    rows = load_rows(directory)
    params = params or ExtentParams()
    out_steps = []
    previous = {}
    for index, row in enumerate(rows):
        step = int(row["step"])
        pose = row["pose"]
        floor_id = int((row.get("method") or {}).get("floor_id", 0) or 0)
        grid, settled, labels = maps.at(step, floor_id)
        world = OccupancyGrid2D(grid, OccupancyGrid2DParams(maps.resolution, maps.origin[0], maps.origin[1], "world"),
                                values=VALUES)
        ledger.observe(world, floor_id, pose["x"], pose["y"], pose["yaw"], pose.get("camera_pitch", 0.0) or 0.0)
        if index % max(1, int(every)) and index != len(rows) - 1:
            continue
        extents = room_extents(grid, labels, maps.resolution, maps.origin, observed=ledger.mask(world, floor_id),
                               resolved=settled, here_xy=(pose["x"], pose["y"]), params=params, previous=previous)
        previous = extents
        search = (row.get("method") or {}).get("search") or {}
        excluded = {int(k): v for k, v in (search.get("excluded") or {}).items()}
        rooms = {}
        for pid, extent in extents.items():
            rooms[pid] = {"area_m2": round(extent.area_m2, 2), "free_m2": round(extent.free_m2, 2),
                          "length_m": round(extent.length_m, 2), "width_m": round(extent.width_m, 2),
                          "bounded": extent.bounded, "walls_seen": extent.walls_seen,
                          "open_sides": list(extent.open_sides), "observed_fraction": round(extent.observed_fraction, 3),
                          "observed_m2": round(extent.observed_m2, 2), "open_perimeter_m": round(extent.open_perimeter_m, 2),
                          "closed_fraction": round(extent.closed_fraction, 3), "here": extent.here,
                          "scanned": excluded.get(pid) if str(excluded.get(pid, "")).startswith("scanned") else None}
        out_steps.append({"step": step, "floor": floor_id, "rooms": rooms, "pids": sorted(rooms)})
        if verbose:
            print("step %3d  rooms %s" % (step, " ".join(
                "R%d[%s%.0fm2 %d/4 obs%.0f%%]" % (pid, "" if r["bounded"] else ">=", r["area_m2"], r["walls_seen"],
                                                 100 * r["observed_fraction"]) for pid, r in sorted(rooms.items()))))
    result = {"scene": episode.get("scene"), "episode": episode.get("episode_id"), "target": episode.get("target"),
              "camera": camera_name, "steps": out_steps}
    result["checks"] = checks(out_steps)
    return result


def checks(steps):
    """What the replay says about the mechanism's promises.

    * ``partition_changes``: actions on which the set of room ids changed, and for each the conservation of
      observed floor and area across it (a merge's survivor holds the parts' coverage; a split's halves hold the
      whole's): ``observed_before_m2`` against ``observed_after_m2`` over the rooms involved.
    * ``final_growth``: for every room id, the area when it was FIRST declared final against its area at the last
      action it existed -- a final size should not grow (ratio about 1); ``lower_bound_growth`` is the same for
      rooms while they read as lower bounds, which may grow.
    * ``final_breaches``: every action on which a room that read FINAL on the previous action grew by more than a
      quarter (and 0.3 m2), classified: ``merge`` when a room vanished on that action (the registry keeps the
      survivor's number, the partition changed, the reading was right for the room it had), ``cut_moved`` when
      the other rooms of the storey lost at least half of what this one gained (the watershed redrew a door cut
      or a basin boundary between neighbours: a partition change too), ``fragment`` when the room was under a
      square metre (a sliver the watershed hands back and forth), ``breach`` otherwise -- the room grew into space
      no other room held, through a side the reading had called located. Only the last is the mechanism's error.
    * ``bounded_share``: the share of (room, action) readings that were final.
    """
    out = {"partition_changes": [], "final_growth": [], "lower_bound_growth": [], "readings": 0, "bounded_readings": 0,
           "final_breaches": []}
    previous = None
    first_final = {}
    first_seen = {}
    last = {}
    for record in steps:
        rooms = record["rooms"]
        pids = set(rooms)
        for pid, r in rooms.items():
            out["readings"] += 1
            out["bounded_readings"] += int(bool(r["bounded"]))
            first_seen.setdefault(pid, (record["step"], r))
            if r["bounded"] and pid not in first_final:
                first_final[pid] = (record["step"], r)
            last[pid] = (record["step"], r)
        if previous is not None and previous["floor"] == record["floor"]:
            vanished = sorted(set(previous["rooms"]) - pids)
            for pid, r in rooms.items():
                p = previous["rooms"].get(pid)
                if p is None or not p["bounded"] or r["area_m2"] <= 1.25 * p["area_m2"] + 0.3:
                    continue
                gain = r["area_m2"] - p["area_m2"]
                lost = sum(max(0.0, previous["rooms"][q]["area_m2"] - rooms[q]["area_m2"])
                           for q in rooms if q != pid and q in previous["rooms"])
                if vanished:
                    kind = "merge"
                elif lost >= 0.5 * gain:
                    kind = "cut_moved"
                elif p["area_m2"] < 1.0:
                    kind = "fragment"
                else:
                    kind = "breach"
                out["final_breaches"].append({"step": record["step"], "pid": pid, "kind": kind,
                                              "area_before_m2": p["area_m2"], "area_after_m2": r["area_m2"],
                                              "vanished": vanished, "neighbours_lost_m2": round(lost, 2)})
        if previous is not None and set(previous["rooms"]) != pids and previous["floor"] == record["floor"]:
            gone = sorted(set(previous["rooms"]) - pids)
            new = sorted(pids - set(previous["rooms"]))
            # The rooms involved: the ones that vanished or appeared, plus any whose area moved by more than a
            # metre -- the survivor of a merge or the remainder of a split.
            involved_before = {pid for pid in previous["rooms"] if pid in gone
                               or (pid in rooms and abs(rooms[pid]["area_m2"] - previous["rooms"][pid]["area_m2"]) > 1.0)}
            involved_after = {pid for pid in rooms if pid in new or pid in involved_before}
            before_obs = sum(previous["rooms"][pid]["observed_m2"] for pid in involved_before)
            after_obs = sum(rooms[pid]["observed_m2"] for pid in involved_after)
            before_area = sum(previous["rooms"][pid]["area_m2"] for pid in involved_before)
            after_area = sum(rooms[pid]["area_m2"] for pid in involved_after)
            out["partition_changes"].append({
                "step": record["step"], "gone": gone, "new": new,
                "involved_before": sorted(involved_before), "involved_after": sorted(involved_after),
                "observed_before_m2": round(before_obs, 2), "observed_after_m2": round(after_obs, 2),
                "area_before_m2": round(before_area, 2), "area_after_m2": round(after_area, 2)})
        previous = record
    for pid, (step, r) in first_final.items():
        end_step, end = last[pid]
        if end_step > step:
            out["final_growth"].append({"pid": pid, "declared_at": step, "area_then_m2": r["area_m2"],
                                        "area_end_m2": end["area_m2"], "end_step": end_step,
                                        "ratio": round(end["area_m2"] / max(r["area_m2"], 1e-6), 3),
                                        "still_bounded": bool(end["bounded"])})
    for pid, (step, r) in first_seen.items():
        if r["bounded"]:
            continue
        end_step, end = last[pid]
        bound_end = first_final.get(pid, (end_step, end))
        if bound_end[0] > step:
            out["lower_bound_growth"].append({"pid": pid, "from": step, "to": bound_end[0], "area_then_m2": r["area_m2"],
                                              "area_later_m2": bound_end[1]["area_m2"],
                                              "ratio": round(bound_end[1]["area_m2"] / max(r["area_m2"], 1e-6), 3),
                                              "became_final": pid in first_final})
    out["bounded_share"] = round(out["bounded_readings"] / max(1, out["readings"]), 3)
    return out


def summarise(result):
    """A few lines on one replay's checks."""
    c = result["checks"]
    lines = ["%s %s target=%s camera=%s: %d actions read, %d (room, action) readings, %.0f%% final" % (
        result["scene"], result["episode"], result["target"], result["camera"], len(result["steps"]), c["readings"],
        100 * c["bounded_share"])]
    changes = c["partition_changes"]
    if changes:
        kept = [abs(ch["observed_after_m2"] - ch["observed_before_m2"]) for ch in changes]
        lines.append("  partition changed on %d actions; observed floor across them: median |delta| %.2f m2, max %.2f m2"
                     % (len(changes), float(np.median(kept)), float(np.max(kept))))
    growth = [g["ratio"] for g in c["final_growth"]]
    if growth:
        lines.append("  %d rooms declared FINAL then kept: area at the end / area when declared: median %.2f, max %.2f; "
                     "%d of them grew over 25%%" % (len(growth), float(np.median(growth)), float(np.max(growth)),
                                                     sum(1 for g in growth if g > 1.25)))
    lower = [g["ratio"] for g in c["lower_bound_growth"]]
    if lower:
        lines.append("  %d rooms first read as LOWER BOUND: later area / first area: median %.2f, max %.2f; %d became final"
                     % (len(lower), float(np.median(lower)), float(np.max(lower)),
                        sum(1 for g in c["lower_bound_growth"] if g["became_final"])))
    kinds = {}
    for b in c["final_breaches"]:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    lines.append("  growth of over a quarter after a FINAL reading: %d merges, %d cuts moved, %d fragments under 1 m2, "
                 "%d BREACHES" % (kinds.get("merge", 0), kinds.get("cut_moved", 0), kinds.get("fragment", 0),
                                  kinds.get("breach", 0)))
    for b in c["final_breaches"]:
        if b["kind"] == "breach":
            lines.append("    breach: action %d room %d %.1f -> %.1f m2" % (b["step"], b["pid"], b["area_before_m2"], b["area_after_m2"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recordings", nargs="+", help="recording directories (with map_journal.npz and steps.jsonl)")
    parser.add_argument("--camera", choices=("hm3d", "mp3d"), default=None,
                        help="the benchmark camera (default: from the scene id)")
    parser.add_argument("--every", type=int, default=1, help="read the extents every N actions (the poses are always replayed)")
    parser.add_argument("--json", default=None, help="write every replay's records to this file")
    parser.add_argument("--verbose", action="store_true", help="print each action's rooms")
    args = parser.parse_args(argv)
    results = []
    for directory in args.recordings:
        result = replay(directory, camera=args.camera, every=args.every, verbose=args.verbose)
        results.append(result)
        print(summarise(result))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
