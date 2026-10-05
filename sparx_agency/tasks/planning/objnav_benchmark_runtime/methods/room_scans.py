"""The ledger of completed look-arounds, and what each one saw.

A room is finished when the agent has stood inside it and turned a full
circle: the camera's ray ends at the nearest object, so a second viewpoint
or a walk to the far frontier shows little a 360-degree scan from the open
floor did not. The ledger remembers WHERE every scan was completed, not
which room id it was credited to -- the watershed renumbers, splits and
merges rooms on every update, and a visit remembered by id is forgotten the
moment the id changes (the Ranchester recordings re-entered the same room
under a new number three times in a row).

Four tests say whether a room is finished, in order:

* a completed scan point lies inside its mask (``scan_point_inside``);
* a completed scan point SAW it: at least :attr:`seen_fraction` of its cells
  lie within the camera's depth range of the point with a clear line of
  sight through observed free space (``seen_from_scan``) -- the small room
  fully visible from the corridor the agent scanned, or the half of a split
  room the scan stood in. Unknown cells block sight, so a region the camera
  never resolved is never credited;
* it has **no live frontier** -- no accessible unexplored boundary left:
  every edge of it is a wall, an object, a pocket, or unknown looked
  through without a return (:class:`~sightlines.SightLedger`) -- and the
  camera has looked into it or walked through it: one recorded pose had
  :attr:`seen_fraction` of its cells inside the camera's cone with a clear
  line of sight (the balcony seen whole from its threshold, the closet
  from its door), or a pose lies inside it and it is **narrow** -- its
  widest point under :attr:`walkthrough_clearance_m` of clearance, a
  corridor or a balcony, which the camera's cone spans as the agent walks
  it (``seen_through``, since 2026-10-05). A wide room the agent merely
  stepped into is not finished by that: the walls beside its door are
  behind the camera, and the scan is what looks at them. The Hanson
  recording put the balcony back in the order at 0.25, "never entered",
  120 actions after the agent had stood at its far end;
* it is a **fragment**: no live frontier, under :attr:`fragment_max_m2`,
  and no confirmed door on it -- the strip behind a bed, the nook beside
  a wardrobe, that the watershed carved into a room of its own
  (``fragment``, since 2026-10-05). A bathroom is small too, but a
  bathroom has a door or an unseen boundary; a fragment has neither.

A verdict is sticky: a room once finished stays finished for the episode
(registry ids never recycle), whatever the watershed does to its edges
afterwards. The warm-up's own rotation is a scan too: the spawn room is
finished before the first room is ever chosen.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_vantage import vantage_point
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.sightlines import cone_seen

#: How a room was finished.
SCAN_POINT_INSIDE = "scan_point_inside"
SEEN_FROM_SCAN = "seen_from_scan"
SEEN_THROUGH = "seen_through"
FRAGMENT = "fragment"


def visible_fraction(world, xy, mask, max_range_m, samples=120):
    """The share of ``mask`` a camera at ``xy`` could see: in range, clear line of sight.

    Sight runs through observed FREE cells only -- unknown and occupied
    cells both end the ray, as they end the camera's. Up to ``samples``
    cells of the mask within range are tested (evenly thinned); cells out
    of range count as unseen.
    """
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    gx0, gy0 = world.world_to_grid(*xy)
    if not world.in_bounds(gx0, gy0):
        return 0.0
    limit = max_range_m / world.resolution
    within = ((xs - gx0) ** 2 + (ys - gy0) ** 2) <= limit * limit
    if not within.any():
        return 0.0
    index = np.nonzero(within)[0]
    if len(index) > samples:
        index = index[np.linspace(0, len(index) - 1, samples).astype(int)]
    blocked = world.grid != world.values.free
    blocked[gy0, gx0] = False                     # the agent's own cell never blocks its view
    hits = sum(1 for i in index if line_of_sight_clear(blocked, int(gx0), int(gy0), int(xs[i]), int(ys[i])))
    return (hits / len(index)) * (int(within.sum()) / len(xs))


class RoomScanLedger:
    """Completed look-arounds on every floor, and whether a room is finished by them.

    Attributes:
        records: One dict per completed scan -- ``floor``, ``xy``, ``step``,
            ``room`` (the pid credited at the time, or None for the warm-up),
            ``cells`` (that room's size then, for the recording), ``source``.
        seen_fraction: The share of a room a scan must have had in view to
            finish it without standing in it.
        fragment_max_m2: Largest room with no live frontier and no door that
            is a fragment rather than a room; 0 disables the test.
        walkthrough_clearance_m: A room whose widest point has no more
            clearance than this is narrow enough to be seen by walking
            through it (a corridor, a balcony); 0 disables that half of
            the ``seen_through`` test.
        pose_batch: Most recorded poses the ``seen_through`` test evaluates
            per room per call; the rest wait for the next call, so one
            action never pays for the whole history.
    """

    def __init__(self, policy, seen_fraction=0.5, fragment_max_m2=3.0, walkthrough_clearance_m=0.9, pose_batch=8):
        if not 0.0 < float(seen_fraction) <= 1.0:
            raise ValueError("seen_fraction must lie in (0, 1], got %r" % (seen_fraction,))
        for name, value in (("fragment_max_m2", fragment_max_m2), ("walkthrough_clearance_m", walkthrough_clearance_m)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("%s must be finite and non-negative, got %r" % (name, value))
        if type(pose_batch) is not int or pose_batch < 1:
            raise ValueError("pose_batch must be a positive integer, got %r" % (pose_batch,))
        self.policy = policy
        self.seen_fraction = float(seen_fraction)
        self.fragment_max_m2 = float(fragment_max_m2)
        self.walkthrough_clearance_m = float(walkthrough_clearance_m)
        self.pose_batch = int(pose_batch)
        self.records = []
        self._finished = {}       # (floor, pid) -> reason, sticky for the episode
        self._cache = {}          # (floor, pid, n_cells, len(records)) -> reason or None
        self._pose_progress = {}  # (floor, pid) -> (n_cells, poses evaluated) for the cone test

    # -- recording ----------------------------------------------------------
    def record(self, obs, room=None, source="room_scan"):
        """A full rotation was completed at the agent's position; remember where."""
        self.records.append(dict(floor=self.policy.mapping.floor_id, xy=(float(obs.pose.x), float(obs.pose.y)),
                                 step=int(obs.step), room=None if room is None else int(room.id),
                                 cells=None if room is None else int(room.n_cells), source=source))
        self._cache.clear()

    def on_floor(self, floor_id=None):
        floor = self.policy.mapping.floor_id if floor_id is None else floor_id
        return [r for r in self.records if r["floor"] == floor]

    # -- the verdict --------------------------------------------------------
    def status(self, world, pid, room, frontier=None, doored=None):
        """None while the room is unfinished, else the test that finished it (sticky).

        Args:
            frontier: How many accessible, unresolved frontier clusters the
                room still has, when known; None skips the ``seen_through``
                and ``fragment`` tests, which need it to be zero.
            doored: Whether a confirmed door stands on the room, when known;
                None is read as "no door known".
        """
        floor = self.policy.mapping.floor_id
        sticky = self._finished.get((floor, int(pid)))
        if sticky is not None:
            return sticky
        key = (floor, int(pid), int(room.n_cells), len(self.records))
        if key in self._cache:
            result = self._cache[key]
        else:
            result = self._status(world, room)
            self._cache[key] = result
        if result is None and frontier is not None and int(frontier) == 0:
            result = self._status_without_frontier(world, pid, room, bool(doored))
        if result is not None:
            self._finished[(floor, int(pid))] = result
        return result

    def _status(self, world, room):
        records = self.on_floor()
        if not records:
            return None
        for record in records:
            gx, gy = world.world_to_grid(*record["xy"])
            if world.in_bounds(gx, gy) and room.mask[gy, gx]:
                return SCAN_POINT_INSIDE
        max_range = float(self.policy.episode.camera.max_depth_m)
        for record in records:
            if visible_fraction(world, record["xy"], room.mask, max_range) >= self.seen_fraction:
                return SEEN_FROM_SCAN
        return None

    def _status_without_frontier(self, world, pid, room, doored):
        """The tests for a room with no live frontier: looked into or walked through (``seen_through``), or a fragment."""
        sight = getattr(self.policy, "sight", None)
        if sight is None:
            return None
        if self.walkthrough_clearance_m > 0.0 and sight.stood_in(world, room.mask):
            found = vantage_point(world, room.mask)
            if found is not None and found[1] <= self.walkthrough_clearance_m:
                return SEEN_THROUGH
        if self._looked_into(world, pid, room, sight):
            return SEEN_THROUGH
        area = float(room.n_cells) * world.resolution ** 2
        if self.fragment_max_m2 > 0.0 and area < self.fragment_max_m2 and not doored:
            return FRAGMENT
        return None

    def _looked_into(self, world, pid, room, sight):
        """Whether one recorded pose had ``seen_fraction`` of the room in its cone with clear sight.

        Poses are tried nearest-bearing first among those within depth
        range of the room, at most ``pose_batch`` new ones per call; the
        poses already tried against a mask of about this size are not
        tried again (a mask grown by a tenth starts over).
        """
        floor = self.policy.mapping.floor_id
        poses = sight.poses(floor)
        if not poses:
            return False
        camera = self.policy.episode.camera
        k = camera.intrinsics
        half_fov = math.atan(0.5 * float(k.width) / float(k.fx))
        max_range = float(camera.max_depth_m)
        mask = np.asarray(room.mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return False
        cx, cy = world.grid_to_world(int(round(xs.mean())), int(round(ys.mean())))
        radius = math.sqrt(len(xs) * world.resolution ** 2 / math.pi)
        key = (floor, int(pid))
        n_cells, done = self._pose_progress.get(key, (0, 0))
        if n_cells and abs(room.n_cells - n_cells) > 0.1 * n_cells:
            done = 0
        total = float(mask.sum())
        tried = 0
        index = done
        while index < len(poses) and tried < self.pose_batch:
            _, x, y, yaw, _ = poses[index]
            index += 1
            distance = math.dist((x, y), (cx, cy))
            if distance > max_range + radius:
                continue
            bearing = math.atan2(cy - y, cx - x) - yaw
            bearing = abs(math.atan2(math.sin(bearing), math.cos(bearing)))
            allowance = half_fov + (math.atan2(radius, distance) if distance > 1e-6 else math.pi)
            if bearing > allowance:
                continue
            tried += 1
            seen = cone_seen(world, (x, y), yaw, half_fov, max_range)
            if float((seen & mask).sum()) / total >= self.seen_fraction:
                self._pose_progress[key] = (int(room.n_cells), len(poses))
                return True
        self._pose_progress[key] = (int(room.n_cells), index)
        return False

    def mark(self, pid, reason, floor_id=None):
        """Finish a room by a verdict reached elsewhere (sticky, like the ledger's own)."""
        floor = self.policy.mapping.floor_id if floor_id is None else floor_id
        self._finished[(floor, int(pid))] = str(reason)

    def finished(self, world, graph):
        """``{pid: reason}`` for every finished room of ``graph`` on this floor."""
        out = {}
        for pid, room in graph.registry.rooms.items():
            reason = self.status(world, pid, room)
            if reason is not None:
                out[pid] = reason
        return out

    def is_finished(self, world, pid, room):
        return self.status(world, pid, room) is not None

    def diagnostics(self):
        floor = getattr(getattr(self.policy, "mapping", None), "floor_id", None)
        return {"seen_fraction": self.seen_fraction, "fragment_max_m2": self.fragment_max_m2,
                "walkthrough_clearance_m": self.walkthrough_clearance_m,
                "records": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in self.records],
                "finished": sorted("f%d/r%d:%s" % (f, pid, reason) for (f, pid), reason in self._finished.items()),
                "finished_here": sorted(pid for (f, pid) in self._finished if f == floor)}
