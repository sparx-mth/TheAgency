"""The ledger of completed look-arounds, and what each one saw.

A room is finished when the agent has stood inside it and turned a full
circle: the camera's ray ends at the nearest object, so a second viewpoint
or a walk to the far frontier shows little a 360-degree scan from the open
floor did not. The ledger remembers WHERE every scan was completed, not
which room id it was credited to -- the watershed renumbers, splits and
merges rooms on every update, and a visit remembered by id is forgotten the
moment the id changes (the Ranchester recordings re-entered the same room
under a new number three times in a row).

Two tests say whether a room is finished, in order:

* a completed scan point lies inside its mask (``scan_point_inside``);
* a completed scan point SAW it: at least :attr:`seen_fraction` of its cells
  lie within the camera's depth range of the point with a clear line of
  sight through observed free space (``seen_from_scan``) -- the small room
  fully visible from the corridor the agent scanned, or the half of a split
  room the scan stood in. Unknown cells block sight, so a region the camera
  never resolved is never credited.

A verdict is sticky: a room once finished stays finished for the episode
(registry ids never recycle), whatever the watershed does to its edges
afterwards. The warm-up's own rotation is a scan too: the spawn room is
finished before the first room is ever chosen.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear

#: How a room was finished.
SCAN_POINT_INSIDE = "scan_point_inside"
SEEN_FROM_SCAN = "seen_from_scan"


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
    """

    def __init__(self, policy, seen_fraction=0.5):
        if not 0.0 < float(seen_fraction) <= 1.0:
            raise ValueError("seen_fraction must lie in (0, 1], got %r" % (seen_fraction,))
        self.policy = policy
        self.seen_fraction = float(seen_fraction)
        self.records = []
        self._finished = {}       # (floor, pid) -> reason, sticky for the episode
        self._cache = {}          # (floor, pid, n_cells, len(records)) -> reason or None

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
    def status(self, world, pid, room):
        """None while the room is unfinished, else the test that finished it (sticky)."""
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
        return {"seen_fraction": self.seen_fraction,
                "records": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in self.records],
                "finished": sorted("f%d/r%d:%s" % (f, pid, reason) for (f, pid), reason in self._finished.items()),
                "finished_here": sorted(pid for (f, pid) in self._finished if f == floor)}
