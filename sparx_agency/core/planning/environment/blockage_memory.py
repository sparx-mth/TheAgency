"""Remembering obstacles the sensors cannot see.

The map is built from depth, so it can only contain what the camera can observe.
Some of the things that stop a drone indoors are not in that set: glass, a thin
pole, a chair leg under the height band, a wall the drone is already touching.
The planner will happily route straight back through such a spot, because as far
as the map is concerned nothing is there — and it will keep doing so, forever,
because flying at it again produces no new map evidence either.

The controller is the only part of the system that can detect these: it notices
that a command is not reaching the world. This module is where that knowledge is
kept, and it is deliberately on the **planning** side of the split — the
controller runs the reflex, the planner owns the memory and the reroute.

A remembered blockage is stamped into every freshly decoded map, because each new
map overwrites the last and would otherwise show the spot as free again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BlockageMemoryParams:
    """Tuning for :class:`BlockageMemory`.

    Attributes:
        radius_m: Radius of the disc marked occupied around a reported point (m).
            Should be at least the drone's own half-width: the report says "the
            drone could not get through here", which is a statement about a
            drone-sized volume, not about a point.
        ttl_s: How long a blockage is remembered (s). 0 means forever, which is
            the right default for a static indoor structure — the wall will still
            be there in five minutes. Set a finite value only where obstacles
            genuinely move.
        max_entries: Cap on remembered blockages. Oldest are dropped first, so a
            pathological loop cannot grow the memory without bound.
    """

    radius_m: float = 0.35
    ttl_s: float = 0.0
    max_entries: int = 32

    def __post_init__(self):
        # type: () -> None
        """Validate the invariants the memory relies on."""
        if self.radius_m <= 0.0:
            raise ValueError("BlockageMemoryParams.radius_m must be > 0")
        if self.ttl_s < 0.0:
            raise ValueError("BlockageMemoryParams.ttl_s must be >= 0 "
                             "(0 means never forget)")
        if self.max_entries < 1:
            raise ValueError("BlockageMemoryParams.max_entries must be >= 1")


class BlockageMemory:
    """World-frame points the drone has proved it cannot get through."""

    def __init__(self, params=None):
        # type: (Optional[BlockageMemoryParams]) -> None
        self.params = params or BlockageMemoryParams()
        self._points = []      # type: List[Tuple[float, float, float]]

    def __len__(self):
        # type: () -> int
        return len(self._points)

    @property
    def points(self):
        # type: () -> List[Tuple[float, float]]
        """The remembered blockage centres, oldest first."""
        return [(x, y) for x, y, _ in self._points]

    def clear(self):
        # type: () -> None
        """Forget everything (call on a new mission, not on a new map)."""
        self._points = []

    def add(self, x, y, t):
        # type: (float, float, float) -> bool
        """Remember a blockage at ``(x, y)``, reported at time ``t``.

        Reports that land inside an existing blockage's radius refresh it rather
        than stacking a second entry — a drone that fails twice against the same
        wall has learned one fact, not two.

        Returns:
            True if this was a new blockage, False if it refreshed a known one.
        """
        r2 = self.params.radius_m ** 2
        for i, (px, py, _) in enumerate(self._points):
            if (px - x) ** 2 + (py - y) ** 2 <= r2:
                self._points[i] = (px, py, float(t))
                return False
        self._points.append((float(x), float(y), float(t)))
        while len(self._points) > self.params.max_entries:
            self._points.pop(0)
        return True

    def prune(self, now):
        # type: (float) -> int
        """Drop blockages older than ``ttl_s``. Returns how many were dropped."""
        if self.params.ttl_s <= 0.0:
            return 0
        before = len(self._points)
        cutoff = now - self.params.ttl_s
        self._points = [p for p in self._points if p[2] >= cutoff]
        return before - len(self._points)

    def _disc_cells(self, grid):
        # type: (object) -> List[Tuple[int, int]]
        """Grid cells covered by every remembered blockage's disc, as (gy, gx)."""
        radius = max(1, int(round(self.params.radius_m / grid.resolution)))
        cells = []
        for x, y, _ in self._points:
            cx, cy = grid.world_to_grid(x, y)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    gx, gy = cx + dx, cy + dy
                    if grid.in_bounds(gx, gy):
                        cells.append((gy, gx))
        return cells

    def stamp(self, grid, now=None):
        # type: (object, Optional[float]) -> int
        """Mark every remembered blockage as OCCUPIED in ``grid``, in place.

        Call this on each freshly decoded map, before planning: a new map
        overwrites the previous one and would otherwise show the spot as free.

        Args:
            grid: An ``OccupancyGrid2D``. Mutated in place.
            now: Current time, used to prune expired entries first. None skips
                pruning.

        Returns:
            The number of cells newly marked.
        """
        if now is not None:
            self.prune(now)
        if not self._points:
            return 0
        occupied = grid.values.occupied
        marked = 0
        for gy, gx in self._disc_cells(grid):
            if int(grid.grid[gy, gx]) != occupied:
                marked += 1
            grid.grid[gy, gx] = occupied
        return marked

    def stamp_confidence(self, confidence, grid, value=1.0):
        # type: (object, object, float) -> int
        """Force full OCCUPIED-confidence at every remembered blockage.

        **This is not optional when a confidence grid is in use.** The planner
        treats an OCCUPIED cell *below* ``lethal_confidence`` as merely expensive
        rather than blocking, so that single-frame depth speckle cannot wall the
        drone in. A remembered blockage is the opposite of speckle — it is the one
        obstacle the drone has physically proved exists — but it carries no map
        confidence at all, because the sensors never saw it. Without this the
        stamped cells stay cheap and A* routes straight back through them, and the
        whole memory silently does nothing.

        Args:
            confidence: ``(H, W)`` array co-registered with ``grid``. Mutated in
                place.
            grid: The ``OccupancyGrid2D`` the confidence belongs to, used for its
                geometry.
            value: Confidence to write. 1.0 (certain) by default.

        Returns:
            The number of cells written.
        """
        if not self._points:
            return 0
        written = 0
        for gy, gx in self._disc_cells(grid):
            confidence[gy, gx] = value
            written += 1
        return written
