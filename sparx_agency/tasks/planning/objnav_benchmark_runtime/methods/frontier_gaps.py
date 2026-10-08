"""Gaps behind furniture: the shallow unknown a frontier opens onto is not a place to go and look.

A frontier is free floor beside unknown, and the frontier logic treats every
one as an exit: a doorway to a room not seen yet, or the slit between the
back of a sofa and the wall. The slit has a long frontier (the sofa's whole
length), so a cell count does not tell it from a doorway; the sightline
ledger's pocket rule needs the unknown behind it to be *enclosed*, and a
strip along a wall leaks through one-cell gaps in the observed wall into the
unknown outside the house, so it never is. Wiconisco/000000 (2026-10-08):
opening O6, a 28-cell "gap" valued at 0.4 by the oracle and peeked for 18
actions, opened onto 0.18 m2 of unknown 0.4 m wide.

What tells them apart is **depth and thickness**: how far the unknown runs
behind the frontier before known cells end it, and how wide the band is.
:func:`probe_gap` marches rays from the frontier cell into the unknown along
the frontier's outward heading and a cone around it
(:attr:`GapSettings.half_cone_deg`), each until the first known cell -- an
obstacle (the wall behind the sofa) or free floor (the camera's own blind
strip in front of floor it has seen) -- or until :attr:`GapSettings.probe_m`,
and reads the unknown's half-width (its distance transform) at the cells the
rays cross. If every ray ends at known cells, and the band is either
shallower or thinner than :attr:`GapSettings.depth_m`, the frontier is a
**gap**: the band along a wall with the frontier down its length is
shallow; the slit behind the sofa entered from its end is thin. A real
opening has a ray that runs deep, or unknown wide enough to be a room.

**Target scale** (:data:`TARGET_FOOTPRINT_M`, :attr:`GapSettings.target_scale`):
a bed is at least 0.9 m across, a sofa 0.8 m; a bounded band shallower than
the target's smallest footprint dimension cannot hold the target, however
reachable it is, so for such a target the depth threshold is that dimension.
A potted plant or a television fits anywhere and gets the plain rule.

A gap is **demoted** this action (``split_exits`` lists it with its reason,
after every exit and every other demoted frontier) and its unknown cells
are **settled** in the sightline ledger (:meth:`SightLedger.settle_gaps`),
so from the next action it is no frontier at all: not an opening node, not
"frontier left" in a room's facts, not an exit for the fallback. Settled
cells are masked to what is still unknown, so a gap that later proves to be
floor seen from another side is known from then on, and the mark is void.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_priors import _ALIASES

#: Target class -> ``(smallest footprint dimension, footprint area)`` in metres, for the macro-target rule.
#: Gibson/HM3D categories in their canonical names (``room_priors._ALIASES``).
TARGET_FOOTPRINT_M = {
    "bed": (0.90, 1.70),
    "sofa": (0.80, 1.40),
    "toilet": (0.38, 0.25),
    "chair": (0.40, 0.18),
    "potted plant": (0.25, 0.06),
    "television": (0.10, 0.05),
}


@dataclass(frozen=True)
class GapSettings:
    """What counts as a gap. Distances in metres.

    Attributes:
        enabled: Off, no frontier is a gap: the former behaviour.
        depth_m: Unknown that ends at known cells within this depth along
            every probe ray is a gap.
        probe_m: How far a ray looks before the unknown counts as deep.
        half_cone_deg: Rays are cast within this many degrees either side
            of the frontier's outward heading.
        rays: How many rays, spread over the cone (odd keeps the centre).
        target_scale: Raise ``depth_m`` to the target's smallest footprint
            dimension when that is larger: a band a bed cannot stand in is
            a gap for a bed search.
        settle: Write a gap's shallow unknown into the sightline ledger so
            it is no frontier from the next action on.
    """

    enabled: bool = True
    depth_m: float = 0.7
    probe_m: float = 2.5
    half_cone_deg: float = 45.0
    rays: int = 5
    target_scale: bool = True
    settle: bool = True

    def __post_init__(self):
        for name in ("enabled", "target_scale", "settle"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("frontier_gaps.%s must be a bool" % name)
        for name in ("depth_m", "probe_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("frontier_gaps.%s must be positive and finite" % name)
        if self.depth_m >= self.probe_m:
            raise ValueError("frontier_gaps.depth_m must lie inside probe_m")
        if isinstance(self.half_cone_deg, bool) or not math.isfinite(self.half_cone_deg) or not 0 <= self.half_cone_deg <= 90:
            raise ValueError("frontier_gaps.half_cone_deg must lie in [0, 90]")
        if type(self.rays) is not int or self.rays < 1:
            raise ValueError("frontier_gaps.rays must be a positive int")


def target_min_dim_m(target) -> Optional[float]:
    """The smallest footprint dimension of the target's class, or None for an unknown class."""
    names: List[str] = []
    if isinstance(target, str):
        names.append(target)
    elif target is not None:
        for attribute in ("query", "category"):
            value = getattr(target, attribute, None)
            if value:
                names.append(str(value))
        names.extend(str(v) for v in getattr(target, "accept", ()) or ())
    dims = [TARGET_FOOTPRINT_M[_ALIASES.get(n.strip().lower(), n.strip().lower())][0]
            for n in names if _ALIASES.get(n.strip().lower(), n.strip().lower()) in TARGET_FOOTPRINT_M]
    return min(dims) if dims else None


def unknown_heading(world, cell: Tuple[int, int], radius_cells: int = 6) -> Optional[float]:
    """Which way the unknown lies from a frontier cell: from the known floor around it toward the unknown.

    The direction from the centroid of the known FREE cells in the window to
    the centroid of its UNKNOWN cells. The mean direction to the unknown
    alone is not it: a frontier cell has unknown on several sides -- the
    strip behind a bed beside it, the gap it stands in -- and Ranchester
    attempt 8 read the stair passage's heading as pointing back into the
    hallway, three times in sixty actions. The known floor is the one side
    the agent came from, so away from it is through the opening. None when
    the window holds no unknown or no free cell.
    """
    gx, gy = int(cell[0]), int(cell[1])
    h, w = world.grid.shape
    x0, x1 = max(0, gx - radius_cells), min(w, gx + radius_cells + 1)
    y0, y1 = max(0, gy - radius_cells), min(h, gy + radius_cells + 1)
    window = world.grid[y0:y1, x0:x1]
    unknown_ys, unknown_xs = np.nonzero(window == world.values.unknown)
    free_ys, free_xs = np.nonzero(window == world.values.free)
    if not len(unknown_xs) or not len(free_xs):
        return None
    dx = float(np.mean(unknown_xs) - np.mean(free_xs))
    dy = float(np.mean(unknown_ys) - np.mean(free_ys))
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


@dataclass(frozen=True)
class GapProbe:
    """What the rays found behind one frontier cell.

    Attributes:
        depth_m: The deepest unknown any ray ran through before a known cell (``probe_m`` when one ran out).
        min_depth_m: The shallowest.
        width_m: The unknown band's greatest thickness at the cells the rays crossed (twice its half-width).
        deep_rays: Rays that reached ``probe_m`` through unknown: the unknown is open that way.
        cells: The unknown cells the rays crossed, ``(gx, gy)``, for settling.
    """

    depth_m: float
    min_depth_m: float
    width_m: float
    deep_rays: int
    cells: Tuple[Tuple[int, int], ...]

    def is_gap(self, scale_m: float) -> bool:
        """Bounded on every ray, and shallower or thinner than ``scale_m``."""
        return self.deep_rays == 0 and (self.depth_m < scale_m or self.width_m < scale_m)

    @property
    def extent_m(self) -> float:
        return min(self.depth_m, self.width_m)


def probe_gap(world, cell: Tuple[int, int], heading: float, settings: GapSettings) -> Optional[GapProbe]:
    """March rays from ``cell`` into the unknown around ``heading``; None when no ray enters unknown at all."""
    res = world.resolution
    grid, unknown = world.grid, world.values.unknown
    h, w = grid.shape
    n = max(1, int(settings.rays))
    offsets = [0.0] if n == 1 else [math.radians(-settings.half_cone_deg + 2 * settings.half_cone_deg * i / (n - 1))
                                     for i in range(n)]
    step = 0.5                                                   # cells
    limit = int(math.ceil(settings.probe_m / res / step))
    depths, deep, crossed, entered = [], 0, set(), False
    for offset in offsets:
        angle = heading + offset
        dx, dy = math.cos(angle) * step, math.sin(angle) * step
        x, y = cell[0] + 0.5, cell[1] + 0.5
        run, ran_out, seen_unknown = 0.0, True, False
        for _ in range(limit):
            x += dx
            y += dy
            gx, gy = int(math.floor(x)), int(math.floor(y))
            if not (0 <= gx < w and 0 <= gy < h):
                ran_out = False                                  # off the map: counts as ended (the map's edge is known)
                break
            if grid[gy, gx] == unknown:
                seen_unknown = True
                crossed.add((gx, gy))
                run += step * res
                continue
            if seen_unknown or run > 0:
                ran_out = False
                break
            # still inside the known floor at the frontier: keep going until the unknown starts
        if not seen_unknown:
            continue
        entered = True
        if ran_out:
            deep += 1
            depths.append(settings.probe_m)
        else:
            depths.append(run)
    if not entered:
        return None
    return GapProbe(depth_m=max(depths), min_depth_m=min(depths), width_m=_band_width_m(world, crossed, settings),
                    deep_rays=deep, cells=tuple(sorted(crossed)))


def _band_width_m(world, cells, settings: GapSettings) -> float:
    """Twice the unknown's greatest half-width at ``cells``: the thickness of the band the rays ran through."""
    if not cells:
        return 0.0
    from scipy import ndimage
    res = world.resolution
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    pad = int(math.ceil(settings.probe_m / res))
    h, w = world.grid.shape
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad + 1)
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad + 1)
    unknown = world.grid[y0:y1, x0:x1] == world.values.unknown
    distance = ndimage.distance_transform_edt(unknown)              # to the nearest known cell, in cells
    half = max(float(distance[gy - y0, gx - x0]) for gx, gy in cells)
    return 2.0 * half * res


class FrontierGaps:
    """The per-action gap verdicts over the frontier inventory, and what they settled.

    Attributes:
        reasons: ``{frontier cell: reason}`` for the gaps found this action.
        stats: Counters for the record.
    """

    def __init__(self, policy, settings: Optional[GapSettings] = None):
        self.policy = policy
        self.settings = settings or GapSettings()
        self.reasons: Dict[Tuple[int, int], str] = {}
        self.step = -1
        self.stats = {"probed": 0, "gaps": 0, "macro_gaps": 0, "cells_settled": 0, "open": 0, "no_heading": 0}
        self.last: List[Dict] = []

    def required_depth_m(self, target) -> float:
        depth = float(self.settings.depth_m)
        if self.settings.target_scale:
            scale = target_min_dim_m(target)
            if scale is not None:
                depth = max(depth, scale)
        return depth

    def probe(self, obs, world, inventory, target) -> Dict[Tuple[int, int], str]:
        """Judge every goal of ``inventory`` this action; settle the gaps in the sightline ledger."""
        self.reasons, self.last = {}, []
        self.step = int(obs.step)
        goals = list(getattr(inventory, "goals", None) or ()) if inventory is not None else []
        if not self.settings.enabled or not goals:
            return self.reasons
        depth_needed = self.required_depth_m(target)
        macro = depth_needed > self.settings.depth_m + 1e-9
        settle = np.zeros(world.grid.shape, dtype=bool) if self.settings.settle else None
        for goal in goals:
            cell = tuple(int(v) for v in goal.cell)
            heading = unknown_heading(world, cell)
            if heading is None:
                self.stats["no_heading"] += 1
                continue
            probe = probe_gap(world, cell, heading, self.settings)
            if probe is None:
                self.stats["no_heading"] += 1
                continue
            self.stats["probed"] += 1
            if not probe.is_gap(depth_needed):
                self.stats["open"] += 1
                continue
            plain = probe.is_gap(self.settings.depth_m)
            shape = "%.1f m deep, %.1f m wide" % (probe.depth_m, probe.width_m)
            reason = ("gap: unknown %s behind it" % shape if plain else
                      "gap for the target: unknown %s, %s needs %.2f m" % (shape, _target_name(target), depth_needed))
            self.reasons[cell] = reason
            self.stats["gaps"] += 1
            self.stats["macro_gaps"] += int(not plain)
            self.last.append({"xy": [round(float(v), 2) for v in goal.xy], "size_cells": int(goal.size_cells),
                              "depth_m": round(probe.depth_m, 2), "width_m": round(probe.width_m, 2), "why": reason})
            if settle is not None:
                for gx, gy in probe.cells:
                    settle[gy, gx] = True
        if settle is not None and settle.any():
            sight = getattr(self.policy, "sight", None)
            if sight is not None:
                self.stats["cells_settled"] += sight.settle_gaps(world, settle, self.policy.mapping.floor_id)
        return self.reasons

    def diagnostics(self) -> Dict:
        from dataclasses import asdict
        return {"settings": asdict(self.settings), "stats": dict(self.stats), "last_step": self.step,
                "last": list(self.last)}


def _target_name(target) -> str:
    for attribute in ("query", "category"):
        value = getattr(target, attribute, None)
        if value:
            return str(value)
    return str(target)
