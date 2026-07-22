"""Tests for the relaxable standoff ladder and the phantom probe.

The scenario throughout is the real one: a wide corridor with a narrow THROAT in
the middle. The throat is what a stably mis-detected depth voxel produces -- a
pinch that leaves no lane wide enough for the preferred standoff, even though the
corridor either side is open.

Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/python -m pytest \
        sparx_agency/core/planning/planners/astar/tests/test_standoff_ladder.py
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.common.types import Pose2D, PlanStatus
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D,
    OccupancyGrid2DParams,
    OccupancyValues,
)
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import (
    WeightedAStarPlanner2D,
)

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
RES = 0.1
H, W = 21, 60
THROAT = slice(28, 32)


def _grid(arr: np.ndarray) -> OccupancyGrid2D:
    return OccupancyGrid2D(
        arr.astype(np.int16),
        OccupancyGrid2DParams(resolution=RES, origin_x=0.0, origin_y=0.0,
                              frame_id="world"),
        values=VALUES,
    )


def _corridor(free_rows: slice = None) -> np.ndarray:
    """Wide corridor (walls at rows 0/20); ``free_rows`` survives in the throat.

    ``None`` leaves the throat as wide as the corridor (no pinch at all).
    """
    arr = np.zeros((H, W), dtype=np.int16)
    arr[0, :] = 100
    arr[H - 1, :] = 100
    if free_rows is not None:
        arr[1:H - 1, THROAT] = 100          # wall off the throat ...
        arr[free_rows, THROAT] = 0          # ... then reopen a lane through it
    return arr


def _params(**kw) -> WeightedAStarParams:
    base = dict(inflate_radius_m=0.40, inflate_floor_m=0.25, relax_step_m=0.05,
                probe_radius_m=0.05, connectivity=8, search_margin_m=6.0,
                start_skip_m=0.0, goal_snap_radius_m=0.0, corner_round=False,
                waypoint_spacing_m=0.5)
    base.update(kw)
    return WeightedAStarParams(**base)


def _request():
    """Across the corridor, both endpoints in the wide (unpinched) part."""
    return PlanRequest(start=Pose2D(0.5, 1.0, 0.0), goal=Pose2D(5.5, 1.0, 0.0),
                       frame_id="world")


# ── the ladder itself ────────────────────────────────────────────────────────
def test_ladder_descends_by_step_and_ends_exactly_on_the_floor():
    ladder = WeightedAStarPlanner2D(_params()).standoff_ladder()
    assert ladder == [0.40, 0.35, 0.30, 0.25]


def test_ladder_is_a_single_rung_when_relaxation_is_disabled():
    # floor == preferred is the default: never relax.
    p = _params(inflate_floor_m=0.40)
    assert WeightedAStarPlanner2D(p).standoff_ladder() == [0.40]


def test_ladder_respects_a_coarser_step():
    p = _params(inflate_floor_m=0.20, relax_step_m=0.10)
    assert WeightedAStarPlanner2D(p).standoff_ladder() == [0.40, 0.30, 0.20]


# ── relaxing only as far as necessary ────────────────────────────────────────
def test_open_corridor_uses_the_preferred_standoff():
    planner = WeightedAStarPlanner2D(_params())
    res = planner.plan(_request(), _grid(_corridor()))
    assert res.ok
    assert res.artifacts["inflate_used_m"] == 0.40
    assert planner.last_inflate_m == 0.40


def test_pinched_corridor_relaxes_to_the_first_rung_that_fits():
    # Throat lane rows 8..12 -> centre row 10 sits 3 cells (0.3 m) off the pinch,
    # so 0.40/0.35/0.30 are all blocked and 0.25 is the first rung that fits.
    planner = WeightedAStarPlanner2D(_params())
    res = planner.plan(_request(), _grid(_corridor(slice(8, 13))))
    assert res.ok, res.message
    assert res.artifacts["inflate_used_m"] == 0.25
    assert "RELAXED" in res.message
    assert res.path.metadata["inflate_used_m"] == 0.25


def test_the_same_pinch_is_no_path_when_relaxation_is_disabled():
    p = _params(inflate_floor_m=0.40, probe_radius_m=0.0)
    res = WeightedAStarPlanner2D(p).plan(_request(), _grid(_corridor(slice(8, 13))))
    assert res.status == PlanStatus.NO_PATH


def test_relaxation_is_not_sticky_across_plans():
    """A squeeze must not lower the standoff of the NEXT plan (replan resets)."""
    planner = WeightedAStarPlanner2D(_params())
    pinched = planner.plan(_request(), _grid(_corridor(slice(8, 13))))
    assert pinched.artifacts["inflate_used_m"] == 0.25
    # The pinch clears (the phantom decayed); the very next plan is back at 0.40.
    reopened = planner.plan(_request(), _grid(_corridor()))
    assert reopened.artifacts["inflate_used_m"] == 0.40
    assert planner.last_inflate_m == 0.40


# ── the phantom probe ────────────────────────────────────────────────────────
def test_one_cell_throat_is_reported_as_a_suspected_phantom():
    # A single free row through the throat: 0.1 m of lane. Nothing flyable fits,
    # but something gets through -- so the blockage is thin, not a wall.
    res = WeightedAStarPlanner2D(_params()).plan(
        _request(), _grid(_corridor(slice(10, 11))))
    assert res.status == PlanStatus.NO_PATH
    assert res.artifacts["phantom_suspected"] is True
    assert res.artifacts["probe_radius_m"] == 0.05
    assert "mis-detected voxel" in res.message
    assert res.path is None, "a probe route is diagnostic and must never be flown"


def test_the_phantom_report_locates_the_obstruction():
    """The verdict must say WHERE to look again, not just that something is there."""
    res = WeightedAStarPlanner2D(_params()).plan(
        _request(), _grid(_corridor(slice(10, 11))))
    assert res.artifacts["phantom_suspected"] is True
    # The throat spans columns 28..31 -> x in [2.8, 3.2); the pinch must be there,
    # not out in the open corridor either side of it.
    px, py = res.artifacts["pinch_xy"]
    assert 2.7 <= px <= 3.3, "pinch reported at x=%.2f, not in the throat" % px
    # And it must be tighter than anything flyable, or it would not have blocked.
    assert res.artifacts["pinch_clearance_m"] < _params().inflate_floor_m
    assert "(%.2f, %.2f)" % (px, py) in res.message


def test_a_solid_wall_is_not_reported_as_a_phantom():
    arr = _corridor()
    arr[1:H - 1, THROAT] = 100          # seal the throat completely
    res = WeightedAStarPlanner2D(_params()).plan(_request(), _grid(arr))
    assert res.status == PlanStatus.NO_PATH
    assert "phantom_suspected" not in res.artifacts


def test_probe_is_skipped_when_disabled():
    p = _params(probe_radius_m=0.0)
    res = WeightedAStarPlanner2D(p).plan(_request(), _grid(_corridor(slice(10, 11))))
    assert res.status == PlanStatus.NO_PATH
    assert "phantom_suspected" not in res.artifacts


# ── detection must judge a route by the standoff it was planned at ───────────
def test_a_relaxed_route_does_not_report_a_collision_against_its_own_walls():
    """The subtle one: after squeezing to 0.25 the route legitimately runs within
    0.40 of the pinch. Testing it against the PREFERRED radius would flag a
    collision on every map frame and replan-storm the drone."""
    planner = WeightedAStarPlanner2D(_params())
    grid = _grid(_corridor(slice(8, 13)))
    res = planner.plan(_request(), grid)
    assert res.ok and planner.last_inflate_m == 0.25
    assert not planner.path_collides(grid, res.path.points)


def test_detection_still_fires_on_a_real_obstacle_across_a_relaxed_route():
    planner = WeightedAStarPlanner2D(_params())
    res = planner.plan(_request(), _grid(_corridor(slice(8, 13))))
    assert res.ok
    blocked = _corridor(slice(8, 13))
    blocked[1:H - 1, 45:48] = 100        # a fresh wall further along the route
    assert planner.path_collides(_grid(blocked), res.path.points)
