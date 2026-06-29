"""Tests for the 2D trajectory simplifier (ROS-free).

Run:
    .venv/bin/python -m pytest \
        sparx_agency/core/planning/path_simplification/tests/test_simplifier_2d.py
"""
from __future__ import annotations

from math import hypot, radians

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.safety.path_correction import InflatedGridCollisionChecker
from sparx_agency.core.planning.path_simplification import (
    SimplifyResult,
    TrajectorySimplifier2D,
    TrajectorySimplifierConfig,
    simplify_collinear_capped_2d,
    smooth_zigzags_2d,
    thin_by_spacing_2d,
)


def _legs(pts):
    return [hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts[:-1], pts[1:])]


def _xy(pts):
    return [(round(p.x, 3), round(p.y, 3)) for p in pts]


# --------------------------------------------------------------------------
# thin_by_spacing_2d  (merge near-duplicates / enforce min spacing)
# --------------------------------------------------------------------------
def test_merge_collapses_near_duplicates():
    # Two points 0.1 m apart with no turn protection -> one is dropped.
    pts = [Pose2D(0, 0), Pose2D(1.0, 0), Pose2D(1.1, 0), Pose2D(3.0, 0)]
    out = thin_by_spacing_2d(pts, min_spacing=0.3, protect_turn_rad=0.0)
    assert _xy(out) == [(0, 0), (1.0, 0), (3.0, 0)]


def test_min_spacing_keeps_endpoints():
    pts = [Pose2D(0, 0), Pose2D(0.2, 0), Pose2D(0.4, 0), Pose2D(0.6, 0)]
    out = thin_by_spacing_2d(pts, min_spacing=1.0, protect_turn_rad=0.0)
    # Everything crowds within 1 m, so only the two endpoints survive.
    assert _xy(out) == [(0, 0), (0.6, 0)]


def test_min_spacing_protects_turns():
    from math import radians
    # The 90 deg corner at (0.5, 0) sits only 0.5 m from the start, so the spacing
    # rule alone would drop it. Turn protection keeps it; without protection it goes.
    pts = [Pose2D(0, 0), Pose2D(0.5, 0), Pose2D(0.5, 0.5), Pose2D(3.0, 0.5)]
    kept = thin_by_spacing_2d(pts, min_spacing=1.0, protect_turn_rad=radians(25))
    assert Pose2D(0.5, 0) in kept           # turn preserved despite crowding
    dropped = thin_by_spacing_2d(pts, min_spacing=1.0, protect_turn_rad=0.0)
    assert Pose2D(0.5, 0) not in dropped    # no protection -> the corner is thinned away


def test_thin_clear_fn_blocks_a_drop():
    # Dropping the middle point would route 0->2 straight; the clear_fn forbids it,
    # so the middle is kept even though it crowds within min_spacing.
    pts = [Pose2D(0, 0), Pose2D(0.2, 0.0), Pose2D(0.4, 0.0)]
    blocked = lambda a, b: not (a.x < 0.3 < b.x or b.x < 0.3 < a.x)  # wall at x=0.3
    out = thin_by_spacing_2d(pts, min_spacing=1.0, protect_turn_rad=0.0, clear_fn=blocked)
    assert len(out) == 3                    # the crowding point survives the block


# --------------------------------------------------------------------------
# simplify_collinear_capped_2d
# --------------------------------------------------------------------------
def test_collinear_drops_straight_middles():
    from math import radians
    # (1,3) (2,3) (4,3) -> (1,3) (4,3): the middle is on the same plane and the
    # bypass gap (3 m) is within the cap.
    pts = [Pose2D(1, 3), Pose2D(2, 3), Pose2D(4, 3)]
    out = simplify_collinear_capped_2d(pts, angle_rad=radians(10), max_segment=3.0)
    assert _xy(out) == [(1, 3), (4, 3)]


def test_collinear_cap_prevents_overlong_leg():
    from math import radians
    # Same straight line but the cap (2 m) forbids the 3 m bypass, so the middle stays.
    pts = [Pose2D(1, 3), Pose2D(2, 3), Pose2D(4, 3)]
    out = simplify_collinear_capped_2d(pts, angle_rad=radians(10), max_segment=2.0)
    assert _xy(out) == [(1, 3), (2, 3), (4, 3)]


def test_collinear_keeps_real_corner():
    from math import radians
    pts = [Pose2D(0, 0), Pose2D(2, 0), Pose2D(2, 2)]   # a 90 deg corner
    out = simplify_collinear_capped_2d(pts, angle_rad=radians(10), max_segment=10.0)
    assert _xy(out) == [(0, 0), (2, 0), (2, 2)]


# --------------------------------------------------------------------------
# smooth_zigzags_2d
# --------------------------------------------------------------------------
def test_zigzag_middle_is_pulled_toward_neighbour_line():
    from math import radians
    # Middle point spikes off the straight x-axis run; smoothing pulls it inward.
    pts = [Pose2D(0, 0), Pose2D(1, 1.0), Pose2D(2, 0)]
    out = smooth_zigzags_2d(pts, angle_rad=radians(45), strength=0.5, passes=1)
    assert out[0] == Pose2D(0, 0) and out[2] == Pose2D(2, 0)   # endpoints fixed
    assert 0.0 < out[1].y < 1.0                                 # pulled toward y=0
    assert abs(out[1].y - 0.5) < 1e-9                           # halfway (strength 0.5)


def test_zigzag_skipped_when_clear_fn_blocks():
    from math import radians
    # The inward move would cross a wall, so the spike is left in place.
    pts = [Pose2D(0, 0), Pose2D(1, 1.0), Pose2D(2, 0)]
    blocked = lambda a, b: False
    out = smooth_zigzags_2d(pts, angle_rad=radians(45), strength=0.5, clear_fn=blocked)
    assert out[1] == Pose2D(1, 1.0)


def test_zigzag_leaves_straight_paths_alone():
    from math import radians
    pts = [Pose2D(0, 0), Pose2D(1, 0), Pose2D(2, 0)]
    out = smooth_zigzags_2d(pts, angle_rad=radians(45), strength=0.5)
    assert _xy(out) == [(0, 0), (1, 0), (2, 0)]


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------
def test_simplifier_short_path_is_untouched():
    s = TrajectorySimplifier2D()
    res = s.simplify([Pose2D(0, 0), Pose2D(1, 0)])
    assert isinstance(res, SimplifyResult)
    assert res.num_in == 2 and res.num_out == 2


def test_simplifier_cleans_a_zigzaggy_corridor():
    # A long straight run sampled densely, with a near-duplicate and a small spike.
    pts = [
        Pose2D(0.0, 0.0), Pose2D(0.5, 0.05), Pose2D(0.55, -0.05),  # crowded + spike
        Pose2D(1.0, 0.0), Pose2D(2.0, 0.0), Pose2D(3.0, 0.0), Pose2D(4.0, 0.0),
    ]
    cfg = TrajectorySimplifierConfig(max_segment_m=5.0, min_spacing_m=1.0)
    res = TrajectorySimplifier2D(cfg).simplify(pts)
    assert res.num_out < res.num_in                  # it got simpler
    assert res.points[0] == Pose2D(0.0, 0.0)         # endpoints preserved
    assert res.points[-1] == Pose2D(4.0, 0.0)
    legs = _legs(res.points)
    assert max(legs) <= 5.0 + 1e-9                   # cap respected
    # straight corridor collapses toward the cap, well under the original 7 points
    assert res.num_out <= 4


def test_simplifier_disabled_passes_are_noops():
    pts = [Pose2D(0, 0), Pose2D(1, 0.4), Pose2D(2, 0)]
    cfg = TrajectorySimplifierConfig(
        merge_enabled=False, zigzag_enabled=False,
        collinear_enabled=False, min_spacing_enabled=False)
    res = TrajectorySimplifier2D(cfg).simplify(pts)
    assert _xy(res.points) == _xy(pts)


# --------------------------------------------------------------------------
# review fixes: max-segment cap on min-spacing, corner-safe merge, validation,
# and the Gauss-Seidel smoothing safety guarantee
# --------------------------------------------------------------------------
def test_min_spacing_respects_max_segment_cap():
    # collinear keeps the middle (3.5 m bypass > 3.0 cap); the min-spacing pass
    # must ALSO honour the cap and not drop it into a single over-long leg.
    res = TrajectorySimplifier2D(TrajectorySimplifierConfig()).simplify(
        [Pose2D(0, 0), Pose2D(0.9, 0), Pose2D(3.5, 0)])
    assert Pose2D(0.9, 0) in res.points
    assert max(_legs(res.points)) <= 3.0 + 1e-9


def test_merge_protects_a_genuine_corner():
    # A 90 deg corner 0.2 m from the start must NOT be merged away (turn-protected).
    res = TrajectorySimplifier2D(TrajectorySimplifierConfig(zigzag_enabled=False)).simplify(
        [Pose2D(0, 0), Pose2D(0.2, 0), Pose2D(0.2, 3)])
    assert Pose2D(0.2, 0) in res.points


def test_config_rejects_out_of_range_strength():
    with pytest.raises(ValueError):
        TrajectorySimplifierConfig(zigzag_strength=2.5)
    with pytest.raises(ValueError):
        TrajectorySimplifierConfig(zigzag_strength=-0.1)
    with pytest.raises(ValueError):
        TrajectorySimplifierConfig(zigzag_passes=0)


def _grid_with_wall():
    """40x40 @0.1m grid, free except a thin vertical wall at col 15, rows 3..6."""
    arr = np.zeros((40, 40), dtype=np.int16)
    arr[3:7, 15] = 100
    return OccupancyGrid2D(
        arr, OccupancyGrid2DParams(0.1, 0.0, 0.0, "world"),
        values=OccupancyValues(free=0, occupied=100, unknown=-1))


def test_smoothing_never_introduces_collision():
    # A,B,C,D where B and C both want to drop to y=0.5; the leg between their
    # MOVED positions crosses the wall. A Jacobi update would move both and
    # collide; the Gauss-Seidel update must reject the second move and stay clear.
    chk = InflatedGridCollisionChecker(_grid_with_wall(), 0.0)
    pts = [Pose2D(0, 0), Pose2D(1, 1), Pose2D(2, 1), Pose2D(3, 0)]
    assert not chk.path_collides(pts)               # input is collision-free
    out = smooth_zigzags_2d(pts, radians(20), strength=1.0, passes=3,
                            clear_fn=chk.segment_clear)
    assert not chk.path_collides(out)               # smoothing kept it clear
    # and the unconstrained (no clear_fn) version WOULD have collided, proving the
    # test actually exercises the guard:
    bad = smooth_zigzags_2d(pts, radians(20), strength=1.0, passes=1, clear_fn=None)
    assert chk.path_collides(bad)
