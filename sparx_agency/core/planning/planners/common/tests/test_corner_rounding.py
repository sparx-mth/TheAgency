"""Unit tests for 2D corner rounding (pure geometry, no numpy)."""
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.planners.common.corner_rounding_2d import (
    _turn_angle,
    chamfer_corners_2d,
    merge_collinear_2d,
)

MAX_TURN = math.radians(14.0)
CHAMFER_MAX = math.radians(28.0)


def _corner(deg, leg=2.0):
    """An L-path turning ``deg`` degrees at the middle vertex, equal legs."""
    a = math.radians(deg)
    return [Pose2D(0, 0), Pose2D(leg, 0),
            Pose2D(leg + leg * math.cos(a), leg * math.sin(a))]


def _max_interior_turn(pts):
    return max((_turn_angle(pts[i - 1], pts[i], pts[i + 1])
               for i in range(1, len(pts) - 1)), default=0.0)


def test_merge_collinear_drops_near_straight_vertex():
    pts = [Pose2D(0, 0), Pose2D(1, 0.01), Pose2D(2, 0), Pose2D(2, 2)]
    out = merge_collinear_2d(pts, math.radians(8.0))
    assert out[0] == pts[0] and out[-1] == pts[-1]
    assert Pose2D(1, 0.01) not in out      # near-straight jog dropped
    assert Pose2D(2, 0) in out             # the real corner is kept


def test_chamfer_halves_moderate_corner():
    pts = _corner(25.0)
    out = chamfer_corners_2d(pts, MAX_TURN, CHAMFER_MAX, 0.5, 0.6)
    assert out[0] == pts[0] and out[-1] == pts[-1]
    assert len(out) == len(pts) + 1                 # one vertex -> two points
    assert _max_interior_turn(out) <= MAX_TURN + 1e-6


def test_chamfer_keeps_sharp_corner():
    pts = _corner(90.0)
    out = chamfer_corners_2d(pts, MAX_TURN, CHAMFER_MAX, 0.5, 0.6)
    assert out == pts                                # too sharp -> untouched


def test_chamfer_keeps_gentle_corner():
    pts = _corner(8.0)                               # already glide-able
    out = chamfer_corners_2d(pts, MAX_TURN, CHAMFER_MAX, 0.5, 0.6)
    assert out == pts


def test_chamfer_skips_when_clear_fn_rejects():
    pts = _corner(25.0)
    out = chamfer_corners_2d(pts, MAX_TURN, CHAMFER_MAX, 0.5, 0.6,
                             clear_fn=lambda a, b: False)
    assert out == pts                                # cut would clip -> keep sharp


def test_chamfer_skips_short_runup():
    pts = _corner(25.0, leg=0.4)                     # legs shorter than min_runup
    out = chamfer_corners_2d(pts, MAX_TURN, CHAMFER_MAX, 0.5, 0.6)
    assert out == pts
