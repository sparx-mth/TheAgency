"""Unit tests for the pure cross-track math (projection, body offset, shaping)."""
import math

from sparx_agency.core.planning.trackers.roll_assist_follower import algorithm as alg


def test_project_interior_point():
    # P above the middle of a horizontal segment -> foot at the midpoint.
    qx, qy, t = alg.project_point_on_segment(1.0, 0.5, 0.0, 0.0, 2.0, 0.0)
    assert abs(qx - 1.0) < 1e-9 and abs(qy) < 1e-9
    assert abs(t - 0.5) < 1e-9


def test_project_clamps_to_endpoints():
    # Before A and past B clamp to the endpoints (t stays in [0, 1]).
    qx, qy, t = alg.project_point_on_segment(-1.0, 3.0, 0.0, 0.0, 2.0, 0.0)
    assert (qx, qy, t) == (0.0, 0.0, 0.0)
    qx, qy, t = alg.project_point_on_segment(5.0, -3.0, 0.0, 0.0, 2.0, 0.0)
    assert (qx, qy, t) == (2.0, 0.0, 1.0)


def test_project_degenerate_segment():
    qx, qy, t = alg.project_point_on_segment(1.0, 1.0, 4.0, 4.0, 4.0, 4.0)
    assert (qx, qy, t) == (4.0, 4.0, 0.0)


def test_body_offset_lateral_when_facing_forward():
    # Facing +x, target 0.3 m to the LEFT (+y) -> e_lat > 0, e_fwd ~ 0.
    e_fwd, e_lat = alg.body_offset_to_point(0.0, 0.0, 0.0, 0.0, 0.3)
    assert abs(e_fwd) < 1e-9
    assert abs(e_lat - 0.3) < 1e-9
    # Target ahead -> forward component, no lateral.
    e_fwd, e_lat = alg.body_offset_to_point(0.0, 0.0, 0.0, 0.5, 0.0)
    assert abs(e_fwd - 0.5) < 1e-9 and abs(e_lat) < 1e-9


def test_body_offset_respects_yaw():
    # Facing +y (yaw=90 deg), a point to the world +x is to the drone's RIGHT.
    e_fwd, e_lat = alg.body_offset_to_point(0.0, 0.0, math.pi / 2, 1.0, 0.0)
    assert abs(e_fwd) < 1e-9
    assert abs(e_lat - (-1.0)) < 1e-9


def test_deadband_continuous():
    assert alg.deadband(0.03, 0.05) == 0.0
    assert abs(alg.deadband(0.20, 0.05) - 0.15) < 1e-9
    assert abs(alg.deadband(-0.20, 0.05) - (-0.15)) < 1e-9
    assert alg.deadband(0.9, 0.0) == 0.9   # width 0 passes through


def test_shape_axis_snap_and_drop():
    # Below release_frac*min -> 0; between -> snapped to min; above -> pass.
    assert alg.shape_axis(0.02, 0.06, 0.5, 1e-3) == 0.0        # < 0.03 drop
    assert abs(alg.shape_axis(0.04, 0.06, 0.5, 1e-3) - 0.06) < 1e-9  # snap up
    assert abs(alg.shape_axis(-0.04, 0.06, 0.5, 1e-3) + 0.06) < 1e-9
    assert abs(alg.shape_axis(0.2, 0.06, 0.5, 1e-3) - 0.2) < 1e-9    # pass


def test_active_segment_selection():
    path = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    # wp_idx 0 -> first real segment.
    assert alg.active_segment(path, 0) == (0.0, 0.0, 2.0, 0.0)
    # wp_idx 1 -> segment ending at waypoint 1.
    assert alg.active_segment(path, 1) == (0.0, 0.0, 2.0, 0.0)
    # wp_idx 2 -> segment ending at waypoint 2.
    assert alg.active_segment(path, 2) == (2.0, 0.0, 4.0, 0.0)
    # Stale/oversized index clamps to the last segment.
    assert alg.active_segment(path, 9) == (2.0, 0.0, 4.0, 0.0)
