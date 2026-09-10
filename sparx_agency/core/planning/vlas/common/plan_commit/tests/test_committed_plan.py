"""Anchoring a body-frame prediction in the world, and where its commitment ends."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.common.plan_commit.committed_plan import (
    anchor_plan,
    commit_index_for,
)


def straight_ahead(count, step=0.2):
    """``count`` body-frame waypoints running straight forward at ``step`` metres."""
    return np.stack([np.arange(1, count + 1) * step, np.zeros(count)], axis=1)


def test_half_of_sixteen_waypoints_is_waypoint_eight():
    assert commit_index_for(16, 0.5) == 8


def test_half_of_navdps_twenty_four_waypoints_is_waypoint_twelve():
    assert commit_index_for(24, 0.5) == 12


def test_a_commitment_is_never_empty():
    """Committing to zero waypoints re-infers on the next tick, which is the bug."""
    assert commit_index_for(16, 0.0) == 1


def test_a_commitment_never_runs_past_the_prediction():
    assert commit_index_for(16, 2.0) == 16


def test_a_prediction_with_no_waypoints_cannot_be_committed_to():
    with pytest.raises(ValueError):
        commit_index_for(0, 0.5)


def test_the_anchor_is_the_polylines_first_vertex():
    plan = anchor_plan(straight_ahead(16), (3.0, -1.0, 0.0), 0.0, 0.5)
    assert plan.world_xy.shape == (17, 2)
    assert plan.world_xy[0] == pytest.approx([3.0, -1.0])
    assert plan.waypoints == 16


def test_waypoints_rotate_into_the_world_by_the_anchor_yaw():
    plan = anchor_plan(straight_ahead(4, step=1.0), (0.0, 0.0, math.pi / 2), 0.0, 0.5)
    # Body +x (forward) at yaw 90 deg is world +y.
    assert plan.world_xy[4] == pytest.approx([0.0, 4.0], abs=1e-9)


def test_anchoring_rotates_about_the_aircraft_not_the_world_origin():
    """Yaw and translation together, which is the only case that separates
    ``R @ body + t`` from ``R @ (body + t)``. Every prediction made away from
    the origin depends on the difference, and with one of the two at zero both
    forms agree."""
    plan = anchor_plan(straight_ahead(4, step=1.0), (10.0, -5.0, math.pi / 2),
                       0.0, 0.5)
    # Facing world +y from (10, -5): four metres forward is (10, -1).
    assert plan.world_xy[4] == pytest.approx([10.0, -1.0], abs=1e-9)
    assert plan.world_xy[0] == pytest.approx([10.0, -5.0])


def test_extra_trajectory_columns_are_ignored():
    """NavDP carries a yaw in column three; it is not part of the route."""
    body = np.concatenate([straight_ahead(8), np.full((8, 1), 0.7)], axis=1)
    plan = anchor_plan(body, (0.0, 0.0, 0.0), 0.0, 0.5)
    assert plan.world_xy.shape == (9, 2)


def test_the_committed_part_stops_at_the_commit_point():
    plan = anchor_plan(straight_ahead(16), (0.0, 0.0, 0.0), 0.0, 0.5)
    assert plan.commit_index == 8
    assert plan.committed_xy.shape == (9, 2)
    assert plan.commit_point == pytest.approx((1.6, 0.0))
    assert plan.commit_arc_m == pytest.approx(1.6)
    assert plan.total_arc_m == pytest.approx(3.2)


def test_progress_is_measured_from_the_aircraft_not_the_first_waypoint():
    """Arc zero is under the aircraft, so a plan starts at 0 % and not at 6 %."""
    plan = anchor_plan(straight_ahead(16), (0.0, 0.0, 0.0), 0.0, 0.5)
    arc, lateral, segment = plan.progress(0.0, 0.0)
    assert arc == pytest.approx(0.0)
    assert lateral == pytest.approx(0.0)
    assert segment == 0


def test_the_carrot_advances_with_the_aircraft():
    """The point of committing: the lookahead moves along the frozen route."""
    plan = anchor_plan(straight_ahead(24), (0.0, 0.0, 0.0), 0.0, 0.5)
    first = plan.carrot(0.0, 0.0, 1.2)
    later = plan.carrot(1.0, 0.0, 1.2)
    assert first[0] == pytest.approx(1.2, abs=0.05)
    assert later[0] == pytest.approx(2.2, abs=0.05)


def test_the_carrot_stops_at_the_end_of_the_plan():
    plan = anchor_plan(straight_ahead(8), (0.0, 0.0, 0.0), 0.0, 0.5)
    assert plan.carrot(1.5, 0.0, 5.0) == pytest.approx((1.6, 0.0, 0.0))


def test_the_heading_is_the_route_tangent_not_the_bearing_to_the_carrot():
    """On a turn the two differ, and only the tangent points the camera down the
    route. It is also what the expert encodes as NavDP's yaw channel."""
    angles = np.linspace(0.0, math.pi / 2, 24)
    body = np.stack([3.0 * np.sin(angles), 3.0 * (1.0 - np.cos(angles))], axis=1)
    plan = anchor_plan(body, (0.0, 0.0, 0.0), 0.0, 0.5)

    cx, cy, heading = plan.carrot(0.0, 0.0, 1.2)
    chord = math.atan2(cy, cx)
    # For a circular arc leaving the aircraft along its nose, the chord to a
    # point at arc angle t subtends t/2 while the tangent there is t -- so the
    # tangent leads by exactly the chord angle, and an aircraft flown on the
    # chord is permanently half a corner behind the route.
    assert heading == pytest.approx(2.0 * chord, rel=0.15)
    assert heading - chord > math.radians(5)


def test_a_stopped_prediction_has_no_heading_to_offer():
    """All-zero trajectories are what an unsure policy emits; atan2(0, 0) would
    silently point the nose east."""
    plan = anchor_plan(np.zeros((24, 2)), (1.0, 2.0, 1.0), 0.0, 0.5)
    assert plan.carrot(1.0, 2.0, 1.2)[2] is None
    assert plan.segment_heading(0) is None


def test_a_prediction_that_stalls_has_no_heading_on_its_dead_tail():
    """A route that moves and then stops offers no heading past the point it
    stopped at, and the carrot that reaches that tail says None too. Only the
    tail is dead here, so a policy that pads an honest first leg with repeated
    waypoints cannot swing the nose to world east the moment the aircraft
    arrives at the end of it."""
    body = np.concatenate([straight_ahead(2, step=1.0),
                           np.tile([2.0, 0.0], (3, 1))], axis=0)
    plan = anchor_plan(body, (0.0, 0.0, 0.0), 0.0, 0.5)
    assert plan.segment_heading(0) == pytest.approx(0.0)
    assert plan.segment_heading(2) is None
    assert plan.segment_heading(3) is None
    assert plan.carrot(0.0, 0.0, 3.5)[2] is None


def test_a_turning_prediction_keeps_its_shape_in_the_world():
    """A right-hand arc anchored at yaw 0 must still curve right in the world."""
    angles = np.linspace(0.0, math.pi / 4, 12)
    body = np.stack([2.0 * np.sin(angles), -2.0 * (1.0 - np.cos(angles))], axis=1)
    plan = anchor_plan(body, (5.0, 5.0, 0.0), 0.0, 0.5)
    assert plan.world_xy[-1][1] < plan.world_xy[1][1]
    assert plan.total_arc_m == pytest.approx(math.pi / 2, abs=0.02)


def switchback(reach=2.4, offset=0.40, per_leg=12):
    """Out and back: the shape a policy makes when it changes its mind."""
    out = np.stack([np.linspace(reach / per_leg, reach, per_leg),
                    np.zeros(per_leg)], axis=1)
    back = np.stack([np.linspace(reach, 0.0, per_leg),
                     np.full(per_leg, offset)], axis=1)
    return np.concatenate([out, back], axis=0)


def fly_along(plan, xs, lookahead_m=1.2):
    """Carrots seen while flying up the outbound leg, cursor advancing as the
    executor advances it. Yields ``(aircraft_xy, carrot_xy, cursor_arc)``."""
    cursor = 0
    seen = []
    for x in xs:
        arc, _, cursor = plan.progress(float(x), 0.0, cursor)
        cx, cy, _ = plan.carrot(float(x), 0.0, lookahead_m, cursor)
        seen.append(((float(x), 0.0), (cx, cy), arc))
    return seen


OUTBOUND_ARC_M = 2.4
"""Arc length of :func:`switchback`'s outbound leg, anchor included."""


def test_the_carrot_stays_ahead_while_the_turn_is_beyond_the_lookahead():
    """A lookahead measured as a RADIUS finds the return leg lying beside the
    aircraft and hands that back: 168 degrees, on exactly this shape, with the
    whole turn skipped. Measured along the route it must stay on the leg the
    aircraft is on until the turn is genuinely within reach.

    Past that point the carrot does swing across -- correctly. A switchback
    tighter than the lookahead is a route the aircraft has to slow down, turn
    around and fly back along, and a carrot that stayed 'ahead' there would be
    the corner-skipping this test exists to forbid.
    """
    plan = anchor_plan(switchback(), (0.0, 0.0, 0.0), 0.0, 0.5)
    checked = 0
    for (ax, _), (cx, cy), here in fly_along(plan, np.arange(0.0, 2.45, 0.1)):
        if here + 1.2 > OUTBOUND_ARC_M:
            continue
        assert abs(math.degrees(math.atan2(cy, cx - ax))) < 1.0
        checked += 1
    assert checked > 5


def test_the_carrot_is_a_fixed_distance_along_the_route():
    """Not a chord across it: around the U-turn tip the two differ by metres."""
    from sparx_agency.core.planning.vlas.common.plan_commit.progress import project

    plan = anchor_plan(switchback(), (0.0, 0.0, 0.0), 0.0, 0.5)
    for _, (cx, cy), here in fly_along(plan, np.arange(0.0, 2.45, 0.1)):
        there, _, _ = project(plan.world_xy, cx, cy, 0, window=None)
        # Exactly a lookahead further along, until the route runs out.
        assert there - here == pytest.approx(min(1.2, plan.total_arc_m - here),
                                             abs=0.02)


# ── a prediction that starts AT the robot ────────────────────────────────
#
# Both of InternVLA-N1's producers do this: the integrated System-1 curve and
# the single step rendered from a discrete action both begin at (0, 0). The
# anchor prepended by `anchor_plan` would otherwise duplicate that first
# waypoint -- invisible on a 33-point curve, fatal on the 2-point action step,
# where `commit_index_for(2, 0.5)` commits through the copy and the commitment
# is two identical points 0.00 m apart. Measured in flight: the executor called
# every such plan TOO_SHORT, re-inferred immediately, and the aircraft hovered
# while the log filled with commitments it never flew.


def test_a_leading_origin_waypoint_is_not_duplicated_by_the_anchor():
    trajectory = np.array([[0.0, 0.0], [0.25, 0.0]])
    plan = anchor_plan(trajectory, (1.0, 2.0, 0.0), issued_s=0.0, fraction=0.5)
    assert plan.world_xy.shape == (2, 2)
    assert plan.world_xy[0] == pytest.approx((1.0, 2.0))
    assert plan.world_xy[1] == pytest.approx((1.25, 2.0))
    assert plan.commit_arc_m == pytest.approx(0.25)


def test_the_two_point_action_step_is_flyable_not_zero_length():
    """The exact shape `geometry.trajectory_from_action` emits."""
    step = np.array([[0.0, 0.0, 0.0], [0.2165, 0.125, 0.5236]])  # 0.25 m, 30 deg
    plan = anchor_plan(step, (0.0, 0.0, 0.0), issued_s=0.0, fraction=0.5)
    assert plan.commit_arc_m == pytest.approx(0.25, abs=1e-3)
    assert len(plan.committed_xy) == 2


def test_a_curve_that_starts_at_the_robot_keeps_every_real_waypoint():
    curve = np.concatenate([np.zeros((1, 2)),
                            np.stack([np.linspace(0.1, 1.6, 16), np.zeros(16)], axis=1)])
    plan = anchor_plan(curve, (0.0, 0.0, 0.0), issued_s=0.0, fraction=0.5)
    assert plan.world_xy.shape == (17, 2)          # anchor + 16, not anchor + 17
    assert plan.world_xy[-1] == pytest.approx((1.6, 0.0))


def test_a_waypoint_a_centimetre_out_is_a_step_not_the_robot():
    """The epsilon is a millimetre: 1 cm is a real, if short, motion."""
    plan = anchor_plan(np.array([[0.01, 0.0], [0.3, 0.0]]), (0.0, 0.0, 0.0),
                       issued_s=0.0, fraction=1.0)
    assert plan.world_xy.shape == (3, 2)
    assert plan.world_xy[1] == pytest.approx((0.01, 0.0))


def test_a_prediction_that_is_only_the_robot_is_rejected():
    with pytest.raises(ValueError):
        anchor_plan(np.array([[0.0, 0.0]]), (0.0, 0.0, 0.0), issued_s=0.0, fraction=1.0)
