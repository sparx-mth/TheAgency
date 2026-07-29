"""Tests for the turn anticipation: find the corner, lead the nose, crab the body.

Three layers, in the order they are built: the corner geometry (pure), the
heading schedule (stateful but frameless), and then the closed-loop flights that
show the manoeuvre actually happening — a corridor with a right turn flown with
the feature off and on, which is the comparison the whole thing exists for.
"""
from __future__ import annotations

from math import cos, degrees, hypot, radians, sin

import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.trackers.drift_pid import (
    DriftPidFollower,
    DriftPidParams,
    DriftPidState,
    EnvelopeParams,
    LocalizationQuality,
    PidGains,
    YawLead,
    YawLookahead,
    YawLookaheadParams,
    find_corner,
)
from sparx_agency.core.planning.trackers.drift_pid import geometry as geo
from sparx_agency.core.planning.trackers.drift_pid.corners import run_heading
from sparx_agency.core.planning.trackers.drift_pid.yaw_lookahead import (
    approach_limit,
    blend_fraction,
)

DT = 0.1  # 10 Hz control loop, matching ctrl_rate_hz on the drone

#: A corridor 5 m long that turns right and runs 3 m more.
RIGHT_TURN = [Pose2D(0.0, 0.0), Pose2D(5.0, 0.0), Pose2D(5.0, -3.0)]


def _good(conf=0.5, eff=1.0):
    """A healthy localization snapshot."""
    return LocalizationQuality(confidence=conf, pos_std_m=0.02, age_s=0.05,
                               coasting=False, cmd_effectiveness=eff, valid=True)


def _xy(poses):
    """The ``(x, y)`` list the corner geometry takes."""
    return [(p.x, p.y) for p in poses]


def _params(**kw):
    """Controller params with the anticipation on and everything else default."""
    return DriftPidParams(yaw_lookahead=YawLookaheadParams(enabled=True, **kw))


# ── Corner geometry ──────────────────────────────────────────────
def test_a_straight_route_has_no_corner():
    path = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
    assert find_corner(path, 1, 0.0, 0.0, 5.0, radians(25.0), 1.0) is None


def test_the_right_turn_is_found_with_its_distance_and_sign():
    corner = find_corner(_xy(RIGHT_TURN), 1, 3.0, 0.0, 5.0, radians(25.0), 1.0)
    assert corner is not None
    assert corner.index == 1, "the corner is the vertex at (5, 0)"
    assert corner.distance_m == pytest.approx(2.0), "walked along the path"
    assert corner.turn_rad == pytest.approx(radians(-90.0)), "right = negative"
    assert corner.heading_out == pytest.approx(radians(-90.0))


def test_a_left_turn_reads_positive():
    path = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    corner = find_corner(path, 1, 1.0, 0.0, 5.0, radians(25.0), 1.0)
    assert corner.turn_rad == pytest.approx(radians(90.0))


def test_distance_is_walked_along_the_path_not_flown_straight():
    """The straight line to a corner two turns away goes through a wall."""
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (4.0, 2.0)]
    corner = find_corner(path, 2, 2.0, 1.0, 6.0, radians(25.0), 1.0)
    assert corner.index == 2
    assert corner.distance_m == pytest.approx(1.0)


def test_route_weave_below_the_threshold_is_not_a_corner():
    """A grid A* route jitters ~10 degrees down a straight corridor."""
    path = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.1), (4.0, 0.0)]
    assert find_corner(path, 1, 0.2, 0.0, 6.0, radians(25.0), 0.5) is None


def test_small_vertices_that_add_up_to_a_turn_are_one_corner():
    """A 90 degree corner can arrive as a run of 15 degree vertices."""
    path = [(0.0, 0.0), (2.0, 0.0), (2.97, -0.26), (3.83, -0.76),
            (4.0, -2.0)]
    corner = find_corner(path, 1, 1.0, 0.0, 5.0, radians(25.0), 1.0)
    assert corner is not None
    assert corner.index == 2, "the vertex where the swing passes the threshold"
    assert corner.turn_rad < radians(-25.0)


def test_the_final_waypoint_is_never_a_corner():
    """The route does not continue past the goal, so there is nothing to lead."""
    path = [(0.0, 0.0), (3.0, 0.0)]
    assert find_corner(path, 1, 0.0, 0.0, 9.0, radians(25.0), 1.0) is None


def test_a_corner_beyond_reach_is_not_reported():
    corner = find_corner(_xy(RIGHT_TURN), 1, 0.0, 0.0, 2.0, radians(25.0), 1.0)
    assert corner is None, "5 m away, and only 2 m of lookahead"


def test_turn_then_turn_never_looks_past_the_first():
    """Right, a metre, right again: line up with the FIRST turn only."""
    path = [(0.0, 0.0), (4.0, 0.0), (4.0, -1.0), (7.0, -1.0)]
    corner = find_corner(path, 1, 2.0, 0.0, 4.0, radians(25.0), 2.0)
    assert corner.index == 1
    assert corner.heading_out == pytest.approx(radians(-90.0)), (
        "the outgoing heading is the leg between the corners, NOT a blend with "
        "the leg after the second one")
    assert corner.turn_rad == pytest.approx(radians(-90.0))


def test_the_outgoing_heading_averages_a_curved_run():
    """A route that keeps bending reads its overall direction, not one segment."""
    path = [(0.0, 0.0), (2.0, 0.0), (2.5, -0.5), (2.9, -1.05), (3.2, -1.65)]
    heading, run_m = run_heading(path, 1, 2.0, radians(60.0))
    assert radians(-70.0) < heading < radians(-45.0)
    assert run_m > 1.0, "the whole bend, not the first segment of it"


def test_a_jog_too_short_to_line_up_with_is_not_a_corner():
    """A vertex that turns hard and turns straight back has no leg to aim at."""
    path = [(0.0, 0.0), (2.0, 0.0), (2.2, -0.2), (2.4, 0.0), (5.0, 0.0)]
    corner = find_corner(path, 1, 1.0, 0.0, 4.0, radians(25.0), 2.0,
                         min_run_m=0.35)
    assert corner is None
    assert find_corner(path, 1, 1.0, 0.0, 4.0, radians(25.0), 2.0) is not None, (
        "without the minimum run, the jog does read as a 45 degree corner")


# ── The blend schedule ───────────────────────────────────────────
def test_the_blend_runs_from_nothing_to_everything():
    p = YawLookaheadParams(enabled=True, start_m=2.0, align_m=0.4)
    turn = radians(90.0)
    assert blend_fraction(2.5, turn, p) == 0.0
    assert blend_fraction(0.4, turn, p) == 1.0
    assert blend_fraction(0.1, turn, p) == 1.0
    mid = blend_fraction(1.2, turn, p)
    assert 0.0 < mid < 1.0
    assert blend_fraction(1.6, turn, p) < mid < blend_fraction(0.8, turn, p)


def test_a_gentle_bend_is_anticipated_later_than_a_sharp_one():
    """A 30 degree bend needs no run-up; giving it one only crabs a corridor."""
    p = YawLookaheadParams(enabled=True, start_m=2.0, align_m=0.4)
    assert blend_fraction(1.5, radians(90.0), p) > 0.0
    assert blend_fraction(1.5, radians(30.0), p) == 0.0


# ── The heading schedule ─────────────────────────────────────────
def _corner(distance, turn=radians(-90.0), index=1):
    from sparx_agency.core.planning.trackers.drift_pid.corners import Corner
    return Corner(index=index, distance_m=distance, turn_rad=turn,
                  heading_out=turn)


def test_disabled_schedule_says_nothing_and_keeps_no_state():
    look = YawLookahead(YawLookaheadParams(enabled=False))
    lead = look.update(_corner(1.0), 0.0, 0.0, DT)
    assert lead == YawLead()
    assert not look.enabled


def test_the_lead_grows_as_the_corner_approaches():
    look = YawLookahead(YawLookaheadParams(enabled=True, rate=5.0))
    seen = []
    offset = 0.0
    for distance in (1.9, 1.5, 1.0, 0.6, 0.3):
        # Feed the lead back as the yaw: a nose that keeps up perfectly, so
        # what is left is the schedule and nothing else.
        for _ in range(40):
            offset = look.update(_corner(distance), 0.0, offset, DT).offset_rad
        seen.append(offset)
    assert all(a >= b for a, b in zip(seen, seen[1:])), (
        "the lead must only ever grow (more negative for a right turn) on the "
        "way in, got %s" % [round(degrees(s), 1) for s in seen])
    assert seen[0] > seen[1] > seen[2], "the lead never actually moved"
    cap = YawLookaheadParams().max_offset_rad
    assert seen[-1] == pytest.approx(-cap, abs=1e-6), (
        "close in, the lead should sit on its cap")


def test_the_lead_never_exceeds_its_cap():
    look = YawLookahead(YawLookaheadParams(enabled=True, rate=5.0,
                                           max_offset_rad=radians(45.0)))
    last = 0.0
    for _ in range(40):
        last = look.update(_corner(0.1, turn=radians(-150.0)), 0.0,
                           last, DT).offset_rad
    assert last == pytest.approx(radians(-45.0))


def test_the_lead_waits_for_a_nose_that_cannot_keep_up():
    """The schedule may never walk away from the drone it is steering."""
    look = YawLookahead(YawLookaheadParams(enabled=True, rate=5.0,
                                           catchup_rad=radians(10.0)))
    frozen_yaw = 0.0
    last = 0.0
    for _ in range(20):
        lead = look.update(_corner(0.4), 0.0, frozen_yaw, DT)
        last = lead.offset_rad
    assert abs(last) <= radians(10.0) + 1e-9, (
        "with the nose pinned, the lead may not open more than the catch-up "
        "band, got %.1f deg" % degrees(last))


def test_the_lead_is_handed_back_the_moment_the_corner_is_gone():
    """Releasing must never be slower than the drone: a real heading error has
    to reach the yaw loop, and the stop-and-turn latch behind it, at once."""
    look = YawLookahead(YawLookaheadParams(enabled=True, rate=0.1))
    for _ in range(30):
        look.update(_corner(0.4), 0.0, radians(-40.0), DT)
    lead = look.update(None, radians(-90.0), radians(-40.0), DT)
    assert lead == YawLead(), "no corner, no lead, no rate limit on saying so"


def test_the_guard_shrinks_a_lead_but_never_manufactures_one():
    """A drone knocked off its heading has a REAL error, not a schedule lead.

    The catch-up guard keeps the lead close to the nose. Left symmetric it does
    that from either side — including by inventing a lead to make a genuine 90
    degree error read as 12, which hides the error from the yaw loop and from
    the stop-and-turn latch behind it. It may only ever shrink."""
    look = YawLookahead(YawLookaheadParams(enabled=True,
                                           catchup_rad=radians(12.0)))
    # Nose 90 degrees off the leg, corner ahead: the schedule must stay out of it.
    for _ in range(30):
        lead = look.update(_corner(1.2), 0.0, radians(90.0), DT)
    assert abs(lead.offset_rad) < radians(5.0), (
        "a gust was absorbed as %.1f deg of 'lead'" % degrees(lead.offset_rad))


def test_the_feed_forward_reports_the_rate_the_schedule_is_asking_for():
    look = YawLookahead(YawLookaheadParams(enabled=True, rate=0.2,
                                           feedforward=1.0))
    lead = look.update(_corner(0.4), 0.0, 0.0, DT)
    assert lead.rate_hint == pytest.approx(-0.2), (
        "rate-limited growth toward a right turn is a -0.2 rad/s demand")
    quiet = YawLookahead(YawLookaheadParams(enabled=True, feedforward=0.0))
    assert quiet.update(_corner(0.4), 0.0, 0.0, DT).rate_hint == 0.0


# ── Easing off into the turn ─────────────────────────────────────
def test_no_corner_means_no_speed_limit():
    p = YawLookaheadParams(enabled=True)
    assert approach_limit(YawLead(), p, 0.1) > 100.0


def test_a_nose_further_behind_forces_a_slower_approach():
    p = YawLookaheadParams(enabled=True, rate=0.25, align_m=0.3)
    on_schedule = YawLead(offset_rad=radians(-60.0), turn_rad=radians(-90.0),
                          corner_distance_m=1.0, corner_index=1)
    behind = YawLead(offset_rad=radians(-10.0), turn_rad=radians(-90.0),
                     corner_distance_m=1.0, corner_index=1)
    assert approach_limit(behind, p, 0.05) < approach_limit(on_schedule, p, 0.05)


def test_the_approach_limit_never_stops_the_drone_dead():
    p = YawLookaheadParams(enabled=True, align_m=0.3)
    at_the_corner = YawLead(offset_rad=0.0, turn_rad=radians(-90.0),
                            corner_distance_m=0.2, corner_index=1)
    assert approach_limit(at_the_corner, p, 0.07) == pytest.approx(0.07)


# ── Frame maths ──────────────────────────────────────────────────
def test_the_travel_frame_is_the_body_frame_when_the_nose_leads_nothing():
    assert geo.travel_frame_offset(0.3, -0.2, 0.0) == pytest.approx((0.3, -0.2))
    assert geo.travel_allocation(0.4, 0.05, 0.0) == pytest.approx((0.4, 0.05))


def test_a_full_lead_turns_the_whole_cruise_into_roll():
    """The crab that ends the manoeuvre: nose 90 degrees off, all of it lateral."""
    vx, vy = geo.travel_allocation(0.3, 0.0, radians(90.0))
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(0.3)


def test_the_offset_across_travel_survives_the_nose_being_turned():
    """Nose 90 degrees off the leg: the line offset is on the FORWARD axis."""
    along, across = geo.travel_frame_offset(0.25, 0.0, radians(90.0))
    assert along == pytest.approx(0.0, abs=1e-9)
    assert across == pytest.approx(-0.25)


def test_the_carrot_stops_at_the_corner_when_asked():
    path = _xy(RIGHT_TURN)
    free = geo.lookahead_point(path, 1, 4.8, 0.0, 0.6)
    clamped = geo.lookahead_point(path, 1, 4.8, 0.0, 0.6, stop_index=1)
    assert free[1] < -0.3, "unclamped, the carrot walks round the corner"
    assert clamped == pytest.approx((5.0, 0.0)), "clamped, it stops on it"


def test_clamping_at_a_corner_still_ahead_changes_nothing():
    path = _xy(RIGHT_TURN)
    assert geo.lookahead_point(path, 1, 1.0, 0.0, 0.6, stop_index=1) == \
        pytest.approx(geo.lookahead_point(path, 1, 1.0, 0.0, 0.6))


# ── Closed loop: the manoeuvre itself ────────────────────────────
class _Plant:
    """A first-order drone, as in test_drift_pid.py."""

    def __init__(self, pose=None, lag=0.45):
        self.pose = pose or Pose2D(0.0, 0.0, 0.0)
        self.lag = lag
        self._v = [0.0, 0.0, 0.0]

    def apply(self, vx, vy, wz, dt):
        for i, target in enumerate((vx, vy, wz)):
            self._v[i] += self.lag * (target - self._v[i])
        bx, by, bwz = self._v
        yaw = self.pose.yaw
        self.pose = Pose2D(self.pose.x + (bx * cos(yaw) - by * sin(yaw)) * dt,
                           self.pose.y + (bx * sin(yaw) + by * cos(yaw)) * dt,
                           self.pose.yaw + bwz * dt)
        return self.pose


def _fly(route, params, ticks=900, plant=None, republish=False):
    """Fly the route, returning the follower, the plant and a per-tick log."""
    follower = DriftPidFollower(params)
    plant = plant or _Plant()
    follower.set_path(route, plant.pose)
    log = []
    for _ in range(ticks):
        if republish:
            follower.set_path(route, plant.pose)
        follower.set_quality(_good())
        cmd = follower.step(plant.pose, DT)
        log.append((plant.pose, cmd))
        plant.apply(cmd.vx, cmd.vy, cmd.wz, DT)
        if cmd.done:
            break
    return follower, plant, log


def test_the_corner_is_flown_without_ever_stopping_to_rotate():
    """The headline case, and the whole point: no TURN, no standstill."""
    follower, plant, log = _fly(RIGHT_TURN, _params())
    assert follower.done
    states = set(cmd.state for _, cmd in log)
    assert DriftPidState.TURN not in states, (
        "the drone stopped to rotate somewhere it should have flown round")
    stalled = sum(1 for _, cmd in log if hypot(cmd.vx, cmd.vy) < 0.02)
    assert stalled == 0, "%d ticks with no translation at all" % stalled


def test_without_the_anticipation_the_same_corner_is_a_stop_and_spin():
    """The behaviour the feature replaces — pinned so the comparison is real."""
    follower, _, log = _fly(RIGHT_TURN, DriftPidParams())
    assert follower.done
    turning = sum(1 for _, cmd in log if cmd.state == DriftPidState.TURN)
    assert turning > 10, "expected the classic stop-and-turn at the corner"


def test_the_nose_is_round_most_of_the_corner_before_the_drone_reaches_it():
    """Most of the way onto the new leg while the body is still on the old one.

    Not ALL of the way: the lead is capped short of sideways on purpose (see
    ``max_offset_rad``), because a crab with no forward speed left is one this
    airframe can no longer yaw out of. The last slice of the turn is finished
    at the corner, moving onto the new leg."""
    _, _, log = _fly(RIGHT_TURN, _params())
    at_corner = [(pose, cmd) for pose, cmd in log
                 if 0.0 < cmd.telemetry.corner_dist_m <= 0.45]
    assert at_corner, "the corner was never approached"
    pose, cmd = at_corner[-1]
    turned = -degrees(pose.yaw)
    assert 55.0 < turned < 90.0, (
        "expected most of the 90 degree turn taken before the corner, got "
        "%.1f deg" % turned)
    assert abs(pose.y) < 0.25, "the body should still be on the old leg"


def test_a_right_turn_is_crabbed_to_the_left():
    """Nose right, roll left — the coupling the pilot described."""
    _, _, log = _fly(RIGHT_TURN, _params())
    crabbing = [cmd for _, cmd in log
                if degrees(cmd.telemetry.yaw_lead_rad) < -45.0]
    assert crabbing, "the drone never took a real lead"
    assert all(cmd.vy > 0.0 for cmd in crabbing), (
        "a nose led right must be paid for with LEFT roll")
    assert all(cmd.vx >= 0.0 for cmd in crabbing), (
        "a crab must never become blind reverse")


def test_the_lead_is_published_for_the_operator():
    _, _, log = _fly(RIGHT_TURN, _params())
    assert min(cmd.telemetry.yaw_lead_rad for _, cmd in log) < radians(-60.0)
    assert max(cmd.telemetry.corner_dist_m for _, cmd in log) > 1.0


def test_a_mis_pointed_drone_still_gets_the_full_stop_and_turn():
    """The anticipation must not swallow a genuine heading error."""
    follower = DriftPidFollower(_params())
    plant = _Plant(Pose2D(0.0, 0.0, radians(90.0)))
    follower.set_path(RIGHT_TURN, plant.pose)
    follower.set_quality(_good())
    cmd = follower.step(plant.pose, DT)
    assert cmd.state == DriftPidState.TURN
    assert cmd.wz < 0.0, "should rotate clockwise to face down the corridor"
    assert cmd.telemetry.yaw_lead_rad == 0.0, "no corner is being anticipated"


@pytest.mark.parametrize("gust_deg", [90.0, -90.0, 45.0])
def test_a_gust_INSIDE_the_corner_window_still_reaches_the_yaw_loop(gust_deg):
    """The dangerous version of the case above: mis-pointed WITH a corner near.

    Here the schedule is live, so a guard that clamps the heading error from
    both sides can rewrite the whole gust as "lead" — the loop is shown 12
    degrees instead of 90, TURN never latches, the rotation stays on the gentle
    tracking cap and the drone keeps flying while pointed at a wall."""
    follower = DriftPidFollower(_params())
    on_line = Pose2D(3.5, 0.0, 0.0)
    follower.set_path(RIGHT_TURN, on_line)
    follower.set_quality(_good())
    follower.step(on_line, DT)                     # settle, corner in the window
    follower.set_quality(_good())
    cmd = follower.step(Pose2D(3.5, 0.0, radians(gust_deg)), DT)
    assert cmd.telemetry.corner_dist_m > 0.0, "the corner left the window"
    assert cmd.state == DriftPidState.TURN, (
        "a %.0f deg heading error was absorbed instead of turned for" % gust_deg)
    assert abs(degrees(cmd.telemetry.heading_err_rad)) > 0.8 * abs(gust_deg), (
        "the yaw loop was shown %.1f deg of a %.0f deg error"
        % (degrees(cmd.telemetry.heading_err_rad), gust_deg))


def _deployed(**yaw_lookahead):
    """Roughly the tuning config/mission.yaml flies, where the crab is fastest."""
    return DriftPidParams(
        cruise_speed=0.38, approach_yaw_rate=0.65, track_yaw_rate=0.25,
        arrive_speed_min=0.10, yaw_engage_rad=radians(30.0),
        yaw_release_rad=radians(12.0), travel_cone_rad=radians(85.0),
        turn_pitch_bias=0.08,
        envelope=EnvelopeParams(max_vx=0.45, max_vx_back=0.12, max_vy=0.15,
                                max_wz=0.65, max_translation=0.45,
                                combined_effort=1.8, min_vx=0.06, min_vy=0.05,
                                min_wz=radians(10.0)),
        lateral_pid=PidGains(kp=0.55, ki=0.06, kd=0.12, i_limit=0.05,
                             d_tau_s=0.4, deadband=0.03, out_limit=0.12),
        yaw_pid=PidGains(kp=0.90, ki=0.08, kd=0.15, i_limit=0.08, d_tau_s=0.35,
                         deadband=radians(2.0), out_limit=0.65),
        yaw_lookahead=YawLookaheadParams(enabled=True, **yaw_lookahead))


def test_the_crab_never_commands_blind_reverse():
    """The lead cannot point backward, but the correction riding on it can.

    At a 60-degree lead the travel vector holds only ~0.09 m/s of forward speed
    and a full cross-track correction pulling the same way is 0.12 — enough to
    tip the total into reverse, which on this airframe is flown blind and is
    only ever an escape move. Measured at the deployed tuning from 1.4 m off
    the line the raw allocation reaches -0.046 m/s, which the minimum-force
    shaper would then snap up to a published -0.06.

    Tested on the allocation directly, because the usual downstream guard hides
    it: ``turn_coordination`` also floors vx, but ONLY while the yaw axis is
    active. A steady lead with a quiet yaw — exactly the middle of a long crab —
    is the case that reaches the published command, and it is the case this
    floor exists for."""
    follower = DriftPidFollower(_deployed())
    follower.set_path(RIGHT_TURN, Pose2D(0.0, 0.0, 0.0))
    follower.set_quality(_good())
    follower.step(Pose2D(0.0, 0.0, 0.0), DT)
    auth = follower._scheduler.evaluate(_good())
    lead = YawLead(offset_rad=radians(-60.0), turn_rad=radians(-90.0),
                   corner_distance_m=0.6, corner_index=1)
    worst = 0.0
    for _ in range(30):                    # let the lateral loop reach its limit
        for travel in (radians(60.0), radians(-60.0)):
            # A metre off the line, on the side the correction pulls toward.
            e_lat = 1.0 if travel > 0 else -1.0
            vx, _, _, _ = follower._crab(0.38, travel, 0.0, e_lat, DT, auth,
                                         True, lead)
            worst = min(worst, vx)
    assert worst >= 0.0, (
        "the crab allocated %.3f m/s of blind reverse" % worst)


def test_a_republished_route_leaves_the_lead_exactly_where_it_was():
    """The planner sends the same path several times a second.

    Driven open-loop off one pose sequence so the ONLY difference between the
    two followers is the re-``set_path`` -- a closed loop would diverge for
    reasons that have nothing to do with this feature (identical republishes
    also reset the derivative memory and the waypoint index, a pre-existing
    trait of the follower that this test is deliberately not measuring)."""
    # A real flight first, to get a physically consistent pose sequence.
    _, _, log = _fly(RIGHT_TURN, _params())
    poses = [pose for pose, _ in log][:260]

    steady = DriftPidFollower(_params())
    churned = DriftPidFollower(_params())
    for follower in (steady, churned):
        follower.set_path(RIGHT_TURN, poses[0])
    for pose in poses:
        churned.set_path(RIGHT_TURN, pose)
        for follower in (steady, churned):
            follower.set_quality(_good())
            follower.step(pose, DT)
    assert churned.yaw_lead.offset_rad == pytest.approx(
        steady.yaw_lead.offset_rad, abs=1e-9)
    assert abs(churned.yaw_lead.offset_rad) > radians(20.0), (
        "the replay never reached a real lead, so it proved nothing")


def test_turn_then_turn_lines_up_with_one_corner_at_a_time():
    route = [Pose2D(0.0, 0.0), Pose2D(4.0, 0.0), Pose2D(4.0, -1.0),
             Pose2D(7.0, -1.0)]
    follower, _, log = _fly(route, _params(), ticks=1400)
    assert follower.done
    # Between the corners the route turns LEFT again, so the lead must change
    # sign rather than carry the first turn's rotation into the second.
    leads = [cmd.telemetry.yaw_lead_rad for _, cmd in log]
    assert min(leads) < radians(-45.0), "never led into the first (right) turn"
    assert max(leads) > radians(45.0), "never led into the second (left) turn"


def test_a_gentle_bend_is_left_alone():
    """Nothing to anticipate below the corner threshold: fly it as before."""
    route = [Pose2D(0.0, 0.0), Pose2D(3.0, 0.0), Pose2D(6.0, 0.5)]
    _, _, log = _fly(route, _params())
    assert max(abs(cmd.telemetry.yaw_lead_rad) for _, cmd in log) == 0.0


def test_the_drone_still_holds_its_line_through_the_corner():
    """The nose may leave the leg; the body may not."""
    _, _, log = _fly(RIGHT_TURN, _params())
    settled = [cmd for i, (_, cmd) in enumerate(log) if i > 30]
    worst = max(abs(cmd.telemetry.cross_track_m) for cmd in settled)
    assert worst < 0.35, "drifted %.2f m off the line" % worst


def test_the_schedules_own_yaw_is_not_evidence_of_a_wedged_drone():
    """Two corners 60 cm apart turn the nose one way and straight back.

    The blockage monitor compares summed commanded yaw against NET rotation, so
    a schedule that reverses inside its window reads as "a lot of yaw, nowhere
    to show for it" — a wedged drone. Measured on a surveyed office route it
    fired a full escape reflex at a drone that was flying perfectly."""
    route = [Pose2D(0.0, 0.0), Pose2D(3.0, 0.0), Pose2D(3.4, 0.4),
             Pose2D(3.4, 2.3), Pose2D(3.0, 2.7)]
    follower, _, log = _fly(route, _params(), ticks=1200)
    states = set(cmd.state for _, cmd in log)
    assert DriftPidState.ESCAPE not in states, (
        "the anticipation's own yaw was read as a blockage and a reflex fired")
    assert not any(cmd.report_blocked for _, cmd in log), (
        "and it would have taught the planner a phantom obstacle")
    assert follower.done


def test_a_genuinely_wedged_drone_is_still_caught_with_the_feature_on():
    """Suppressing the yaw evidence must not blind the monitor outright."""
    params = _params()
    follower = DriftPidFollower(params)
    follower.set_path(RIGHT_TURN, Pose2D(0.0, 0.0, 0.0))
    pinned = Pose2D(0.0, 0.0, 0.0)          # never moves, whatever we command
    reports = 0
    states = set()
    for _ in range(400):
        follower.set_quality(_good(eff=0.02))
        cmd = follower.step(pinned, DT)
        states.add(cmd.state)
        reports += int(cmd.report_blocked)
    assert DriftPidState.ESCAPE in states, "the drone never tried to get free"
    assert reports >= 1, "a wedged drone was never reported to the planner"


def test_holding_mid_crab_gives_the_lead_back_and_resumes():
    follower = DriftPidFollower(_params())
    plant = _Plant()
    follower.set_path(RIGHT_TURN, plant.pose)
    held = None
    for i in range(900):
        follower.set_quality(_good())
        cmd = follower.step(plant.pose, DT, hold=200 <= i < 240)
        if i == 220:
            held = cmd
        plant.apply(cmd.vx, cmd.vy, cmd.wz, DT)
        if cmd.done:
            break
    assert held is not None and held.state == DriftPidState.HOLD
    assert held.telemetry.yaw_lead_rad == 0.0, (
        "a stopped drone is not mid-manoeuvre and must not claim to be")
    assert follower.done, "the flight did not survive the hold"


# ── Parameter contracts ──────────────────────────────────────────
def test_a_schedule_faster_than_the_tracking_yaw_cap_is_rejected():
    with pytest.raises(ValueError, match="track_yaw_rate"):
        DriftPidParams(track_yaw_rate=0.20,
                       yaw_lookahead=YawLookaheadParams(enabled=True, rate=0.30))


def test_a_catch_up_band_past_the_engage_threshold_is_rejected():
    with pytest.raises(ValueError, match="catchup_rad"):
        DriftPidParams(yaw_engage_rad=radians(20.0),
                       yaw_lookahead=YawLookaheadParams(
                           enabled=True, catchup_rad=radians(25.0)))


def test_an_empty_blend_span_is_rejected():
    with pytest.raises(ValueError, match="align_m"):
        YawLookaheadParams(start_m=1.0, align_m=1.0)


def test_a_lead_past_sideways_is_rejected():
    with pytest.raises(ValueError, match="max_offset_rad"):
        YawLookaheadParams(max_offset_rad=radians(120.0))


def test_the_disabled_default_leaves_the_contracts_alone():
    """Every invariant above only binds a controller that actually flies it."""
    DriftPidParams(track_yaw_rate=0.20, yaw_engage_rad=radians(12.0),
                   yaw_lookahead=YawLookaheadParams(rate=0.90,
                                                    catchup_rad=radians(80.0)))
