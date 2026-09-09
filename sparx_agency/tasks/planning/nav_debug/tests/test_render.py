"""The renderer produces a valid image for full, partial and empty frames."""
import numpy as np
import pytest

from sparx_agency.tasks.planning.nav_debug.frame import (
    Actuator, AxisTrace, BevMap, Drift, DroneState, GaugeScales, NavFrame,
    Quality, Reference, ReplanEvent, Routes, VelocityTarget,
)
from sparx_agency.tasks.planning.nav_debug.render import render
from sparx_agency.tasks.planning.nav_debug.render_lanes import (
    build_lane_column, _reference_lane,
)
from sparx_agency.tasks.planning.nav_debug.render_panel import build_panel


def _bev():
    grid = np.full((30, 40), 0, np.int8)
    grid[0, :] = 100
    return BevMap(grid=grid, resolution=0.1, origin_x=-2.0, origin_y=-1.0)


def _full_frame():
    return NavFrame(
        stamp=1.0, x=0.2, y=0.3, yaw=1.2, z=1.0,
        trail=[(0.0, 0.0), (0.1, 0.15)],
        our_cmd=(0.30, 0.05, 0.08, -0.20), drone_cmd=(400, -80, 60, 320),
        quality=Quality(0.62, 0.18, 0.8, False, 0.3, "TRACK"),
        drift=Drift(0.0, 0.05, 0.0, 0.04, 0.0, 3.0, 0.5, 1.0,
                    "holding roll", "TRACK", "IDLE", ""),
        target=(2, 5, 2.0, 0.0), advanced=True,
        bev=_bev(), routes=Routes(astar=[(0, 0), (1, 0)], safe=[(0, 0), (1, 0)],
                                  final=[(0, 0), (1, 0), (2, 0)], goal=(2.0, 0.0),
                                  lookahead=(0.5, 0.0)),
        replan=ReplanEvent(0.9, "rotation", "REPLAN: rotated 34 deg", 0.1),
        cmd_history=[0.1, 0.2, 0.3, 0.3], conf_history=[0.6, 0.62, 0.62],
        why="holding 5cm/s right roll vs drift")


def test_render_full_frame():
    img = render(_full_frame(), GaugeScales())
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 100 and img.shape[1] > 400   # map + panel side by side


def test_render_without_map_or_commands():
    fr = NavFrame(stamp=0.0, x=0.0, y=0.0, yaw=0.0)     # nothing but a pose
    img = render(fr)
    assert img.ndim == 3 and img.shape[2] == 3          # must not raise


def test_render_handles_extreme_route_coords():
    fr = _full_frame()
    fr.routes.final = [(0, 0), (1e6, -1e6)]             # far off the map -> clipped
    img = render(fr)
    assert img.shape[2] == 3


# ── the blocks the four upgrades added ───────────────────────────────────────
def _panel(frame, scales=None):
    """The command column alone, so a block's presence is a pixel-count question."""
    return build_panel(frame, scales or GaugeScales())


def _drawn(before, after):
    """True when ``after`` has content ``before`` did not -- i.e. something drew."""
    return after.shape[0] > before.shape[0]


def test_the_command_column_names_who_sends_cmd_vel_and_who_receives_it():
    """Request: 'ours' must say what it is, who produces it and who consumes it.

    Asserted on the strings rather than on pixels, because the point of the
    change is the specific names -- a block that drew but said nothing would
    still be the bug.
    """
    from sparx_agency.tasks.planning.nav_debug import render_panel
    assert "falcon_exploration_follower" in render_panel._CMD_VEL_FROM
    assert "GO gate" in render_panel._CMD_VEL_FROM
    assert "rooster_twist_control_adapter" in render_panel._CMD_VEL_INTO


def test_an_upstream_disagreement_grows_the_command_column():
    """SHAPER CLIPPED / GO GATE BLOCKED only draw when a stage changed the command."""
    agreed = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                      our_cmd=(0.50, 0.0, 0.0, 0.0),
                      velocity_target=VelocityTarget(vx=0.50),
                      velocity_requested=VelocityTarget(vx=0.50))
    clipped = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                       our_cmd=(0.50, 0.0, 0.0, 0.0),
                       velocity_target=VelocityTarget(vx=0.50),
                       velocity_requested=VelocityTarget(vx=0.90))
    assert _drawn(_panel(agreed), _panel(clipped))


def test_a_gate_that_swallowed_the_command_is_reported():
    """The follower asked for real motion and nothing came out of the GO gate."""
    passed = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                      our_cmd=(0.50, 0.0, 0.0, 0.0),
                      velocity_target=VelocityTarget(vx=0.50))
    blocked = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                       our_cmd=(0.0, 0.0, 0.0, 0.0),
                       velocity_target=VelocityTarget(vx=0.50))
    assert _drawn(_panel(passed), _panel(blocked))


def test_missing_counts_say_why_on_a_sphera_run():
    """'no cmd_nav' alone cannot distinguish 'not recorded' from 'told nothing'."""
    bare = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0)
    sphera = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                      reference=Reference(x=1.0, y=2.0, z=1.5))
    # The Sphera frame earns the extra two lines naming run_nav_debug_recorder.sh.
    assert _drawn(_panel(bare), _panel(sphera))


def test_the_counts_are_drawn_axis_by_axis_when_they_exist():
    without = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0)
    with_counts = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                           drone_cmd=(500, -120, 700, 340),
                           actuator=Actuator(cmd_nav=(500.0, -120.0, 340.0),
                                             manual=(500.0, -120.0, 700.0, 340.0)))
    assert _drawn(_panel(without), _panel(with_counts))


def test_the_axis_servo_lane_no_longer_replaces_the_counts():
    """They used to be either/or, which left the counts with nowhere to be drawn."""
    counts = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                      drone_cmd=(500, -120, 700, 340),
                      actuator=Actuator(cmd_nav=(500.0, -120.0, 340.0),
                                        manual=(500.0, -120.0, 700.0, 340.0)))
    both = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                    drone_cmd=counts.drone_cmd, actuator=counts.actuator,
                    axes=[AxisTrace(name="forward", requested=0.6, measured=0.55,
                                    feed_forward=480.0, correction=20.0,
                                    pre_slew=500.0, counts=500.0)])
    assert _drawn(_panel(counts), _panel(both))


def test_the_reference_lane_compares_against_the_measured_state():
    """The table must draw with a reference, with a state, or with both.

    Asserted against the lane function's own advance rather than the column
    height: the column contains four other lanes, so a height check passes even
    when this lane draws nothing at all.
    """
    ref = Reference(x=10.0, y=2.0, z=1.5, yaw=0.1, vx=0.6, moving=True)
    state = DroneState(x=9.6, y=2.0, z=1.5, yaw=0.1, vx=0.9, vy=0.0, vz=0.0,
                       wz=0.02, source="odom")
    panel = np.full((1600, 360, 3), 0, np.uint8)
    empty = _reference_lane(panel, 12, 26, 360, NavFrame(stamp=1.0, x=0.0, y=0.0,
                                                         yaw=0.0), None)
    for frame in (NavFrame(stamp=1.0, x=9.6, y=2.0, yaw=0.1, reference=ref),
                  NavFrame(stamp=1.0, x=9.6, y=2.0, yaw=0.1, state=state),
                  NavFrame(stamp=1.0, x=9.6, y=2.0, yaw=0.1, reference=ref,
                           state=state)):
        drawn = _reference_lane(panel, 12, 26, 360, frame, None)
        # One row per compared quantity, plus a header, a section and a note.
        assert drawn > empty + 8 * 18, "lane advanced only %d px" % (drawn - 26)
        assert build_lane_column(frame, GaugeScales()).ndim == 3


def test_a_bridge_that_changed_the_command_is_reported():
    """cmd_vel and the twist the servo says it acted on must be the same twist."""
    agreed = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                      our_cmd=(0.50, 0.0, 0.0, 0.0),
                      velocity_target=VelocityTarget(vx=0.50),
                      velocity_received=VelocityTarget(vx=0.50))
    staled = NavFrame(stamp=1.0, x=0.0, y=0.0, yaw=0.0,
                      our_cmd=(0.50, 0.0, 0.0, 0.0),
                      velocity_target=VelocityTarget(vx=0.50),
                      velocity_received=VelocityTarget(vx=0.10))
    assert _drawn(_panel(agreed), _panel(staled))


def test_the_heading_error_is_folded_not_reported_as_a_full_turn():
    """A reference at +179 deg and an aircraft at -179 deg are 2 deg apart.

    Without the fold the row reads -358 deg, which grades red and looks like the
    aircraft is facing backwards at exactly the moments it is not.
    """
    from sparx_agency.tasks.planning.nav_debug.render_lanes import _error
    assert _error("yaw deg", 179.0, -179.0) == pytest.approx(2.0)
    assert _error("yaw deg", -179.0, 179.0) == pytest.approx(-2.0)
    # A linear row must NOT be folded: 179 m of cross-track is 179 m.
    assert _error("x m", 179.0, -179.0) == pytest.approx(-358.0)


# ── the transmitter sticks ───────────────────────────────────────────────────
def _knob(stick):
    """(col, row) of the stick knob's centre.

    The centroid of the knob-coloured pixels, not ``argmax``: the knob is a
    filled disc, and argmax returns its topmost pixel, which sits a radius above
    where the stick actually is.
    """
    from sparx_agency.tasks.planning.hud import gauges
    mask = np.all(stick == np.array(gauges.GAUGE_DOT, np.uint8), axis=2)
    rows, cols = np.nonzero(mask)
    assert rows.size, "no knob drawn"
    return int(round(cols.mean())), int(round(rows.mean()))


def test_the_stick_deflects_the_way_a_pilots_hand_would():
    """The raw ManualControl counts ARE the stick: right is right, up is up.

    The twist adapter has already inverted lateral and yaw by the time these
    counts exist, so a negation here would draw a stick nobody is holding --
    and an inverted stick is the kind of defect that reads as plausible forever.
    """
    from sparx_agency.tasks.planning.hud import gauges
    centre = gauges.STICK_SIZE // 2
    right = _knob(gauges.draw_stick(800, 0, 1000, 1000))
    left = _knob(gauges.draw_stick(-800, 0, 1000, 1000))
    up = _knob(gauges.draw_stick(0, 800, 1000, 1000))
    down = _knob(gauges.draw_stick(0, -800, 1000, 1000))
    assert right[0] > centre and left[0] < centre
    assert up[1] < centre and down[1] > centre     # screen rows grow downward
    # Neutral sits on the crosshair, not off in a corner.
    assert _knob(gauges.draw_stick(0, 0, 1000, 1000)) == (centre, centre)


def test_the_sticks_are_not_drawn_without_counts():
    """A stick resting at centre is a zero command, not an unrecorded one."""
    from sparx_agency.tasks.planning.nav_debug.render_counts import _sticks
    panel = np.full((600, 340, 3), 0, np.uint8)
    assert _sticks(panel, 12, 26, 340, None, GaugeScales()) == 26
    assert _sticks(panel, 12, 26, 340, (500, 0), GaugeScales()) == 26   # short tuple
    assert _sticks(panel, 12, 26, 340, (500, 0, 700, 0), GaugeScales()) > 26


def test_the_throttle_axis_reaches_the_sticks():
    """It has no needle gauge at all, and it is the axis the hold loop writes."""
    from sparx_agency.tasks.planning.hud import gauges
    hover = _knob(gauges.draw_stick(0, 700, 1000, 1000))
    descending = _knob(gauges.draw_stick(0, -700, 1000, 1000))
    assert hover[1] < descending[1]
