"""A Sphera run, end to end: two recorders, two clocks, one replayable timeline.

The Sphera recording is written by two processes that cannot see each other's
ROS graph -- a ROS1 recorder in the ``falcon`` container and a ROS2 recorder in
``it`` -- so the thing most worth pinning down is the join between them. These
tests build a synthetic run in :mod:`.schema`'s layout with a deliberate skew
between the two ROS clocks and assert that the host wall clock still lands the
right actuator/truth row on the right frame, that every lane reaches the
:class:`NavFrame`, and that a run missing any lane still loads and renders.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sparx_agency.tasks.planning.nav_debug import schema
from sparx_agency.tasks.planning.nav_debug.render import render
from sparx_agency.tasks.planning.nav_debug.session import NavSession

# The two recorders run in different containers on different ROS clocks; only
# the host wall clock is shared, so the fixture skews them apart on purpose.
_WALL0 = 1788335937.0
_ROS1_SKEW = -0.75      # ROS1 t = wall + this
_ROS2_SKEW = +12.5      # ROS2 t = wall + this (a wildly different epoch)
_N = 40
_DT = 0.05              # 20 Hz spine


def _write(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _wall(i):
    return _WALL0 + i * _DT


def _build_run(tmp_path, *, ros2=True, lanes=True):
    """Write a synthetic Sphera run folder and return its path."""
    run = tmp_path / "nav_debug_20260902_120000"
    run.mkdir()

    # The spine: telemetry.jsonl (no certainty CSV exists on this path).
    _write(run / schema.TELEMETRY_FILE, [
        schema.row(_wall(i) + _ROS1_SKEW, _wall(i),
                   x=float(i), y=2.0, z=1.5, yaw=0.1,
                   vx=0.5, vy=0.0, vz=0.0, wz=0.0)
        for i in range(_N)])

    if lanes:
        _write(run / schema.REFERENCE_FILE, [
            schema.row(_wall(i) + _ROS1_SKEW, _wall(i),
                       x=float(i) + 0.4, y=2.0, z=1.5, yaw=0.1,
                       vx=0.6, vy=0.0, vz=0.0, yaw_dot=0.0,
                       age_s=0.02, traj_id=7, moving=True)
            for i in range(_N)])
        _write(run / schema.CONTROL_FILE, [
            schema.row(_wall(i) + _ROS1_SKEW, _wall(i),
                       tracking={"position_error_m": 0.4,
                                 "along_track_lag_m": 0.35,
                                 "cross_track_error_m": 0.12,
                                 "yaw_error_rad": 0.03,
                                 "diverged": False, "holding": False,
                                 "reference_age_s": 0.02},
                       terms={"feed_forward": [0.6, 0.0, 0.0],
                              "damping": [-0.05, 0.0, 0.0],
                              "correction": [0.08, 0.0, 0.0],
                              "commanded": [0.63, 0.0, 0.0],
                              "clamped": [0.63, 0.0, 0.0],
                              "smoothed": [0.60, 0.0, 0.0],
                              "limits": ["speed"]},
                       gate={"reason": "", "published": True,
                             "demo_mode": "exploring"})
            for i in range(_N)])
        _write(run / schema.MAPPING_FILE, [
            schema.row(_wall(i) + _ROS1_SKEW, _wall(i),
                       pose_age_s=0.08, occupied_cells=120,
                       free_cells=900, unknown_cells=40)
            for i in range(0, _N, 10)])

    if ros2:
        r2 = run / schema.ROS2_DIR
        r2.mkdir()
        # Same instants, a completely different ROS epoch.
        _write(r2 / schema.ACTUATOR_FILE, [
            schema.row(_wall(i) + _ROS2_SKEW, _wall(i),
                       cmd_nav=[500.0, 0.0, 0.0],
                       manual=[500.0, 0.0, 700.0, 0.0],
                       buttons=0, cmd_nav_age_s=0.01, manual_age_s=0.01)
            for i in range(_N)])
        _write(r2 / schema.TRUTH_FILE, [
            schema.row(_wall(i) + _ROS2_SKEW, _wall(i),
                       vx=0.55, vy=0.01, vz=0.0, roll=0.02, pitch=-0.03,
                       battery_pct=88.0, armed=True, flight_mode="OFFBOARD")
            for i in range(_N)])
        _write(r2 / schema.ALTITUDE_FILE, [
            schema.row(_wall(i) + _ROS2_SKEW, _wall(i),
                       target_m=1.6, ranger_m=1.55, error_m=0.05,
                       wanted_z=720.0, sent_z=700.0, nudge_m=0.0,
                       at_ceiling=False, guard_rejected=False,
                       guard_rejects_total=3)
            for i in range(_N)])
        _write(r2 / schema.AXIS_TRACE_FILE, [
            schema.row(_wall(i) + _ROS2_SKEW, _wall(i),
                       axes=[{"name": "forward", "requested": 0.6,
                              "measured": 0.55, "error": 0.05,
                              "feed_forward": 480.0, "integral": 12.0,
                              "correction": 20.0, "pre_slew": 500.0,
                              "counts": 500.0, "saturated": False,
                              "slew_limited": False, "capped": False,
                              "feedback_stale": False},
                             {"name": "yaw", "requested": 0.0, "measured": 0.0,
                              "error": 0.0, "feed_forward": 0.0,
                              "integral": 0.0, "correction": 0.0,
                              "pre_slew": 0.0, "counts": 0.0,
                              "saturated": False, "slew_limited": False,
                              "capped": False, "feedback_stale": False}])
            for i in range(_N)])

    # A BEV snapshot so the map pane has something to draw.
    bev = run / schema.BEV_DIR
    bev.mkdir()
    grid = np.full((20, 20), -1, np.int8)
    grid[5:8, 5:15] = 100
    ms = int(_wall(0) * 1000)
    np.save(str(bev / ("%d.npy" % ms)), grid)
    with open(bev / ("%d.json" % ms), "w") as fh:
        json.dump({"t": _wall(0) + _ROS1_SKEW, "resolution": 0.5,
                   "origin_x": -5.0, "origin_y": -5.0, "frame_id": "world"}, fh)

    routes = run / schema.ROUTES_DIR
    routes.mkdir()
    with open(routes / ("%d.json" % ms), "w") as fh:
        json.dump({"t": _wall(0) + _ROS1_SKEW,
                   "final": [[0.0, 2.0], [10.0, 2.0]],
                   "executed": [[0.0, 2.0], [3.0, 2.0]],
                   "goal": [10.0, 2.0]}, fh)

    with open(run / schema.MANIFEST_FILE, "w") as fh:
        json.dump({"created_wall": "2026-09-02 12:00:00", "frame_id": "world"}, fh)
    return str(run)


@pytest.fixture()
def sphera_run(tmp_path):
    return _build_run(tmp_path)


def test_timeline_comes_from_telemetry_without_a_csv(sphera_run):
    """Sphera writes no certainty CSV, so telemetry.jsonl is the spine."""
    session = NavSession(sphera_run)
    assert len(session) == _N
    assert session.csv_path is None


def test_every_lane_reaches_the_frame(sphera_run):
    session = NavSession(sphera_run)
    frame = session.build(_N // 2)

    assert frame.reference is not None and frame.reference.moving
    assert frame.tracking is not None
    assert frame.tracking.cross_track_error_m == pytest.approx(0.12)
    assert frame.terms is not None
    assert frame.terms.feed_forward == pytest.approx((0.6, 0.0, 0.0))
    assert frame.map_stats is not None and frame.map_stats.occupied_cells == 120
    # These four come from the OTHER recorder, on the other clock.
    assert frame.actuator is not None and frame.actuator.manual[2] == pytest.approx(700.0)
    assert frame.truth is not None and frame.truth.battery_pct == pytest.approx(88.0)
    assert frame.altitude is not None and frame.altitude.guard_rejects_total == 3
    assert [a.name for a in frame.axes] == ["forward", "yaw"]
    assert frame.bev is not None
    assert frame.routes.final and frame.routes.executed


def test_ros2_lanes_join_on_wall_clock_despite_a_skewed_ros_clock(sphera_run):
    """The ROS2 recorder's epoch is 13 s off; the wall clock must still align it.

    A join on ROS time would land ~13 s away -- past every freshness window --
    and blank the actuator/truth panels, so this is the regression that matters.
    """
    session = NavSession(sphera_run)
    assert abs(session.clock.offset - (-_ROS1_SKEW)) < 0.05
    assert abs(session.ros2_clock.offset - (-_ROS2_SKEW)) < 0.05
    # Every frame resolves the cross-recorder lanes, not just the middle one.
    for i in (0, _N // 3, _N - 1):
        assert NavSession(sphera_run).build(i).truth is not None


def test_truth_speed_beats_commanded_speed_for_the_history_strip(sphera_run):
    """Achieved speed must come from truth when it exists, not from the command."""
    session = NavSession(sphera_run)
    frame = session.build(_N - 1)
    assert frame.speed_history
    assert frame.speed_history[-1] == pytest.approx(0.55, abs=0.02)
    assert frame.err_history[-1] == pytest.approx(0.4)


def test_renders_a_sphera_frame(sphera_run):
    img = render(NavSession(sphera_run).build(_N // 2), None, 400)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def test_run_without_the_ros2_recorder_still_loads_and_renders(tmp_path):
    """Only half the rig ran. That must degrade, not raise."""
    run = _build_run(tmp_path, ros2=False)
    session = NavSession(run)
    frame = session.build(_N // 2)
    assert frame.tracking is not None          # ROS1 lanes survive
    assert frame.actuator is None and frame.truth is None
    assert render(frame, None, 400).ndim == 3


def test_bare_run_with_only_telemetry_still_loads_and_renders(tmp_path):
    """The shape of all 1111 recordings made before this schema existed."""
    run = _build_run(tmp_path, ros2=False, lanes=False)
    session = NavSession(run)
    frame = session.build(0)
    assert frame.reference is None and frame.tracking is None
    assert render(frame, None, 400).ndim == 3


def test_ros2_lanes_can_live_outside_the_run_folder(tmp_path):
    """The real topology: `it` cannot see the FALCON log dir, so the halves split.

    The ROS2 recorder writes under its own container's workspace bind. Rather
    than require a copy before replay, the loader takes that directory directly.
    """
    run = _build_run(tmp_path)
    moved = tmp_path / "elsewhere" / "ros2"
    moved.parent.mkdir()
    os.rename(os.path.join(run, schema.ROS2_DIR), moved)

    # Without pointing at it, the cross-recorder lanes are simply absent.
    assert NavSession(run).build(_N // 2).truth is None
    # Pointed at it, they join exactly as if they had been collected in.
    frame = NavSession(run, None, str(moved)).build(_N // 2)
    assert frame.truth is not None and frame.truth.battery_pct == pytest.approx(88.0)
    assert frame.actuator is not None and len(frame.axes) == 2


# ── regressions from the review pass ─────────────────────────────────────────

def test_explicit_null_section_does_not_fabricate_a_perfect_tracking_panel():
    """`"tracking": null` means the tracker did not run -- not "zero error".

    The follower writes an explicit null on every tick it returned early (muted
    demo mode, no reference, tilt cut). Falling back to the whole row there built
    a Tracking of all-zero defaults, i.e. a panel reading "perfect tracking" at
    exactly the moments the aircraft was not being flown.
    """
    from sparx_agency.tasks.planning.nav_debug import records
    muted = {"t": 1.0, "wall": 1.0, "tracking": None, "terms": None,
             "gate": {"reason": "demo_mode", "published": False}}
    assert records.tracking(muted) is None
    assert records.control_terms(muted) is None
    # A real section is still read, and a flat lane still works.
    assert records.tracking({"t": 1.0, "wall": 1.0,
                             "tracking": {"position_error_m": 0.4}}).position_error_m == 0.4
    assert records.truth({"t": 1.0, "wall": 1.0, "vx": 0.5}).vx == 0.5


def test_altitude_reason_survives_into_the_frame():
    """A skipped hold tick must be distinguishable from a healthy one.

    On every reason but ``held`` the loop never computed error/wanted_z/sent_z,
    so without the reason the lane reads like a hold sitting at zero error.
    """
    from sparx_agency.tasks.planning.nav_debug import records
    alt = records.altitude({"t": 1.0, "wall": 1.0, "reason": "guard_rejected",
                            "altitude": {"target_m": 1.6, "ranger_m": 1.5,
                                         "error_m": None, "guard_rejected": True,
                                         "guard_rejects_total": 7}})
    assert alt.reason == "guard_rejected"
    assert alt.error_m is None and alt.guard_rejects_total == 7


def test_gauge_scales_do_not_flip_when_the_axis_trace_momentarily_blanks(sphera_run):
    """The envelope is a property of the run; the axis lane blanks routinely.

    The Rooster and XTEND envelopes differ by ~3.5x, so flipping between them
    mid-replay makes every command gauge lie.
    """
    import dataclasses
    from sparx_agency.tasks.planning.nav_debug.render import ROOSTER_SCALES, default_scales

    session = NavSession(sphera_run)
    frame = session.build(_N // 2)
    assert default_scales(frame) == ROOSTER_SCALES
    # The adapter publishes an empty trace while stopped; other Rooster lanes remain.
    assert default_scales(dataclasses.replace(frame, axes=[])) == ROOSTER_SCALES

    # And the player resolves one envelope for the whole run, once.
    from sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug import resolve_scales
    assert resolve_scales(session) == ROOSTER_SCALES


def test_export_pads_rather_than_squashes_a_narrower_frame():
    """The screen grows a lane column only on frames that have lane data.

    Rescaling a narrower frame up to the canvas would change the pixel scale of
    the map and every gauge from frame to frame.
    """
    from sparx_agency.tasks.planning.nav_debug.run_folder_nav_debug import _fit
    small = np.full((10, 20, 3), 7, np.uint8)
    fitted = _fit(small, (30, 12))
    assert fitted.shape == (12, 30, 3)
    assert (fitted[:10, :20] == 7).all()      # content preserved, not stretched
    assert (fitted[10:, :] == 0).all()        # padded, not resampled
