"""Loading a run folder + certainty CSV into a frame timeline with as-of joins."""
import csv
import json
import os

import numpy as np
import pytest

from sparx_agency.tasks.planning.nav_debug.session import NavSession, classify_event

_CSV_COLS = [
    "wall_clock", "ros_stamp", "pos_x", "pos_y", "pos_z", "yaw_deg",
    "confidence", "pos_std_m", "cmd_effectiveness", "coasting", "age_s",
    "target_wp_idx", "num_waypoints", "target_x", "target_y",
    "drift_vx", "drift_vy", "drift_wz", "cross_track_m", "along_track_m",
    "heading_err_deg", "effort", "speed_scale", "lead_s", "deadband_extra_m",
    "authority", "state", "blocked_axis", "escape_state",
    "cmd_vx", "cmd_vy", "cmd_wz",
    "axis_forward", "axis_lateral", "axis_vertical", "axis_yaw",
]


def _csv_row(t, x, y, wp, **kw):
    row = {c: "" for c in _CSV_COLS}
    row.update(ros_stamp="%.3f" % t, pos_x="%.3f" % x, pos_y="%.3f" % y,
               pos_z="1.00", yaw_deg="90.0", confidence="0.62", pos_std_m="0.18",
               cmd_effectiveness="0.8", coasting="False", age_s="0.30",
               target_wp_idx=str(wp), num_waypoints="5", target_x="2.0",
               target_y="0.0", drift_vy="0.05", authority="holding roll",
               state="TRACK", escape_state="IDLE", blocked_axis="",
               cmd_vx="0.30", cmd_vy="0.05", cmd_wz="-0.20",
               axis_forward="400", axis_lateral="-80", axis_vertical="60",
               axis_yaw="320")
    row.update(kw)
    return row


def _make_run(tmp_path, with_csv=True):
    run = tmp_path / "nav_debug_run"
    (run / "bev").mkdir(parents=True)
    (run / "routes").mkdir()

    grid = np.full((20, 30), -1, np.int8)
    grid[0, :] = 100
    for t_ms in (1000, 1050):                         # two snapshots to test as-of
        np.save(run / "bev" / ("%d.npy" % t_ms), grid)
        (run / "bev" / ("%d.json" % t_ms)).write_text(json.dumps(
            {"t": t_ms / 1000.0, "resolution": 0.1, "origin_x": -1.0,
             "origin_y": -1.0, "frame_id": "world"}))

    (run / "routes" / "1000.json").write_text(json.dumps({
        "t": 1.0, "astar": [[0, 0], [1, 0]], "safe": [[0, 0], [1, 0]],
        "final": [[0, 0], [1, 0], [2, 0]], "goal": [2, 0], "lookahead": [0.5, 0]}))
    (run / "events.jsonl").write_text(json.dumps(
        {"t": 0.9, "kind": None, "text": "REPLAN: rotated 34 deg"}) + "\n")
    (run / "telemetry.jsonl").write_text(
        json.dumps({"t": 1.0, "x": 0.0, "y": 0.0, "z": 1.0, "yaw": 1.57,
                    "vx": 0.3, "vy": 0.05, "vz": 0.08, "wz": -0.2}) + "\n"
        + json.dumps({"t": 1.1, "x": 0.1, "y": 0.0, "z": 1.05, "yaw": 1.57,
                      "vx": 0.3, "vy": 0.05, "vz": 0.10, "wz": -0.2}) + "\n")

    manifest = {"bev": {"resolution": 0.1, "origin_x": -1.0, "origin_y": -1.0}}
    if with_csv:
        csv_path = run / "certainty_20260721.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_COLS)
            w.writeheader()
            w.writerow(_csv_row(1.0, 0.0, 0.0, 2))
            w.writerow(_csv_row(1.1, 0.1, 0.0, 3))
        manifest["certainty_csv"] = str(csv_path)
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def test_classify_event_buckets():
    assert classify_event("REPLAN: rotated 34 deg") == "rotation"
    assert classify_event("periodic 7.0s check") == "time"
    assert classify_event("REPLAN: obstacle on route -> new route") == "obstacle"
    assert classify_event("BLOCKAGE: unseen obstacle at (1,2) -> reroute") == "blockage"
    assert classify_event("STOP: boxed in - no A* route") == "boxed_in"


def test_loads_csv_frames_and_asof_joins(tmp_path):
    run = _make_run(tmp_path)
    s = NavSession(str(run))
    assert len(s) == 2

    fr = s.build(1)                      # t = 1.1
    assert fr.x == pytest.approx(0.1) and fr.yaw == pytest.approx(90.0 * np.pi / 180)
    # OURS command: vx/vy/wz from the CSV, vz backfilled from telemetry.jsonl.
    assert fr.our_cmd[0] == pytest.approx(0.3)
    assert fr.our_cmd[2] == pytest.approx(0.10)
    # TO DRONE command from the axis_* columns.
    assert fr.drone_cmd == (400, -80, 60, 320)
    # As-of BEV is the 1.05 snapshot (latest <= 1.1), routes the 1.0 snapshot.
    assert fr.bev is not None and fr.bev.stamp == pytest.approx(1.05)
    assert fr.routes.final == [(0, 0), (1, 0), (2, 0)]
    # The rotation replan (t=0.9) is still within the banner window at t=1.1.
    assert fr.replan is not None and fr.replan.kind == "rotation"
    assert fr.replan.age_s == pytest.approx(0.2, abs=1e-6)
    # Advanced flag fires because wp_idx grew 2 -> 3 between the two rows.
    assert fr.advanced is True
    assert fr.quality is not None and fr.drift is not None
    assert "roll" in fr.why


def test_telemetry_only_when_no_csv(tmp_path):
    run = _make_run(tmp_path, with_csv=False)
    s = NavSession(str(run))
    assert len(s) == 2                   # falls back to telemetry.jsonl
    fr = s.build(0)
    assert fr.our_cmd is not None and fr.quality is None   # no rich fields w/o CSV
    assert fr.bev is not None


def test_missing_run_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        NavSession(str(empty))
