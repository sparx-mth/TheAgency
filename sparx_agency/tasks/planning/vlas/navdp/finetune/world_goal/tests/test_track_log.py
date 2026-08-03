"""Recording what the policy proposed, so a map panel can be drawn later."""
import json
import math

import numpy as np

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.track_log import TrackLog


def test_a_body_frame_plan_is_stored_in_world_coordinates():
    """The panel is drawn on the map, so the rotation has to happen at capture
    time -- the pose is gone by the time anything reads the log."""
    log = TrackLog(goal_xy=(10.0, 0.0), start_xy=(0.0, 0.0))
    straight_ahead = np.array([[1.0, 0.0], [2.0, 0.0]])

    log.add(1.0, (5.0, 5.0, math.pi / 2), straight_ahead, (5.0, 6.0))

    # Facing +y at (5, 5): one metre "forward" is (5, 6), two is (5, 7).
    assert log.entries[0]["traj"] == [[5.0, 6.0], [5.0, 7.0]]


def test_a_dropped_inference_is_recorded_rather_than_skipped():
    """A transport failure is part of the story: the panel says so instead of
    silently repeating the previous plan."""
    log = TrackLog(goal_xy=(1.0, 1.0), start_xy=(0.0, 0.0))

    log.add(2.0, (0.0, 0.0, 0.0), None, (0.0, 0.0))

    assert len(log.entries) == 1
    assert "traj" not in log.entries[0]


def test_the_flown_path_is_subsampled():
    """The control loop appends at 250 Hz; a minute of that is 15,000 points
    two millimetres apart, which no panel can draw and no one should store."""
    log = TrackLog(goal_xy=(0.0, 0.0), start_xy=(0.0, 0.0))

    log.set_flown([(float(i), 0.0) for i in range(100)], stride=10)

    assert len(log.flown) == 10
    assert log.flown[1] == [10.0, 0.0]


def test_round_trip_through_disk(tmp_path):
    log = TrackLog(goal_xy=(3.0, 4.0), start_xy=(0.0, 0.0))
    log.add(0.5, (0.0, 0.0, 0.0), np.array([[1.0, 0.0]]), (1.0, 0.0))
    log.set_flown([(0.0, 0.0), (1.0, 0.0)], stride=1)
    path = tmp_path / "mission_00_track.json"

    log.write(path, extra={"scene": "warehouse_shelves", "arm": "trained"})
    restored = TrackLog.read(path)

    assert restored["scene"] == "warehouse_shelves"
    assert restored["goal_xy"] == [3.0, 4.0]
    assert restored["flown"] == [[0.0, 0.0], [1.0, 0.0]]
    assert restored["inferences"][0]["traj"] == [[1.0, 0.0]]


def test_flight_metadata_cannot_overwrite_the_trajectories(tmp_path):
    """The bug that destroyed a whole run's plans.

    A flight result dict has its own ``inferences`` key holding the *count*.
    Merged over the log it replaced the list of proposed trajectories with an
    integer, and since the plans exist nowhere else, the only way to get them
    back was to fly everything again.
    """
    log = TrackLog(goal_xy=(1.0, 1.0), start_xy=(0.0, 0.0))
    log.add(0.0, (0.0, 0.0, 0.0), np.array([[1.0, 0.0]]), (1.0, 0.0))
    path = tmp_path / "track.json"

    log.write(path, extra={"inferences": 240, "goal_xy": [9.0, 9.0],
                           "arm": "trained"})
    restored = TrackLog.read(path)

    assert isinstance(restored["inferences"], list)
    assert restored["inferences"][0]["traj"] == [[1.0, 0.0]]
    assert restored["goal_xy"] == [1.0, 1.0]      # the log's own, not the result's
    assert restored["arm"] == "trained"           # non-colliding keys still land
    assert restored["inference_count"] == 1


def test_the_log_stays_small(tmp_path):
    """It is written next to gigabytes of imagery and must never be the reason
    a recording is dropped."""
    log = TrackLog(goal_xy=(0.0, 0.0), start_xy=(0.0, 0.0))
    plan = np.zeros((24, 2))
    for step in range(240):                      # a minute at the inference rate
        log.add(step * 0.25, (0.0, 0.0, 0.0), plan, (0.0, 0.0))
    path = tmp_path / "track.json"
    log.write(path)

    assert path.stat().st_size < 1_000_000
