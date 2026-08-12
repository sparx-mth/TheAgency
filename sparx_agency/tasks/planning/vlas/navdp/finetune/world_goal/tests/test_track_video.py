"""Placing a flight's map panel on the flight's own clock.

The panel used to draw one frame per inference and index the flown path by
*fraction of the flight*. Those are different timelines -- the aircraft is
recorded from the ground up while inference only starts at cruise altitude -- so
the trail sat metres from the aircraft marker and the whole clip ran ahead of the
camera it was stacked against. It was read as the aircraft being mislocalised. It
was not; these keep it that way.

No matplotlib and no ffmpeg here: only the frame plan, which is where the bug was.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import track_video

# A flight recorded from t=50 s, 10 s of climb, then inference at 4 Hz. The shape
# of every real flight, and the shape the old code got wrong.
FLOWN_DT = 0.04
CLIMB_S = 10.0
START_S = 50.0


def _log(flown_samples: int = 900, inferences: int = 66, timing: bool = True) -> dict:
    """A track log of a flight that climbs before it infers anything."""
    log = {
        "goal_xy": [10.0, 10.0],
        "start_xy": [0.0, 0.0],
        "flown": [[index * 0.01, 0.0] for index in range(flown_samples)],
        "inferences": [{"t": START_S + CLIMB_S + step * 0.25,
                        "pose": [0.0, 0.0, 0.0],
                        "traj": [[1.0, 0.0]]}
                       for step in range(inferences)],
    }
    if timing:
        log.update({"schema": 2, "started_s": START_S, "flown_dt": FLOWN_DT})
    return log


# --- the timeline -----------------------------------------------------------

def test_the_timeline_starts_when_the_recording_did():
    times = track_video.timeline(_log(), fps=10.0)
    assert times[0] == pytest.approx(START_S)


def test_the_timeline_covers_the_whole_flight_not_just_the_inferences():
    """The old panel began at the first inference, ten seconds in."""
    log = _log()
    times = track_video.timeline(log, fps=10.0)
    span = times[-1] - times[0]
    assert span == pytest.approx((len(log["flown"]) - 1) * FLOWN_DT, abs=0.1)
    assert span > CLIMB_S + 1.0


def test_the_panel_is_the_same_length_as_the_camera_clip():
    """Both are one frame per capture over the same flight, which is what lets
    compare_videos stack them without one running ahead."""
    log = _log()
    fps = 10.0
    frames = len(track_video.timeline(log, fps=fps))
    camera_frames = round((len(log["flown"]) - 1) * FLOWN_DT * fps) + 1
    assert frames == camera_frames


def test_a_higher_frame_rate_gives_more_frames_over_the_same_span():
    slow = track_video.timeline(_log(), fps=10.0)
    fast = track_video.timeline(_log(), fps=25.0)
    assert len(fast) > len(slow)
    assert fast[-1] == pytest.approx(slow[-1], abs=0.1)


def test_a_log_with_no_flown_path_has_no_timeline():
    log = _log()
    log["flown"] = []
    assert track_video.timeline(log, fps=10.0) == []


# --- which inference a frame shows ------------------------------------------

def test_nothing_is_shown_before_the_first_inference():
    """During the climb the policy has not been asked anything, and saying so is
    the honest frame -- not the first plan drawn ten seconds early."""
    entries = _log()["inferences"]
    assert track_video.latest_inference(entries, START_S + 1.0) is None


def test_the_most_recent_inference_is_shown():
    entries = _log()["inferences"]
    found = track_video.latest_inference(entries, START_S + CLIMB_S + 0.6)
    assert found is entries[2]        # 0.00, 0.25, 0.50 have happened; 0.75 has not


def test_the_last_inference_holds_to_the_end_of_the_flight():
    entries = _log()["inferences"]
    assert track_video.latest_inference(entries, 1e6) is entries[-1]


# --- the frame plan ---------------------------------------------------------

def test_the_trail_ends_at_the_aircraft():
    """The property that was violated: at every frame, the last flown sample drawn
    is the aircraft's position at that instant."""
    log = _log()
    for when, upto, _ in track_video._frame_plan(log, fps=10.0):
        expected = min(len(log["flown"]),
                       int((when - START_S) / FLOWN_DT) + 1)
        assert upto == expected


def test_the_trail_is_short_while_the_aircraft_is_still_climbing():
    """The old fraction mapping drew a tenth of the whole flight on the first
    frame, which is why the video opened with the aircraft already displaced."""
    plan = track_video._frame_plan(_log(), fps=10.0)
    _, upto, entry = plan[0]
    assert upto == 1
    assert entry is None


def test_the_trail_never_runs_past_the_end_of_the_flown_path():
    log = _log()
    for _, upto, _ in track_video._frame_plan(log, fps=10.0):
        assert 0 < upto <= len(log["flown"])


def test_a_log_without_timing_still_renders_one_frame_per_inference():
    """Old logs have no clock. The shape is right and the timing is not, and
    render_frames says so rather than refusing."""
    log = _log(timing=False)
    plan = track_video._frame_plan(log, fps=10.0)
    assert len(plan) == len(log["inferences"])


def test_timing_is_detected_from_the_log():
    assert track_video.has_timing(_log())
    assert not track_video.has_timing(_log(timing=False))


# --- the committed prefix ---------------------------------------------------

def _proposal(points: int = 24, commit: int = 12) -> dict:
    """One inference entry with a straight proposal and a commitment on it."""
    entry = {"t": 0.0, "pose": [0.0, 0.0, 0.0],
             "traj": [[step * 0.2, 0.0] for step in range(1, points + 1)]}
    if commit is not None:
        entry["commit"] = commit
    return entry


def test_the_committed_prefix_is_the_first_commit_waypoints():
    """The anchor pose is not in traj, so commit=12 means traj[:12]."""
    held = track_video.committed_prefix(_proposal(24, 12))
    assert held.shape == (12, 2)
    assert held[-1][0] == pytest.approx(2.4)


def test_a_log_from_before_commitments_promises_nothing():
    """Those flights really did treat every waypoint as provisional."""
    assert track_video.committed_prefix(_proposal(24, commit=None)) is None


def test_an_empty_commitment_draws_nothing():
    assert track_video.committed_prefix(_proposal(24, commit=0)) is None


def test_a_dropped_inference_has_no_committed_prefix():
    assert track_video.committed_prefix({"t": 0.0, "commit": 12}) is None


# --- "the policy declined to move" is a result, not a blank panel -----------

def test_a_stacked_trajectory_is_recognised_as_a_stop():
    """What the pretrained checkpoint emits on most frames: 24 waypoints, all
    on the same point. Drawn as a route it is an invisible dot."""
    entry = {"t": 0.0, "pose": [1.0, 2.0, 0.0],
             "traj": [[1.0, 2.0]] * 24, "commit": 12}
    assert track_video.is_stop(entry)


def test_a_real_route_is_not_a_stop():
    assert not track_video.is_stop(_proposal(24, 12))


def test_a_dropped_inference_is_not_a_stop():
    """No trajectory at all is a transport failure, which says something else."""
    assert not track_video.is_stop({"t": 0.0, "pose": [0.0, 0.0, 0.0]})
