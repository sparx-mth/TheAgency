from sparx_agency.tasks.planning.objnav_benchmark_runtime.trace_tools.trace_summary import (
    outcome_of, render_report, section_stop_rooms, summarise_episode, takeover_windows)

COUNTERS = dict(memory_sight_refusals=0)


def row(step, active=False, locked=False, boxes=(), xy=None, best=0.0, release=None, phase="SEARCH", room=None):
    candidate_room = None if room is None else {"room": 1, "label": room[0], "strength": room[1]}
    return {"step": step, "phase": phase, "active": active, "locked": locked, "best_conf": best, "candidate_xy": xy, "last_release": release,
            "target_boxes": [{"cls": "toilet", "conf": 0.9, "clipped": clipped} for clipped in boxes], "counters": dict(COUNTERS),
            "candidate_room": candidate_room}


def record(success=False, stop=False, dtg=5.0, phases=None):
    return {"success": success, "stop_called": stop, "steps": 80, "distance_to_goal_m": dtg, "spl": 0.0, "path_length_m": 12.0,
            "shortest_path_m": 4.0, "target_category": "toilet",
            "agent_info": {"policy": {"exploration_metrics": {"actions_by_phase": phases or {}}}}}


def takeover(xy=(1.0, 1.0)):
    return [row(10, True, boxes=(False,), xy=xy, best=0.6, phase="VERIFY"), row(11, True, boxes=(False,), xy=xy, best=0.7, phase="VERIFY"),
            row(12, True, xy=xy, phase="VERIFY"), row(13, True, boxes=(True,), xy=xy, phase="VERIFY"),
            row(14, release="unverified", phase="RELEASED")]


def test_a_takeover_window_counts_hit_frames_and_the_longest_consecutive_run():
    ((window, following),) = takeover_windows(takeover())
    assert [r["step"] for r in window] == [10, 11, 12, 13] and following["last_release"] == "unverified"


def test_a_summary_describes_each_window_and_whether_it_sat_on_a_goal():
    summary = summarise_episode("ep", record(), takeover(), poses=[(0, 0)], goals=[(1.5, 1.0)])
    (window,) = summary["windows"]
    assert (window["steps"], window["hit_frames"], window["longest_run"], window["clipped_hit_frames"]) == (4, 3, 2, 1)
    assert window["on_real_goal"] and not window["locked"] and window["ended_by"] == "unverified" and window["max_conf"] == 0.7
    far = summarise_episode("ep", record(), takeover(), poses=[(0, 0)], goals=[(9.0, 9.0)])
    assert not far["windows"][0]["on_real_goal"]


def test_frames_with_a_target_box_outside_a_takeover_are_split_into_border_dropped_and_seen_not_taken():
    rows = [row(1, boxes=(True,)), row(2, boxes=(True, True)), row(3, boxes=(False,)), row(4, boxes=(True, False))]
    summary = summarise_episode("ep", record(), rows, poses=[(0, 0)], goals=[(5.0, 5.0)])
    assert (summary["frames_dropped_by_border"], summary["frames_seen_not_taken"]) == (2, 2)


def test_the_outcome_separates_timeouts_near_misses_and_wrong_stops():
    goals = [(1.0, 1.0)]
    assert outcome_of(record(success=True, stop=True), (9, 9), goals) == "success"
    assert outcome_of(record(), (9, 9), goals) == "timeout"
    assert outcome_of(record(stop=True), (1.5, 1.0), goals) == "near_miss_stop"
    assert outcome_of(record(stop=True), (9.0, 9.0), goals) == "wrong_stop"


def test_the_summary_keeps_the_closest_approach_the_escape_steps_and_the_memory_refusals():
    rows = [row(1)] + [dict(row(2), counters=dict(memory_sight_refusals=31))]
    summary = summarise_episode("ep", record(phases={"escape": 120, "frontier": 5}), rows, poses=[(4, 0), (1, 0)], goals=[(1.0, 1.0)])
    assert summary["closest_approach_m"] == 1.0 and summary["escape_steps"] == 120 and summary["memory_sight_refusals"] == 31
    assert summary["path_ratio"] == 3.0


def test_numbered_transit_phases_are_reported_as_one_phase():
    summary = summarise_episode("ep", record(success=True, stop=True, phases={"transit/0": 30, "transit/7": 30, "frontier": 40}), [], [(0, 0)], [(1.0, 1.0)])
    text = render_report([summary])
    assert "transit: 60% of actions" in text and "transit/0" not in text


def test_the_summary_names_the_room_of_the_last_candidate():
    rows = [row(10, True, room=("kitchen", "strong")), row(11, True, room=("kitchen", "weak")), row(12, True)]
    summary = summarise_episode("ep", record(stop=True), rows, [(0, 0)], [(9.0, 9.0)])
    assert summary["stop_room"] == {"label": "kitchen", "strength": "weak"}
    assert summarise_episode("ep", record(), [row(1)], [(0, 0)], [(9.0, 9.0)])["stop_room"] is None


def test_a_row_without_the_room_fields_from_an_older_trace_is_accepted():
    old = {key: value for key, value in row(1).items() if key != "candidate_room"}
    assert summarise_episode("ep", record(), [old], [(0, 0)], [(9.0, 9.0)])["stop_room"] is None


def stop_summary(outcome, label):
    return {"outcome": outcome, "stop_room": None if label is None else {"label": label, "strength": "strong"}}


def test_the_stop_room_section_counts_the_room_labels_of_false_stops_and_successes_for_the_model_replay():
    summaries = [stop_summary("wrong_stop", "kitchen"), stop_summary("wrong_stop", "kitchen"), stop_summary("wrong_stop", None),
                 stop_summary("success", "bathroom")]
    text = "\n".join(section_stop_rooms(summaries))
    assert "false stops: 3 (2 with a room label, 1 without)" in text and "successes: 1 (1 with a room label, 0 without)" in text
    assert "'kitchen': 2" in text and "'bathroom': 1" in text


def test_the_report_has_a_section_for_each_test():
    summary = summarise_episode("ep", record(), takeover(), poses=[(0, 0)], goals=[(1.5, 1.0)])
    text = render_report([summary])
    for heading in ("T1", "T2", "T3", "T4", "T5", "T8"):
        assert heading in text
    assert "ep" in text
