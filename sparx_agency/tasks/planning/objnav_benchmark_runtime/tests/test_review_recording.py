"""The log-side review of a recording: every question a frame answers, answered from ``steps.jsonl`` and the journal.

A review pointed at an action must read the log, not a video frame; this
is the tool that makes the log sufficient (2026-10-22). The recording here
is synthetic -- three rows and a two-step map journal -- so every number
the commands print can be checked by hand.
"""
from __future__ import annotations

import json
import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime import review_recording as review
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import JOURNAL_LAYERS, FloorPanels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_floor_panels import panels_with_rooms

OPENING_BASE = 200000


def write_recording(directory):
    """Three actions on a 80 m map: the agent at the origin faces +x; floor ahead, a wall to the right, unknown behind."""
    panels = panels_with_rooms()
    slot = panels.slots[0]
    slot["grid"].fill(-1)
    slot["room_labels"].fill(0)
    for layer in JOURNAL_LAYERS:
        slot["journal"][layer] = []
    res, (ox, oy) = panels.resolution, panels.origin
    gx, gy = int((0.0 - ox) / res), int((0.0 - oy) / res)
    grid = np.full(panels.shape, -1, np.int8)
    grid[gy - 10:gy + 11, gx - 5:gx + 40] = 0                 # a 2 m wide strip of floor, 4 m ahead and 0.5 m behind
    grid[gy - 12:gy - 10, gx - 5:gx + 40] = 100               # a wall along the right-hand side (-y)
    labels = np.where(grid == 0, 1, 0).astype(np.int32)
    FloorPanels._journal(slot, "grid", 0, slot["grid"], grid)
    FloorPanels._journal(slot, "settled", 0, slot["settled"], np.zeros(panels.shape, bool))
    FloorPanels._journal(slot, "room_labels", 0, slot["room_labels"], labels)
    later = grid.copy()
    later[gy + 11:gy + 30, gx:gx + 20] = 0                    # action 2: floor opens up to the left (+y)
    settled = np.zeros(panels.shape, bool)
    settled[gy - 20:gy - 12, gx:gx + 10] = True               # ... and a pocket beyond the wall is settled
    FloorPanels._journal(slot, "grid", 2, slot["grid"], later)
    FloorPanels._journal(slot, "settled", 2, slot["settled"], settled)
    FloorPanels._journal(slot, "room_labels", 2, slot["room_labels"], np.where(later == 0, 1, 0).astype(np.int32))
    np.savez_compressed(directory / "map_journal.npz", **panels.journal_arrays())
    rows = []
    counters = [{"astar": 1, "room_classifier": 0, "room_llm": 1, "rpt": 1}, {"astar": 1, "room_classifier": 1, "room_llm": 1, "rpt": 1},
                {"astar": 2, "room_classifier": 1, "room_llm": 2, "rpt": 2}]
    events = [[{"step": 0, "floor_id": 0, "event": "transit", "room": 0, "goal": [2.5, 0.0], "entry": "frontier"}],
              [{"step": 0, "floor_id": 0, "event": "transit", "room": 0, "goal": [2.5, 0.0], "entry": "frontier"},
               {"step": 1, "floor_id": 0, "event": "reconsider_deferred", "room": 0, "state": "transit"}],
              [{"step": 1, "floor_id": 0, "event": "reconsider_deferred", "room": 0, "state": "transit"},
               {"step": 2, "floor_id": 0, "event": "release", "room": 0, "verdict": "unreachable", "local_steps": 0},
               {"step": 2, "floor_id": 0, "event": "room_rebound", "room": 0, "to": 5, "state": "transit"}]]
    for step in range(3):
        search = {"target": "toilet", "p_present": 0.5, "supervisor_state": "transit", "room_in_force": None, "next_room": 0,
                  "order": [0], "visit": "explore", "local_steps": 0, "local_budget": 60,
                  "accessible_frontiers": [[2.5, 0.0], [1.0, 1.1]],
                  "frontier_goals": [{"xy": [2.5, 0.0], "cell": [gx + 25, gy], "size_cells": 20, "geodesic_m": 2.5,
                                      "heading_error_deg": 0.0, "utility": 1.2, "room": 0},
                                     {"xy": [1.0, 1.1], "cell": [gx + 10, gy + 11], "size_cells": 40, "geodesic_m": 1.5,
                                      "heading_error_deg": 48.0, "utility": 2.5, "room": 0}],
                  "transit": {"room": 0 if step < 2 else 5, "committed_room": 0, "entry": [2.5, 0.0], "from_frontier": True,
                              "room_missing_actions": 0},
                  "openings": [{"id": OPENING_BASE + 3, "kind": "opening", "room_pid": 0, "centroid": [1.0, 1.1],
                                "heading_deg": 90.0, "gain_m2": 12.5, "folded": True}],
                  "rooms": [{"id": 0, "label": "hallway", "prob": 0.5, "frontier_clusters": 2, "searched_s": 3.0,
                             "entry": "frontier", "distance_m": 2.5, "objects": ["chair"], "cooling": False}],
                  "reading": {"home": "bathroom", "house": "one hallway", "stage": "early"},
                  "glance": {"here": {"evaluated_step": step, "left_m2": 4.0, "right_m2": 0.0}}, "events": events[step],
                  "excluded": {} if step < 2 else {"3": "unreachable:2"}}
        rows.append({"step": step, "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "camera_pitch": 0.0},
                     "decision": {"action": "MOVE_FORWARD", "info": {"status": "forward", "heading_error_rad": 0.0,
                                                                     "cross_track_m": 0.0, "remaining_path_m": 2.5 - step}},
                     "method": {"state": "transit", "room_id": 0, "floor_id": 0, "control_counters": counters[step],
                                "reasoning": {"partition_revision": step}, "search": search, "detections": [],
                                "target_closing": {"active": False, "phase": "SEARCH"}},
                     "command": {"waypoints": [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0], [2.5, 0.0]],
                                 "info": {"kind": "transit/0", "phase": "transit/0", "route": "committed_safe_route",
                                          "camera": {"owner": "SEARCH"}}}})
    (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (directory / "metrics.json").write_text(json.dumps({
        "steps": 3, "success": False, "spl": 0.0, "distance_to_goal_m": 4.2, "termination": "step_limit", "wall_s": 6.0,
        "agent_info": {"policy": {"llm_queries": 2, "room_label_queries": 1, "plan_calls": 2,
                                  "route_commitment": {"adoptions": 2},
                                  "room_search_loop": {"stats": {"rooms_rebound": 1, "relabels": 1}, "entry_failures": []},
                                  "exploration_metrics": {"latency": {"scene_graph": {"count": 3, "total_ms": 3000.0, "median_ms": 1000.0,
                                                                                       "p95_ms": 1200.0, "max_ms": 1300.0}},
                                                          "actions_by_phase": {"transit/0": 3}}}}}))
    (directory / "episode.json").write_text(json.dumps({"episode_id": "Synthetic/000001", "target": "toilet"}))
    return rows


def run(directory, *argv):
    """One command's lines, through the module's functions (the CLI is exercised separately)."""
    lines = []
    rows = review.load_steps(directory)
    command = argv[0]
    out = lines.append
    if command == "summary":
        review.summary(directory, rows, out)
    elif command == "timeline":
        review.timeline(rows, out)
    elif command == "step":
        review.step_report(directory, rows, int(argv[1]), out, radius_m=float(argv[2]) if len(argv) > 2 else 6.0)
    elif command == "map":
        review.ascii_map(directory, rows, int(argv[1]), out, radius_m=3.0, cell_m=0.5)
    elif command == "events":
        review.events_command(rows, out, argv[1] if len(argv) > 1 else None)
    return lines


def test_the_summary_names_the_outcome_the_latency_ledger_and_the_release_verdicts(tmp_path):
    write_recording(tmp_path)
    text = "\n".join(run(tmp_path, "summary"))
    assert "episode Synthetic/000001  target toilet  steps 3" in text
    assert "scene_graph        n=3    total     3.0 s  median    1000 ms" in text
    assert "LLM: 2 room-reasoning calls, 1 classifier calls" in text
    assert 'selections 1; releases by verdict: {"unreachable": 1}' in text
    assert "rooms_rebound=1" in text


def test_the_timeline_shows_each_action_once_with_its_new_events_and_counter_deltas(tmp_path):
    write_recording(tmp_path)
    lines = run(tmp_path, "timeline")
    assert len(lines) == 3
    assert "transit(room=0, goal=[2.5, 0.0], entry=frontier)" in lines[0] and "reconsider_deferred" not in lines[0]
    assert "reconsider_deferred(room=0, state=transit)" in lines[1] and "transit(" not in lines[1], "new events only"
    assert "A*+1 room LLM+1 RPT*+1" in lines[2] and "release(room=0, verdict=unreachable, local_steps=0)" in lines[2]
    assert "room_rebound(room=0, to=5, state=transit)" in lines[2] and "rebound->5" in lines[2]
    assert "classifier+1" in lines[1]


def test_the_step_report_reads_the_map_of_that_action_by_sector_from_the_journal(tmp_path):
    write_recording(tmp_path)
    before = "\n".join(run(tmp_path, "step", "0", "3.0"))
    assert "== step 0 ==" in before and "pose (0.00, 0.00) yaw 0°  action MOVE_FORWARD  phase transit/0" in before
    assert "transit: room 0 (committed as 0)  entry [2.5, 0.0]  from_frontier True" in before
    assert "R0   hallway        p=0.50  frontier=2" in before and "LLM reading: home=bathroom" in before
    assert "(no map_journal.npz" not in before, "the journal is there: the sectors read the map of the action"
    ahead = next(line for line in before.splitlines() if line.strip().startswith("ahead"))
    left = next(line for line in before.splitlines() if line.strip().startswith("left"))
    right = next(line for line in before.splitlines() if line.strip().startswith("right"))
    behind = next(line for line in before.splitlines() if line.strip().startswith("behind"))
    assert "frontier goals 1" in ahead and "openings 0" in ahead and "rooms [0]" in ahead
    assert "frontier goals 1" in left and "openings 1" in left
    assert "opening O3 room 0 at [1.0, 1.1] heading 90.0° gain 12.5 m2 folded True" in before
    assert "goal [1.0, 1.1] size_cells=40, geodesic_m=1.5, heading_error_deg=48.0, utility=2.5, room=0" in before
    assert "occupied   0.0 m2" not in right and "unknown" in right, "the wall lies to the right"
    assert float(behind.split("unknown")[1].split("m2")[0]) > float(ahead.split("unknown")[1].split("m2")[0]), (
        "the strip runs ahead; behind the agent the map is unknown")
    after = "\n".join(run(tmp_path, "step", "2", "3.0"))
    left_after = next(line for line in after.splitlines() if line.strip().startswith("left"))
    assert float(left_after.split("unknown")[1].split("m2")[0]) < float(left.split("unknown")[1].split("m2")[0]), (
        "action 2 mapped the floor to the left: less unknown there than at action 0")
    right_after = next(line for line in after.splitlines() if line.strip().startswith("right"))
    assert "(settled   0.0)" not in right_after, "the pocket beyond the wall was settled by action 2"
    assert "excluded rooms: {\"3\": \"unreachable:2\"}" in after
    assert "transit: room 5 (committed as 0)" in after


def test_the_ascii_map_draws_the_action_map_with_the_agent_goals_openings_and_route(tmp_path):
    write_recording(tmp_path)
    lines = run(tmp_path, "map", "0")
    assert lines[0].startswith("map at step 0 round (0.00, 0.00), 3 m radius, 0.5 m per character, heading > (0°)")
    rows = lines[2:]
    assert len(rows) == 13 and all(len(r) == 13 for r in rows), "a 6 m square at 0.5 m per character"
    centre = rows[6]
    assert centre[6] == "@" and "*" in centre and "F" in centre[7:], "the agent, the route ahead, the goal at its end"
    assert any("O" in r for r in rows[:6]), "the opening to the left (north, drawn up)"
    assert any("#" in r for r in rows[8:11]), "the wall to the right (south), 1.0-1.2 m off"
    assert all(set(r) <= set("#. :FO*@") for r in rows)
    later = run(tmp_path, "map", "2")[2:]
    assert sum(r.count(".") for r in later) > sum(r.count(".") for r in rows), "more floor mapped by action 2"
    assert any(":" in r for r in later), "the settled pocket"


def test_events_lists_every_event_once_and_filters_by_name(tmp_path):
    write_recording(tmp_path)
    everything = run(tmp_path, "events")
    assert [line.split()[0] for line in everything] == ["0", "1", "2", "2"]
    assert run(tmp_path, "events", "release") == ["   2 release(room=0, verdict=unreachable, local_steps=0)"]


def test_a_recording_without_a_journal_falls_back_to_the_final_map_and_says_so(tmp_path):
    write_recording(tmp_path)
    (tmp_path / "map_journal.npz").unlink()
    panels = panels_with_rooms()
    panels.save(tmp_path)
    (tmp_path / "map_journal.npz").unlink()
    (tmp_path / "floor_panels.json").write_text(json.dumps([{"floor_id": 0, "origin": list(panels.origin),
                                                             "resolution_m": panels.resolution}]))
    text = "\n".join(run(tmp_path, "step", "1", "3.0"))
    assert "(no map_journal.npz: the sectors below read the FINAL map, not the map at this action)" in text
    assert "map by sector within 3 m" in text


def test_the_cli_entry_point_runs_each_command(tmp_path, capsys):
    write_recording(tmp_path)
    for argv in (["summary"], ["timeline", "--start", "1", "--end", "2"], ["step", "1", "--radius", "2"],
                 ["map", "1", "--radius", "2", "--cell", "0.5"], ["events", "--name", "transit"]):
        assert review.main([str(tmp_path)] + argv) == 0
        assert capsys.readouterr().out.strip()


def test_the_panels_journal_the_real_policys_map_on_capture_and_the_snapshot_carries_the_inventory(tmp_path):
    """The journal is written from the policy's own maps on every capture; the replay of any step equals the
    world the policy planned on at that step, and the step snapshot names every frontier goal with its account."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import replay_journal
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import search_snapshot
    policy, episode = setup_policy(label="bed")               # not the target: the search owns every action
    panels = FloorPanels(1, policy.settings, observation(episode, 0).pose)
    grids = {}
    for step in range(3):
        obs = observation(episode, step, depth=3.0 - 0.5 * step)
        policy.plan(obs)
        panels.capture(policy, obs)
        grids[step] = policy.mapping.worlds[policy.mapping.floor_id].grid.copy()
        snapshot = search_snapshot(policy)
        assert "frontier_goals" in snapshot and "transit" in snapshot and "entry_failures" in snapshot
        for goal in snapshot["frontier_goals"]:
            assert set(goal) == {"xy", "cell", "size_cells", "geodesic_m", "heading_error_deg", "utility", "room"}
        assert [g["xy"] for g in snapshot["frontier_goals"]] == [[round(v, 3) for v in xy] for xy in snapshot["accessible_frontiers"]]
    panels.save(tmp_path)
    journal = dict(np.load(tmp_path / "map_journal.npz", allow_pickle=False))
    assert journal["floor_ids"].tolist() == [policy.mapping.floor_id]
    for step, grid in grids.items():
        np.testing.assert_array_equal(replay_journal(journal, 0, "grid", step, panels.shape, -1), grid.astype(np.int8))
    assert (replay_journal(journal, 0, "grid", 2, panels.shape, -1) >= 0).sum() >= (
        replay_journal(journal, 0, "grid", 0, panels.shape, -1) >= 0).sum(), "the map grows"
    labels = replay_journal(journal, 0, "room_labels", 2, panels.shape, 0)
    np.testing.assert_array_equal(labels, panels.slots[0]["room_labels"])
    assert set(np.load(tmp_path / "floor_maps.npz").files) == {"slot_0"}, "the final maps are still written"
