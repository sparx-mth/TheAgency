"""The offline replay of a recording through the frontier-view selector: the nodes, the score-once rule, the numbers and the sheets.

A synthetic recording with a frame log beside it -- eleven actions of an
agent turning in place in a 5 x 4 m room -- is built so that every answer
follows from the plan by hand:

* a **north opening** is on the map from the start, out of the first
  frame's picture: its node waits one action for the frame that shows it;
* a **strip of unknown along the south wall**, a cell deep with the wall
  behind it: a boundary nothing can be seen PAST. The first frame has its
  far end in a corner of the picture -- enough to be scored on at once, and
  a fifth of what the last frame of the turn shows of it;
* an **east doorway** opens on action 6, when the agent faces west: the
  frames of actions 0 and 1 looked down it, the first with it in the
  middle -- so the selector with a frame log scores it at once on the frame
  of action 0, and one with the action's own frame never does;
* the corridor behind that doorway is **mapped further on action 9**: the
  space behind the node, later, which the frame of action 0 looked straight
  into;
* a **window in the west wall**: floor up to it and unknown beyond that the
  search has settled (looked through, nothing came back). It is no boundary
  to anywhere and must not become a node.

A real MP3D recording with its frame log is replayed too when one is on
this machine (``FRONTIER_VIEW_REPLAY_RECORDING`` and ``..._FRAMES``).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frame_log.disk_store import DiskFrameStore, episode_directory_name
from sparx_agency.core.planning.object_search.frame_log.episode_frame_log import EpisodeFrameLog
from sparx_agency.core.planning.object_search.frontier_views.selection import NOT_FRAMED
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec, NAVIGATION_ACTIONS
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_frames as frames_tool
from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_hindsight as hindsight
from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_metrics as metrics
from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_replay as replay_tool
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import JOURNAL_LAYERS, FloorPanels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_floor_panels import panels_with_rooms

EPISODE_ID = "synthetic-room/000004"
KEY = hashlib.sha256(EPISODE_ID.encode("utf-8")).hexdigest()[:12]
CAMERA = CameraSpec(intrinsics_from_hfov(160, 120, 79.0), 0.88, 0.5, 5.0)
STEPS = 11
HERE = (1.05, 0.05)
NORTH, SOUTH, EAST = "north opening", "south strip", "east doorway"


def pose(step):
    """The agent stands still and turns 30 degrees left per action, from facing east."""
    return AgentPose(HERE[0], HERE[1], 0.0, math.radians(30.0 * step), 0.0)


def write_run(root):
    """The recording (``root/recordings/<key>``) and the frame log (``root/frames``) of the module docstring's episode."""
    panels = panels_with_rooms()
    slot = panels.slots[0]
    slot["grid"].fill(-1)
    slot["room_labels"].fill(0)
    for layer in JOURNAL_LAYERS:
        slot["journal"][layer] = []
    res, (ox, oy) = panels.resolution, panels.origin

    def box(x0, y0, x1, y1):
        centres_x = (np.arange(panels.shape[1]) + 0.5) * res + ox
        centres_y = (np.arange(panels.shape[0]) + 0.5) * res + oy
        cols = np.nonzero((centres_x >= x0) & (centres_x < x1))[0]
        rows = np.nonzero((centres_y >= y0) & (centres_y < y1))[0]
        return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)

    settled = np.zeros(panels.shape, bool)
    settled[box(-0.8, 0.8, -0.1, 1.6)] = True                 # beyond the west window: looked through, settled

    def journal(step, grid):
        FloorPanels._journal(slot, "grid", step, slot["grid"], grid)
        FloorPanels._journal(slot, "settled", step, slot["settled"], settled)
        FloorPanels._journal(slot, "room_labels", step, slot["room_labels"], np.where(grid == 0, 1, 0).astype(np.int32))

    first = np.full(panels.shape, -1, np.int8)
    first[box(-0.1, -2.1, 5.1, 2.1)] = 100                    # the room's walls ...
    first[box(0.0, -2.0, 5.0, 2.0)] = 0                       # ... round its floor
    first[box(2.0, 2.0, 2.8, 2.9)] = 0                        # the north opening and the floor seen through it
    first[box(2.0, -2.0, 4.0, -1.9)] = -1                     # a cell-deep strip along the south wall never seen
    first[box(-0.1, 0.8, 0.0, 1.6)] = 0                       # the west window's sill: floor, unknown beyond
    journal(0, first)
    opened = first.copy()
    opened[box(5.0, -0.4, 6.0, 0.4)] = 0                      # action 6: the east doorway, and a metre of corridor
    journal(6, opened)
    further = opened.copy()
    further[box(6.0, -0.4, 7.5, 0.4)] = 0                     # action 9: the corridor a metre and a half further
    journal(9, further)
    recording = root / "recordings" / KEY
    recording.mkdir(parents=True)
    np.savez_compressed(recording / "map_journal.npz", **panels.journal_arrays())
    rows = []
    for step in range(STEPS):
        p = pose(step)
        rows.append({"step": step, "pose": {"x": p.x, "y": p.y, "z": p.z, "yaw": p.yaw, "camera_pitch": p.camera_pitch},
                     "decision": {"action": "TURN_LEFT", "info": {}}, "method": {"floor_id": 0, "search": {}},
                     "command": None})
    (recording / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (recording / "episode.json").write_text(json.dumps({"episode_id": EPISODE_ID, "scene": "synthetic-room", "target": "bed"}))
    episode = ObjNavEpisode(EPISODE_ID, "synthetic-room", "synthetic", "val", "bed", CAMERA,
                            DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0, tilt_angle_deg=30.0,
                                               actions=NAVIGATION_ACTIONS), 500)
    log = EpisodeFrameLog(DiskFrameStore(root / "frames"))
    log.begin(episode)
    k = CAMERA.intrinsics
    for step in range(STEPS):
        rgb = np.full((k.height, k.width, 3), 40 + 15 * step, np.uint8)
        log.add(ObjNavObservation(rgb, np.full((k.height, k.width), 2.0, np.float32), pose(step), CAMERA, "bed", step))
    return recording, root / "frames"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("frontier_view_run")
    recording, frames = write_run(root)
    replayed = replay_tool.replay(recording, frames)
    hindsight.add_hindsight(replayed)
    hindsight.add_patience(replayed)
    return recording, frames, replayed


def named(replayed):
    """``{name: node id}`` of the episode's three boundaries, told apart by where their goal point lies."""
    names = {}
    for node_id, (_, _, node, _) in replayed.last_nodes.items():
        x, y = node.xy
        names[NORTH if y > 2.0 else SOUTH if y < -1.0 else EAST] = node_id
    return names


def scoring(replayed, variant, node_id):
    found = [event for event in replayed.events[variant] if event.node.node_id == node_id]
    assert len(found) <= 1, "a node is scored once"
    return found[0] if found else None


def test_the_frame_log_is_the_recordings_own_and_supplies_the_camera(run):
    recording, frames, replayed = run
    assert episode_directory_name(EPISODE_ID).endswith("-" + KEY)
    assert replayed.frames.source == frames_tool.FROM_LOG == "frame log"
    assert replayed.frames.directory == frames / episode_directory_name(EPISODE_ID)
    assert [record.step for record in replayed.frames.records] == list(range(STEPS))
    assert replayed.frames.camera == CAMERA
    assert [record.step for record in replayed.frames.until(3)] == [0, 1, 2, 3]
    assert [record.step for record in replayed.frames.at(3)] == [3] and replayed.frames.at(99) == []
    # One episode's own folder is accepted in place of the root.
    assert frames_tool.frame_folder(frames / episode_directory_name(EPISODE_ID), KEY) == replayed.frames.directory
    assert frames_tool.frame_folder(None, KEY) is None and frames_tool.frame_folder(frames, "0" * 12) is None


def test_the_three_boundaries_are_three_nodes_with_ids_that_stay(run):
    _, _, replayed = run
    ids = named(replayed)
    assert set(ids) == {NORTH, SOUTH, EAST} and len(replayed.ledgers[replay_tool.SELECTED].nodes) == 3
    # (The window in the west wall borders unknown and is none of them: what lies beyond it is settled.)
    by_step = {step["step"]: set(step["nodes"]) for step in replayed.steps}
    assert all(by_step[step] == {ids[NORTH], ids[SOUTH]} for step in range(6))
    assert all(by_step[step] == {ids[NORTH], ids[SOUTH], ids[EAST]} for step in range(6, STEPS)), (
        "the east doorway keeps its id when the corridor behind it is mapped a metre and a half further")
    last_step, floor_id, east, _ = replayed.last_nodes[ids[EAST]]
    # The corridor's boundary on the last action: 24 cells along each side (the doorway's own cell has the wall
    # beside it, not the unknown) and 8 across its end, the two corners counted once.
    assert (last_step, floor_id, east.size_cells) == (STEPS - 1, 0, 24 + 24 + 8 - 2)


def test_a_node_is_scored_once_on_the_first_action_a_frame_shows_it(run):
    _, _, replayed = run
    ids = named(replayed)
    north = scoring(replayed, replay_tool.SELECTED, ids[NORTH])
    # Facing east the north opening is out of the picture; one turn of 30 degrees brings it in.
    assert (north.first_step, north.step, north.views.names) == (0, 1, ("step_000001.jpg",))
    assert north.views.through_fraction > 0.3, "across the opening the line runs on into the unknown"
    south = scoring(replayed, replay_tool.SELECTED, ids[SOUTH])
    # The strip's far end is in the corner of the first frame: five of its 22 cells, from 3.4 m.
    assert (south.first_step, south.step, south.views.names) == (0, 0, ("step_000000.jpg",))
    assert south.views.seen_fraction == pytest.approx(5 / 22) and south.views.views[0].range_m == pytest.approx(3.4, abs=0.1)
    assert south.views.through_fraction == 0.0, "a cell of unknown with the wall behind it is not seen past"
    east = scoring(replayed, replay_tool.SELECTED, ids[EAST])
    # The doorway opens while the agent faces west. Two old frames looked down it from this very spot, worth the
    # same: the one of action 0 has it in the middle of the picture.
    assert (east.first_step, east.step, east.views.names) == (6, 6, ("step_000000.jpg",))
    assert east.views.seen_fraction == 1.0 and east.views.through_fraction == pytest.approx(8 / 24)
    assert east.views.views[0].range_m == pytest.approx(4.6, abs=0.3)
    ledger = replayed.ledgers[replay_tool.SELECTED]
    assert ledger.counts()["scored"] == 3 and ledger.nodes[ids[EAST]].frames == east.views.names
    assert [event.step for event in replayed.events[replay_tool.SELECTED]] == [0, 1, 6]


def test_the_strict_rule_and_the_frame_of_the_action_alone_each_lose_a_node(run):
    _, _, replayed = run
    ids = named(replayed)
    strict, current = replay_tool.THROUGH_ONLY, replay_tool.CURRENT_ONLY
    assert scoring(replayed, strict, ids[SOUTH]) is None and scoring(replayed, strict, ids[EAST]).step == 6
    assert scoring(replayed, strict, ids[NORTH]).step == 1
    # Without a frame log the east doorway is never in the picture: the agent does not face it again.
    assert scoring(replayed, current, ids[EAST]) is None
    assert replayed.last_views[current][ids[EAST]].reason == NOT_FRAMED
    assert scoring(replayed, current, ids[NORTH]).step == 1 and scoring(replayed, current, ids[SOUTH]).step == 0
    assert all(event.views.names == ("step_%06d.jpg" % event.step,) for event in replayed.events[current])


def test_in_hindsight_the_old_frame_showed_the_corridor_that_was_mapped_later(run):
    _, _, replayed = run
    ids = named(replayed)
    east = scoring(replayed, replay_tool.SELECTED, ids[EAST])
    # Mapped on action 9: the corridor from 6.0 to 7.5 m, 0.8 m wide -- 120 cells, all behind the node of action 6.
    assert east.hindsight["revealed_cells"] == 120
    assert east.hindsight["shown"] == 1.0 == east.hindsight["available"] == east.hindsight["best_frame"]
    assert east.hindsight["current_frame"] == 0.0, "the frame of action 6 faces the other way"
    north = scoring(replayed, replay_tool.SELECTED, ids[NORTH])
    assert north.hindsight["revealed_cells"] == 0 and "shown" not in north.hindsight
    assert north.hindsight["behind_cells"] > 500, "two metres of unknown round the opening, none of it mapped later"
    south = scoring(replayed, replay_tool.SELECTED, ids[SOUTH])
    assert south.hindsight["behind_cells"] == 20, "the strip itself: the wall behind it bounds what lies behind"


def test_what_waiting_would_have_bought_is_measured_on_each_nodes_last_action(run):
    _, _, replayed = run
    ids = named(replayed)
    south = scoring(replayed, replay_tool.SELECTED, ids[SOUTH])
    # Scored on a corner of the first frame; the last frame of the turn faces the strip.
    assert south.later["step"] == STEPS - 1 and south.later["usable"]
    assert south.later["score"] > 4 * south.views.score and south.later["range_m"] < south.views.views[0].range_m
    north = scoring(replayed, replay_tool.SELECTED, ids[NORTH])
    assert north.later["score"] >= north.views.score, "a later frame may only add to what the first showed"
    east = scoring(replayed, replay_tool.SELECTED, ids[EAST])
    # The doorway's node has grown with the corridor: the old frame shows all of it, from farther.
    assert east.later["usable"] and east.later["range_m"] > east.views.views[0].range_m
    # Without the frame log nothing shows the north opening on the last action: the agent has turned away.
    assert not scoring(replayed, replay_tool.CURRENT_ONLY, ids[NORTH]).later["usable"]
    selected = metrics.summarise(replayed)["variants"][replay_tool.SELECTED]["on_its_last_action"]
    assert selected["scorings"] == 3 and selected["no_view_by_then"] == 0
    assert selected["score_doubled_share"] == pytest.approx(1 / 3, abs=1e-3)


def test_the_summary_counts_calls_coverage_and_reasons_and_is_json(run):
    _, _, replayed = run
    result = metrics.summarise(replayed)
    json.dumps(result)
    assert result["steps"] == STEPS and result["frames"] == {"count": STEPS, "source": "frame log"}
    assert result["inventory"] == {"compared": 0, "as_recorded": 0}, "this recording lists no frontier goals to compare with"
    assert result["nodes"]["ids"] == 3 and result["nodes"]["per_action_max"] == 3
    selected, strict, current = (result["variants"][name] for name in (
        replay_tool.SELECTED, replay_tool.THROUGH_ONLY, replay_tool.CURRENT_ONLY))
    assert (selected["calls"], strict["calls"], current["calls"]) == (3, 2, 2)
    assert selected["scored_on_appearance"] == 2 and selected["scored_later"] == 1 and selected["never_scored"] == 0
    assert selected["median_wait_steps"] == 1.0 and selected["scored_share"] == 1.0
    assert selected["calls_per_100_steps"] == pytest.approx(100.0 * 3 / STEPS, abs=0.1)
    # The strip and the doorway were both asked about with the frame of action 0.
    assert selected["distinct_pictures"] == 2 and selected["frames_per_call"] == {"1": 3}
    # Node-actions: north 11, south 11, east 5. With a score in hand: north from action 1 (10), south 11, east 5.
    assert selected["node_actions_with_a_score"] == pytest.approx(26 / 27, abs=1e-3)
    assert selected["calls_on_nodes_gone_soon"] == 0 and selected["blocked"] == 0
    assert current["never_scored_long_lived"] == 1 and current["never_scored_reasons"] == {NOT_FRAMED: 1}
    assert current["node_actions_with_a_score"] == pytest.approx(21 / 27, abs=1e-3)
    assert strict["never_scored_long_lived"] == 1 and strict["looked_past_share"] == 1.0
    assert selected["looked_past_share"] == pytest.approx(2 / 3, abs=1e-3)
    assert selected["hindsight"]["scorings"] == 1 and selected["hindsight"]["too_little_mapped_later"] == 2
    assert selected["hindsight"]["shown_of_available"] == 1.0 and selected["hindsight"]["nothing_available_share"] == 0.0
    assert [row["step"] for row in result["scorings"]] == [0, 1, 6]
    assert result["scorings"][2]["views"][0]["age_steps"] == 6 and result["scorings"][2]["wait_steps"] == 0
    assert result["scorings"][2]["hindsight"]["revealed_cells"] == 120 and "cells" not in result["scorings"][2]["hindsight"]
    assert result["scorings"][0]["on_its_last_action"]["step"] == STEPS - 1
    text = metrics.report(result)
    for needle in ("nodes scored (= model calls)", "3 of 3", "2 of 3", "node-actions with a score in hand",
                   "in hindsight", "had the node been scored on its LAST action",
                   "never scored, long-lived, current_frame: 1 in no frame's picture"):
        assert needle in text, needle
    pooled = metrics.pooled([result, result])
    assert pooled["steps"] == 2 * STEPS and pooled["variants"][replay_tool.SELECTED]["calls"] == 6
    assert pooled["variants"][replay_tool.SELECTED]["frames_per_call"] == {"1": 6}, "values are pooled, not replaced"
    assert pooled["variants"][replay_tool.SELECTED]["hindsight"]["scorings"] == 2
    assert pooled["variants"][replay_tool.SELECTED]["node_actions_with_a_score"] == selected["node_actions_with_a_score"]
    assert pooled["variants"][replay_tool.SELECTED]["distinct_pictures"] == 4
    assert pooled["variants"][replay_tool.CURRENT_ONLY]["never_scored_reasons"] == {NOT_FRAMED: 2}
    assert pooled["nodes"]["ids"] == 6 and pooled["nodes"]["per_action_mean"] == result["nodes"]["per_action_mean"]
    assert "2 episodes pooled" in metrics.report(pooled)


def test_the_rebuilt_inventory_is_compared_with_the_recordings_own_where_that_is_of_the_same_action(run, tmp_path):
    """The nodes are only as true as the inventory they are cut from: the replay says how often it rebuilt the policy's."""
    recording, frames, replayed = run
    rows = [json.loads(line) for line in (recording / "steps.jsonl").read_text().splitlines()]
    planner = WeightedAStarPlanner2D(RPTSettings().planner_params())
    built = {}
    for row in rows:
        inventory = replay_tool.inventory_at(replayed.maps, planner, row["step"], 0, frames_tool.pose_of(row))[2]
        built[row["step"]] = [{"cell": list(goal.cell), "size_cells": goal.size_cells} for goal in inventory.goals]
    assert [len(built[step]) for step in (0, 6, 9)] == [2, 3, 3]
    # What a policy records: the goals of the inventory it last built and the action it built it on -- here on
    # actions 0, 6 and 9, with one goal of action 9 written a cell off.
    for row in rows:
        last = max(step for step in (0, 6, 9) if step <= row["step"])
        goals = [dict(goal) for goal in built[last]]
        if row["step"] == 9:
            goals[0]["cell"] = [goals[0]["cell"][0] + 1, goals[0]["cell"][1]]
        row["method"]["search"] = {"frontier_goals": goals, "frontier_inventory_step": last}
    copy = tmp_path / recording.name
    copy.mkdir()
    for path in recording.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    (copy / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    again = replay_tool.replay(copy, frames)
    # Between its rebuilds the recording's snapshot is older than the map: nothing is compared there.
    assert [step["as_recorded"] for step in again.steps] == [True] + [None] * 5 + [True, None, None, False, None]
    result = metrics.summarise(again)
    assert result["inventory"] == {"compared": 3, "as_recorded": 2}
    assert "frontier inventory rebuilt = the recorded one on 2 of the 3 actions" in metrics.report(result)
    assert metrics.pooled([result, result])["inventory"] == {"compared": 6, "as_recorded": 4}
    # A goal is its cluster: the same cell with another size is another inventory.
    step_0 = dict(rows[0])
    inventory = replay_tool.inventory_at(replayed.maps, planner, 0, 0, frames_tool.pose_of(step_0))[2]
    assert replay_tool.same_as_recorded(step_0, 0, inventory) is True
    step_0["method"] = {"search": {"frontier_inventory_step": 0, "frontier_goals": [
        dict(goal, size_cells=goal["size_cells"] + 1) for goal in built[0]]}}
    assert replay_tool.same_as_recorded(step_0, 0, inventory) is False
    assert replay_tool.same_as_recorded({"method": {"search": {"frontier_inventory_step": 0}}}, 0, inventory) is None
    assert replay_tool.same_as_recorded({}, 0, inventory) is None


def test_the_selector_is_handed_the_map_the_settled_unknown_and_the_storey_of_the_action(run, monkeypatch):
    recording, frames, _ = run
    calls = []
    real = replay_tool.select_views

    def spy(world, nodes, records, camera, settings=None, settled=None, reference_z=None):
        calls.append((world, [record.step for record in records], camera, settled, reference_z))
        return real(world, nodes, records, camera, settings, settled, reference_z=reference_z)

    monkeypatch.setattr(replay_tool, "select_views", spy)
    replay_tool.replay(recording, frames, variants=[replay_tool.Variant("only", ViewSettings())])
    # North waits one action for its view; the doorway is new on action 6: three calls in all.
    assert [steps for _, steps, _, _, _ in calls] == [[0], [0, 1], list(range(7))]
    for world, _, camera, settled, reference_z in calls:
        assert camera == CAMERA and reference_z == 0.0, "the height of the floor under the agent on that action"
        assert settled is not None and settled.shape == world.grid.shape and settled.sum() == 7 * 8, (
            "the unknown beyond the west window, as the journal has it")
    assert (calls[0][0].grid == calls[1][0].grid).all() and (calls[2][0].grid != calls[1][0].grid).any(), (
        "the map of the action, not of the first")


def test_without_a_frame_log_the_frames_are_the_recorded_poses(run, monkeypatch):
    recording, _, replayed = run
    monkeypatch.setattr(frames_tool, "benchmark_camera", lambda name: CAMERA)
    bare = replay_tool.replay(recording)
    assert bare.frames.source.startswith("recorded poses") and bare.frames.directory is None
    assert [record.name for record in bare.frames.records] == [record.name for record in replayed.frames.records]
    assert ([(event.step, event.node.node_id, event.views.names) for event in bare.events[replay_tool.SELECTED]]
            == [(event.step, event.node.node_id, event.views.names) for event in replayed.events[replay_tool.SELECTED]])


def test_an_age_keeps_the_scoring_back_and_the_frame_log_is_what_still_shows_the_node(run):
    recording, frames, _ = run
    aged = replay_tool.replay(recording, frames, min_age_steps=3)
    ids = named(aged)
    east = scoring(aged, replay_tool.SELECTED, ids[EAST])
    assert (east.first_step, east.step, east.views.names) == (6, 9, ("step_000000.jpg",))
    # At three actions old the north opening is offered the whole turn so far: the frame of action 2 faces it.
    north = scoring(aged, replay_tool.SELECTED, ids[NORTH])
    assert (north.step, north.views.names[0]) == (3, "step_000002.jpg")
    # The frame of the action alone has the opening at the edge of its picture by then, and never the doorway.
    alone = scoring(aged, replay_tool.CURRENT_ONLY, ids[NORTH])
    assert (alone.step, alone.views.names) == (3, ("step_000003.jpg",)) and alone.views.score < north.views.score
    assert scoring(aged, replay_tool.CURRENT_ONLY, ids[EAST]) is None
    assert aged.ledgers[replay_tool.SELECTED].nodes[ids[EAST]].visited_step is None
    # The agent is "at" a node within visit_m of its goal point: the strip's is 2.6 m away.
    near = replay_tool.replay(recording, frames, visit_m=3.0)
    assert near.ledgers[replay_tool.SELECTED].nodes[named(near)[SOUTH]].visited_step == 0
    assert near.ledgers[replay_tool.SELECTED].nodes[named(near)[EAST]].visited_step is None


def test_a_recording_without_a_journal_and_a_frame_log_of_another_run_are_refused(run, tmp_path):
    recording, frames, _ = run
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("steps.jsonl", "episode.json"):
        (bare / name).write_bytes((recording / name).read_bytes())
    np.savez_compressed(bare / "floor_maps.npz", slot_0=np.zeros((4, 4), np.int8))
    np.savez_compressed(bare / "room_partitions.npz", slot_0=np.zeros((4, 4), np.int32))
    with pytest.raises(SystemExit, match="map_journal"):
        replay_tool.replay(bare)
    with pytest.raises(SystemExit, match="no frame log"):
        replay_tool.replay(recording, tmp_path)
    other = tmp_path / "other" / recording.name
    other.mkdir(parents=True)
    for path in recording.iterdir():
        (other / path.name).write_bytes(path.read_bytes())
    rows = [json.loads(line) for line in (other / "steps.jsonl").read_text().splitlines()]
    rows[4]["pose"]["x"] += 0.25
    (other / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(SystemExit, match="not of this recording"):
        replay_tool.replay(other, frames)


def test_the_command_line_prints_the_report_writes_the_json_and_the_sheets(run, tmp_path, capsys):
    cv2 = pytest.importorskip("cv2")
    recording, frames, _ = run
    out = tmp_path / "out" / "replay.json"
    assert replay_tool.main([str(recording), "--frames", str(frames), "--json", str(out), "--sheets", str(tmp_path / "sheets"),
                             "--at-weight", "0.4", "--min-cells", "5"]) == 0
    printed = capsys.readouterr().out
    assert "nodes scored (= model calls)" in printed and "contact sheets under" in printed
    saved = json.loads(out.read_text())
    assert len(saved["episodes"]) == 1 and saved["pooled"] is None
    assert saved["episodes"][0]["variants"][replay_tool.SELECTED]["about"].endswith("(at_weight 0.40)")
    sheets = sorted((tmp_path / "sheets" / recording.name).glob("*.jpg"))
    assert [path.name.split("_node_")[0] for path in sheets] == ["step_0000", "step_0001", "step_0006"]
    k = CAMERA.intrinsics
    for path in sheets:
        image = cv2.imread(str(path))
        assert image is not None and image.shape[1] >= 520 + int(k.width * 2 / 3) and image.shape[0] > 520
    with pytest.raises(SystemExit, match="--frames"):
        replay_tool.main([str(recording), "--sheets", str(tmp_path / "nowhere")])


def test_a_node_never_scored_gets_a_sheet_of_the_frames_that_came_closest(run, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_sheets as sheets
    recording, frames, _ = run
    strict = replay_tool.replay(recording, frames, view_settings=ViewSettings(at_weight=0.0))
    written = sheets.write_sheets(strict, tmp_path, limit=1)
    names = sorted(path.name for path in written)
    assert sum(name.startswith("step_") for name in names) == 1, "the limit thins the scorings"
    missed = [name for name in names if name.startswith("unscored_")]
    assert len(missed) == 1 and missed[0].endswith("step_%04d.jpg" % (STEPS - 1))
    assert cv2.imread(str(tmp_path / missed[0])).shape[1] >= 1000
    bare = replay_tool.replay(recording)
    with pytest.raises(SystemExit, match="frame log"):
        sheets.write_sheets(bare, tmp_path / "none")


@pytest.mark.skipif(not os.environ.get("FRONTIER_VIEW_REPLAY_RECORDING"),
                    reason="set FRONTIER_VIEW_REPLAY_RECORDING (and ..._FRAMES) to replay a real recording")
def test_a_real_recording_replays_and_every_scoring_names_frames_that_exist():
    recording = Path(os.environ["FRONTIER_VIEW_REPLAY_RECORDING"])
    replayed = replay_tool.replay(recording, os.environ.get("FRONTIER_VIEW_REPLAY_FRAMES") or None)
    names = {record.name: record.step for record in replayed.frames.records}
    events = replayed.events[replay_tool.SELECTED]
    assert events, "a real episode has frontier nodes with views"
    for event in events:
        assert 1 <= len(event.views.views) <= ViewSettings().max_frames
        for view in event.views.views:
            assert names[view.name] == view.step <= event.step
    assert len({event.node.node_id for event in events}) == len(events)
    json.dumps(metrics.summarise(replayed))
