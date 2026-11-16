"""The offline replay of a recording through the room extents: the observed record from the poses, the extents from the journal.

A synthetic recording -- a room that is first seen from its door, then
walked into and scanned, then split in two by a cut and merged back under
the old number -- checks what the tool promises for a real one: the
coverage follows the floor across the partition's changes, a size called
final stays final, a lower bound turns final when the far wall is seen,
and the checks classify a merge as a merge. A real HM3D recording is
replayed too when one is on this machine.

The second half is the live side of the same reading: the scene graph turns a
room of the map into the ``size=`` and ``covered=`` the oracle is shown, the
policy reads the extents on every action, and what it holds is exactly what
the occupancy grid, the partition cut out of it and the sight ledger give
when read again -- the size has no other source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime import room_extent_replay as replay_tool
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import JOURNAL_LAYERS, FloorPanels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_floor_panels import panels_with_rooms


def write_recording(directory):
    """Six actions in a 5 x 4 m room entered through a door in its -x wall.

    Action 0: the agent stands in the doorway facing +x; the near 2.5 m of the room is mapped, the far wall is not.
    Action 1: the agent has walked to the middle; the whole room is mapped, every wall seen.
    Actions 2-4: the agent turns in place (the observed record grows).
    Action 5: a door cut splits the room into two numbers (1 keeps the west half, 7 takes the east).
    Action 6: the halves merge back under number 1.
    """
    panels = panels_with_rooms()
    slot = panels.slots[0]
    slot["grid"].fill(-1)
    slot["room_labels"].fill(0)
    for layer in JOURNAL_LAYERS:
        slot["journal"][layer] = []
    res, (ox, oy) = panels.resolution, panels.origin
    gx, gy = int((0.0 - ox) / res), int((0.0 - oy) / res)           # the doorway cell: x = 0, y = 0
    x0, x1, y0, y1 = gx, gx + 50, gy - 20, gy + 20                      # the room: x in [0, 5], y in [-2, 2]

    def frame(step, grid, labels):
        FloorPanels._journal(slot, "grid", step, slot["grid"], grid)
        FloorPanels._journal(slot, "settled", step, slot["settled"], np.zeros(panels.shape, bool))
        FloorPanels._journal(slot, "room_labels", step, slot["room_labels"], labels)

    first = np.full(panels.shape, -1, np.int8)
    first[y0:y1, x0:x0 + 25] = 0                                        # the near half of the floor
    first[y0 - 1, x0 - 1:x0 + 25] = 100                                 # the side walls as far as seen
    first[y1, x0 - 1:x0 + 25] = 100
    first[y0 - 1:y1 + 1, x0 - 1] = 100                                  # the door wall ...
    first[gy - 4:gy + 5, x0 - 1] = 0                                    # ... with the doorway open behind the agent
    first[gy - 4:gy + 5, x0 - 6:x0 - 1] = 0                             # the hallway floor the agent came from
    labels = np.where(first == 0, 2, 0).astype(np.int32)                # room number 1 (label 2)
    labels[gy - 4:gy + 5, x0 - 6:x0 - 1] = 4                            # the hallway is room 3
    frame(0, first, labels)
    whole = first.copy()
    whole[y0:y1, x0:x1] = 0
    whole[y0 - 1, x0 - 1:x1 + 1] = 100
    whole[y1, x0 - 1:x1 + 1] = 100
    whole[y0 - 1:y1 + 1, x1] = 100                                      # the far wall
    labels_whole = np.where(whole == 0, 2, 0).astype(np.int32)
    labels_whole[gy - 4:gy + 5, x0 - 6:x0 - 1] = 4
    frame(1, whole, labels_whole)
    split = labels_whole.copy()
    split[y0:y1, x0 + 25:x1] = 8                                        # the east half becomes room 7 (label 8)
    frame(5, whole, split)
    frame(6, whole, labels_whole)
    np.savez_compressed(directory / "map_journal.npz", **panels.journal_arrays())
    poses = [(-0.3, 0.0, 0.0), (2.5, 0.0, 0.0), (2.5, 0.0, 1.57), (2.5, 0.0, 3.14), (2.5, 0.0, -1.57), (2.5, 0.0, 0.0), (2.5, 0.0, 0.0)]
    rows = []
    for step, (x, y, yaw) in enumerate(poses):
        rows.append({"step": step, "pose": {"x": x, "y": y, "z": 0.0, "yaw": yaw, "camera_pitch": 0.0},
                     "decision": {"action": "MOVE_FORWARD", "info": {}},
                     "method": {"state": "transit", "room_id": 1, "floor_id": 0,
                                "search": {"excluded": {} if step < 3 else {"1": "scanned:scan_point_inside"}}},
                     "command": None})
    (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (directory / "episode.json").write_text(json.dumps({"episode_id": "00800-synthetic/000000", "scene": "00800-synthetic",
                                                         "target": "toilet"}))


def test_the_replay_reads_the_room_as_first_seen_then_scanned_then_split_and_merged(tmp_path):
    write_recording(tmp_path)
    result = replay_tool.replay(tmp_path)
    assert result["camera"] == "hm3d" and len(result["steps"]) == 7
    by_step = {record["step"]: record["rooms"] for record in result["steps"]}
    door = by_step[0][1]
    assert not door["bounded"] and door["walls_seen"] == 3, "the far wall is beyond the first frame"
    assert door["area_m2"] == pytest.approx(10.0, abs=0.2) and door["observed_fraction"] > 0.6
    assert door["covered_fraction"] == pytest.approx(door["covered_m2"] / door["area_m2"], abs=0.01), (
        "the covered share is of the mapped footprint: an upper bound while the far wall is unseen")
    inside = by_step[1][1]
    assert inside["bounded"] and inside["walls_seen"] == 4 and inside["area_m2"] == pytest.approx(20.0, abs=0.2)
    assert by_step[4][1]["observed_fraction"] > by_step[1][1]["observed_fraction"], "the turn in place observes more"
    assert by_step[4][1]["observed_fraction"] >= 0.9, "a full turn in the middle of a 5 x 4 m room sees nearly all of it"
    assert by_step[4][1]["covered_m2"] + by_step[4][1]["left_m2"] == pytest.approx(by_step[4][1]["area_m2"], abs=0.05)
    assert by_step[4][1]["scanned"] == "scanned:scan_point_inside", "the recorded verdict rides along"
    halves = by_step[5]
    assert set(halves) == {1, 3, 7}
    assert halves[1]["bounded"] and halves[7]["bounded"], "each half is closed by walls and the cut"
    assert halves[1]["observed_m2"] + halves[7]["observed_m2"] == pytest.approx(by_step[4][1]["observed_m2"], abs=0.3), (
        "the coverage belongs to the floor, not the number")
    assert halves[1]["covered_m2"] + halves[7]["covered_m2"] == pytest.approx(by_step[4][1]["covered_m2"], abs=0.3)
    merged = by_step[6][1]
    assert merged["bounded"] and merged["area_m2"] == pytest.approx(20.0, abs=0.2)
    assert merged["observed_fraction"] == pytest.approx(by_step[4][1]["observed_fraction"], abs=0.02)
    checks = result["checks"]
    assert [c["step"] for c in checks["partition_changes"]] == [5, 6]
    merge = [b for b in checks["final_breaches"] if b["step"] == 6]
    assert merge and merge[0]["kind"] == "merge" and merge[0]["vanished"] == [7]
    assert not any(b["kind"] == "breach" for b in checks["final_breaches"])
    grown = [g for g in checks["lower_bound_growth"] if g["pid"] == 1]
    assert grown and grown[0]["became_final"] and grown[0]["ratio"] == pytest.approx(2.0, abs=0.1)
    text = replay_tool.summarise(result)
    assert "1 merges" in text and "0 BREACHES" in text


def test_the_command_line_writes_the_records(tmp_path, capsys):
    write_recording(tmp_path)
    out = tmp_path / "replay.json"
    assert replay_tool.main([str(tmp_path), "--every", "2", "--json", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved[0]["episode"] == "00800-synthetic/000000"
    assert [s["step"] for s in saved[0]["steps"]] == [0, 2, 4, 6], "every second action, and the last"
    assert "00800-synthetic" in capsys.readouterr().out


def test_the_camera_follows_the_scene_id():
    assert replay_tool.camera_for_scene("00880-Nfvxx8J5NCo") == "hm3d"
    assert replay_tool.camera_for_scene("2azQ1b91cZZ") == "mp3d"
    assert replay_tool.camera_for_scene("Collierville") == "hm3d", "no Gibson camera here: the shared runtime stays clear of it"
    hm3d, mp3d = replay_tool.benchmark_camera("hm3d"), replay_tool.benchmark_camera("mp3d")
    assert (hm3d.min_depth_m, hm3d.max_depth_m, hm3d.height_m) == (mp3d.min_depth_m, mp3d.max_depth_m, mp3d.height_m) == (0.5, 5.0, 0.88)


#: A recording to replay, when one is on this machine: ``OBJNAV_EXTENT_RECORDING``, else the HM3D episode the
#: mechanism was first checked on, under this checkout's (gitignored) ``runs/``.
RECORDING = os.environ.get("OBJNAV_EXTENT_RECORDING") or str(
    Path(__file__).resolve().parents[5] / "runs" / "rerun2_relook_lookdown_20261105" / "00880-Nfvxx8J5NCo"
    / "recordings" / "83d1d9fe8271")


@pytest.mark.skipif(not (Path(RECORDING) / "map_journal.npz").is_file(), reason="no HM3D recording on this machine")
def test_a_real_hm3d_recording_keeps_the_mechanisms_promises():
    """00880-Nfvxx8J5NCo/000009 (2026-11-05): 354 actions, the watershed renumbering rooms on 54 of them."""
    result = replay_tool.replay(RECORDING, every=3)
    checks = result["checks"]
    assert checks["readings"] > 500 and checks["partition_changes"], "a real episode: many rooms, the partition moving"
    deltas = [abs(c["observed_after_m2"] - c["observed_before_m2"]) for c in checks["partition_changes"]]
    assert float(np.median(deltas)) < 1.0, "the observed floor is conserved across the partition's changes"
    kinds = [b["kind"] for b in checks["final_breaches"]]
    assert kinds.count("breach") == 0, "no room of a square metre or more grew through a side read as located"
    assert "merge" in kinds, "the rooms that did grow after a final reading were merges"


# -- the live policy: the ledger feeds the scene graph, the scene graph the oracle --------------------------------------
class PromptKeepingLLM:
    """The neutral fake that keeps every oracle prompt it was shown."""

    def __init__(self):
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM
        self.inner = FakeLLM()
        self.oracle_prompts = []

    def chat_json(self, system, user, **kwargs):
        if kwargs.get("reasoning"):
            self.oracle_prompts.append((system, user))
        return self.inner.chat_json(system, user, **kwargs)


def test_the_policy_reads_the_extents_every_action_and_the_oracle_is_shown_size_status_and_coverage():
    from sparx_agency.core.mapping.topology.room_extent import room_extents
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import method_snapshot
    llm = PromptKeepingLLM()
    policy, episode = setup_policy(label="desk", llm=llm)          # a context class: the search runs, no takeover
    for step in range(4):
        policy.plan(observation(episode, step, depth=3.0))           # a wall 3 m ahead, floor before it
    assert policy.graph.observed_provider is not None, "the floor's scene graph reads the ledger's observed cells"
    assert policy.sight.observed_mask(policy.last_world).any(), "the frames observed the floor ahead"
    assert policy.graph.extents, "every update reads the rooms' extents"
    extent = next(iter(policy.graph.extents.values()))
    assert extent.observed_fraction > 0.0 and extent.free_m2 > 0.0 and extent.covered_m2 > 0.0
    assert not extent.bounded, "one flat wall ahead: the other walls are not seen"
    nodes = policy.graph.nodes(policy.last_world, 3)
    assert nodes and nodes[0].bounded is False and nodes[0].covered_fraction is not None
    assert nodes[0].footprint_m2 == pytest.approx(extent.area_m2) and nodes[0].walls_seen == extent.walls_seen
    assert llm.oracle_prompts, "the loop asked the oracle"
    system, user = llm.oracle_prompts[-1]
    assert "covered=" in user and "observed=" not in user, user
    assert "at most" in user or "all of the mapped part" in user, "a lower-bound size: the share is a bound, or all mapped"
    assert "wall not seen yet" in user or "walls not seen yet" in user, user
    assert "is a LOWER BOUND" in system and "COVERAGE FIRST" in system and "TIME IS SECONDARY" in system
    record = policy.episode_info()["last_reasoning"]
    assert record["extents"] and all("size_status" in item for item in record["extents"].values())
    assert any(node.get("bounded") is False for node in record["nodes"] if node.get("kind") == "room")
    snapshot = method_snapshot(policy)
    rooms = [room for room in snapshot["search"]["rooms"] if room.get("extent")]
    assert rooms and rooms[0]["extent"]["size_status"] == "lower_bound"
    assert rooms[0]["extent"]["covered_is_bound"] and rooms[0]["extent"]["walls_seen"] == extent.walls_seen
    assert all(node.get("covered_m2", 0.0) >= 0.0 for node in record["nodes"] if node.get("kind") == "room")
    configuration = policy.configuration()
    assert configuration["room_extents"]["doorway_max_m"] == 1.2 and "OBSERVED" in configuration["sight"]["rule"]
    assert configuration["sight"]["detect_range_m"] == 4.0
    assert configuration["room_extents"]["rule"].startswith("observed map only"), "the frozen record says where it comes from"
    # The size is the agent's own map and nothing else: read again here from the occupancy grid, the partition cut
    # out of that grid and the sight ledger's two records, it is what the scene graph holds and the oracle was shown.
    world = policy.last_world
    again = room_extents(world.grid, policy.graph.labels, world.resolution, (world.origin_x, world.origin_y),
                         free_value=world.values.free, occupied_value=world.values.occupied,
                         unknown_value=world.values.unknown, observed=policy.sight.observed_mask(world),
                         resolved=policy.sight.resolved(world), params=policy.extent_params)
    assert set(again) == set(policy.graph.extents)
    for pid, held in policy.graph.extents.items():
        for name in ("area_m2", "free_m2", "length_m", "width_m", "covered_m2", "left_m2", "open_perimeter_m"):
            assert getattr(again[pid], name) == pytest.approx(getattr(held, name)), (pid, name)
        assert (again[pid].bounded, again[pid].open_sides) == (held.bounded, held.open_sides)


def _walled_room_world(far_wall=True):
    """A 5 x 4 m room on a 12 m map, walled on all four sides, or with its far (+x) wall never seen."""
    from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
    grid = np.full((120, 120), -1, np.int8)
    grid[20:60, 20:70] = 0
    grid[19, 19:71] = 100
    grid[60, 19:71] = 100
    grid[19:61, 19] = 100
    if far_wall:
        grid[19:61, 70] = 100
    return OccupancyGrid2D(grid, OccupancyGrid2DParams(0.1, -6.0, -6.0, "world"),
                           values=OccupancyValues(free=0, occupied=100, unknown=-1))


def test_the_scene_graph_turns_a_room_of_the_map_into_the_size_and_coverage_the_oracle_reads():
    """From cells to words, through the live scene graph and nothing else: the partition is the watershed's own, the
    size is read off the grid it was cut from, and the line is the one the oracle is shown."""
    from sparx_agency.core.mapping.topology.search_node_oracle import format_node
    from sparx_agency.core.planning.object_search.rooms.scene_graph import ObservedSceneGraph
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM
    graph = ObservedSceneGraph(FakeLLM())
    world = _walled_room_world()
    graph.update(world, [], None, reason=False)
    assert list(graph.extents) == list(graph.registry.rooms) == [0]
    assert format_node(graph.nodes(world, 0)[0]) == (
        "id=0  ROOM  type=unknown  size=20m2 (5x4m, all walls seen)  frontier=0  exits=0  status=never_entered  "
        "searched=0s  seen: nothing yet"), "no sight record was given: the size, and no coverage"
    observed = np.zeros(world.grid.shape, dtype=bool)
    observed[20:60, 20:45] = True                                   # the half nearer the door was looked at
    graph.observed_provider = lambda world: observed
    graph.update(world, [], None, reason=False)
    assert "  size=20m2 (5x4m, all walls seen)  covered=10m2 (50% of the room, 10m2 left)  frontier=0  " in format_node(
        graph.nodes(world, 0)[0])
    # The far wall beyond the camera's reach: the same floor is a lower bound, and its covered share an upper one.
    open_world = _walled_room_world(far_wall=False)
    graph.update(open_world, [], None, reason=False)
    assert ("  size=at least 20m2 (5+ x 4m, 1 wall not seen yet)  covered=10m2 (at most 50% of the room, at least 10m2 "
            "left)  frontier=1  ") in format_node(graph.nodes(open_world, 0)[0])
    assert graph.extent_of(0).open_sides and graph.extent_records()["0"]["size_status"] == "lower_bound"


def test_a_scene_graph_without_a_sight_record_says_the_size_and_nothing_of_the_coverage():
    """No ledger, or a disabled one, is not "0% covered": the room line then carries the size alone."""
    from sparx_agency.core.mapping.topology.search_node_oracle import format_node
    from sparx_agency.core.planning.object_search.world_model.sightlines import SightLedger, SightSettings
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
    policy, episode = setup_policy(label="desk")
    for step in range(4):
        policy.plan(observation(episode, step, depth=3.0))
    world, graph = policy.last_world, policy.graph
    assert "covered=" in format_node(graph.nodes(world, 3)[0])
    assert SightLedger(policy, SightSettings(enabled=False)).observed_mask(world) is None, "a disabled ledger keeps no record"
    for provider in (None, lambda world: None):
        graph.observed_provider = provider
        graph.extents = graph._room_extents(world, graph.labels, graph.frontier_world(world), None)
        node = graph.nodes(world, 3)[0]
        assert node.bounded is False and node.covered_fraction is None
        line = format_node(node)
        assert "size=at least " in line and "covered=" not in line, line
