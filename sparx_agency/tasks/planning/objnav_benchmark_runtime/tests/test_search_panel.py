"""The dashboard's search column and the node overlay: what the loop believes is what the video shows."""
from __future__ import annotations
import numpy as np
from sparx_agency.tasks.planning.objnav_benchmark_runtime.floor_panels import FloorPanels
from sparx_agency.tasks.planning.objnav_benchmark_runtime.search_panel import render_search_panel
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, NamingLLM, obs_at, see, stair_policy)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import (
    method_snapshot, render_dashboard, search_snapshot)
def test_the_snapshot_carries_rooms_order_next_node_objects_and_the_oracles_split():
    # The relabel in transit is the historical rule (``transit_commitment=False``); the panel is what is under test.
    policy, episode, world, rooms, _ = stair_policy(order=(1, 0), transit_commitment=False)
    policy.graph.label_tracker.classifier._client = NamingLLM()
    policy.last_world = world
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    see(policy, {1: ["toilet"]}, step=1)
    policy.loop.plan(obs_at(episode, 1, IN_A), world)            # room B named a bathroom: re-solved, chosen again
    search = search_snapshot(policy)
    assert search["target"] == "chair" and search["order"] == [1, 0] and search["next_room"] == 1
    assert search["room_in_force"] is None and search["supervisor_state"] == "transit"
    assert 0.0 <= search["elsewhere"] <= 1.0 and search["p_present"] + search["elsewhere"] >= 1.0 - 1e-9
    by_id = {room["id"]: room for room in search["rooms"]}
    assert by_id[1]["label"] == "bathroom" and by_id[1]["kind"] == "room" and by_id[1]["objects"] == ["toilet"]
    assert by_id[0]["prob"] == 0.6 and by_id[0]["centroid"] == list(rooms[0].centroid)
    assert search["rooms"][0]["id"] == 0, "sorted by probability: the room worth most first"
    assert "stairs" not in search, "a seen staircase is no node of the search: the snapshot lists rooms and openings"
    assert [e["event"] for e in search["events"]][-3:] == ["relabel", "release", "transit"]
    assert search["local_budget"] == policy.loop.settings.local_steps
    assert search["floor"]["phase"] == "SEARCH" and search["floor"]["active_portal"] is None
    snapshot = method_snapshot(policy)
    assert snapshot["search"]["next_room"] == 1
def test_the_search_column_and_the_node_overlay_render_from_the_snapshot():
    policy, episode, world, rooms, _ = stair_policy(order=(0, 1))
    policy.last_world = world
    policy.loop.plan(obs_at(episode, 0, IN_A), world)
    search = search_snapshot(policy)
    assert search["next_room"] == 0 and search["floor"]["active_portal"] is None, "room A, where the agent stands"
    column = render_search_panel(search, (320, 720))
    assert column.shape == (720, 320, 3) and (column > 60).any(), "text was drawn"
    blank = render_search_panel({}, (320, 720))
    assert blank.shape == (720, 320, 3), "an empty block still renders"
    obs = obs_at(episode, 0, IN_A)
    policy.mapping.update(obs)                                   # a mapped floor for the panel to bind a slot to
    panels = FloorPanels(2, policy.settings, obs.pose)
    panels.capture(policy, obs)
    assert panels.bindings == {0: 0}
    plain = panels.render(0, size=(640, 720))
    marked = panels.render(0, size=(640, 720), search=search)
    assert plain.shape == marked.shape == (720, 640, 3)
    assert (plain != marked).any(), "the rooms, the order and the next node are drawn on the active floor"
    assert (panels.render(1, size=(640, 720), search=search) == panels.render(1, size=(640, 720))).all(), (
        "an inactive slot carries no node overlay")
def test_the_dashboard_frame_keeps_its_size_with_the_search_column():
    policy, episode = setup_policy("bed")
    obs = observation(episode, 0, depth=3)
    command = policy.plan(obs)
    snapshot = method_snapshot(policy)
    assert "search" in snapshot and snapshot["search"]["target"] == "chair"
    assert "stairs" not in snapshot["search"], "no stair node exists to list"
    panels = FloorPanels(1, policy.settings, obs.pose)
    panels.capture(policy, obs)
    for floor_panels in (None, panels):
        frame = render_dashboard(policy, obs, [(0, 0)], {"action": "TURN_LEFT", "info": command.info},
                                 episode.episode_id, snapshot, floor_panels=floor_panels)
        assert frame.shape == (900, 1600, 3)
        assert (frame[40:760, 1280:] > 60).any(), "the search column is drawn"
    assert isinstance(np.asarray(frame), np.ndarray)
def test_a_rooms_size_and_coverage_ride_in_the_snapshot_and_are_drawn_under_its_row():
    """The room extents (2026-11-16) of every room: the snapshot's ``extent`` and a line of the search column."""
    policy, episode = setup_policy("desk")
    for step in range(4):
        policy.plan(observation(episode, step, depth=3.0))
    search = search_snapshot(policy)
    rooms = [room for room in search["rooms"] if room.get("extent")]
    assert rooms and rooms[0]["extent"]["size_status"] == "lower_bound" and rooms[0]["extent"]["area_m2"] > 0.0
    assert set(rooms[0]["extent"]["wall_distances_m"]) == {"+u", "-u", "+v", "-v"}
    column = render_search_panel(search, (320, 720))
    bare = dict(search, rooms=[dict(room, extent=None) for room in search["rooms"]])
    assert (column != render_search_panel(bare, (320, 720))).any(), "the size and coverage line is drawn"
    final = dict(search, rooms=[dict(room, extent=dict(room["extent"], size_status="final", walls_seen=4))
                                for room in rooms])
    assert render_search_panel(final, (320, 720)).shape == (720, 320, 3), "a final size renders too"
