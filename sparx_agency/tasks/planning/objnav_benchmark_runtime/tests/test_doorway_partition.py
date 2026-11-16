"""Regressions for the doorway partition (2026-10-21): an observed open doorway parts the space beyond it into a node of its own.

The grid is ``room_doorway_hall`` of the watershed tests at 0.1 m: a room,
a 0.8 m doorway in a 0.3 m wall, and a hall beyond of which only a sliver
has been seen through the door. The scene graph is the real one
(``ObservedSceneGraph.update`` with the classifier not asked); the openings
module's record of the exit is built by hand from the sliver's frontier.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.mapping.topology.tests.test_room_watershed import HOME_RES, room_doorway_hall
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.object_search.rooms.opening_nodes import Opening
from sparx_agency.core.planning.object_search.rooms.scene_graph import DEFAULT_SEGMENTATION, ObservedSceneGraph
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM

FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)
IN_ROOM, IN_SLIVER, DOORWAY_XY = (1.5, 2.0), (3.5, 4.4), (3.5, 4.1)


def glimpsed_world(depth_rows=4):
    """The room seen whole, the hall seen ``depth_rows`` deep through the doorway, unknown beyond; walls occupied."""
    free, _, _, _ = room_doorway_hall()
    grid = np.full(free.shape, OCC, np.int8)
    grid[free] = FREE
    grid[43:70, 10:60] = UNK                                   # the hall: unknown ...
    grid[43:43 + depth_rows, 29:41] = FREE                     # ... but for the sliver seen through the door
    return OccupancyGrid2D(grid, OccupancyGrid2DParams(HOME_RES, 0.0, 0.0, "world"), values=VALUES)


def exit_through_the_door(world, index=0):
    """The openings module's record of the sliver's far edge: a frontier at the mapped floor's edge, the unknown north."""
    return Opening(index=index, xy=IN_SLIVER, heading=math.pi / 2, size_cells=12, geodesic_m=3.0, room_pid=0,
                   door_id=None, glimpsed=())


def graph_on(world, step=0):
    graph = ObservedSceneGraph(FakeLLM(), segmentation=DEFAULT_SEGMENTATION)
    target = gibson_label_mapper().target_labels("toilet")
    graph.update(world, [], target, doors=(), step=step, reason=False)
    return graph, target


def test_an_observed_open_doorway_parts_the_floor_beyond_it_into_a_node_of_its_own_joined_by_an_edge():
    world = glimpsed_world()
    graph, target = graph_on(world)
    assert len(graph.registry.rooms) == 1, "nothing says there is a door yet: the sliver hangs off the room"
    room = graph.room_at(world, IN_ROOM)
    assert graph.room_at(world, IN_SLIVER) == room and graph.doors == []
    # The exit is read off the live map from the room; the sightline to it passes the doorway.
    added = graph.note_openings(world, [exit_through_the_door(world)], IN_ROOM, step=1)
    assert added == 1 and len(graph.thresholds) == 1
    assert math.dist(graph.thresholds[0], DOORWAY_XY) <= DEFAULT_SEGMENTATION.threshold_snap_reach_m + 0.1, (
        "the threshold is at the doorway (within the snap's reach of its choke), not at the frontier")
    assert graph.threshold_events[-1]["source"] == "sightline" and graph.threshold_events[-1]["opening"] == 0
    assert graph.note_openings(world, [exit_through_the_door(world)], IN_ROOM, step=2) == 0, "known already"
    graph.update(world, [], target, doors=(), step=2, reason=False)
    rooms = graph.registry.rooms
    assert len(rooms) == 2
    beyond = graph.room_at(world, IN_SLIVER)
    assert beyond is not None and beyond != graph.room_at(world, IN_ROOM) == room, "the space beyond the door is a node"
    assert rooms[beyond].n_cells < 2 * DEFAULT_SEGMENTATION.min_room_cells, "the sliver and the doorway's cells"
    assert graph.facts[beyond].frontier_clusters >= 1, "with the unknown beyond it as its frontier"
    # A thinner sliver -- two rows beyond the door -- is under the room floor and is a node all the same.
    thin = glimpsed_world(depth_rows=2)
    thin_graph, _ = graph_on(thin)
    thin_graph.note_openings(thin, [Opening(index=0, xy=(3.5, 4.3), heading=math.pi / 2, size_cells=12, geodesic_m=3.0,
                                            room_pid=0, door_id=None, glimpsed=())], IN_ROOM, step=1)
    thin_graph.update(thin, [], target, doors=(), step=2, reason=False)
    assert len(thin_graph.registry.rooms) == 2
    stub = thin_graph.room_at(thin, (3.5, 4.35))
    assert stub is not None and stub != thin_graph.room_at(thin, IN_ROOM)
    assert thin_graph.registry.rooms[stub].n_cells < DEFAULT_SEGMENTATION.min_room_cells, "a stub, under the floor, kept"
    assert graph.room_at(world, (3.5, 3.8)) == room, "the room keeps its floor up to the doorway"
    [edge] = graph.doors
    assert edge["kind"] == "threshold" and edge["index"] == "t0" and edge["room_pairs"] == [sorted([room, beyond])]
    assert math.dist(edge["xy"], DOORWAY_XY) <= DEFAULT_SEGMENTATION.threshold_snap_reach_m + 0.1, (
        "the edge passes through the doorway coordinate")
    assert sorted(edge["rooms"]) == sorted([room, beyond])
    # The hall is seen deeper: the node beyond the door grows under its number, the partition holds.
    deeper = glimpsed_world(depth_rows=12)
    graph.update(deeper, [], target, doors=(), step=3, reason=False)
    assert graph.room_at(deeper, IN_SLIVER) == beyond and graph.room_at(deeper, (3.5, 5.2)) == beyond
    assert graph.room_at(deeper, IN_ROOM) == room and len(graph.registry.rooms) == 2
    assert graph.registry.rooms[beyond].n_cells > rooms[beyond].n_cells
    assert [e for e in graph.doors if e["kind"] == "threshold"][0]["room_pairs"] == [sorted([room, beyond])]
    # The whole hall seen: still two rooms, the same two numbers, joined by the same edge.
    whole = glimpsed_world(depth_rows=27)
    graph.update(whole, [], target, doors=(), step=4, reason=False)
    assert graph.room_at(whole, (3.5, 6.5)) == beyond and graph.room_at(whole, IN_ROOM) == room
    assert len(graph.registry.rooms) == 2


def test_a_sightline_to_an_open_rooms_own_frontier_adds_no_threshold_and_a_detected_door_still_parts_the_sliver():
    grid = np.full((60, 60), OCC, np.int8)
    grid[5:30, 5:55] = FREE                                    # an open room, its north half unseen
    grid[30:58, 5:55] = UNK
    world = OccupancyGrid2D(grid, OccupancyGrid2DParams(HOME_RES, 0.0, 0.0, "world"), values=VALUES)
    graph, target = graph_on(world)
    frontier = Opening(index=3, xy=(3.0, 2.9), heading=math.pi / 2, size_cells=40, geodesic_m=2.0, room_pid=0,
                       door_id=None, glimpsed=())
    assert graph.note_openings(world, [frontier], (3.0, 0.8), step=1) == 0 and graph.thresholds == []
    graph.update(world, [], target, doors=(), step=2, reason=False)
    assert len(graph.registry.rooms) == 1
    # A door landmark detected at the doorway (no sightline needed): the sliver beyond is a node too.
    from types import SimpleNamespace
    world = glimpsed_world()
    graph, target = graph_on(world)
    door = SimpleNamespace(id=7, xy=DOORWAY_XY, count=3)
    graph.update(world, [], target, doors=(door,), step=1, reason=False)
    assert len(graph.registry.rooms) == 2
    assert graph.room_at(world, IN_SLIVER) != graph.room_at(world, IN_ROOM)
    [edge] = [e for e in graph.doors if e["kind"] == "door"]
    assert edge["index"] == 7 and len(edge["room_pairs"]) == 1


def test_the_graphs_edges_name_each_rooms_neighbours_by_type_and_the_oracles_node_carries_them():
    """``connections()`` -- what the oracle's ``connected to:`` is made of: the types of the rooms a door or a
    threshold joins a room to. A type and nothing of where: adjacency is in the scene graph, distance is RPT*'s."""
    from types import SimpleNamespace
    from sparx_agency.core.mapping.topology.search_node_oracle import format_node
    world = glimpsed_world()
    graph, target = graph_on(world)
    assert graph.connections() == {}, "one room and no edge: nothing is connected to anything"
    graph.update(world, [], target, doors=(SimpleNamespace(id=7, xy=DOORWAY_XY, count=3),), step=1, reason=False)
    room, beyond = graph.room_at(world, IN_ROOM), graph.room_at(world, IN_SLIVER)
    assert graph.connections() == {room: ["unknown"], beyond: ["unknown"]}, "a neighbour not identified yet"
    tracker = graph.label_tracker
    tracker.labels[room], tracker.metadata[room] = SimpleNamespace(label="bedroom"), {"strength": "strong"}
    tracker.labels[beyond], tracker.metadata[beyond] = SimpleNamespace(label="hallway"), {"strength": "weak"}
    assert graph.connections() == {room: ["hallway?"], beyond: ["bedroom"]}, "a weakly held type keeps its question mark"
    nodes = {node.id: node for node in graph.nodes(world, step=2, here_xy=IN_ROOM)}
    assert nodes[room].connected == ("hallway?",) and nodes[beyond].connected == ("bedroom",)
    line = format_node(nodes[room])
    assert "connected to: hallway?" in line
    assert "here" not in line and "%.1f" % IN_ROOM[0] not in line, "the room the agent stands in is not marked: that is distance"
    # An edge to a room the registry no longer holds (a merge) names nothing.
    graph.doors.append({"index": 9, "kind": "door", "room_pairs": [[room, 99]]})
    assert graph.connections() == {room: ["hallway?"], beyond: ["bedroom"]}


def test_a_room_holding_a_confirmed_object_of_the_targets_own_kind_is_marked_and_the_oracle_is_told():
    """``target_seen`` -- what lets the oracle's prompt say the covered floor was looked at WITHOUT finding the
    target: a room whose confirmed objects include one of the target's own kind (found, let go by the closing, the
    search still on) says so on its line; every other room says nothing; without the target the mark is unknown."""
    from types import SimpleNamespace
    world = glimpsed_world()
    graph, target = graph_on(world)                                # the target is a toilet
    door = SimpleNamespace(id=7, xy=DOORWAY_XY, count=3)
    sink = SimpleNamespace(class_name="sink", xy=IN_ROOM, radius_m=0.3)
    toilet = SimpleNamespace(class_name="toilet", xy=IN_SLIVER, radius_m=0.3)
    graph.update(world, [sink, toilet], target, doors=(door,), step=1, reason=False)
    room, beyond = graph.room_at(world, IN_ROOM), graph.room_at(world, IN_SLIVER)
    marked = {node.id: node for node in graph.nodes(world, step=2, target=target)}
    assert marked[room].target_seen is False and marked[beyond].target_seen is True
    assert "target:" not in format_line(marked[room]) and format_line(marked[beyond]).endswith(
        "seen: toilet  target: detected here, not reached")
    assert all(node.target_seen is None for node in graph.nodes(world, step=2)), "no target given: not known, not said"
    other = gibson_label_mapper().target_labels("bed")
    assert not any(node.target_seen for node in graph.nodes(world, step=2, target=other)), "another target's kind is no mark"
    # The loop's own query passes the episode's target, so what the model is shown carries the mark.
    graph.reason(world, target, step=3)
    shown = {node["id"]: node for node in graph.last_reasoning["nodes"]}
    assert shown[beyond]["target_seen"] is True and shown[room]["target_seen"] is False


def format_line(node):
    from sparx_agency.core.mapping.topology.search_node_oracle import format_node
    return format_node(node, 1.0)
