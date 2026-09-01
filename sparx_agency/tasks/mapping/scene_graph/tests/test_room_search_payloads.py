"""Tests for the room-search wire seam — the parts that fail silently.

Nothing in ``room_search_payloads`` can be verified by looking at a running
system: a BEV decoded with the wrong occupancy encoding produces a grid that
A* happily plans through, a merge that loses half the rooms produces a shorter
candidate list nobody counts, and an info payload with a renamed key produces
a dashboard panel that is simply blank. So each of those is asserted here, with
no rclpy context anywhere -- these run in the plain ``.venv``.

The centrepiece is :func:`test_a_sealed_wall_makes_the_goal_unreachable`. The
FALCON BEV encodes OCCUPIED as **100**, and ``OccupancyValues`` defaults it to
**1**; a grid built without saying so has no occupied cells at all, and the
only symptom is a route through a wall. The pair of wall tests below is the
only place that argument is checked end to end.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.exploration.room_search_policy import (
    Hold, RoomSearchParams, RoomSearchPolicy)
from sparx_agency.core.planning.interfaces import PlanRequest
from sparx_agency.core.planning.planners.astar import (WeightedAStarParams,
                                                       WeightedAStarPlanner2D)
from sparx_agency.tasks.mapping.scene_graph.ros2.room_search_payloads import (
    BEV_VALUES, centroids_from_scene_graph, grid_from_bev, room_options,
    route_points, search_info_payload)

RES = 0.2
SHAPE = (60, 60)          # (height, width) -> 12 m x 12 m
WALL_COLUMN = 30          # x = 6.0 m
GAP_ROWS = slice(26, 34)  # y = 5.2 .. 6.8 m


def bev_data(gap):
    """A 12 m room split by a wall at x = 6 m, with or without a doorway."""
    cells = np.zeros(SHAPE, dtype=np.int8)
    cells[:, WALL_COLUMN] = 100
    if gap:
        cells[GAP_ROWS, WALL_COLUMN] = 0
    return cells.reshape(-1).tolist()


def planner():
    return WeightedAStarPlanner2D(WeightedAStarParams(
        inflate_radius_m=0.4, unknown_blocked=False, waypoint_spacing_m=1.0,
        goal_snap_radius_m=2.0))


def plan_across(gap):
    world = grid_from_bev(bev_data(gap), SHAPE[0], SHAPE[1], RES, 0.0, 0.0,
                          "world")
    return planner().plan(
        PlanRequest(start=Pose2D(1.0, 6.0, 0.0), goal=Pose2D(11.0, 6.0, 0.0),
                    frame_id="world"), world=world)


# -- the decode ------------------------------------------------------------

def test_the_decode_carries_the_bev_encoding_not_the_default():
    world = grid_from_bev(bev_data(gap=False), SHAPE[0], SHAPE[1], RES,
                          -3.0, 2.0, "world")
    assert (world.values.free, world.values.occupied, world.values.unknown) == (0, 100, -1)
    assert world.values == BEV_VALUES
    assert (world.height, world.width) == SHAPE
    assert world.resolution == pytest.approx(RES)
    assert (world.origin_x, world.origin_y) == (-3.0, 2.0)
    assert world.frame_id == "world"


def test_row_zero_is_minimum_y_with_no_flip():
    cells = np.zeros(SHAPE, dtype=np.int8)
    cells[0, 0] = 100          # the corner AT the origin
    world = grid_from_bev(cells.reshape(-1).tolist(), SHAPE[0], SHAPE[1], RES,
                          -3.0, 2.0)
    assert world.is_occupied(*world.world_to_grid(-3.0 + 0.5 * RES, 2.0 + 0.5 * RES))
    assert world.world_to_grid(-3.0, 2.0) == (0, 0)


def test_unknown_survives_the_decode_as_unknown():
    cells = np.full(SHAPE, -1, dtype=np.int8)
    world = grid_from_bev(cells.reshape(-1).tolist(), SHAPE[0], SHAPE[1], RES, 0.0, 0.0)
    assert world.is_unknown(5, 5)
    assert not world.is_occupied(5, 5)


def test_a_truncated_grid_raises_rather_than_reshaping_out_of_phase():
    with pytest.raises(ValueError, match="expected"):
        grid_from_bev([0] * (SHAPE[0] * SHAPE[1] - 1), SHAPE[0], SHAPE[1],
                      RES, 0.0, 0.0)


def test_an_empty_grid_shape_raises():
    with pytest.raises(ValueError, match="positive"):
        grid_from_bev([], 0, 0, RES, 0.0, 0.0)


def test_every_decode_is_a_new_object_because_the_planner_caches_on_identity():
    args = (bev_data(gap=True), SHAPE[0], SHAPE[1], RES, 0.0, 0.0)
    assert grid_from_bev(*args) is not grid_from_bev(*args)


# -- the decode, proved against the planner --------------------------------

def test_a_doorway_in_the_wall_is_routed_through():
    result = plan_across(gap=True)
    assert result.ok
    assert result.path.points[-1].x > 6.0


def test_a_sealed_wall_makes_the_goal_unreachable():
    # THE decode test. Fails -- with a route straight through the wall -- the
    # moment the grid is built without values=OccupancyValues(free=0,
    # occupied=100, unknown=-1).
    assert not plan_across(gap=False).ok


def test_the_default_encoding_would_have_flown_through_that_wall():
    # Why the argument above is not optional, stated as an assertion rather
    # than a comment: the same cells under the DEFAULT OccupancyValues have no
    # occupied cell in them at all.
    cells = np.asarray(bev_data(gap=False), dtype=np.int8).reshape(SHAPE)
    naive = OccupancyGrid2D(cells, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"))
    assert not naive.is_occupied(WALL_COLUMN, 5)
    assert planner().plan(
        PlanRequest(start=Pose2D(1.0, 6.0, 0.0), goal=Pose2D(11.0, 6.0, 0.0),
                    frame_id="world"), world=naive).ok


# -- the merge -------------------------------------------------------------

def test_centroids_are_read_off_the_scene_graph_by_pid():
    payload = {"rooms": [{"id": 3, "centroid": [1.5, -2.0]},
                         {"id": 7, "centroid": [4.0, 4.0]}]}
    assert centroids_from_scene_graph(payload) == {3: (1.5, -2.0), 7: (4.0, 4.0)}


def test_a_room_without_a_usable_centroid_is_left_out():
    payload = {"rooms": [{"id": 1, "centroid": [0.0]},
                         {"id": 2},
                         {"id": "x", "centroid": [1.0, 1.0]},
                         {"id": 4, "centroid": [1.0, 1.0]}]}
    assert centroids_from_scene_graph(payload) == {4: (1.0, 1.0)}


def test_an_empty_scene_graph_is_an_empty_map():
    assert centroids_from_scene_graph({}) == {}


def test_the_ranking_and_the_centroids_join_on_the_pid():
    ranked = [{"id": 3, "prob": 0.6, "label": "ward"},
              {"id": 7, "prob": 0.4, "label": "office"}]
    options = room_options(ranked, {3: (1.5, -2.0)})
    assert [(o.room_id, o.prob, o.xy, o.label) for o in options] == [
        (3, 0.6, (1.5, -2.0), "ward"),
        (7, 0.4, None, "office")]


def test_a_room_the_ranking_scored_but_the_map_has_not_found_is_not_flyable():
    options = room_options([{"id": 9, "prob": 1.0}], {})
    state = RoomSearchPolicy(RoomSearchParams()).update(options, (0.0, 0.0), 0.0)
    assert isinstance(state.action, Hold)
    assert state.candidates == ()


def test_a_malformed_ranking_entry_is_skipped_not_raised_on():
    ranked = [{"prob": 0.5}, {"id": None, "prob": 0.5},
              {"id": 2, "prob": "later"}, {"id": 5, "prob": 0.5}]
    assert [o.room_id for o in room_options(ranked, {})] == [5]


# -- the route -------------------------------------------------------------

def test_the_route_starts_where_the_aircraft_is():
    points = route_points((1.0, 2.0), [(3.0, 2.0), (5.0, 2.0)], 1.75)
    assert points[0] == (1.0, 2.0, 1.75)
    assert points == [(1.0, 2.0, 1.75), (3.0, 2.0, 1.75), (5.0, 2.0, 1.75)]


def test_a_waypoint_on_top_of_the_aircraft_collapses():
    points = route_points((1.0, 2.0), [(1.0, 2.0), (4.0, 2.0)], 1.0)
    assert points == [(1.0, 2.0, 1.0), (4.0, 2.0, 1.0)]


def test_an_empty_plan_leaves_nothing_flyable():
    assert route_points((1.0, 2.0), [], 1.0) == [(1.0, 2.0, 1.0)]


def test_a_planned_route_survives_the_assembly_intact():
    result = plan_across(gap=True)
    points = route_points((1.0, 6.0),
                          [(p.x, p.y) for p in result.path.points], 1.75)
    assert len(points) >= 2
    assert all(p[2] == 1.75 for p in points)
    assert points[0] == (1.0, 6.0, 1.75)


# -- the operator payload --------------------------------------------------

def pursuing_state():
    ranked = [{"id": 3, "prob": 0.75, "label": "ward"},
              {"id": 7, "prob": 0.25, "label": "office"}]
    options = room_options(ranked, {3: (1.5, -2.0), 7: (4.0, 4.0)})
    return RoomSearchPolicy(RoomSearchParams()).update(options, (0.0, 0.0), 12.5)


def test_the_info_payload_keeps_every_key_the_old_topic_had():
    state = pursuing_state()
    payload = search_info_payload(12.5, state, "wheelchair", False, True, 4,
                                  state.action.note, {"samples": 1})
    for key in ("stamp", "state", "room_id", "label", "prob", "goal",
                "candidates"):
        assert key in payload
    assert payload["state"] == "pursuing"
    assert payload["goal"] == [pytest.approx(state.goal_xy[0]),
                               pytest.approx(state.goal_xy[1])]
    assert {c["id"] for c in payload["candidates"]} == {3, 7}
    assert all("prob_renorm" in c for c in payload["candidates"])
    assert sum(c["prob_renorm"] for c in payload["candidates"]) == pytest.approx(1.0)


def test_the_info_payload_survives_a_json_round_trip():
    state = pursuing_state()
    payload = search_info_payload(12.5, state, "wheelchair", True, True, 4,
                                  "drew a room", {"samples": 1, "arrivals": 0})
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["room_id"], int)
    assert isinstance(payload["prob"], float)
    assert payload["fly"] is True
    assert payload["target"] == "wheelchair"


def test_an_idle_payload_says_so_without_a_goal():
    state = RoomSearchPolicy(RoomSearchParams()).update([], (0.0, 0.0), 1.0)
    payload = search_info_payload(1.0, state, "wheelchair", False, False, 0,
                                  state.action.note, {})
    assert payload["state"] == "idle"
    assert payload["room_id"] is None
    assert payload["goal"] is None
    assert payload["candidates"] == []
    assert json.loads(json.dumps(payload))["planned"] is False


def test_numpy_scalars_from_a_ranking_do_not_reach_json_dumps():
    # The failure this guards is a TypeError inside a timer callback, in
    # flight: np.float32 is not JSON-serialisable and is exactly what a numpy
    # pipeline hands the oracle.
    ranked = [{"id": np.int64(3), "prob": np.float32(0.9), "label": "ward"}]
    options = room_options(ranked, {3: (np.float64(1.0), np.float64(2.0))})
    state = RoomSearchPolicy(RoomSearchParams()).update(options, (0.0, 0.0), 0.0)
    payload = search_info_payload(0.0, state, "wheelchair", False, False, 0,
                                  "", {"samples": np.int64(1)})
    assert json.loads(json.dumps(payload))["room_id"] == 3
