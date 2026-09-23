"""Tests for the object-search wire seam.

Three things are worth a test here and the rest is plumbing:

* **the room mask**, because it is the only thing keeping the in-room sweep
  inside its room. The label grid carries small GRID VALUES rather than pids,
  so a consumer that forgets the ``grid_pid_map`` indirection masks a
  different room and the aircraft flies out through a door -- and nothing
  raises;
* **the instance**, because the solver's whole answer is a function of it;
* **the payloads being plain**, because a numpy scalar reaches ``json.dumps``
  as a TypeError at publish time, in flight, on the one tick that mattered.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.object_search_supervisor import (
    ObjectSearchSupervisor, RoomFacts)
from sparx_agency.tasks.mapping.scene_graph.ros2.object_search_payloads import (
    costs_payload, facts_from_scene_graph, grid_from_bev, instance_from_wire,
    room_mask_from_labels, room_options, search_info_payload)
from sparx_agency.tasks.mapping.scene_graph.tests.test_payloads import (
    assert_plain)

FREE, OCC, UNK = 0, 100, -1
RES = 0.5


def bev():
    """The same three-room corridor the core tests use, as a wire message."""
    g = np.full((6, 9), OCC, dtype=np.int8)
    g[1:3, 1:9] = FREE
    g[3:5, 1:3] = FREE
    g[3:5, 4:6] = FREE
    g[3:5, 7:9] = FREE
    return g


def world():
    g = bev()
    return grid_from_bev(g.ravel().tolist(), 6, 9, RES, 0.0, 0.0)


def flat_cost(w):
    cost = np.full(w.grid.shape, np.inf, dtype=np.float64)
    cost[w.grid == FREE] = 1.0
    return cost


def scene_graph():
    """A ``/scene_graph`` payload shaped exactly as the mapper publishes one."""
    return {
        "stamp": 12.5,
        "resolution": RES,
        "origin": [0.0, 0.0],
        "rooms": [
            {"id": 1, "centroid": [1.0, 2.0], "cells": 4,
             "time_in_room_s": 30.0, "frontier_clusters": 2,
             "color": [0.1, 0.2, 0.3], "objects": [], "doors": []},
            {"id": 2, "centroid": [2.5, 2.0], "cells": 4,
             "time_in_room_s": 0.0, "frontier_clusters": 0,
             "color": [0.4, 0.5, 0.6], "objects": [], "doors": []},
        ],
        "doors": [],
        "drone": {"xy": [2.5, 0.75], "room_id": None},
        "grid_pid_map": {"1": 1, "2": 2},
    }


# -- facts ----------------------------------------------------------------
def test_facts_carry_the_two_effort_terms_and_the_size():
    facts = facts_from_scene_graph(scene_graph())
    assert set(facts) == {1, 2}
    assert facts[1] == RoomFacts(room_id=1, frontier_clusters=2,
                                 time_in_room_s=30.0, cells=4)


def test_a_room_without_an_id_is_dropped_not_guessed():
    facts = facts_from_scene_graph({"rooms": [{"centroid": [0.0, 0.0]},
                                              {"id": 5}]})
    assert set(facts) == {5}


def test_a_new_room_with_no_credited_dwell_still_produces_a_fact():
    facts = facts_from_scene_graph({"rooms": [{"id": 7}]})
    assert facts[7].frontier_clusters == 0
    assert facts[7].time_in_room_s == 0.0


def test_facts_survive_a_malformed_field():
    facts = facts_from_scene_graph(
        {"rooms": [{"id": 3, "frontier_clusters": None,
                    "time_in_room_s": "nope", "cells": []}]})
    assert facts[3] == RoomFacts(room_id=3, frontier_clusters=0,
                                 time_in_room_s=0.0, cells=0)


# -- the room mask --------------------------------------------------------
def label_grid():
    """Grid value 1 over room A's cells, 2 over room B's, 0 elsewhere."""
    lbl = np.zeros((6, 9), dtype=np.int8)
    lbl[3:5, 1:3] = 1
    lbl[3:5, 4:6] = 2
    return lbl


def test_the_mask_resolves_a_pid_through_the_grid_value_indirection():
    lbl = label_grid()
    mask = room_mask_from_labels(lbl.ravel().tolist(), 6, 9, {"1": 1, "2": 2}, 2)
    assert mask is not None
    assert mask.shape == (6, 9)
    assert mask[3:5, 4:6].all()
    assert not mask[3:5, 1:3].any(), "room A leaked into room B's mask"


def test_the_mask_follows_the_map_not_the_pid_number():
    """A pid of 130 cannot be a cell value; only the map can resolve it."""
    lbl = label_grid()
    mask = room_mask_from_labels(lbl.ravel().tolist(), 6, 9, {"2": 130}, 130)
    assert mask is not None
    assert mask[3:5, 4:6].all()


def test_a_renumbered_pid_returns_none_rather_than_an_empty_mask():
    lbl = label_grid()
    assert room_mask_from_labels(lbl.ravel().tolist(), 6, 9,
                                 {"1": 1, "2": 2}, 99) is None


def test_a_truncated_label_grid_raises_rather_than_reshaping_out_of_phase():
    with pytest.raises(ValueError):
        room_mask_from_labels([0, 0, 0], 6, 9, {"1": 1}, 1)


# -- the instance ---------------------------------------------------------
def test_instance_from_wire_joins_the_scene_graph_and_the_ranking():
    w = world()
    ranked = [{"id": 1, "prob": 0.8, "label": "ward"},
              {"id": 2, "prob": 0.2, "label": "corridor"}]
    inst, dropped = instance_from_wire(w, flat_cost(w), scene_graph(), ranked,
                                       depot_xy=(2.5, 0.75))
    assert inst is not None
    assert dropped == []
    assert inst.index_to_pid[:2] == (1, 2)
    assert inst.p[0] > inst.p[1], "the ranking should survive into p"
    assert inst.index_to_pid[inst.depot] == -1, "the aircraft is its own depot"


def test_instance_from_wire_is_none_before_the_first_room_exists():
    w = world()
    inst, dropped = instance_from_wire(w, flat_cost(w), {"rooms": []}, [])
    assert inst is None
    assert dropped == []


def test_a_malformed_ranking_entry_does_not_sink_the_instance():
    w = world()
    ranked = [{"id": "nope"}, {"prob": 0.5}, {"id": 1, "prob": 0.9}]
    inst, _ = instance_from_wire(w, flat_cost(w), scene_graph(), ranked)
    assert inst is not None
    assert inst.p[0] == pytest.approx(0.9)


# -- the payloads ---------------------------------------------------------
def test_costs_payload_is_plain_and_round_trips_through_json():
    w = world()
    ranked = [{"id": 1, "prob": 0.8}, {"id": 2, "prob": 0.2}]
    inst, dropped = instance_from_wire(w, flat_cost(w), scene_graph(), ranked,
                                       depot_xy=(2.5, 0.75))
    payload = costs_payload(inst, 12.5, 13.0, dropped, "llm")
    assert_plain(payload)
    back = json.loads(json.dumps(payload))
    assert len(back["cost"]) == len(back["rooms"])
    assert all(len(row) == len(back["rooms"]) for row in back["cost"])
    assert len(back["p"]) == len(back["rooms"])
    assert back["units"] == "seconds"
    assert back["prob_source"] == "llm"


def test_costs_payload_records_a_square_symmetric_matrix():
    w = world()
    inst, dropped = instance_from_wire(
        w, flat_cost(w), scene_graph(),
        [{"id": 1, "prob": 0.5}, {"id": 2, "prob": 0.5}])
    payload = costs_payload(inst, 12.5, 13.0, dropped, "llm")
    C = np.array(payload["cost"], dtype=float)
    assert C.shape[0] == C.shape[1]
    assert np.allclose(C, C.T)


def test_search_info_payload_is_plain_and_carries_the_committed_order():
    sup = ObjectSearchSupervisor()
    ranked = [{"id": 1, "prob": 0.8, "label": "ward"},
              {"id": 2, "prob": 0.2, "label": "corridor"}]
    options = room_options(ranked, {1: (1.0, 2.0), 2: (2.5, 2.0)})
    state = sup.update(options, facts_from_scene_graph(scene_graph()),
                       (2.5, 0.75), now=1.0, last_plan_s=1.0)
    payload = search_info_payload(
        2.0, state, "wheelchair", True, True, 4, state.action.note, sup.stats,
        room_facts=facts_from_scene_graph(scene_graph()).get(state.room_id))
    assert_plain(payload)
    back = json.loads(json.dumps(payload))
    assert back["state"] == "transit"
    assert back["order"], "the committed order must be visible in a recording"
    assert back["target"] == "wheelchair"
    assert back["candidates"][0]["prob_renorm"] > 0.0
    assert back["completed"] is None


def test_search_info_payload_reports_the_verdict_on_the_tick_a_room_ends():
    sup = ObjectSearchSupervisor(solver=lambda c, i=None: [c[0].room_id])
    options = room_options([{"id": 1, "prob": 1.0, "label": "ward"}],
                           {1: (1.0, 2.0)})
    facts = facts_from_scene_graph(scene_graph())
    sup.update(options, facts, (0.0, 0.0), now=0.0, last_plan_s=None)
    state = sup.update(options, facts, (0.0, 0.0), now=9.0, last_plan_s=None)
    payload = search_info_payload(9.0, state, "wheelchair", True, False, 0,
                                  state.action.note, sup.stats)
    assert_plain(payload)
    assert payload["completed"]["room_id"] == 1
    assert payload["completed"]["verdict"] == "unreachable"


def test_search_info_payload_survives_numpy_scalars_in_the_stats():
    sup = ObjectSearchSupervisor()
    state = sup.update([], {}, None, now=0.0)
    payload = search_info_payload(np.float64(1.0), state, "x", False, False,
                                  np.int64(0), "note",
                                  {"selections": np.int64(3)})
    assert_plain(payload)
