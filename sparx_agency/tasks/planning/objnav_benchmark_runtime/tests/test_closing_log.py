from types import SimpleNamespace

from sparx_agency.tasks.planning.objnav_benchmark_runtime.trace_tools.closing_log import closing_row, worth_logging


def detection(cls, conf, xyxy):
    return SimpleNamespace(cls=cls, conf=conf, xyxy=xyxy)


def graph_with(rooms):
    """``rooms``: ``{(x, y): (room id, label record)}``; the graph answers like the policy's room graph."""
    return SimpleNamespace(labels=object(), room_near=lambda world, xy, reach: rooms.get(xy, (None, None))[0],
                           label_info=lambda pid: next(info for room, info in rooms.values() if room == pid))


def policy_with(detections, graph=None, **closing_state):
    state = dict(phase="SEARCH", active=False, locked=False, count=0, suspect=None, best_conf=0.0, xyz=None, last_release=None,
                 border_rejections=0, context_rejections=0, glimpse_steps=0, verify_pauses=0, weak_resightings=0, releases=0,
                 memory_sight_refusals=0, inspection_steps=0, support_holds=0, history_holds=0, inspection_releases=0)
    state.update(closing_state)
    closing = SimpleNamespace(settings=SimpleNamespace(confidence=0.25), _clipped=lambda box, k: k == "K" and box[3] >= 470, **state)
    return SimpleNamespace(closing=closing, perception=SimpleNamespace(detections=detections), graph=graph,
                           last_world=object() if graph else None, target=SimpleNamespace(accepts=lambda cls: cls == "toilet"))


OBS = SimpleNamespace(step=41, pose=SimpleNamespace(x=1.0, y=2.0), camera=SimpleNamespace(intrinsics="K"))


def test_a_target_box_touching_the_border_is_reported_as_clipped():
    policy = policy_with([detection("toilet", 0.9, (100, 300, 200, 479)), detection("toilet", 0.8, (300, 100, 400, 200)),
                          detection("sink", 0.9, (0, 0, 10, 10))])
    row = closing_row(policy, OBS)
    assert row["target_boxes"] == [{"cls": "toilet", "conf": 0.9, "clipped": True}, {"cls": "toilet", "conf": 0.8, "clipped": False}]


def test_a_box_below_the_start_confidence_is_not_listed():
    row = closing_row(policy_with([detection("toilet", 0.1, (1, 1, 5, 5))]), OBS)
    assert row["target_boxes"] == []


def test_the_row_carries_the_verification_state_and_distance_to_the_candidate():
    policy = policy_with([], phase="VERIFY", active=True, count=1, best_conf=0.6, xyz=(4.0, 6.0, 0.5), border_rejections=2)
    row = closing_row(policy, OBS)
    assert (row["step"], row["phase"], row["count"], row["best_conf"]) == (41, "VERIFY", 1, 0.6)
    assert row["candidate_distance_m"] == 5.0 and row["counters"]["border_rejections"] == 2


def test_the_row_counts_the_refusals_that_keep_a_memory_stop_from_happening():
    policy = policy_with([], memory_sight_refusals=31, support_holds=7, history_holds=2, inspection_steps=24, inspection_releases=1)
    counters = closing_row(policy, OBS)["counters"]
    assert {name: counters[name] for name in ("memory_sight_refusals", "support_holds", "history_holds", "inspection_steps", "inspection_releases")} == {
        "memory_sight_refusals": 31, "support_holds": 7, "history_holds": 2, "inspection_steps": 24, "inspection_releases": 1}


def test_the_row_carries_the_candidates_map_position():
    assert closing_row(policy_with([], xyz=(4.123, 6.987, 0.5)), OBS)["candidate_xy"] == [4.12, 6.99]
    assert closing_row(policy_with([]), OBS)["candidate_xy"] is None


def test_the_row_names_the_room_of_the_candidate_and_of_the_agent():
    graph = graph_with({(4.0, 6.0): (3, {"label": "kitchen", "strength": "strong"}), (1.0, 2.0): (5, {"label": "hallway", "strength": "weak"})})
    row = closing_row(policy_with([], graph=graph, xyz=(4.0, 6.0, 0.5)), OBS)
    assert row["candidate_room"] == {"room": 3, "label": "kitchen", "strength": "strong"}
    assert row["agent_room"] == {"room": 5, "label": "hallway", "strength": "weak"}


def test_the_rooms_are_none_without_a_room_graph_or_a_known_room():
    row = closing_row(policy_with([]), OBS)
    assert row["candidate_room"] is None and row["agent_room"] is None
    assert closing_row(policy_with([], graph=graph_with({}), xyz=(4.0, 6.0, 0.5)), OBS)["candidate_room"] is None


def test_a_search_step_with_no_target_box_is_not_worth_logging():
    assert not worth_logging(closing_row(policy_with([]), OBS))
    assert worth_logging(closing_row(policy_with([detection("toilet", 0.9, (1, 1, 5, 5))]), OBS))
    assert worth_logging(closing_row(policy_with([], phase="VERIFY"), OBS))
