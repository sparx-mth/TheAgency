from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_scores import planner_view_of_room


def step(room_id, probs=None, nodes=()):
    reasoning = {"nodes": list(nodes)}
    if probs is not None:
        reasoning["oracle"] = {"probs": {str(k): v for k, v in probs.items()}}
    return {"method": {"room_id": room_id, "reasoning": reasoning}}


NODE = {"id": 3, "exits": 2, "label": "bedroom"}


def test_the_room_is_the_one_the_planner_stood_in_most_of_the_visit():
    steps = [step(1), step(3), step(3), step(1), step(3, nodes=[NODE])]
    assert planner_view_of_room(steps, 1, 4)["planner_room"] == 3


def test_the_probability_is_the_latest_one_offered_before_arrival():
    steps = [step(1, {3: 0.6}), step(1, {3: 0.8, 4: 0.2}), step(1), step(3, {3: 0.1}), step(3, nodes=[NODE])]
    assert planner_view_of_room(steps, 3, 4)["probability"] == 0.8


def test_a_room_never_offered_has_no_probability():
    steps = [step(1, {4: 0.5}), step(3), step(3, nodes=[NODE])]
    assert planner_view_of_room(steps, 1, 2)["probability"] is None


def test_exits_and_label_come_from_the_rooms_node_during_the_visit():
    steps = [step(1), step(3), step(3, nodes=[{"id": 1, "exits": 5, "label": "hall"}, NODE])]
    seen = planner_view_of_room(steps, 1, 2)
    assert (seen["exits"], seen["label"]) == (2, "bedroom")


def test_a_visit_without_a_planner_room_gives_nothing():
    assert planner_view_of_room([step(None), step(None)], 0, 1) is None


def test_exits_become_a_percent_within_the_episode_in_either_direction():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.planner_scores import exits_percent
    assert exits_percent([1, 3, 2]) == [0.0, 100.0, 50.0]
    assert exits_percent([1, 3, 2], few_first=True) == [100.0, 0.0, 50.0]
    assert exits_percent([2, 2]) == [50.0, 50.0]
