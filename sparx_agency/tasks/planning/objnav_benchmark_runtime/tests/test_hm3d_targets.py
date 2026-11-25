from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.hm3d_targets import target_categories
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.target_score import room_holds


def test_raw_names_map_to_the_benchmarks_goal_categories():
    assert target_categories(["armchair", "wall", "tv stand"]) == ["chair"]
    assert target_categories(["couch", "toilet", "Sofa"]) == ["sofa", "toilet"]
    assert target_categories(["potted plant", "monitor", "bunk bed"]) == ["bed", "plant", "tv_monitor"]


def test_names_with_no_goal_category_give_none():
    assert target_categories(["stool", "bar stool", "plant pot", "wall"]) == []


def test_a_room_with_goal_categories_holds_exactly_those():
    room = {"target_categories": ["bed", "chair"], "objects": []}
    assert room_holds(room, "bed") and room_holds(room, "chair") and not room_holds(room, "toilet")


def test_a_room_without_them_is_read_from_its_annotated_names():
    assert room_holds({"objects": ["couch", "lamp"]}, "sofa")
    assert not room_holds({"objects": ["couch"]}, "bed")
