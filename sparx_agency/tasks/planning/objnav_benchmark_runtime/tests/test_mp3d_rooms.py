import json
import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_house import read_house, region_box_xz, to_habitat
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_room_types import (
    ROOM_GROUPS, ROOM_POOL, group_of_answer, group_of_label)

HOUSE = """ASCII 1.0
H  -  -  2088 116 95 14 41311 479 1659 14 0  1  0 0 0 0 0 -35 -20 -1 7 26 9  0 0 0 0 0
L  0  0  -  -10 3 0  -34 -17 -1  7 26 9  0 0 0 0 0
R  0 0  0 0  e  -26.0 2.6 0.0  -27.0 1.4 0.0  -24.0 3.8 2.9  2.9  0 0 0 0
R  1 0  0 0  k  -31.0 2.9 0.5  -32.0 1.4 0.5  -27.0 3.8 2.6  2.1  0 0 0 0
R  2 0  0 0  -  -1.0 2.9 0.5  -2.0 1.4 0.5  -1.0 3.8 2.6  2.1  0 0 0 0
C  0  1 wall  1 wall  0 0 0 0 0
C  1  2 door  4 door  0 0 0 0 0
O  0 1 0  -25.6 2.68 1.49  0 0 1  1 0 0  1.5 1.4 1.2  0 0 0 0 0 0 0 0
O  1 1 1  -26.4 1.53 1.28  0 0 1  1 0 0  1.2 0.6 0.07  0 0 0 0 0 0 0 0
O  2 -1 1  -26.4 1.53 1.28  0 0 1  1 0 0  1.2 0.6 0.07  0 0 0 0 0 0 0 0
"""


def house(tmp_path):
    path = tmp_path / "s.house"
    path.write_text(HOUSE)
    return read_house(path)


def test_regions_carry_their_label_character_floor_and_box(tmp_path):
    regions = house(tmp_path)["regions"]
    assert [r["label"] for r in regions] == ["e", "k", "-"]
    assert regions[0]["lo"] == (-27.0, 1.4, 0.0) and regions[0]["hi"] == (-24.0, 3.8, 2.9)


def test_objects_name_their_region_and_category_and_unassigned_objects_are_dropped(tmp_path):
    objects = house(tmp_path)["objects"]
    assert [(o["region"], o["category"]) for o in objects] == [(1, "wall"), (1, "door")]


def test_mp3d_z_up_becomes_habitat_y_up_with_north_flipped():
    assert to_habitat((1.0, 2.0, 3.0)) == (1.0, 3.0, -2.0)


def test_a_regions_box_on_the_floor_plan_is_x_range_and_flipped_y_range(tmp_path):
    assert region_box_xz(house(tmp_path)["regions"][0]) == (-27.0, -3.8, -24.0, -1.4)


@pytest.mark.parametrize("letter, group", [("a", "bathroom"), ("t", "bathroom"), ("l", "living room"), ("f", "living room"), ("n", "living room"),
                                           ("d", "dining room"), ("k", "kitchen"), ("h", "hallway"), ("p", "outdoor"), ("z", "other"), ("-", None)])
def test_label_characters_fold_into_groups(letter, group):
    assert group_of_label(letter) == group


@pytest.mark.parametrize("answer, group", [("Living Room", "living room"), ("a master bedroom", "bedroom"), ("restroom", "bathroom"),
                                           ("family room", "living room"), ("corridor", "hallway"), ("walk-in closet", "closet"),
                                           ("patio", "outdoor"), ("home office", "office"), ("basement", "other"), ("", None), ("spaceship", None)])
def test_a_free_text_answer_is_read_as_a_group_by_its_words(answer, group):
    assert group_of_answer(answer) == group


def test_every_option_the_model_may_be_offered_reads_back_as_its_own_group():
    assert all(group_of_answer(option) in ROOM_GROUPS for option in ROOM_POOL) and set(map(group_of_answer, ROOM_POOL)) == set(ROOM_GROUPS)


from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import in_box_share

CAMERA = dict(width=640, height=480, hfov_deg=79.0)


def depth_of(metres):
    return np.full((480, 640), metres)


def test_a_view_that_only_sees_floor_inside_the_box_is_all_inside():
    assert in_box_share(depth_of(1.0), (0.0, 0.0, 0.0), (-5, -5, 5, 5), **CAMERA) == 1.0


def test_a_view_looking_out_of_the_box_is_all_outside():
    assert in_box_share(depth_of(3.0), (0.0, 0.0, 0.0), (-1, 0.5, 1, 5), margin_m=0.0, **CAMERA) == 0.0


def test_the_share_counts_only_pixels_with_valid_depth():
    depth = depth_of(1.0)
    depth[:, :320] = 0.0
    assert in_box_share(depth, (0.0, 0.0, 0.0), (-5, -5, 5, 5), **CAMERA) == 1.0


def test_a_wall_on_the_edge_of_the_box_counts_inside_with_the_margin():
    assert in_box_share(depth_of(2.2), (0.0, 0.0, 0.0), (-5, -2.0, 5, 5), margin_m=0.3, **CAMERA) == 1.0


def test_category_names_lose_the_hash_spaces_and_annotation_placeholders_are_dropped(tmp_path):
    path = tmp_path / "s.house"
    path.write_text("""R  0 0  0 0  k  0 0 0  0 0 0  1 1 1  1  0 0 0 0
C  0  1 side#table  1 x  0 0 0 0 0
C  1  2 remove  1 x  0 0 0 0 0
C  2  3 unknown  1 x  0 0 0 0 0
C  3  4 wall#/#other#room  1 x  0 0 0 0 0
C  4  5 sofa  1 x  0 0 0 0 0
""" + "".join("O  %d 0 %d  0 0 0  0 0 1  1 0 0  1 1 1  0 0 0 0 0 0 0 0\n" % (i, i) for i in range(5)))
    assert [o["category"] for o in read_house(path)["objects"]] == ["side table", "sofa"]


def test_annotated_object_lists_leave_out_placeholder_names(tmp_path):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.mp3d_text_rooms import object_lists
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene" / "regions.json").write_text(json.dumps({"rooms": [{"room": "scene_r0", "objects": ["bed", "delete", "lamp"]}]}))
    assert object_lists(tmp_path, "annotated", "inside") == {"scene_r0": ["bed", "lamp"]}


def test_a_region_counts_only_once_every_frame_any_method_kept_has_an_answer(tmp_path):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.report_mp3d_selection import VLM_FILE, answers_by_method
    (tmp_path / "scene").mkdir()
    selection = {"done": {"frames": {"a": {}, "b": {}}, "methods": {"floor": ["a", "b"], "random": ["a"]}},
                 "half": {"frames": {"c": {}, "d": {}}, "methods": {"floor": ["c"], "random": ["d"]}}}
    (tmp_path / "scene" / "selection.json").write_text(json.dumps(selection))
    (tmp_path / VLM_FILE).write_text("".join(json.dumps({"room": r, "frame": f, "answer": "kitchen"}) + "\n" for r, f in (("done", "a"), ("done", "b"), ("half", "c"))))
    assert answers_by_method(tmp_path) == {"floor": {"done": ["kitchen", "kitchen"]}, "random": {"done": ["kitchen"]}}
