"""The fixed representative set: equal scenes, categories as even as the split allows, seeded, and the shipped file is the rule's output."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d import representative_set as rs

CATEGORIES = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")


def synthetic_split():
    """Four scenes of 30 episodes; ``plant`` offered by one scene only, ``sofa`` by none of the last."""
    episodes = {}
    offers = {"s1": ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa"),
              "s2": ("chair", "bed", "toilet", "tv_monitor", "sofa"),
              "s3": ("chair", "bed", "toilet", "sofa"),
              "s4": ("chair", "bed", "toilet", "tv_monitor")}
    for scene, categories in offers.items():
        for index in range(30):
            category = categories[index % len(categories)]
            episode_id = "%s/%06d" % (scene, index)
            episodes[episode_id] = SimpleNamespace(scene_key=scene, category=category, episode_id=episode_id)
    return episodes


def test_the_scene_quotas_are_equal_and_the_remainder_goes_to_the_largest_scenes():
    assert rs.scene_quotas(["a", "b", "c"], {"a": 99, "b": 99, "c": 119}, 9) == {"a": 3, "b": 3, "c": 3}
    assert rs.scene_quotas(["a", "b", "c"], {"a": 99, "b": 99, "c": 119}, 10) == {"a": 3, "b": 3, "c": 4}
    assert rs.scene_quotas(["a", "b", "c"], {"a": 99, "b": 99, "c": 99}, 11) == {"a": 4, "b": 4, "c": 3}, "ties by publisher order"
    with pytest.raises(ValueError):
        rs.scene_quotas(["a"], {"a": 2}, 5)


def test_the_allocation_balances_categories_under_the_per_scene_cap():
    episodes = synthetic_split()
    result = rs.build(episodes, CATEGORIES, size=40, seed=7, cap=4)
    meta = result["meta"]
    assert meta["size"] == 40 and all(sum(per.values()) == 10 for per in meta["per_scene"].values())
    assert all(n <= 4 for per in meta["per_scene"].values() for n in per.values()), "a scene stays mixed"
    assert meta["per_category"]["plant"] == 4, "one scene offers plant: it gives the cap"
    others = [meta["per_category"][c] for c in CATEGORIES if c != "plant"]
    assert max(others) - min(others) <= 2, others
    assert len(set(result["ids"])) == 40 and all(episode_id in episodes for episode_id in result["ids"])
    categories = [episodes[i].category for i in result["ids"]]
    assert {c: categories.count(c) for c in CATEGORIES} == meta["per_category"]


def test_the_draw_is_deterministic_and_a_cell_does_not_reshuffle_when_another_changes():
    episodes = synthetic_split()
    first = rs.build(episodes, CATEGORIES, size=40, seed=7)
    again = rs.build(episodes, CATEGORIES, size=40, seed=7)
    assert first["ids"] == again["ids"]
    other = rs.build(episodes, CATEGORIES, size=40, seed=8)
    assert other["ids"] != first["ids"], "another seed is another set"
    # A cell's draw depends on the seed, the scene and the category alone: another cell beside it changes nothing.
    alone = rs.draw({("s", "c"): ["a", "b", "c", "d"]}, {"s": {"c": 2}}, seed=1)
    beside = rs.draw({("s", "c"): ["a", "b", "c", "d"], ("t", "c"): ["e", "f"]}, {"s": {"c": 2}, "t": {"c": 1}}, seed=1)
    assert [i for i in beside if i in "abcd"] == alone
    # The order is the published one, whatever order the picks were gathered in.
    cells = {("s", "x"): ["s/0", "s/2", "s/4"], ("s", "y"): ["s/1", "s/3", "s/5"]}
    published = ["s/0", "s/1", "s/2", "s/3", "s/4", "s/5"]
    drawn = rs.draw(cells, {"s": {"x": 3, "y": 3}}, seed=1, published_order=published)
    assert drawn == published
    with pytest.raises(ValueError):
        rs.draw({("s", "c"): ["a"]}, {"s": {"c": 2}}, seed=1)


def test_the_ids_come_out_in_published_order_and_the_files_round_trip(tmp_path):
    episodes = synthetic_split()
    result = rs.build(episodes, CATEGORIES, size=12, seed=3)
    order = list(episodes)
    assert result["ids"] == sorted(result["ids"], key=order.index)
    ids_path, meta_path = rs.write("probe", result, tmp_path)
    assert json.loads(ids_path.read_text()) == result["ids"]
    meta = json.loads(meta_path.read_text())
    assert meta["benchmark_number"] is False and meta["seed"] == 3 and len(meta["episodes"]) == 12


SPLIT = Path(os.environ.get("HM3D_EPISODES_DIR", os.path.expanduser("~/datasets/objectnav/hm3d/v1/val")))


@pytest.mark.skipif(not (SPLIT / "val.json.gz").is_file(), reason="HM3D v1 val episodes not installed")
def test_the_shipped_rep200_set_is_the_rules_output_on_the_published_split():
    """``sets/hm3d_v1_val_rep200.json`` (seed 20261111, cap 4): 10 per scene, plant 20, the rest 35-37, 200 in all."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import CATEGORIES as DATASET_CATEGORIES, HM3DDataset
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V1
    dataset = HM3DDataset(SPLIT, None, HM3D_V1, full=True, require_scenes=False)
    levels = rs.load_levels(rs.SETS_DIR / "hm3d_v1_val_levels.csv")
    result = rs.build(dataset.episodes, DATASET_CATEGORIES, 200, seed=20261111, cap=4, levels=levels or None)
    shipped = json.loads((rs.SETS_DIR / "hm3d_v1_val_rep200.json").read_text())
    assert result["ids"] == shipped, "the shipped set must be regenerable: a different list is a different experiment"
    meta = json.loads((rs.SETS_DIR / "hm3d_v1_val_rep200.meta.json").read_text())
    assert meta["size"] == 200 and meta["scenes"] == 20 and all(sum(p.values()) == 10 for p in meta["per_scene"].values())
    assert meta["per_category"]["plant"] == 20 and all(35 <= meta["per_category"][c] <= 37 for c in DATASET_CATEGORIES if c != "plant")
    assert meta["goal_same_level"] + meta["goal_cross_level"] == 200
    assert result["meta"]["per_category"] == meta["per_category"]
