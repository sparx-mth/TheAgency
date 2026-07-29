"""The shipped multi-building split, checked against the surveyed maps.

An empty or mis-assigned split is silent: `evaluate.py` refuses to run on an
empty test set, but only after training has already cost hours, and a split that
quietly leaks a training building into the test set reports a number that means
nothing at all. Both are cheap to check here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.splits import (
    SPLITS, load_split_plan,
)

CONFIG = (Path(__file__).resolve().parents[1] / "configs" / "splits_multiscene.yaml")


@pytest.fixture(scope="module")
def plan():
    return load_split_plan(CONFIG)


def test_every_split_has_somewhere_to_come_from(plan):
    """load_split_plan raises on a split with no region; this pins the intent."""
    covered = set(plan.scene_split.values())
    for zones in plan.zones.values():
        covered |= {s for s in SPLITS if zones.boxes.get(s)}
    assert covered == set(SPLITS)


def test_the_test_set_is_an_entire_unseen_building(plan):
    """The point of the multi-scene split.

    An unseen wing of a building the policy trained on shares its architecture,
    lighting and assets. An unseen building does not.
    """
    assert plan.scene_split.get("warehouse_shelves") == "test"


def test_no_building_is_both_trained_on_and_tested_on(plan):
    """The leak that would make the headline number meaningless."""
    for scene, split in plan.scene_split.items():
        assert scene not in plan.zones, (
            f"{scene} is assigned whole to {split} and also carries zones")
    trained = {s for s, v in plan.scene_split.items() if v == "train"}
    tested = {s for s, v in plan.scene_split.items() if v == "test"}
    assert not (trained & tested)


def test_the_office_is_trained_on(plan):
    """It is the only scene with corridors and doorways, and the closest to the
    building the real aircraft flies. Holding it out would train the wrong policy."""
    assert plan.scene_split.get("office") == "train"


def test_a_whole_scene_accepts_any_route(plan):
    """A scene assigned whole has no internal boundary to cross.

    This is what makes the held-out *building* cheaper in samples than a band
    split, which rejects every label whose route leaves its band.
    """
    anywhere = np.array([[-1000.0, -1000.0], [1000.0, 1000.0]])
    assert plan.route_inside("warehouse_shelves", "test", anywhere)


def test_the_validation_band_is_wider_than_the_label_horizon(plan):
    """A band shorter than the expert horizon keeps only labels that point
    along it, which biases the split it selects checkpoints on."""
    from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.expert import (
        ExpertConfig,
    )

    for box in plan.zones["full_warehouse"].boxes["val"]:
        assert box.y_max - box.y_min > 4 * ExpertConfig().horizon_m


def test_the_bands_partition_the_scene_without_overlapping(plan):
    """A point may not be both trained on and validated against."""
    zones = plan.zones["full_warehouse"]
    for y in (-35.0, -20.0, 0.0, 20.0, 30.0):
        hits = [s for s in ("train", "val", "test")
                if any(b.y_min <= y < b.y_max and b.x_min <= 0.0 < b.x_max
                       for b in zones.boxes.get(s, ()))]
        assert len(hits) <= 1, f"y={y} lands in {hits}"


@pytest.mark.parametrize("scene", ["office", "warehouse_shelves", "full_warehouse"])
def test_every_scene_named_here_has_been_surveyed(scene):
    """A split naming a scene with no map produces an empty split, not an error.

    Asks the platform where its maps live rather than counting ``parents[..]``,
    which silently points at the wrong directory the moment a file moves.
    """
    from sparx_agency.robots.PEGASUS.adapters.scene_map import map_path

    assert map_path(scene, 1.5).is_file(), f"{scene} is not surveyed"
