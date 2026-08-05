"""The two guarantees the earlier fine-tune did not have.

*Goals are never on an obstacle* -- structurally, because they are drawn from
the map's goal-eligible cells rather than back-projected from a pixel.

*Train and test are different places* -- a sample belongs to a split only if the
aircraft is inside that split's region and the whole route it is being taught
stays there.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.goal_sampler import (
    GOAL_KINDS, GoalSampler, GoalSamplerConfig, route_ahead_world,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.splits import (
    Box, SceneZones, SplitPlan, load_split_plan, split_counts,
)
from .test_expert import CENTRE_Y, corridor_scene


@pytest.fixture(scope="module")
def scene():
    return corridor_scene()


# ------------------------------------------------------------------ goals
def test_every_proposed_goal_is_in_the_goal_region(scene):
    """The failure that motivated this package: goals landing on geometry."""
    sampler = GoalSampler(scene, GoalSamplerConfig(goal_stride_cells=2))
    rng = np.random.default_rng(0)
    proposed = list(sampler.candidates((0.6, CENTRE_Y, 0.0), rng, None, limit=200))
    assert len(proposed) > 20
    for candidate in proposed:
        assert scene.in_goal_region(candidate.x, candidate.y), candidate
        assert scene.clearance(candidate.x, candidate.y) > 0.0


def test_goals_behind_the_aircraft_are_never_proposed(scene):
    """NavDP collapses a goal at or behind the camera plane to a fixed stub, so
    such a sample would carry no information about where it was meant to go."""
    sampler = GoalSampler(scene, GoalSamplerConfig(goal_stride_cells=2))
    rng = np.random.default_rng(1)
    pose = (3.0, CENTRE_Y, 0.0)
    for candidate in sampler.candidates(pose, rng, None, limit=150):
        forward = candidate.distance_m * np.cos(candidate.bearing_rad)
        assert forward >= sampler.config.min_forward_m - 1e-9
        assert abs(np.degrees(candidate.bearing_rad)) <= sampler.config.max_bearing_deg


def test_each_kind_stays_inside_its_own_bearing_band(scene):
    """Per-kind bands are what keep the dataset from being almost all turns."""
    config = GoalSamplerConfig(goal_stride_cells=2)
    sampler = GoalSampler(scene, config)
    rng = np.random.default_rng(5)
    for candidate in sampler.candidates((0.6, CENTRE_Y, 0.0), rng, None, limit=400):
        band = config.bearing_deg.get(candidate.kind)
        if band is None:
            continue                                    # 'route' follows the flight
        magnitude = abs(np.degrees(candidate.bearing_rad))
        assert band[0] - 1e-6 <= magnitude <= band[1] + 1e-6, (candidate.kind, magnitude)


def test_kind_mixture_is_respected(scene):
    """Every distance band should appear. The bands are shrunk to fit the 6 m
    test corridor -- the real ones reach 35 m, which no synthetic scene has."""
    sampler = GoalSampler(scene, GoalSamplerConfig(
        goal_stride_cells=2, near_range_m=(0.6, 1.5), mid_range_m=(1.5, 3.0),
        far_range_m=(3.0, 5.5),
        bearing_deg={"near": (0.0, 45.0), "mid": (0.0, 45.0),
                     "far": (0.0, 30.0), "corner": (15.0, 75.0)}))
    rng = np.random.default_rng(2)
    kinds = {c.kind for c in sampler.candidates((0.6, CENTRE_Y, 0.0), rng, None, 400)}
    assert {"near", "mid", "far"} <= kinds <= set(GOAL_KINDS)


def test_route_kind_snaps_the_flown_path_onto_a_legal_cell(scene):
    """The flown path may pass closer to a wall than a goal is allowed to sit."""
    sampler = GoalSampler(scene, GoalSamplerConfig(
        goal_stride_cells=2, kind_weights={"route": 1.0}))
    ahead = np.stack([np.linspace(1.0, 5.0, 40), np.full(40, CENTRE_Y)], axis=1)
    rng = np.random.default_rng(3)
    proposed = list(sampler.candidates((0.6, CENTRE_Y, 0.0), rng, ahead, limit=60))
    assert proposed and all(c.kind == "route" for c in proposed)
    assert all(scene.in_goal_region(c.x, c.y) for c in proposed)


def test_snap_refuses_a_point_that_is_too_far_from_anywhere_legal(scene):
    sampler = GoalSampler(scene, GoalSamplerConfig(snap_radius_m=0.2))
    assert sampler.snap(3.0, CENTRE_Y) is not None
    assert sampler.snap(3.0, -5.0) is None


def test_route_ahead_returns_none_at_the_end_of_a_recording():
    poses = np.zeros((5, 4), dtype=np.float32)
    assert route_ahead_world(poses, 4) is None
    assert route_ahead_world(poses, 0).shape == (5, 2)


# ----------------------------------------------------------------- splits
def zones() -> SceneZones:
    return SceneZones(boxes={
        "train": (Box(-10.0, 0.0, 10.0, 20.0),),
        "val": (Box(-10.0, -10.0, 10.0, 0.0),),
        "test": (Box(-10.0, -30.0, 10.0, -10.0),),
    }, buffer_m=1.5)


def test_anchor_in_the_buffer_strip_is_dropped_not_guessed():
    plan = SplitPlan(zones={"office": zones()}, scene_split={})
    assert plan.assign("office", 0.0, 10.0) == "train"
    assert plan.assign("office", 0.0, -5.0) == "val"
    assert plan.assign("office", 0.0, -20.0) == "test"
    assert plan.assign("office", 0.0, 0.5) is None       # inside the buffer
    assert plan.assign("office", 0.0, -9.5) is None
    assert plan.assign("office", 0.0, 99.0) is None      # outside every region


def test_a_route_leaving_its_region_disqualifies_the_sample():
    plan = SplitPlan(zones={"office": zones()}, scene_split={})
    inside = np.array([[0.0, 5.0], [0.0, 8.0], [0.0, 12.0]])
    crossing = np.array([[0.0, 2.0], [0.0, 0.5], [0.0, -3.0]])
    assert plan.route_inside("office", "train", inside)
    assert not plan.route_inside("office", "train", crossing)


def test_a_whole_scene_can_be_a_split():
    plan = SplitPlan(zones={"office": zones()}, scene_split={"warehouse": "test"})
    assert plan.assign("warehouse", 0.0, 0.0) == "test"
    assert plan.route_inside("warehouse", "test", np.zeros((3, 2)))


def test_a_plan_with_an_empty_split_is_refused(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("scenes:\n  office:\n    train: [[0, 0, 1, 1]]\n")
    with pytest.raises(ValueError, match="no region"):
        load_split_plan(path)


def test_split_counts_tallies_drops():
    counts = split_counts(["train", "train", "val", None, "test", None])
    assert counts == {"train": 2, "val": 1, "test": 1, "dropped": 2}
