"""The expert label: encoding, truncation, arrival, centring, and the rejections.

Built on a synthetic 2 m-wide corridor rather than the surveyed office, so every
assertion has an analytically known right answer -- the corridor centre-line is
exactly y = 2.0 m, and a label that claims otherwise is wrong rather than
merely different.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues,
)
from sparx_agency.core.mapping.costmap.sdf import compute_sdf
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import expert as E
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import polyline as P
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
    Scene, SceneConfig,
)

RESOLUTION = 0.1
WALL_LOW_M = 1.0      # corridor runs along +x between y = 1.0 and y = 3.0
WALL_HIGH_M = 3.0
CENTRE_Y = 0.5 * (WALL_LOW_M + WALL_HIGH_M)


def corridor_scene() -> Scene:
    """A straight 2 m corridor along +x, 6 m long, walls top and bottom."""
    height, width = 40, 60                     # 4.0 m x 6.0 m at 10 cm
    grid = np.zeros((height, width), dtype=np.int16)
    grid[: int(WALL_LOW_M / RESOLUTION), :] = OccupancyValues().occupied
    grid[int(WALL_HIGH_M / RESOLUTION):, :] = OccupancyValues().occupied
    occupancy = OccupancyGrid2D(
        grid, OccupancyGrid2DParams(resolution=RESOLUTION, origin_x=0.0, origin_y=0.0))
    sdf = compute_sdf((grid != 0).astype(np.uint8), RESOLUTION)
    free = grid == 0
    return Scene(SceneConfig(scene="corridor", inflate_radius_m=0.3,
                             inflate_floor_m=0.25, goal_clearance_m=0.4),
                 occupancy, sdf, free, free, {})


@pytest.fixture(scope="module")
def scene() -> Scene:
    return corridor_scene()


def test_label_encodes_and_decodes_to_the_route(scene):
    """A straight run down the corridor decodes back to a straight forward path."""
    pose = (0.5, CENTRE_Y, 0.0)
    label, reason = E.build_label(scene, pose, (5.5, CENTRE_Y), "mid")
    assert reason is None and label is not None
    assert label.action.shape == (24, 3)
    waypoints = label.waypoints_body
    assert waypoints.shape == (24, 2)
    assert waypoints[-1, 0] == pytest.approx(label.horizon_used_m, abs=0.15)
    assert np.abs(waypoints[:, 1]).max() < 0.15          # no lateral wander
    assert label.turn_deg < 10.0


def test_horizon_is_arc_length_and_a_near_goal_decelerates(scene):
    """A goal inside the horizon spreads over all 24 steps, which is the stop cue."""
    pose = (0.5, CENTRE_Y, 0.0)
    far, _ = E.build_label(scene, pose, (5.5, CENTRE_Y), "far")
    near, _ = E.build_label(scene, pose, (2.0, CENTRE_Y), "near")
    assert far.horizon_used_m == pytest.approx(4.8, abs=0.05)
    assert near.horizon_used_m < 2.0
    assert near.reaches_goal and not far.reaches_goal
    # Per-step displacement is what NavDP's stop rule keys on.
    assert np.abs(near.action[:, 0]).mean() < np.abs(far.action[:, 0]).mean()


def test_centring_pulls_a_wall_hugging_route_toward_the_middle(scene):
    """Starting against a wall, the label should aim for the corridor centre-line."""
    pose = (0.5, WALL_LOW_M + 0.45, 0.0)
    centred, _ = E.build_label(scene, pose, (5.5, WALL_LOW_M + 0.45), "mid",
                               E.ExpertConfig())
    raw, _ = E.build_label(scene, pose, (5.5, WALL_LOW_M + 0.45), "mid",
                           E.ExpertConfig(center=False))
    world_centred = P.to_world(centred.waypoints_body.astype(float), pose)
    world_raw = P.to_world(raw.waypoints_body.astype(float), pose)
    assert abs(world_centred[-1, 1] - CENTRE_Y) < abs(world_raw[-1, 1] - CENTRE_Y)
    assert centred.min_clearance_m >= raw.min_clearance_m - 1e-6


def test_correction_starts_tangent_to_the_aircraft(scene):
    """The first metre must not jump sideways -- no aircraft can fly that."""
    original = np.stack([np.linspace(0.0, 4.0, 40), np.full(40, 1.5)], axis=1)
    corrected = original + np.array([0.0, 0.6])
    clearance = np.full(40, 0.2)
    blended = E.blend_correction(original, corrected, clearance, ramp_m=1.0, taper_m=1.5)
    assert blended[0, 1] == pytest.approx(original[0, 1], abs=1e-9)
    assert blended[-1, 1] > original[-1, 1] + 0.3
    assert np.all(np.diff(blended[:, 1]) >= -1e-9)        # monotone fade-in


def test_openness_taper_leaves_an_already_clear_route_alone(scene):
    """A waypoint with plenty of room is not moved, so open halls add no turning."""
    original = np.stack([np.linspace(0.0, 4.0, 40), np.full(40, 1.5)], axis=1)
    corrected = original + np.array([0.0, 0.6])
    blended = E.blend_correction(original, corrected, np.full(40, 3.0),
                                 ramp_m=1.0, taper_m=1.5)
    assert np.allclose(blended, original)


def test_goal_inside_a_wall_is_rejected_not_quietly_relocated(scene):
    """The weighted A* snaps a blocked goal onto a nearby free cell rather than
    failing. Accepting that would produce a sample whose goal token points into
    a wall while its label goes somewhere else, so it must be dropped."""
    label, reason = E.build_label(scene, (0.5, CENTRE_Y, 0.0), (3.0, 0.2), "mid")
    assert label is None and reason == E.REJECT_GOAL_MOVED


def test_a_label_that_turns_too_hard_is_dropped(scene):
    """A near-reversal inside the horizon is not flyable and is not taught."""
    label, reason = E.build_label(scene, (3.0, CENTRE_Y, 0.0), (5.5, CENTRE_Y), "mid",
                                  E.ExpertConfig(max_turn_deg=1.0))
    assert label is None and reason == E.REJECT_TURN


def test_label_clearance_is_measured_on_the_decoded_action(scene):
    """The audit runs on what the network is asked to output, clamping included."""
    label, _ = E.build_label(scene, (0.5, CENTRE_Y, 0.0), (5.5, CENTRE_Y), "mid")
    world = P.to_world(label.waypoints_body.astype(float), (0.5, CENTRE_Y, 0.0))
    assert label.min_clearance_m == pytest.approx(
        float(scene.clearance(world[:, 0], world[:, 1]).min()), abs=1e-5)


def test_goal_token_is_clipped_the_way_inference_clips_it(scene):
    """Training and deployment must encode a far goal identically."""
    label, _ = E.build_label(scene, (0.5, CENTRE_Y, 0.0), (5.5, CENTRE_Y), "far")
    assert label.goal_token[0] == pytest.approx(5.0, abs=1e-3)   # 5.5 - 0.5 forward
    assert label.goal_world.tolist() == pytest.approx([5.5, CENTRE_Y], abs=1e-5)


def test_body_and_world_transforms_are_exact_inverses():
    pose = (3.2, -1.4, 0.7)
    points = np.array([[1.0, 0.0], [2.0, -0.5], [0.0, 1.5]])
    assert np.allclose(P.to_body(P.to_world(points, pose), pose), points, atol=1e-9)
