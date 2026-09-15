"""The AI2-THOR bridge's frames, depth and refusals, with no ai2thor installed.

A frame conversion is the one adapter bug that produces a plausible number: a
mirrored or transposed frame leaves every path length untouched -- a sum of
chords survives any rotation or reflection -- so ``cross_checks.py`` cannot see
it, and only the per-action motion check can. So the central test here drives a
scripted episode through a controller that moves the way AI2-THOR moves, in
AI2-THOR's own frame, and asserts the harness's own ``check_motion`` accepts
every action against the episode's spec.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics, normalize_angle
from sparx_agency.core.planning.objnav.camera_intrinsics import (
    intrinsics_from_hfov, intrinsics_from_vfov,
)
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS, DiscreteAction, DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import (
    KinematicTolerance, check_motion,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.fake_thor import (
    FakeThorController,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.thor.simulator import (
    AI2ThorRGBDSimulator, metric_depth, thor_pose,
)

WIDTH, HEIGHT = 64, 48
ORIGIN_HEIGHT_M = 0.9
CAMERA_OFFSET_M = 0.0312


def camera(min_depth_m=0.1, max_depth_m=5.0):
    return CameraSpec(Intrinsics(WIDTH, HEIGHT, 40.0, 40.0, 31.5, 23.5),
                      height_m=ORIGIN_HEIGHT_M - CAMERA_OFFSET_M,
                      min_depth_m=min_depth_m, max_depth_m=max_depth_m)


def actions():
    return DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0,
                              tilt_angle_deg=30.0, min_pitch_deg=-30.0,
                              max_pitch_deg=30.0, actions=ALL_ACTIONS)


def bridge(controller=None, **overrides):
    controller = controller or FakeThorController(
        WIDTH, HEIGHT, camera_offset_m=CAMERA_OFFSET_M,
        origin_height_m=ORIGIN_HEIGHT_M)
    options = dict(height_m=0.9, origin_height_m=ORIGIN_HEIGHT_M,
                   initialize={"gridSize": 0.25, "rotateStepDegrees": 30},
                   commit_id="deadbeef", width=WIDTH, height=HEIGHT,
                   controller_factory=lambda _: controller)
    options.update(overrides)
    return AI2ThorRGBDSimulator(camera(), actions(), 0.175, **options), controller


# --------------------------------------------------------------- frames


def test_bearing_zero_faces_north_and_rotate_right_turns_clockwise():
    """The two anchors that fix the whole mapping."""
    at = {"x": 1.0, "y": ORIGIN_HEIGHT_M, "z": 2.0}
    facing_z = thor_pose(at, 0.0, 0.0, ORIGIN_HEIGHT_M)
    assert (facing_z.x, facing_z.y) == (1.0, 2.0)
    assert facing_z.z == pytest.approx(0.0)
    assert math.degrees(facing_z.yaw) == pytest.approx(90.0)
    # RotateRight raises AI2-THOR's bearing and must lower the ENU yaw.
    assert math.degrees(thor_pose(at, 30.0, 0.0, ORIGIN_HEIGHT_M).yaw) == pytest.approx(60.0)
    assert math.degrees(thor_pose(at, -30.0, 0.0, ORIGIN_HEIGHT_M).yaw) == pytest.approx(120.0)
    # Bearing 90 is AI2-THOR's +x, which is ENU east.
    assert math.degrees(thor_pose(at, 90.0, 0.0, ORIGIN_HEIGHT_M).yaw) == pytest.approx(0.0)


def test_a_forward_step_advances_along_the_converted_heading():
    """The frame is not mirrored: the realised chord matches the reported yaw."""
    sim, controller = bridge()
    for bearing in (0.0, 30.0, 90.0, 137.0, 210.0, 330.0):
        controller.position = {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0}
        controller.rotation = {"x": 0.0, "y": bearing, "z": 0.0}
        controller.horizon = 0.0
        sim._controller = controller
        sim._event = controller._event()
        before = sim.observe()[2]
        after = sim.step(DiscreteAction.MOVE_FORWARD)[2]
        travelled = math.hypot(after.x - before.x, after.y - before.y)
        course = math.atan2(after.y - before.y, after.x - before.x)
        assert travelled == pytest.approx(0.25, abs=1e-6)
        assert abs(normalize_angle(course - before.yaw)) < 1e-6


def test_a_scripted_episode_passes_the_harness_motion_check():
    """The only check that can see a mirrored or transposed frame."""
    sim, controller = bridge()
    spec = actions()
    tolerance = KinematicTolerance()
    _, _, pose = sim.reset("FloorPlan_Val1_1",
                           {"x": 3.0, "y": ORIGIN_HEIGHT_M, "z": -1.5}, 270.0, 0.0)
    script = [DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_LEFT,
              DiscreteAction.MOVE_FORWARD, DiscreteAction.TURN_RIGHT,
              DiscreteAction.TURN_RIGHT, DiscreteAction.LOOK_DOWN,
              DiscreteAction.MOVE_FORWARD, DiscreteAction.LOOK_UP,
              DiscreteAction.LOOK_UP, DiscreteAction.STOP]
    for action in script:
        after = sim.step(action)[2]
        check_motion(action, pose, after, spec, tolerance)
        pose = after


def test_a_look_refused_at_the_clamp_spends_the_step_without_moving():
    """AI2-THOR fails the LOOK rather than clipping it; the motion check allows that."""
    sim, controller = bridge()
    sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0},
              0.0, 30.0)
    before = sim.observe()[2]
    after = sim.step(DiscreteAction.LOOK_DOWN)[2]
    assert not sim.last_action_succeeded
    assert after.camera_pitch == pytest.approx(before.camera_pitch)
    check_motion(DiscreteAction.LOOK_DOWN, before, after, actions(),
                 KinematicTolerance())


def test_a_blocked_move_keeps_the_pose_and_still_returns_a_frame():
    controller = FakeThorController(WIDTH, HEIGHT, camera_offset_m=CAMERA_OFFSET_M,
                                    origin_height_m=ORIGIN_HEIGHT_M,
                                    blocked=lambda x, z: z > 0.1)
    sim, _ = bridge(controller)
    sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0}, 0.0, 0.0)
    before = sim.observe()[2]
    rgb, depth, after = sim.step(DiscreteAction.MOVE_FORWARD)
    assert not sim.last_action_succeeded
    assert (after.x, after.y) == (before.x, before.y)
    assert rgb.shape == (HEIGHT, WIDTH, 3) and depth.shape == (HEIGHT, WIDTH)
    check_motion(DiscreteAction.MOVE_FORWARD, before, after, actions(),
                 KinematicTolerance())


def test_a_mirrored_conversion_would_be_caught_by_the_motion_check():
    """Guard the guard: if the conversion were mirrored, this test would fail."""
    sim, controller = bridge()
    sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0}, 0.0, 0.0)
    before = sim.observe()[2]
    after = sim.step(DiscreteAction.TURN_LEFT)[2]
    mirrored = type(after)(x=after.x, y=-after.y, z=after.z, yaw=-after.yaw,
                           camera_pitch=after.camera_pitch)
    with pytest.raises(EnvContractError):
        check_motion(DiscreteAction.TURN_LEFT,
                     type(before)(x=before.x, y=-before.y, z=before.z,
                                  yaw=-before.yaw,
                                  camera_pitch=before.camera_pitch),
                     mirrored, actions(), KinematicTolerance())


def test_a_toppled_body_is_refused_rather_than_skewing_every_map():
    sim, controller = bridge()
    sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0}, 0.0, 0.0)
    controller.rotation = {"x": 12.0, "y": 0.0, "z": 0.0}
    sim._event = controller._event()
    with pytest.raises(EnvContractError):
        sim.observe()


# --------------------------------------------------------------- camera


def test_the_challenge_vertical_fov_is_the_79_degree_horizontal_one():
    """``fieldOfView: 63.453048374758716`` is vertical; 79 is the horizontal it means."""
    horizontal = intrinsics_from_hfov(640, 480, 79.0)
    vertical = intrinsics_from_vfov(640, 480, 63.453048374758716)
    assert horizontal.fx == pytest.approx(vertical.fx, abs=1e-9)
    assert horizontal.fy == pytest.approx(vertical.fy, abs=1e-9)


def test_a_build_whose_camera_sits_elsewhere_is_refused():
    controller = FakeThorController(WIDTH, HEIGHT, camera_offset_m=0.25,
                                    origin_height_m=ORIGIN_HEIGHT_M)
    sim, _ = bridge(controller)
    with pytest.raises(EnvContractError):
        sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0},
                  0.0, 0.0)


def test_a_failed_teleport_is_refused_rather_than_silently_restarting_elsewhere():
    class Refusing(FakeThorController):
        def step(self, action=None, **kwargs):
            event = super().step(action=action, **kwargs)
            if isinstance(action, dict):
                event.metadata["lastActionSuccess"] = False
                event.metadata["errorMessage"] = "outside the navmesh"
            return event

    sim, _ = bridge(Refusing(WIDTH, HEIGHT, camera_offset_m=CAMERA_OFFSET_M,
                             origin_height_m=ORIGIN_HEIGHT_M))
    with pytest.raises(EnvContractError):
        sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0},
                  0.0, 0.0)


# ---------------------------------------------------------------- depth


def test_depth_clips_at_both_ends_into_the_shared_convention():
    values = metric_depth(np.array([[0.0, 0.1, 1.0, 5.0, 7.0]], np.float32),
                          camera())
    assert np.isnan(values[0, :2]).all()
    assert values[0, 2] == 1.0
    assert np.isinf(values[0, 3]) and np.isinf(values[0, 4])


def test_ray_depth_is_corrected_to_optical_z_only_when_asked():
    spec = camera()
    flat = np.full((HEIGHT, WIDTH), 2.0, np.float32)
    planar = metric_depth(flat, spec)
    assert planar[0, 0] == pytest.approx(planar[HEIGHT // 2, WIDTH // 2])
    corrected = metric_depth(flat, spec, ray_distance=True)
    # A ray of constant length is nearer the image plane at the corners.
    assert corrected[0, 0] < corrected[HEIGHT // 2, WIDTH // 2] <= 2.0


def test_a_missing_depth_frame_is_refused_rather_than_mapped_as_empty():
    class NoDepth(FakeThorController):
        def _event(self):
            event = super()._event()
            event.depth_frame = None
            return event

    sim, _ = bridge(NoDepth(WIDTH, HEIGHT, camera_offset_m=CAMERA_OFFSET_M,
                            origin_height_m=ORIGIN_HEIGHT_M))
    with pytest.raises(EnvContractError):
        sim.reset("FloorPlan_Val1_1", {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0},
                  0.0, 0.0)


# ------------------------------------------------------------- lifecycle


def test_the_scene_is_reloaded_only_when_it_changes():
    sim, controller = bridge()
    start = {"x": 0.0, "y": ORIGIN_HEIGHT_M, "z": 0.0}
    sim.reset("FloorPlan_Val1_1", start, 0.0, 0.0)
    sim.reset("FloorPlan_Val1_1", start, 90.0, 0.0)
    assert controller.scene_resets == 1
    sim.reset("FloorPlan_Val1_2", start, 0.0, 0.0)
    assert controller.scene_resets == 2


def test_embodiment_and_intrinsics_must_agree_with_the_controller():
    with pytest.raises(ValueError):
        bridge(width=WIDTH * 2)
    with pytest.raises(ValueError):
        bridge(height_m=0.0)


def test_stepping_before_reset_is_refused():
    sim, _ = bridge()
    with pytest.raises(EnvContractError):
        sim.step(DiscreteAction.MOVE_FORWARD)
