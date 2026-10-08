"""Close-range closing: the backing manoeuvre, the approach-history STOP, support-surface resilience, verification pitch.

Synthetic geometry through the real converter, as the other closing tests;
never a Habitat score.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosingSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_target_closing import freeze_global


def open_floor_policy(monkeypatch, label="chair", tilt=False, **closing):
    """A policy on an open 20 m floor (every cell free), optionally with the tilt-enabled protocol."""
    p, original = setup_policy(target_closing=closing) if closing else setup_policy()
    ep = original
    if tilt or label != "chair":
        ep = replace(original, action_spec=MULTIFLOOR_PROTOCOL.actions() if tilt else original.action_spec,
                     target_category=label)
        p.reset(ep, gibson_label_mapper().target_labels(label))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    return p, ep


def box_at(label, u, v, conf=0.9, half=20):
    return DetectionWire(label, conf, (u - half, v - half, u + half, v + half))


def drive(p, ep, target_xyz, label, detections, steps=80, until=None):
    """Run the policy through the converter against a fixed world target.

    ``detections(step, pose, u, v, depth, visible)`` returns the frame's
    boxes -- or ``(boxes, depth)`` to set the uniform depth image too (the
    frame shows a different surface). Returns ``(actions, poses, phases,
    lock_step)``; the last pose is where STOP was issued, when it was.
    """
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0)
    actions, poses, phases, lock_step = [], [pose], [], None
    for step in range(steps):
        pixels, depths = project_to_image(np.array([target_xyz]), ep.camera, pose)
        u, v = pixels[0]
        visible = depths[0] > 0 and 22 < u < 618 and 22 < v < 458
        frame = detections(step, pose, u, v, float(depths[0]), visible)
        depth = float(depths[0]) if depths[0] > 0.3 else 4.0
        if isinstance(frame, tuple):
            frame, depth = frame
        boxes = list(frame)
        p.detector.detect = lambda rgb, boxes=boxes: boxes
        obs = replace(observation(ep, step, depth=depth), pose=pose, target_category=label)
        command = p.plan(obs)
        phases.append(p.closing.phase)
        if lock_step is None and p.closing.locked:
            lock_step = step
        action = converter.step(pose, command).action
        assert action is not None
        actions.append(action)
        p.notify_action(obs, action)
        if action == DiscreteAction.STOP or (until is not None and until(p)):
            break
        pose = apply_action(pose, action, ep.action_spec)
        poses.append(pose)
    return actions, poses, phases, lock_step


# -- sudden close proximity: back off, face, re-verify, then close -----------------------------------
def test_a_target_seen_suddenly_at_arms_length_is_re_verified_from_farther_back(monkeypatch):
    p, ep = open_floor_policy(monkeypatch)
    target = (0.8, 0.0, 0.5)
    actions, poses, phases, lock_step = drive(
        p, ep, target, "chair", lambda step, pose, u, v, d, visible: [box_at("chair", u, v)] if visible else [])
    closing = p.closing
    assert actions[-1] == DiscreteAction.STOP and closing.locked
    assert closing.backoffs == 1 and closing.backoff_skips == 0
    assert closing.backoff.state == "done" and closing.backoff.source == "ring" and closing.backoff.skipped is None
    assert "BACK_OFF" in phases and phases[0] == "BACK_OFF"
    assert min(pose.x for pose in poses) <= -0.4, "the agent really backed away from the target"
    assert lock_step is not None and closing.backoff.finished < lock_step, "no lock during the manoeuvre"
    assert DiscreteAction.MOVE_FORWARD in actions
    final = poses[-1]
    assert math.dist((final.x, final.y), target[:2]) <= p.settings.target_closing.terminal_distance_m + 0.05
    assert closing.memory.n >= 4 and closing.memory.sigma_m < closing.memory.noise_floor_m
    diag = p.episode_info()["target_closing"]
    assert diag["backoff"]["actions"] >= 5 and diag["approach"]["path_m"] > 1.0 and not diag["approach"]["qualifies"]


def test_no_clear_space_behind_skips_the_manoeuvre_and_verifies_in_place():
    """The default synthetic frame: a wall 0.7 m ahead, nothing observed behind. Two frames STOP, as before."""
    p, ep = setup_policy()
    first = p.plan(observation(ep, 0))
    assert p.closing.active and not first.stop and p.closing.phase == "VERIFY"
    assert p.closing.backoff is not None and p.closing.backoff.state == "done"
    assert p.closing.backoff_skips == 1 and p.closing.backoffs == 0
    assert "no navigable clear space" in p.closing.backoff.skipped
    assert p.plan(observation(ep, 1)).stop and p.closing.locked
    assert p.episode_info()["target_closing"]["backoff"]["skipped"]


def test_a_far_candidate_and_a_disabled_setting_make_no_manoeuvre(monkeypatch):
    p, ep = open_floor_policy(monkeypatch)
    p.plan(observation(ep, 0, depth=2.0))
    assert p.closing.active and p.closing.backoff is None, "2.0 m is beyond sudden_proximity_m"
    p, ep = open_floor_policy(monkeypatch, backoff=False)
    p.plan(observation(ep, 0, depth=0.7))
    assert p.closing.active and p.closing.backoff is None


@pytest.mark.parametrize("kwargs", [{"backoff_steps": 0}, {"backoff_max_actions": 0}, {"sudden_proximity_m": 0.0},
                                    {"backoff": 1}, {"approach_history_m": -1.0}, {"history_stop_after_missing": 0},
                                    {"support_surface_classes": ()}, {"support_surface_resilience": "yes"},
                                    {"memory_noise_floor_m": 0.0}])
def test_close_range_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        TargetClosingSettings(**kwargs)


# -- extended approach history: the trajectory is completed and STOPped ----------------------------------
def near_frames(mode, label="chair"):
    """Boxes along an approach: whole and centred from far; inside 1.15 m, dropped or cropped off-centre."""
    def detections(step, pose, u, v, depth, visible):
        if not visible:
            return []
        if depth >= 1.15:
            return [box_at(label, u, v)]
        if mode == "drop":
            return []
        return [DetectionWire(label, 0.9, (min(u + 230, 598), v - 20, min(u + 270, 639), v + 20))]   # cropped, off-centre
    return detections


@pytest.mark.parametrize("mode", ["drop", "crop"])
def test_a_target_approached_over_metres_is_stopped_on_despite_a_drop_or_a_crop(monkeypatch, mode):
    p, ep = open_floor_policy(monkeypatch)
    target = (4.6, 0.0, 0.5)
    actions, poses, phases, lock_step = drive(p, ep, target, "chair", near_frames(mode), steps=60)
    closing = p.closing
    assert actions[-1] == DiscreteAction.STOP and closing.locked and lock_step is not None
    assert closing.history_stops == 1 and closing.inspection_releases == 0 and closing.releases == 0
    assert closing.history.qualifies(3.0, 1.0) and closing.history.path_m > 3.0
    final = poses[-1]
    assert math.dist((final.x, final.y), target[:2]) <= p.settings.target_closing.terminal_distance_m + 0.05
    reason = p._decision_command.info["reason"]
    if mode == "drop":
        assert reason.startswith("approach history") and "3-D memory" in reason and closing.terminal_missing >= 3
    else:
        assert "alignment not required" in reason
    assert closing.backoff is None, "a candidate first seen from 4.6 m has no manoeuvre"


def test_without_the_history_rule_the_same_drop_releases_the_lock(monkeypatch):
    p, ep = open_floor_policy(monkeypatch, approach_history_m=0.0, max_reacquire_steps=6)
    actions, poses, phases, lock_step = drive(p, ep, (4.6, 0.0, 0.5), "chair", near_frames("drop"), steps=60,
                                              until=lambda policy: not policy.closing.active)
    assert DiscreteAction.STOP not in actions and lock_step is not None
    assert p.closing.inspection_releases == 1 and p.closing.history_stops == 0 and p.closing.phase == "RELEASED"


def test_a_history_stop_from_too_far_resumes_the_approach_instead_of_releasing(monkeypatch):
    """The filtered estimate entered the inspection early and the box dropped: with history, step closer."""
    p, ep = open_floor_policy(monkeypatch, max_reacquire_steps=3, history_stop_after_missing=50)
    target = (4.6, 0.0, 0.5)
    seen = {"near": 0}

    def detections(step, pose, u, v, depth, visible):
        if not visible:
            return []
        if depth >= 1.15:
            return [box_at("chair", u, v)]
        seen["near"] += 1
        return []
    actions, poses, phases, lock_step = drive(p, ep, target, "chair", detections, steps=60)
    assert p.closing.history_holds + p.closing.history_stops >= 1 and p.closing.inspection_releases == 0
    assert actions[-1] == DiscreteAction.STOP


# -- support-surface resilience: the television becomes the dresser -----------------------------------------
def tv_on_dresser(step, pose, u, v, depth, visible, ep=None):
    """The tv box from afar; inside 1.3 m (planar) only a ``cabinet`` box, where the dresser top beneath it shows."""
    if math.dist((pose.x, pose.y), (2.4, 0.0)) >= 1.3:
        return [box_at("tv", u, v)] if visible else []
    pixels, depths = project_to_image(np.array([[2.4, 0.0, 0.7]]), ep.camera, pose)
    cu, cv = pixels[0]
    if not (depths[0] > 0.3 and 22 < cu < 618 and 22 < cv < 458):
        return [], 4.0
    return [box_at("cabinet", cu, cv, conf=0.8, half=30)], float(depths[0])


def test_a_support_surface_at_the_memory_keeps_the_lock_and_the_closing_stops(monkeypatch):
    p, ep = open_floor_policy(monkeypatch, label="tv", tilt=True, max_reacquire_steps=4)
    target = (2.4, 0.0, 1.6)
    actions, poses, phases, lock_step = drive(
        p, ep, target, "tv", lambda *args: tv_on_dresser(*args, ep=ep), steps=60)
    closing = p.closing
    assert actions[-1] == DiscreteAction.STOP and closing.locked
    assert closing.support_sightings >= 1 and closing.support_stops == 1 and closing.inspection_releases == 0
    assert closing.support_xyz is not None and closing.support_xyz[2] < closing.memory.xyz[2]
    assert DiscreteAction.LOOK_UP in actions, "an elevated target is looked UP at inside the close band"
    assert p._decision_command.info["reason"].startswith("support surface consistent")
    final = poses[-1]
    assert math.dist((final.x, final.y), target[:2]) <= p.settings.target_closing.terminal_distance_m + 0.05


def test_without_support_resilience_the_same_frames_release_the_lock(monkeypatch):
    p, ep = open_floor_policy(monkeypatch, label="tv", tilt=True, max_reacquire_steps=4, support_surface_resilience=False)
    actions, poses, phases, lock_step = drive(
        p, ep, (2.4, 0.0, 1.6), "tv", lambda *args: tv_on_dresser(*args, ep=ep), steps=60,
        until=lambda policy: not policy.closing.active)
    assert DiscreteAction.STOP not in actions and lock_step is not None
    assert p.closing.inspection_releases == 1 and p.closing.support_sightings == 0 and p.closing.support_stops == 0


# -- dynamic pitch during close verification -------------------------------------------------------------
@pytest.mark.parametrize("label, z, sign", [("toilet", 0.45, +1), ("tv", 1.3, -1)])
def test_the_verification_step_toward_a_close_candidate_tilts_by_its_elevation(monkeypatch, label, z, sign):
    p, ep = open_floor_policy(monkeypatch, label=label, tilt=True, backoff=False)
    pose = AgentPose(0, 0, 0, 0)
    pixels, depths = project_to_image(np.array([[1.28, 0.0, z]]), ep.camera, pose)
    u, v = pixels[0]
    p.detector.detect = lambda rgb: [box_at(label, u, v)]
    command = p.plan(replace(observation(ep, 0, depth=float(depths[0])), pose=pose, target_category=label))
    assert p.closing.active and command.info["verify_step"] == "towards the candidate"
    assert command.info["elevation"] == -sign, "+1 above the camera is looked UP at (negative pitch), -1 below DOWN"
    assert command.camera_pitch == pytest.approx(sign * ep.action_spec.tilt_angle_rad)
    expected = DiscreteAction.LOOK_DOWN if sign > 0 else DiscreteAction.LOOK_UP
    assert DiscreteActionConverter(ep.action_spec).step(pose, command).action == expected
