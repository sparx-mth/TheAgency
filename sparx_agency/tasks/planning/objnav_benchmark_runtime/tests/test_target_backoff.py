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


def box_at(label, u, v, conf=0.9, half=20, width=640, height=480):
    """A square box of ``2 * half`` px centred on ``(u, v)``, clipped to the frame as a detector's box is."""
    return DetectionWire(label, conf, (max(0.0, u - half), max(0.0, v - half),
                                       min(float(width - 1), u + half), min(float(height - 1), v + half)))


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
        try:
            command = p.plan(obs)
        except AssertionError:
            if until is not None and not p.closing.active:
                break                                        # released inside observe(): the frozen search took over
            raise
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
    # A chair at arm's length fills a good part of the frame: 120 px, not a 40 px sliver.
    actions, poses, phases, lock_step = drive(
        p, ep, target, "chair", lambda step, pose, u, v, d, visible: [box_at("chair", u, v, half=60)] if visible else [])
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
    assert diag["backoff"]["actions"] >= 5 and diag["approach"]["path_m"] >= 0.75 and not diag["approach"]["qualifies"]
    assert diag["backoff"]["retreat_actions"] <= 10 and diag["backoff"]["outcome"].startswith("facing"), "arrived, no idle turns"


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


# -- the manoeuvre runs once per object, and a candidate the wider perspective does not confirm is disproved ------
def test_a_candidate_not_confirmed_from_the_wider_perspective_is_released_for_good_and_not_re_manoeuvred(monkeypatch):
    """Markleeville 2026-10-08: a table read as a bed from 0.9 m, eight manoeuvres, eight unverified releases, each
    re-taken at 0.80 past the 'unverified' memory. Now: one manoeuvre, a non-overridable release, no repeat."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosing
    p, ep = open_floor_policy(monkeypatch)
    target = (0.8, 0.0, 0.5)

    def only_from_close(step, pose, u, v, depth, visible):
        near = math.dist((pose.x, pose.y), target[:2]) <= 1.0
        return [box_at("chair", u, v, conf=0.9, half=60)] if visible and near else []
    actions, poses, phases, lock_step = drive(p, ep, target, "chair", only_from_close, steps=60,
                                              until=lambda policy: not policy.closing.active)
    closing = p.closing
    assert lock_step is None and not closing.active and closing.phase == "RELEASED"
    assert closing.backoffs == 1 and closing.wider_releases == 1 and closing.last_release == TargetClosing.WIDER_RELEASE
    assert closing.rejected[-1][4] == TargetClosing.WIDER_RELEASE
    assert closing.rejected[-1][3] == pytest.approx(2.0 * p.settings.target_closing.rejection_radius_m), "twice the radius"
    released_at = closing.rejected[-1][5]
    # Back at the spot where the box fires, at 0.9 confidence: refused, override or not -- same evidence, same place.
    pose = AgentPose(0.0, 0.0, 0.0, 0.0)
    pixels, depths = project_to_image(np.array([target]), ep.camera, pose)
    p.detector.detect = lambda rgb: [box_at("chair", pixels[0][0], pixels[0][1], conf=0.9, half=60)]
    freeze_global(monkeypatch, p)
    try:
        p.plan(replace(observation(ep, 100, depth=float(depths[0])), pose=pose, target_category="chair"))
    except AssertionError:
        pass                                                         # the search owned the action: no takeover
    assert not closing.active and closing.same_spot_rejections >= 1
    # From a spot well away from the release (more than twice boxed_in_radius_m) the box may start a takeover
    # again -- but no second manoeuvre: this object had its wider look.
    far = AgentPose(1.5, 0.75, 0.0, math.atan2(-0.75, -0.7))
    assert math.dist((far.x, far.y), released_at) > 2.0 * p.settings.target_closing.boxed_in_radius_m
    pixels, depths = project_to_image(np.array([target]), ep.camera, far)
    p.detector.detect = lambda rgb: [box_at("chair", pixels[0][0], pixels[0][1], conf=0.9, half=60)]
    p.plan(replace(observation(ep, 101, depth=float(depths[0])), pose=far, target_category="chair"))
    assert closing.active and closing.backoff is None and closing.backoff_repeats == 1 and closing.backoffs == 1


# -- proactive pitch and the no-step rule (Wiconisco toilet, 2026-10-08) -------------------------------------
def toilet_frame(ep, pose, target, top_m=0.75, half_width=55):
    """A toilet's box as a detector draws it: from the tank top down to where the base meets the floor."""
    centroid, depths = project_to_image(np.array([target]), ep.camera, pose)
    top, _ = project_to_image(np.array([[target[0], target[1], top_m]]), ep.camera, pose)
    base, _ = project_to_image(np.array([[target[0], target[1], 0.0]]), ep.camera, pose)
    u = centroid[0][0]
    box = DetectionWire("toilet", 0.95, (max(0.0, u - half_width), max(0.0, top[0][1]),
                                          min(639.0, u + half_width), min(479.0, base[0][1])))
    return box, float(depths[0])


def test_without_look_actions_no_verify_step_pushes_a_bottom_cut_box_out_of_the_frame(monkeypatch):
    """Wiconisco/000000, action 430: toilet box on the bottom edge at 1.32 m, one MOVE_FORWARD, gone. The step
    is withheld and the lock comes from the next frame; the published protocol has no LOOK."""
    p, ep = open_floor_policy(monkeypatch, label="toilet", backoff=False)
    assert not ep.action_spec.has_camera_tilt
    pose, target = AgentPose(0, 0, 0, 0), (1.32, 0.0, 0.45)
    box, depth = toilet_frame(ep, pose, target)
    assert box.xyxy[3] >= 479, "the frame of the recording: cut by the bottom edge"
    p.detector.detect = lambda rgb: [box]
    command = p.plan(replace(observation(ep, 0, depth=depth), pose=pose, target_category="toilet"))
    assert p.closing.active and p.closing.count == 1
    assert not command.waypoints and command.info.get("verify_step") != "towards the candidate"
    assert p.closing.no_step_holds == 1 and command.info.get("verify_step_withheld")
    p.plan(replace(observation(ep, 1, depth=depth), pose=pose, target_category="toilet"))
    assert p.closing.locked and p.closing.count == 2, "the second frame from the same spot locks it"


def test_with_look_actions_the_verify_step_toward_a_low_target_looks_down_proactively(monkeypatch):
    p, ep = open_floor_policy(monkeypatch, label="toilet", tilt=True, backoff=False)
    pose, target = AgentPose(0, 0, 0, 0), (1.6, 0.0, 0.45)
    box, depth = toilet_frame(ep, pose, target)
    assert box.xyxy[3] < 472, "whole in the frame from 1.6 m"
    p.detector.detect = lambda rgb: [box]
    command = p.plan(replace(observation(ep, 0, depth=depth), pose=pose, target_category="toilet"))
    assert command.info["verify_step"] == "towards the candidate" and command.waypoints
    # 1.35 m after the step, the box's centroid ~0.5 m below the camera: 21 degrees, nearer one tilt step than level.
    assert command.camera_pitch == pytest.approx(ep.action_spec.tilt_angle_rad)
    assert DiscreteActionConverter(ep.action_spec).step(pose, command).action == DiscreteAction.LOOK_DOWN
    # From 2.3 m the predicted angle (14 degrees at 2.05 m) is nearer level: no tilt yet, the box stays whole.
    pose, target = AgentPose(0, 0, 0, 0), (2.3, 0.0, 0.45)
    p, ep = open_floor_policy(monkeypatch, label="toilet", tilt=True, backoff=False)
    box, depth = toilet_frame(ep, pose, target)
    p.detector.detect = lambda rgb: [box]
    command = p.plan(replace(observation(ep, 0, depth=depth), pose=pose, target_category="toilet"))
    assert command.info["verify_step"] == "towards the candidate" and command.camera_pitch == pytest.approx(0.0)


def test_the_approach_carries_the_predicted_pitch_on_every_step_not_only_inside_the_close_band(monkeypatch):
    p, ep = open_floor_policy(monkeypatch, label="toilet", tilt=True, backoff=False)
    target = (2.6, 0.0, 0.45)
    pitches = []

    def detections(step, pose, u, v, depth, visible):
        return [box_at("toilet", u, v, conf=0.95, half=50)] if visible else []
    actions, poses, phases, lock_step = drive(p, ep, target, "toilet", detections, steps=40)
    assert actions[-1] == DiscreteAction.STOP and lock_step is not None
    assert DiscreteAction.LOOK_DOWN in actions
    first_look = actions.index(DiscreteAction.LOOK_DOWN)
    assert math.dist((poses[first_look].x, poses[first_look].y), target[:2]) > 1.3, "looked down before the close band"


def test_a_fresh_near_edge_inside_the_terminal_radius_stops_before_the_standoff():
    """The benchmark measures to the object: a couch whose near edge is 0.95 m away is reached, wherever
    its centroid is -- and a low object without LOOK actions is still in the frame from here."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy
    p, original = setup_policy(target_closing={"backoff": False})
    ep = replace(original, target_category="couch")
    p.reset(ep, gibson_label_mapper().target_labels("couch"))
    far = DetectionWire("couch", .9, (200, 240, 440, 360))             # a couch's height, 1.3 m out
    p.detector.detect = lambda rgb: [far]
    p.plan(replace(observation(ep, 0, depth=1.3), target_category="couch"))
    p.plan(replace(observation(ep, 1, depth=1.3), target_category="couch"))
    assert p.closing.locked
    # The next frame: the box's pixels measure 0.95 m (the near edge of a long couch) while the fused centroid
    # is still beyond the terminal distance.
    obs = replace(observation(ep, 2, depth=1.3), target_category="couch")
    obs.depth_m[240:360, 200:440] = 0.95
    command = p.plan(obs)
    assert command.stop and p.closing.phase == "STOP"
    assert math.dist((0.0, 0.0), p.closing.xyz[:2]) > p.settings.target_closing.terminal_distance_m
    assert command.info["reason"] == "fresh terminal target confirmation"
