"""The semantic sanity check on a target candidate and the verification frame that keeps the box whole.

Allensville/2 of the 5x3 benchmark (``807d9d0107aa``): a kitchen counter front read as ``bed`` from
0.55 m, its surface 0.67-0.70 m above the agent's base, inside a STRONG kitchen (refrigerator,
oven), locked on two frames from one spot and STOPped on, 2.18 m from the real bed. The toilet run
(``b536f3c4fe05``): a centred toilet at 1.18 m counted one frame, the satisfied hold was idle, the
headless agent turned, the box left the frame, and the candidate was released ``unverified``.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosingSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_context import (
    CLASS_HEIGHT_BANDS, height_conflict, room_conflict, suspect)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_target_closing import freeze_global


# -- the pure check ---------------------------------------------------------------------------
def test_a_counter_top_read_as_a_bed_conflicts_by_height_and_a_real_bed_does_not():
    base = 0.178                                                    # Allensville's navmesh height
    assert height_conflict("bed", (3.1, -4.4, base + 0.69), base) == "height:0.69"
    assert height_conflict("bed", (3.1, -4.4, base + 0.52), base) is None, "the real bed, from 1 m"
    assert height_conflict("bed", (3.1, -4.4, base + 0.11), base) is None, "the real bed, from 4 m"
    assert height_conflict("toilet", (0, 0, base + 0.47), base) is None
    assert height_conflict("toilet", (0, 0, base + 0.88), base) == "height:0.88", "a toilet at eye level"
    assert height_conflict("couch", (0, 0, base + 0.74), base) is None, "an alias, within the sofa band"
    assert height_conflict("refrigerator", (0, 0, base + 1.3), base) is None, "no band: no opinion"
    assert height_conflict("bed", None, base) is None
    assert set(CLASS_HEIGHT_BANDS) >= {"bed", "toilet", "sofa", "chair", "potted plant", "television"}


def test_a_strong_kitchen_conflicts_with_a_bed_but_a_weak_one_or_a_confirmed_bed_in_it_does_not():
    target = gibson_label_mapper().target_labels("bed")
    assert room_conflict(target, "kitchen", "strong", ["refrigerator", "oven"]) == "room:kitchen"
    assert room_conflict(target, "kitchen", "weak", ["refrigerator"]) is None, "a weak label is a guess"
    assert room_conflict(target, "bedroom", "strong", ["bed"]) is None
    assert room_conflict(target, "kitchen", "strong", ["oven", "bed"]) is None, "the map confirmed a bed here"
    assert room_conflict(target, "unknown", "strong", []) is None
    assert suspect(target, "bed", (0, 0, 0.9), 0.178, label="kitchen", strength="strong",
                   objects=["oven"]) == "room:kitchen", "the room first"
    assert suspect(target, "bed", (0, 0, 0.9), 0.178) == "height:0.72"
    assert suspect(target, "bed", (0, 0, 0.5), 0.178, label="bedroom", strength="strong") is None
    assert suspect("office chair", "office chair", (0, 0, 2.0), 0.0) is None, "a target the tables do not know"


def test_context_settings_are_validated():
    for bad in ({"context_penalty": 0.0}, {"context_penalty": 1.5}, {"context_confirmation_frames": 1},
                {"context_baseline_m": -0.1}, {"context_check": "yes"}, {"verify_keep_in_frame": 1}):
        with pytest.raises(ValueError):
            TargetClosingSettings(**bad)
    s = TargetClosingSettings()
    assert s.context_penalty == 0.70 and s.context_confirmation_frames == 4 and s.context_baseline_m == 0.30
    assert s.look_down_distance_m == 1.30 and s.override_confidence == 0.80


# -- the closing under the check --------------------------------------------------------------
def bed_policy(monkeypatch, **closing):
    p, original = setup_policy(target_closing=closing) if closing else setup_policy()
    ep = replace(original, target_category="bed")
    p.reset(ep, gibson_label_mapper().target_labels("bed"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    return p, ep


def test_a_bed_box_at_counter_height_is_penalised_and_two_frames_from_one_spot_do_not_lock_it(monkeypatch):
    """The box at the image centre at 2 m projects to camera height (0.88 m above the base): a counter,
    not a bed. At 0.60 it is under the start threshold once penalised; at 0.75 it starts, and needs four
    frames from two viewpoints -- turning in place at the counter never locks."""
    p, ep = bed_policy(monkeypatch)
    p.detector.detect = lambda rgb: [DetectionWire("bed", .60, (280, 200, 360, 280))]
    try:
        p.plan(replace(observation(ep, 0, depth=2), target_category="bed"))
    except AssertionError:
        pass                                   # the frozen search was handed the action: no takeover
    assert not p.closing.active and p.closing.context_rejections == 1
    p.detector.detect = lambda rgb: [DetectionWire("bed", .75, (280, 200, 360, 280))]
    for step in range(1, 5):
        p.plan(replace(observation(ep, step, depth=2), target_category="bed"))
    assert p.closing.active and not p.closing.locked and p.closing.count >= 4
    assert p.closing.suspect == "height:0.88" and p.closing.suspect_locks_held >= 1
    assert p.closing.diagnostics()["suspect"] == "height:0.88"
    # The same box seen after a 0.5 m step: the second viewpoint the lock was waiting for.
    moved = replace(observation(ep, 5, depth=1.5), pose=AgentPose(0.5, 0, 0, 0), target_category="bed")
    p.plan(moved)
    assert p.closing.locked
    # Without the check: two frames, as before.
    q, ep2 = bed_policy(monkeypatch, context_check=False)
    q.detector.detect = lambda rgb: [DetectionWire("bed", .60, (280, 200, 360, 280))]
    for step in (0, 1):
        try:
            q.plan(replace(observation(ep2, step, depth=2), target_category="bed"))
        except AssertionError:
            pass
    assert q.closing.locked and q.closing.context_rejections == 0


def test_a_low_bed_box_is_not_suspect_and_locks_on_two_frames(monkeypatch):
    """A box low in the frame at 1.5 m projects 0.3 m under the camera: a bed's surface."""
    p, ep = bed_policy(monkeypatch)
    k = ep.camera.intrinsics
    row = int(k.cy + k.fy * 0.55 / 1.5)                              # 0.55 m under the camera, 1.5 m out
    p.detector.detect = lambda rgb: [DetectionWire("bed", .60, (200, row - 40, 440, row + 40))]
    p.plan(replace(observation(ep, 0, depth=1.5), target_category="bed"))
    p.plan(replace(observation(ep, 1, depth=1.5), target_category="bed"))
    assert p.closing.locked and p.closing.suspect is None and p.closing.context_rejections == 0


# -- the verification frame ---------------------------------------------------------------------
def tilt_toilet_policy(monkeypatch):
    p, original = setup_policy()
    ep = replace(original, action_spec=MULTIFLOOR_PROTOCOL.actions(), target_category="toilet")
    p.reset(ep, gibson_label_mapper().target_labels("toilet"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    return p, ep


def test_a_centred_low_toilet_too_close_to_step_toward_is_looked_down_at_not_turned_away_from(monkeypatch):
    """Allensville toilet, action 77: centred at 1.18 m, one frame counted, then TURN_LEFT -- the hold
    was satisfied, idle, and idle is a turn. The verification frame is a LOOK_DOWN."""
    p, ep = tilt_toilet_policy(monkeypatch)
    k = ep.camera.intrinsics
    # A toilet whose box touches the bottom edge, 1.18 m out: the frame of the recording.
    box = DetectionWire("toilet", .95, (252, 320, 470, 479))
    p.detector.detect = lambda rgb: [box]
    command = p.plan(replace(observation(ep, 0, depth=1.18), target_category="toilet"))
    assert p.closing.active and p.closing.count == 1 and not p.closing.locked
    assert command.waypoints == () and command.camera_pitch == pytest.approx(ep.action_spec.tilt_angle_rad)
    assert command.info["verify_step"] == "look down at a low target"
    converter = DiscreteActionConverter(ep.action_spec)
    assert converter.step(AgentPose(0, 0, 0, 0), command).action == DiscreteAction.LOOK_DOWN
    # The pitched frame shows the toilet whole: the consecutive frame, and the lock.
    pitched = replace(observation(ep, 1, depth=1.18), pose=AgentPose(0, 0, 0, 0, camera_pitch=ep.action_spec.tilt_angle_rad),
                      target_category="toilet")
    p.detector.detect = lambda rgb: [DetectionWire("toilet", .95, (252, 100, 470, 260))]
    p.plan(pitched)
    assert p.closing.locked and p.closing.count == 2
    with pytest.raises(ValueError):
        TargetClosingSettings(verify_keep_in_frame="no")


def test_a_bottom_cut_toilet_at_1_7_m_is_counted_and_stepped_toward(monkeypatch):
    """Allensville toilet, action 55: ``toilet 0.97`` at 1.69 m, box 113 x 187 px on the bottom edge,
    refused as clipped (``border_rejections`` 3 -> 26) -- never two frames in a row."""
    p, ep = tilt_toilet_policy(monkeypatch)
    p.detector.detect = lambda rgb: [DetectionWire("toilet", .97, (187, 290, 300, 477))]
    command = p.plan(replace(observation(ep, 0, depth=1.7), target_category="toilet"))
    assert p.closing.active and p.closing.border_rejections == 0 and p.closing.count == 1
    assert command.info.get("verify_step") == "towards the candidate" and command.waypoints
    p.plan(replace(observation(ep, 1, depth=1.45), target_category="toilet"))
    assert p.closing.locked, "the second consecutive frame, from one step closer"


# -- the dynamic pitch: the target's elevation decides the direction (2026-10-07) --------------
def tilt_tv_policy(monkeypatch):
    p, original = setup_policy()
    ep = replace(original, action_spec=MULTIFLOOR_PROTOCOL.actions(), target_category="tv")
    p.reset(ep, gibson_label_mapper().target_labels("tv"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    return p, ep


def test_a_wall_mounted_television_above_the_camera_is_looked_up_at_in_verification_not_down(monkeypatch):
    """A hardcoded LOOK_DOWN drove high targets out of the frame. A box in the upper half of a level frame
    at 1.2 m projects 0.5 m ABOVE the camera: the verification frame is a LOOK_UP."""
    p, ep = tilt_tv_policy(monkeypatch)
    k = ep.camera.intrinsics
    row = int(k.cy - k.fy * 0.5 / 1.2)                                   # 0.5 m above the camera, 1.2 m out
    box = DetectionWire("tv", .95, (240, row - 50, 400, row + 50))
    p.detector.detect = lambda rgb: [box]
    command = p.plan(replace(observation(ep, 0, depth=1.2), target_category="tv"))
    assert p.closing.active and p.closing.count == 1 and not p.closing.locked
    assert p.closing._elevation(replace(observation(ep, 0, depth=1.2), target_category="tv"), 1.2) == 1
    assert command.waypoints == () and command.camera_pitch == pytest.approx(-ep.action_spec.tilt_angle_rad)
    assert command.info["verify_step"] == "look up at a high target" and command.info["elevation"] == 1
    action = DiscreteActionConverter(ep.action_spec).step(AgentPose(0, 0, 0, 0), command).action
    assert action == DiscreteAction.LOOK_UP
    # Pitched up, the television sits lower in the frame: the consecutive frame, and the lock.
    up = replace(observation(ep, 1, depth=1.2), pose=AgentPose(0, 0, 0, 0, camera_pitch=-ep.action_spec.tilt_angle_rad),
                 target_category="tv")
    p.detector.detect = lambda rgb: [DetectionWire("tv", .95, (240, 220, 400, 320))]
    p.plan(up)
    assert p.closing.locked and p.closing.count == 2


def test_the_geometric_pitch_commits_to_one_tilt_in_the_targets_direction_inside_the_close_range_band(monkeypatch):
    """``_pitch``: a target slightly above the camera within 1.3 m gets LOOK_UP (not the rounded-to-zero
    level), one slightly below gets LOOK_DOWN, one within the elevation band gets neither; beyond the band
    the geometry alone decides."""
    p, ep = tilt_tv_policy(monkeypatch)
    tilt = ep.action_spec.tilt_angle_rad
    obs = replace(observation(ep, 0, depth=1.2), target_category="tv")
    camera_z = obs.pose.z + ep.camera.height_m
    p.closing.xyz = (1.2, 0.0, camera_z + 0.2)                            # 9.5 degrees up: rounds to level ...
    assert p.closing._pitch(obs, 1.2) == pytest.approx(-tilt), "... but inside the band it is one tilt UP"
    p.closing.xyz = (1.2, 0.0, camera_z - 0.2)
    assert p.closing._pitch(obs, 1.2) == pytest.approx(tilt), "and one tilt DOWN below the camera"
    p.closing.xyz = (1.2, 0.0, camera_z + 0.1)
    assert p.closing._pitch(obs, 1.2) == pytest.approx(0.0), "within elevation_band_m: level"
    p.closing.xyz = (2.5, 0.0, camera_z + 0.2)
    assert p.closing._pitch(obs, 2.5) == pytest.approx(0.0), "beyond the band the geometry rounds to level"
    p.closing.xyz = (2.5, 0.0, camera_z + 1.6)
    assert p.closing._pitch(obs, 2.5) == pytest.approx(-tilt), "a high television at 2.5 m: 33 degrees up"
    with pytest.raises(ValueError):
        TargetClosingSettings(elevation_band_m=0.0)


def test_the_close_approach_pitches_toward_the_target_inside_the_band_and_stays_level_outside_it(monkeypatch):
    """CLOSE used to force a level camera until the terminal inspection. Inside ``look_down_distance_m`` the
    approach carries the target's own pitch: up at a high television, down at a low toilet."""
    p, ep = tilt_tv_policy(monkeypatch)
    tilt = ep.action_spec.tilt_angle_rad
    k = ep.camera.intrinsics
    row = int(k.cy - k.fy * 0.6 / 3.0)                                   # 0.6 m above the camera, 3 m out
    p.detector.detect = lambda rgb: [DetectionWire("tv", .95, (280, row - 40, 360, row + 40))]
    p.plan(replace(observation(ep, 0, depth=3.0), target_category="tv"))
    p.plan(replace(observation(ep, 1, depth=3.0), target_category="tv"))
    assert p.closing.locked and p.closing.xyz[2] > ep.camera.height_m + 0.4
    far = p.plan(replace(observation(ep, 2, depth=3.0), target_category="tv"))
    assert p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED") and far.camera_pitch == pytest.approx(0.0), "level in transit"
    # Standing 1.2 m from it: inside the band, the approach looks UP one tilt.
    near_obs = replace(observation(ep, 3, depth=1.2), pose=AgentPose(1.8, 0, 0, 0), target_category="tv")
    row_near = int(k.cy - k.fy * 0.6 / 1.2)
    p.detector.detect = lambda rgb: [DetectionWire("tv", .95, (280, row_near - 60, 360, row_near + 60))]
    near = p.plan(near_obs)
    if p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED"):
        assert near.camera_pitch == pytest.approx(-tilt), "the approach pitches toward a high target"
    else:
        assert p.closing.phase in ("INSPECT", "STOP") and (near.camera_pitch is None or near.camera_pitch <= 0.0)
    # The same geometry below the camera: a low toilet is looked DOWN at on the way in.
    q, ep2 = tilt_toilet_policy(monkeypatch)
    row_low = int(k.cy + k.fy * 0.5 / 3.0)
    q.detector.detect = lambda rgb: [DetectionWire("toilet", .95, (280, row_low - 40, 360, row_low + 40))]
    q.plan(replace(observation(ep2, 0, depth=3.0), target_category="toilet"))
    q.plan(replace(observation(ep2, 1, depth=3.0), target_category="toilet"))
    assert q.closing.locked
    low_obs = replace(observation(ep2, 2, depth=1.2), pose=AgentPose(1.8, 0, 0, 0), target_category="toilet")
    q.detector.detect = lambda rgb: [DetectionWire("toilet", .95, (280, int(k.cy + k.fy * 0.5 / 1.2) - 60, 360, k.height - 1))]
    low = q.plan(low_obs)
    assert low.camera_pitch is not None and low.camera_pitch >= tilt - 1e-9, "down at a low target, whatever the phase"


def test_a_centred_candidate_without_tilt_turns_toward_the_side_that_keeps_the_box_in_frame(monkeypatch):
    """No LOOK actions in the Gibson protocol: the one turn whose shift leaves the box inside the frame."""
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    k = ep.camera.intrinsics
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (240, 150, 360, 330))]     # centred, 120 px wide
    command = p.plan(observation(ep, 0, depth=1.1))                                           # too close to step
    assert p.closing.active and command.waypoints == () and command.final_yaw is not None
    assert command.info["verify_step"].startswith("turn")
    shift = k.fx * math.tan(ep.action_spec.turn_angle_rad)
    assert abs(command.final_yaw) == pytest.approx(ep.action_spec.turn_angle_rad)
    assert 240 - shift >= 8 or 360 + shift <= k.width - 8, "a 30-degree turn leaves a 120 px box in the frame"
    action = DiscreteActionConverter(ep.action_spec).step(AgentPose(0, 0, 0, 0), command).action
    assert action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT)



