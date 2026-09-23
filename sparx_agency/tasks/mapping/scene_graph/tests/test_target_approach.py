"""Tests for the target approach — the decisions that end with a landed aircraft.

Nothing here needs rclpy: every judgement the node makes that could be *wrong
rather than broken* lives in ``target_approach_payloads`` /
``target_approach_config``, and this is where each is pinned.

Four things are asserted, because each fails silently in flight:

* **Which class to lock onto.** ``/target_seen/info`` carries both the mission's
  target word and the vocabulary class the matcher actually matched; locking
  onto the wrong one produces a node that flies, tracks nothing, and times out.
* **The RGB->depth bbox rescale.** Both front cameras render 600x600, so a box
  copied across looks right and reads a *different part of the scene*. The pair
  of range tests below is the only place that rescale is checked end to end —
  and the unrescaled answer is asserted too, so "it happens to work" cannot pass.
* **The land streak.** A single close depth reading must never land the drone;
  a sustained one must. This is the difference between a mission that ends
  beside the target and one that puts the aircraft down in a corridor.
* **The status shape.** A renamed key here blanks the operator panel and there
  is no other record of how the mission ended.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sparx_agency.core.planning.visual_servo import (ACQUIRE_STOP, APPROACH,
                                                     LAND, SEARCH,
                                                     TargetConfirmationGate)
from sparx_agency.tasks.mapping.scene_graph.ros2.target_approach_config import (
    PARAM_DEFAULTS, build_fsm, build_gate_config, build_servo, build_tracker)
from sparx_agency.tasks.mapping.scene_graph.ros2.target_approach_payloads import (
    approach_info_payload, bbox_range_m, detections_to_core,
    target_info_from_json)

# Two coaxial 600x600 cameras with different focal lengths: the RGB sensor sees
# twice as wide as the depth sensor, which is the SJTU drone's actual situation
# and the reason a box cannot be copied between them.
RGB_K = (300.0, 300.0, 300.0, 300.0)
DEPTH_K = (600.0, 600.0, 300.0, 300.0)
SIZE = 600


def watcher_info(**overrides):
    """A ``/target_seen/info`` payload shaped exactly as target_watcher emits."""
    payload = {
        "stamp": 1234.5,
        "target": "bed",
        "matched_class": "Hospital Bed",
        "object_id": 7,
        "xy": [3.25, -1.5],
        "count": 4,
        "reason": "llm: a hospital bed is a bed",
    }
    payload.update(overrides)
    return json.dumps(payload)


# ── which class to lock onto ─────────────────────────────────────────
def test_the_matched_class_wins_over_the_target_word():
    """The servo matches DETECTOR labels, so the matched class is what to lock."""
    info = target_info_from_json(watcher_info())
    assert info.lock_class == "hospital bed"     # not "bed"
    assert info.target == "bed"
    assert info.matched_class == "Hospital Bed"
    assert info.object_id == 7
    assert info.xy == pytest.approx((3.25, -1.5))
    assert info.count == 4


def test_target_is_the_fallback_when_no_class_was_matched():
    info = target_info_from_json(watcher_info(matched_class=""))
    assert info.lock_class == "bed"


@pytest.mark.parametrize("text", [
    "not json at all",
    "[1, 2, 3]",                                  # JSON, but not an object
    "null",
    json.dumps({"target": "", "matched_class": ""}),   # nothing to lock onto
    json.dumps({"stamp": 1.0}),                        # neither key present
])
def test_malformed_info_raises_rather_than_locking_onto_nothing(text):
    """A node that shrugged this off would take the aircraft and hunt for nothing."""
    with pytest.raises(ValueError):
        target_info_from_json(text)


def test_a_broken_position_does_not_break_the_lock():
    """xy is reported, never flown to, so a bad one must not lose the class."""
    info = target_info_from_json(watcher_info(xy="nonsense", object_id="x",
                                              count=None))
    assert info.lock_class == "hospital bed"
    assert info.xy == (0.0, 0.0)
    assert info.object_id == -1
    assert info.count == 0


# ── wire detections into our pixel space ─────────────────────────────
def test_detections_are_rescaled_into_our_intrinsics():
    """A box posted at half resolution must double, not be read as-is."""
    items = [{"cls": "wheelchair", "conf": 0.42, "xyxy": [10.0, 20.0, 50.0, 60.0]}]
    dets = detections_to_core(items, src_w=300, src_h=300, dst_w=600, dst_h=600)
    assert len(dets) == 1
    assert dets[0].bbox_xyxy == (20, 40, 100, 120)
    assert dets[0].frame_w == 600 and dets[0].frame_h == 600
    assert dets[0].label == "wheelchair"
    assert dets[0].score == pytest.approx(0.42)


def test_a_malformed_detection_item_raises():
    with pytest.raises(ValueError):
        detections_to_core([{"cls": "chair", "conf": 0.5}], 600, 600, 600, 600)


def test_a_zero_frame_size_raises_instead_of_dividing_by_it():
    with pytest.raises(ValueError):
        detections_to_core([], 0, 600, 600, 600)


def test_the_gate_confirms_on_the_configured_streak():
    """Wire format -> gate: the shipped defaults confirm on 3 frames, not 1."""
    gate = TargetConfirmationGate("hospital bed",
                                  build_gate_config(PARAM_DEFAULTS))
    dets = detections_to_core(
        [{"cls": "hospital bed", "conf": 0.31, "xyxy": [10, 10, 90, 90]}],
        SIZE, SIZE, SIZE, SIZE)
    assert not gate.update(dets).confirmed
    assert not gate.update(dets).confirmed
    state = gate.update(dets)
    assert state.confirmed and state.streak == 3
    assert state.best is not None


def test_a_detection_under_min_score_never_confirms():
    gate = TargetConfirmationGate("hospital bed",
                                  build_gate_config(PARAM_DEFAULTS))
    weak = detections_to_core(
        [{"cls": "hospital bed", "conf": 0.05, "xyxy": [10, 10, 90, 90]}],
        SIZE, SIZE, SIZE, SIZE)
    for _ in range(10):
        assert not gate.update(weak).confirmed


# ── the RGB -> depth rescale, and the range it produces ──────────────
def depth_scene(near_m=1.5, far_m=3.0):
    """A far wall with one near object, placed where the DEPTH camera sees it.

    An RGB box centred at ``(400, 300)`` maps through the two pinholes to
    ``(500, 300)`` in depth pixels, so the near patch is put there. Read with
    the RGB box directly, the same image returns the far wall — which is the
    whole point of the test below.
    """
    depth = np.full((SIZE, SIZE), float(far_m), dtype=np.float32)
    depth[280:320, 460:540] = float(near_m)
    return depth


RGB_BBOX = (380.0, 290.0, 420.0, 310.0)   # centre (400, 300)


def test_the_range_comes_from_the_depth_camera_pixels():
    rng = bbox_range_m(depth_scene(), RGB_BBOX, RGB_K, DEPTH_K)
    assert rng == pytest.approx(1.5, abs=1e-3)


def test_skipping_the_rescale_measures_the_wrong_part_of_the_scene():
    """Load-bearing: without the rescale the same box reads the far wall.

    Both cameras are 600x600, so the wrong answer is a plausible number rather
    than a crash — the drone would simply never reach ``land_range_m`` and
    would hover until the approach timed out.
    """
    wrong = bbox_range_m(depth_scene(), RGB_BBOX, RGB_K, RGB_K)
    assert wrong == pytest.approx(3.0, abs=1e-3)
    assert wrong != pytest.approx(1.5, abs=1e-3)


def test_no_valid_depth_is_a_missing_measurement_not_a_zero():
    """Holes must read as None: a 0.0 range would land the drone instantly."""
    depth = np.zeros((SIZE, SIZE), dtype=np.float32)
    assert bbox_range_m(depth, RGB_BBOX, RGB_K, DEPTH_K) is None
    nan = np.full((SIZE, SIZE), np.nan, dtype=np.float32)
    assert bbox_range_m(nan, RGB_BBOX, RGB_K, DEPTH_K) is None


def test_depth_beyond_the_max_is_rejected():
    far = np.full((SIZE, SIZE), 50.0, dtype=np.float32)
    assert bbox_range_m(far, RGB_BBOX, RGB_K, DEPTH_K, max_depth_m=8.0) is None


# ── the status payload ───────────────────────────────────────────────
EXPECTED_STATUS_KEYS = {
    "stamp", "state", "target", "lock_class", "engaged", "confirmed",
    "streak", "tracking", "range_m", "ticks", "elapsed_s", "ended", "reason",
}


def test_the_status_payload_keeps_its_shape_and_survives_json():
    payload = approach_info_payload(
        stamp=12.0, state=APPROACH, target="bed", lock_class="hospital bed",
        engaged=True, confirmed=True, streak=5, tracking=True,
        range_m=np.float32(1.42), ticks=97, elapsed_s=9.7, ended=False,
        reason="closing")
    assert set(payload) == EXPECTED_STATUS_KEYS
    round_tripped = json.loads(json.dumps(payload))   # numpy would raise here
    assert round_tripped["range_m"] == pytest.approx(1.42, abs=1e-4)
    assert round_tripped["state"] == "APPROACH"
    assert round_tripped["lock_class"] == "hospital bed"
    assert round_tripped["engaged"] is True
    assert round_tripped["ended"] is False


def test_an_unmeasured_range_stays_null_rather_than_becoming_zero():
    payload = approach_info_payload(
        stamp=0.0, state=SEARCH, target="bed", lock_class="bed", engaged=False,
        confirmed=False, streak=0, tracking=False, range_m=None, ticks=0,
        elapsed_s=0.0, ended=False, reason="searching")
    assert json.loads(json.dumps(payload))["range_m"] is None


# ── the flown configuration ──────────────────────────────────────────
def test_the_shipped_defaults_land_outside_the_hover_standoff():
    """"Beside it", not on top of it — and reachable at all."""
    assert PARAM_DEFAULTS["land_range_m"] > PARAM_DEFAULTS["target_range_m"]
    assert build_servo(PARAM_DEFAULTS).p.mode == "holonomic"
    assert build_tracker(PARAM_DEFAULTS).propagates      # detect-once/track-many


def test_a_land_range_inside_the_standoff_is_refused():
    """The servo stops closing at target_range_m, so this could never fire."""
    bad = dict(PARAM_DEFAULTS, land_range_m=0.4, target_range_m=0.5)
    with pytest.raises(ValueError, match="land_range_m"):
        build_fsm(bad)


def test_an_unknown_lock_mode_is_refused():
    with pytest.raises(ValueError, match="lock_mode"):
        build_tracker(dict(PARAM_DEFAULTS, lock_mode="telepathy"))


# ── the scripted approach: what actually lands the aircraft ──────────
DT = 0.1                       # 10 Hz, the shipped approach_rate_hz


def run_to_approach(fsm):
    """Drive the machine from SEARCH through ACQUIRE_STOP into APPROACH."""
    decision = fsm.update(confirmed=True, track_valid=True, at_target=False,
                          dt=DT, range_m=4.0)
    assert decision.mode == ACQUIRE_STOP
    while fsm.state == ACQUIRE_STOP:
        decision = fsm.update(confirmed=True, track_valid=True,
                              at_target=False, dt=DT, range_m=4.0)
    assert fsm.state == APPROACH
    return decision


def close(fsm, range_m):
    """One APPROACH tick at a given range."""
    return fsm.update(confirmed=True, track_valid=True, at_target=False,
                      dt=DT, range_m=range_m)


def test_nothing_is_driven_while_the_target_is_not_confirmed():
    """The guarantee that today's flight is unchanged: SEARCH drives nothing."""
    fsm = build_fsm(PARAM_DEFAULTS)
    for _ in range(50):
        decision = fsm.update(confirmed=False, track_valid=False,
                              at_target=False, dt=DT)
        assert decision.mode == SEARCH
        assert decision.drive_cmd_vel is False
        assert decision.land is False


def test_a_sustained_closing_run_reaches_land():
    fsm = build_fsm(PARAM_DEFAULTS)
    run_to_approach(fsm)
    for far in (3.0, 2.5, 2.0, 1.5, 1.05):        # all outside land_range_m=1.0
        assert close(fsm, far).mode == APPROACH

    ticks = int(PARAM_DEFAULTS["land_confirm_ticks"])
    for _ in range(ticks - 1):                    # in range, streak not yet full
        assert close(fsm, 0.9).mode == APPROACH
    decision = close(fsm, 0.9)                    # the tick that commits
    assert decision.mode == LAND
    assert decision.land is True
    assert decision.drive_cmd_vel is False


def test_one_spurious_close_reading_does_not_land_the_aircraft():
    """A single depth glitch mid-approach must not put the drone down."""
    fsm = build_fsm(PARAM_DEFAULTS)
    run_to_approach(fsm)
    for _ in range(40):
        assert close(fsm, 3.0).mode == APPROACH
        assert close(fsm, 0.4).mode == APPROACH   # the glitch: one in-range tick
    assert fsm.state == APPROACH


def test_an_interrupted_streak_starts_over():
    """Three in-range ticks then one out-of-range must not land on the fourth."""
    fsm = build_fsm(PARAM_DEFAULTS)
    run_to_approach(fsm)
    for _ in range(int(PARAM_DEFAULTS["land_confirm_ticks"]) - 1):
        assert close(fsm, 0.9).mode == APPROACH
    assert close(fsm, 2.0).mode == APPROACH       # streak broken
    assert close(fsm, 0.9).mode == APPROACH       # would have been the commit
    assert fsm.state == APPROACH


def test_a_missing_range_breaks_the_streak_too():
    """No measurement is not "still close": holes must not accumulate toward LAND."""
    fsm = build_fsm(PARAM_DEFAULTS)
    run_to_approach(fsm)
    for _ in range(20):
        assert close(fsm, 0.9).mode == APPROACH
        assert close(fsm, None).mode == APPROACH
    assert fsm.state == APPROACH


def test_land_is_terminal_and_a_late_detection_cannot_restart_the_approach():
    fsm = build_fsm(PARAM_DEFAULTS)
    run_to_approach(fsm)
    for _ in range(int(PARAM_DEFAULTS["land_confirm_ticks"])):
        close(fsm, 0.9)
    assert fsm.state == LAND
    for range_m in (5.0, None, 0.2):
        decision = fsm.update(confirmed=True, track_valid=True, at_target=True,
                              dt=DT, range_m=range_m)
        assert decision.mode == LAND
        assert decision.land is True
        assert decision.drive_cmd_vel is False
