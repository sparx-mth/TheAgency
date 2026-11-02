from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosing, TargetClosingSettings
from sparx_agency.core.planning.object_search.closing.vlm_confirmation import (
    MIN_CROP_PX, NO, UNSURE, YES, Verdict, crop_around, parse_answer, question_for)

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_crop_grows_the_box_by_the_margin():
    crop = crop_around(FRAME, (300, 100, 400, 300), margin=0.5)
    assert crop.shape[:2] == (400, 224)


def test_crop_is_never_smaller_than_the_floor_and_stays_in_the_frame():
    assert crop_around(FRAME, (300, 200, 310, 210), margin=0.0).shape[:2] == (MIN_CROP_PX, MIN_CROP_PX)
    assert crop_around(FRAME, (0, 0, 20, 20), margin=0.3).shape[:2] == (MIN_CROP_PX // 2 + 10, MIN_CROP_PX // 2 + 10)


@pytest.mark.parametrize("content, answer", [('{"answer": "No"}', NO), ('{"answer": "yes"}', YES), ('{"answer": "maybe"}', UNSURE),
                                             ("not json", UNSURE), ("[1]", UNSURE), ("{}", UNSURE)])
def test_parse_answer(content, answer):
    assert parse_answer(content) == answer


def test_question_names_the_class_and_defines_confusable_ones():
    assert "not a stool" in question_for("armchair")
    assert "a gizmo" in question_for("gizmo")


@pytest.mark.parametrize("field, value", [("vlm_confirm", 1), ("vlm_model", " "), ("vlm_crop_margin", -1.0)])
def test_invalid_vlm_settings_are_refused(field, value):
    with pytest.raises(ValueError):
        TargetClosingSettings(**{field: value})


class StubConfirmer:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def confirm(self, rgb, box, label):
        self.calls.append((box, label))
        return Verdict(self.answer, 1.0, "stub")


def closing(answer, fresh=True):
    closing = TargetClosing.__new__(TargetClosing)
    closing.settings = TargetClosingSettings()
    closing.confirmer, closing.vlm_checks = StubConfirmer(answer), []
    closing.box, closing.label, closing.conf, closing.last_seen = (10, 10, 50, 50), "chair", 0.8, 7 if fresh else 3
    closing.policy = SimpleNamespace(episode=SimpleNamespace(action_spec=SimpleNamespace(turn_angle_rad=0.26)))
    closing.released = []
    closing._release = lambda obs, radius_factor, why: closing.released.append((radius_factor, why))
    return closing


OBS = SimpleNamespace(step=7, rgb=FRAME, pose=SimpleNamespace(yaw=0.0))


def test_no_releases_the_candidate_and_turns():
    c = closing(NO)
    command = c._vlm_veto(OBS)
    assert command is not None and c.released == [(1.0, "VLM: not a chair")]
    assert c.vlm_checks[0]["answer"] == NO


@pytest.mark.parametrize("answer", [YES, UNSURE])
def test_yes_and_unsure_leave_the_stop_alone(answer):
    c = closing(answer)
    assert c._vlm_veto(OBS) is None and c.released == [] and len(c.vlm_checks) == 1


def test_a_stale_box_is_not_asked_about():
    c = closing(NO, fresh=False)
    assert c._vlm_veto(OBS) is None and c.confirmer.calls == []


def test_the_adapter_attaches_only_when_asked():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.vlm_confirmer import OllamaConfirmer, attach_confirmer

    def policy(on):
        settings = SimpleNamespace(target_closing=TargetClosingSettings(vlm_confirm=on))
        return SimpleNamespace(settings=settings, target_confirmer=None)
    off, on = policy(False), policy(True)
    attach_confirmer(off), attach_confirmer(on)
    assert off.target_confirmer is None and isinstance(on.target_confirmer, OllamaConfirmer)


def test_a_dead_service_is_unsure():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.vlm_confirmer import OllamaConfirmer

    verdict = OllamaConfirmer("m", "http://127.0.0.1:9", 0.3, timeout_s=2).confirm(FRAME, (10, 10, 50, 50), "chair")
    assert verdict.answer == UNSURE and verdict.detail.startswith("error")
