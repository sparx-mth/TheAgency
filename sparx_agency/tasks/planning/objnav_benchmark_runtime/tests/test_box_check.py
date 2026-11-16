import json

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.closing.box_check import box_prompt, category_phrase, crop_around_box
from sparx_agency.tasks.planning.objnav_benchmark_runtime.box_check_client import OllamaBoxCheck
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_target_closing import decided_at

RED = (255, 0, 0)


@pytest.fixture
def frame_640x480():
    return np.zeros((480, 640, 3), np.uint8)


def test_the_crop_grows_the_box_by_half_its_size_on_each_side(frame_640x480):
    crop = crop_around_box(frame_640x480, (200, 150, 400, 350))
    assert crop.shape == (400, 400, 3)


def test_a_tiny_box_gets_a_crop_of_at_least_224_pixels(frame_640x480):
    crop = crop_around_box(frame_640x480, (300, 200, 310, 210))
    assert crop.shape[:2] == (224, 224)


def test_the_crop_stays_inside_the_frame_at_the_border(frame_640x480):
    crop = crop_around_box(frame_640x480, (0, 0, 50, 50))
    assert 0 < crop.shape[0] <= 480 and 0 < crop.shape[1] <= 640
    assert tuple(crop[0, 0]) == RED


def test_the_box_is_drawn_in_red_where_it_lies_in_the_crop(frame_640x480):
    crop = crop_around_box(frame_640x480, (200, 150, 400, 350))
    assert tuple(crop[100, 200]) == RED and tuple(crop[200, 100]) == RED
    assert tuple(crop[200, 200]) == (0, 0, 0)


def test_the_source_frame_is_not_drawn_on(frame_640x480):
    crop_around_box(frame_640x480, (100, 100, 200, 200))
    assert not frame_640x480.any()


def test_the_prompt_is_the_fixed_question_and_nothing_else():
    assert box_prompt("couch") == ('In this photo, look only at the object inside the red rectangle. '
                                   'Is it a couch? Answer with JSON {"answer": "yes"} or {"answer": "no"}.')


@pytest.mark.parametrize("category, phrase", [("sofa", "couch"), ("tv_monitor", "tv"), ("plant", "potted plant"),
                                              ("chair", "chair"), ("bed", "bed"), ("toilet", "toilet")])
def test_the_question_uses_the_benchmark_category_name(category, phrase):
    assert category_phrase(category) == phrase


class FakePost:
    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.error:
            raise self.error
        return type("Reply", (), {"json": lambda _: {"message": {"content": self.content}}})()


def verdict(content=None, error=None, frame=None):
    check = OllamaBoxCheck("http://x/api/chat", "m", post=FakePost(content, error))
    rgb = np.zeros((480, 640, 3), np.uint8) if frame is None else frame
    return check.verdict(rgb, (100, 100, 200, 200), "couch"), check


@pytest.mark.parametrize("content, expected", [('{"answer": "yes"}', "yes"), ('{"answer": "no"}', "no"),
                                               ('{"answer": "No"}', "no"), ('{"answer": "maybe"}', "unsure"),
                                               ("not json", "unsure"), ('{"other": 1}', "unsure"), (None, "unsure")])
def test_the_answer_is_parsed_and_anything_else_is_unsure(content, expected):
    assert verdict(content)[0] == expected


@pytest.mark.parametrize("error", [TimeoutError("slow"), ConnectionError("down"), ValueError("bad")])
def test_a_failed_call_is_unsure(error):
    assert verdict(error=error)[0] == "unsure"


def test_the_request_is_one_deterministic_call_with_the_prompt_and_one_image():
    result, check = verdict('{"answer": "yes"}')
    request = check.post.calls[0]["json"]
    assert result == "yes" and len(check.post.calls) == 1
    assert request["model"] == "m" and request["think"] is False and request["options"]["temperature"] == 0
    assert request["messages"][0]["content"] == box_prompt("couch") and len(request["messages"][0]["images"]) == 1
    assert check.post.calls[0]["timeout"] is not None


class FakeCheck:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def verdict(self, rgb, xyxy, category):
        self.calls.append((rgb, tuple(xyxy), category))
        return self.answer


def decided(monkeypatch, answer, **closing):
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84, **closing)
    p.closing.box_check = FakeCheck(answer)
    return p, frame


def test_a_no_releases_the_candidate_and_leaves_a_rejection(monkeypatch):
    p, frame = decided(monkeypatch, "no", box_check_enabled=True)
    command = p.plan(frame)
    assert not command.stop and command.info["kind"] == "target_released"
    assert p.closing.releases == 1 and p.closing.last_release.startswith("vlm") and p.closing.rejected


@pytest.mark.parametrize("answer", ["yes", "unsure"])
def test_a_yes_or_unsure_leaves_the_stop_as_it_was(monkeypatch, answer):
    p, frame = decided(monkeypatch, answer, box_check_enabled=True)
    command = p.plan(frame)
    assert command.waypoints and p.closing.phase == "ADVANCE" and p.closing.releases == 0


def test_the_check_gets_the_full_frame_the_box_and_the_benchmark_category(monkeypatch):
    p, frame = decided(monkeypatch, "yes", box_check_enabled=True)
    p.plan(frame)
    (rgb, xyxy, category), = p.closing.box_check.calls
    assert rgb is frame.rgb and xyxy == (0, 25, 392, 404) and category == "chair"


def test_with_the_flag_off_the_check_is_never_called(monkeypatch):
    p, frame = decided(monkeypatch, "no")
    command = p.plan(frame)
    assert command.waypoints and not p.closing.box_check.calls and p.closing.releases == 0
