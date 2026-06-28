"""Unit tests for ``core.common.frame_path_message``."""

import pytest

from sparx_agency.core.common.frame_path_message import (
    ParsedFramePath,
    parse_frame_path_message,
)


def test_parses_rgb_example():
    p = parse_frame_path_message(
        "/tmp/xtend_frames/frame_00000216.jpg 1780843795 329645196")
    assert p == ParsedFramePath("/tmp/xtend_frames/frame_00000216.jpg",
                                1780843795, 329645196)


def test_parses_depth_example():
    p = parse_frame_path_message(
        "/tmp/xtend_depth/frame_00006888.npy 1780845414 842679492")
    assert p.path == "/tmp/xtend_depth/frame_00006888.npy"
    assert p.sec == 1780845414
    assert p.nsec == 842679492


def test_stamp_seconds():
    p = parse_frame_path_message("/a/b.npy 10 500000000")
    assert p.stamp_seconds == pytest.approx(10.5)


def test_surrounding_whitespace_is_ignored():
    p = parse_frame_path_message("  /a/b.jpg 1 2 \n")
    assert p == ParsedFramePath("/a/b.jpg", 1, 2)


def test_path_with_spaces_is_preserved():
    # rsplit from the right keeps spaces in the path; only the last two tokens
    # are the stamp.
    p = parse_frame_path_message("/tmp/my frames/f.jpg 7 8")
    assert p.path == "/tmp/my frames/f.jpg"
    assert (p.sec, p.nsec) == (7, 8)


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "/a/b.jpg",            # missing both stamps
    "/a/b.jpg 1",          # missing nsec
    "/a/b.jpg 1 x",        # non-integer nsec
    "/a/b.jpg y 2",        # non-integer sec
])
def test_malformed_messages_raise(bad):
    with pytest.raises(ValueError):
        parse_frame_path_message(bad)
