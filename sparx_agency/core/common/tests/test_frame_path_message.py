"""Unit tests for ``core.common.frame_path_message``."""

import pytest

from sparx_agency.core.common.frame_path_message import (
    ParsedFramePath,
    parse_frame_path_message,
    resolve_frame_path,
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


def test_resolve_no_search_dir_returns_original():
    # Empty search_dir = live behaviour: the recorded path is used verbatim.
    assert resolve_frame_path("/tmp/xtend_depth/frame_1.npy") == \
        "/tmp/xtend_depth/frame_1.npy"


def test_resolve_search_dir_hit_wins(tmp_path):
    # The basename exists under search_dir -> that location overrides the
    # (stale) recorded directory. This is the offline/recording fix.
    (tmp_path / "frame_1.npy").write_bytes(b"x")
    got = resolve_frame_path("/tmp/gone/frame_1.npy", str(tmp_path))
    assert got == str(tmp_path / "frame_1.npy")


def test_resolve_search_dir_miss_falls_back(tmp_path):
    # Basename absent under search_dir -> return the original so the caller's
    # load raises and surfaces the real miss (no silent wrong file).
    got = resolve_frame_path("/tmp/xtend_depth/frame_9.npy", str(tmp_path))
    assert got == "/tmp/xtend_depth/frame_9.npy"


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
