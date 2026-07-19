"""Tests for the flight-log sink behind thought_logger_node.

``ThoughtJournal`` is deliberately ROS-free, so these drive it directly -- no
rospy stubs needed. The contract worth locking in is durability: a journal whose
lines are still buffered when the battery goes has recorded nothing about
exactly the flight you wanted to read back.

Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 venv/bin/python -m pytest \
        sparx_agency/tasks/planning/falcon/adapter/scripts/tests/test_thought_journal.py
"""
import pathlib
import sys

import pytest

_SCRIPTS = str(pathlib.Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from sparx_agency.core.common.thought_message import Thought  # noqa: E402
from thought_journal import ThoughtJournal, default_journal_path  # noqa: E402


def _thought(text="No route at 0.40m", level="warn", source="astar_planner",
             category="plan", stamp=12.345):
    return Thought(stamp=stamp, text=text, category=category, level=level,
                   source=source)


def test_writes_a_readable_line_per_thought(tmp_path):
    path = tmp_path / "thoughts.log"
    j = ThoughtJournal(str(path), wall_clock=lambda: 1_700_000_000.25)
    j.write(_thought())
    j.close()
    body = path.read_text()
    assert "astar_planner[plan]" in body
    assert "WARN" in body
    assert "No route at 0.40m" in body
    assert j.lines == 1


def test_ros_stamps_are_absolute_so_writers_share_one_scale(tmp_path):
    """Several nodes append to ONE file, so the ROS column must be absolute: a
    per-journal origin would make the column mean something different on every
    other line. Absolute is also what lines a thought up with a bag."""
    path = tmp_path / "thoughts.log"
    j = ThoughtJournal(str(path), wall_clock=lambda: 1_700_000_000.0)
    j.write(_thought(text="first", stamp=100.0))
    j.write(_thought(text="second", stamp=101.5))
    j.close()
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    assert "100.00" in lines[0] and "101.50" in lines[1]


def test_many_writers_share_one_file_with_one_header(tmp_path):
    """Every narrating node opens its own journal on the same path. The header
    belongs to the FILE, not the writer, or a twelve-node stack starts its log
    with twelve headers."""
    path = tmp_path / "thoughts.log"
    planner = ThoughtJournal(str(path))
    follower = ThoughtJournal(str(path))
    mapper = ThoughtJournal(str(path))
    planner.write(_thought(text="planner thought", source="astar_planner"))
    follower.write(_thought(text="follower thought", source="waypoint_follower"))
    mapper.write(_thought(text="mapper thought", source="mapping_sync"))
    body = path.read_text()
    assert body.count("# falcon thought journal") == 1
    for text in ("planner thought", "follower thought", "mapper thought"):
        assert text in body
    # Every line must be whole: interleaved appends would corrupt the record.
    for line in body.splitlines():
        assert line.startswith("#") or "[" in line, "torn line: %r" % line


def test_the_size_cap_is_shared_across_writers(tmp_path):
    """A per-writer byte count would let N nodes write N times the cap between
    them, which on the Orin is how the rootfs fills mid-mission."""
    path = tmp_path / "thoughts.log"
    first = ThoughtJournal(str(path), max_bytes=400)
    while not first.capped:
        first.write(_thought(text="filling the journal up"))
    size_at_cap = path.stat().st_size
    # A node starting later must see the file is already full, not start at zero.
    late = ThoughtJournal(str(path), max_bytes=400)
    late.write(_thought(text="from a late-starting node"))
    assert late.capped
    assert "from a late-starting node" not in path.read_text()
    assert path.stat().st_size < size_at_cap + 200


def test_each_line_is_flushed_so_a_kill_keeps_the_record(tmp_path):
    """The whole point: readable on disk BEFORE close()."""
    path = tmp_path / "thoughts.log"
    j = ThoughtJournal(str(path))
    j.write(_thought(text="first thought"))
    # No close(), no flush() -- simulating the node being killed right here.
    assert "first thought" in path.read_text()


def test_appends_rather_than_truncating_an_existing_log(tmp_path):
    path = tmp_path / "thoughts.log"
    ThoughtJournal(str(path)).write(_thought(text="from the first run"))
    ThoughtJournal(str(path)).write(_thought(text="from the second run"))
    body = path.read_text()
    assert "from the first run" in body and "from the second run" in body


def test_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "thoughts.log"
    ThoughtJournal(str(path)).write(_thought())
    assert path.exists()


def test_stops_at_the_size_cap_and_says_so(tmp_path):
    path = tmp_path / "thoughts.log"
    j = ThoughtJournal(str(path), max_bytes=200)
    for i in range(200):
        j.write(_thought(text="thought number %d" % i))
    assert j.capped, "an unbounded journal can fill the root filesystem mid-mission"
    body = path.read_text()
    assert "journal capped" in body
    assert j.lines < 200
    # And it must stop dead rather than keep appending past the notice.
    size_after = path.stat().st_size
    j.write(_thought(text="should not appear"))
    assert path.stat().st_size == size_after
    assert "should not appear" not in path.read_text()


def test_zero_max_bytes_means_unlimited(tmp_path):
    path = tmp_path / "thoughts.log"
    j = ThoughtJournal(str(path), max_bytes=0)
    for i in range(300):
        j.write(_thought(text="thought number %d" % i))
    assert not j.capped
    assert j.lines == 300


def test_close_is_idempotent(tmp_path):
    j = ThoughtJournal(str(tmp_path / "thoughts.log"))
    j.close()
    j.close()          # must not raise on a second shutdown callback


def test_write_after_close_is_a_no_op(tmp_path):
    path = tmp_path / "thoughts.log"
    j = ThoughtJournal(str(path))
    j.close()
    assert j.write(_thought(text="after close")) is False
    assert "after close" not in path.read_text()


def test_default_path_is_timestamped_under_the_given_root(tmp_path):
    import time
    stamp = time.localtime(1_700_000_000.0)
    p = default_journal_path(root=str(tmp_path), now=stamp)
    assert p.startswith(str(tmp_path))
    assert p.endswith(".log") and "thoughts_" in p
    # Two runs a second apart must not collide into one file.
    assert p != default_journal_path(root=str(tmp_path),
                                     now=time.localtime(1_700_000_060.0))


def test_unwritable_path_raises_rather_than_pretending_to_record(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    with pytest.raises((IOError, OSError)):
        ThoughtJournal(str(blocker / "thoughts.log"))
