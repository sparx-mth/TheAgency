"""Tests for the consumer-side rolling thought log.

The log is what the operator actually reads, so the tests pin the two properties
that decide whether it stays readable: it is bounded, and a repeated line
collapses instead of flushing the surrounding reasoning off the top.
"""
import pytest

from sparx_agency.core.common.thought_log import ThoughtEntry, ThoughtLog
from sparx_agency.core.common.thought_message import Thought


def _thought(text, stamp=0.0, source="follower", level="info", category="nav"):
    return Thought(stamp=stamp, text=text, category=category, level=level,
                   source=source)


def test_thoughts_are_kept_oldest_first():
    log = ThoughtLog()
    log.add(_thought("Aligning to waypoint 3"))
    log.add(_thought("Flying forward to waypoint 3"))
    assert [e.display_text() for e in log.entries()] == [
        "Aligning to waypoint 3", "Flying forward to waypoint 3"]


def test_add_reports_whether_it_made_a_new_entry():
    log = ThoughtLog()
    assert log.add(_thought("Stopping to turn"))
    assert not log.add(_thought("Stopping to turn", stamp=1.0))
    assert log.add(_thought("Flying forward"))


def test_consecutive_repeats_collapse_with_a_count():
    log = ThoughtLog()
    for i in range(3):
        log.add(_thought("Stopping to turn", stamp=float(i)))
    assert [e.display_text() for e in log.entries()] == ["Stopping to turn (x3)"]
    assert len(log) == 1


def test_a_collapsed_repeat_keeps_the_latest_stamp():
    log = ThoughtLog()
    log.add(_thought("Stopping to turn", stamp=1.0))
    log.add(_thought("Stopping to turn", stamp=9.0))
    assert log.entries()[0].thought.stamp == 9.0


def test_interleaved_repeats_stay_separate_entries():
    # Same line twice, but genuinely separate events -- the drone stopped, flew,
    # and stopped again. Collapsing those would misreport the flight.
    log = ThoughtLog()
    log.add(_thought("Stopping to turn"))
    log.add(_thought("Flying forward"))
    log.add(_thought("Stopping to turn"))
    assert [e.display_text() for e in log.entries()] == [
        "Stopping to turn", "Flying forward", "Stopping to turn"]


def test_same_text_from_a_different_node_does_not_collapse():
    log = ThoughtLog()
    log.add(_thought("Replanning", source="astar_planner"))
    log.add(_thought("Replanning", source="hybrid_planner"))
    assert len(log) == 2


def test_capacity_drops_the_oldest_entries():
    log = ThoughtLog(capacity=3)
    for i in range(5):
        log.add(_thought("thought %d" % i))
    assert [e.display_text() for e in log.entries()] == [
        "thought 2", "thought 3", "thought 4"]


def test_collapsing_does_not_consume_capacity():
    # The point of collapsing: a spamming line must not push the reasoning that
    # explains it out of the log.
    log = ThoughtLog(capacity=3)
    log.add(_thought("Replanning: obstacle on route"))
    for i in range(50):
        log.add(_thought("Stopping to turn", stamp=float(i)))
    assert [e.display_text() for e in log.entries()] == [
        "Replanning: obstacle on route", "Stopping to turn (x50)"]


def test_entries_limit_returns_the_newest():
    log = ThoughtLog()
    for i in range(5):
        log.add(_thought("thought %d" % i))
    assert [e.display_text() for e in log.entries(limit=2)] == [
        "thought 3", "thought 4"]


@pytest.mark.parametrize("n", range(1, 9))
def test_entries_limit_larger_than_the_log_returns_everything(n):
    # The viewer asks for exactly `capacity` lines every frame, so limit > len
    # for the whole pre-fill window. Getting this wrong drops the drone's
    # reasoning silently -- and only in that window, which is the one the panel
    # exists to explain. Swept, because a single case can land in a region that
    # happens to work: limit >= 2*len clamps past the front of the list and
    # returns everything even with a broken slice.
    log = ThoughtLog(capacity=8)
    for i in range(n):
        log.add(_thought("thought %d" % i))
    assert [e.display_text() for e in log.entries(limit=8)] == [
        "thought %d" % i for i in range(n)]


def test_entries_limit_zero_returns_nothing():
    log = ThoughtLog()
    log.add(_thought("only"))
    assert log.entries(limit=0) == []


def test_entries_rejects_a_negative_limit():
    with pytest.raises(ValueError, match="limit must be >= 0"):
        ThoughtLog().entries(limit=-1)


def test_clear_empties_the_log():
    log = ThoughtLog()
    log.add(_thought("Stopping to turn"))
    log.clear()
    assert log.entries() == [] and len(log) == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        ThoughtLog(capacity=0)


def test_entry_without_repeats_has_no_marker():
    assert ThoughtEntry(_thought("Stopping to turn")).display_text() == \
        "Stopping to turn"
