"""The score-once rule: a node is asked about the first action it has a view, never again, and blocked when a visit does not resolve it."""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.object_search.frontier_views.score_ledger import BLOCKED, PENDING, SCORED, ScoreLedger


class Views:
    """What the ledger reads of a node's selection: whether it is usable."""

    def __init__(self, usable):
        self.usable = usable


def test_a_node_is_due_the_first_action_it_has_a_usable_view_and_never_again():
    ledger = ScoreLedger()
    ledger.observe(4, [7, 9])
    assert ledger.pending() == [7, 9]
    assert ledger.due({7: Views(False), 9: Views(True)}) == [9]
    ledger.mark_scored(9, 4, ["step_000004.jpg", "step_000001.jpg"])
    ledger.observe(5, [7, 9])
    assert ledger.pending() == [7] and ledger.due({7: Views(False), 9: Views(True)}) == []
    ledger.observe(11, [7, 9, 12])
    assert ledger.due({7: Views(True), 12: Views(True)}) == [7, 12]         # 7 at last, 12 on the action it appeared
    ledger.mark_scored(7, 11, ["step_000010.jpg"])
    ledger.mark_scored(12, 11, ["step_000011.jpg"])
    nine, seven = ledger.nodes[9], ledger.nodes[7]
    assert (nine.state, nine.scored_step, nine.frames, nine.wait_steps) == (SCORED, 4, ("step_000004.jpg", "step_000001.jpg"), 0)
    assert (seven.first_step, seven.scored_step, seven.wait_steps, seven.last_step) == (4, 11, 7, 11)
    assert ledger.counts() == {PENDING: 0, SCORED: 3, BLOCKED: 0, "nodes": 3}
    with pytest.raises(ValueError, match="scored once"):
        ledger.mark_scored(9, 12, ["step_000012.jpg"])


def test_a_node_that_is_gone_is_not_pending_and_keeps_its_answer_if_it_comes_back():
    ledger = ScoreLedger()
    ledger.observe(0, [1, 2])
    ledger.mark_scored(1, 0, ["step_000000.jpg"])
    ledger.observe(1, [])
    assert ledger.pending() == [] and ledger.due({1: Views(True), 2: Views(True)}) == []
    ledger.observe(8, [1, 2])                                    # the same places again, eight actions on
    assert ledger.pending() == [2]
    assert ledger.nodes[1].state == SCORED and ledger.nodes[1].last_step == 8 and ledger.nodes[2].wait_steps is None
    # A node with no entry in the views (not offered a frame) is not due.
    assert ledger.due({}) == []


def test_a_node_still_there_well_after_its_first_visit_is_blocked_scored_or_not():
    ledger = ScoreLedger(settle_steps=5)
    ledger.observe(0, [1, 2, 3])
    ledger.mark_scored(1, 0, ["step_000000.jpg"])
    ledger.mark_visited(1, 10)
    ledger.mark_visited(2, 10)
    ledger.mark_visited(1, 13)                                   # standing beside it does not restart the clock
    assert ledger.observe(14, [1, 2, 3]) == []
    assert ledger.observe(15, [1, 3]) == [1]                     # five actions after its FIRST visit, still there
    assert ledger.nodes[1].state == BLOCKED and ledger.nodes[1].blocked_step == 15 and ledger.nodes[1].scored_step == 0
    assert ledger.nodes[2].state == PENDING                      # gone when its time came: the visit resolved it
    assert ledger.observe(40, [2]) == [2] and ledger.pending() == []
    with pytest.raises(ValueError, match="blocked"):
        ledger.mark_scored(2, 41, ["step_000041.jpg"])
    assert ledger.observe(41, [1, 2]) == []                      # blocked once
    assert ledger.counts() == {PENDING: 1, SCORED: 0, BLOCKED: 2, "nodes": 3}
    with pytest.raises(KeyError):
        ledger.mark_visited(99, 3)


def test_an_age_keeps_a_passing_node_from_being_asked_about():
    ledger = ScoreLedger(min_age_steps=3)
    ledger.observe(10, [1])
    assert ledger.pending() == [] and ledger.due({1: Views(True)}) == []
    ledger.observe(12, [1, 2])
    assert ledger.pending() == []
    ledger.observe(13, [1, 2])
    assert ledger.pending() == [1] and ledger.due({1: Views(True), 2: Views(True)}) == [1]
    assert ScoreLedger().min_age_steps == 0                      # the rule as decided: on the action it appears


def test_the_ledger_refuses_time_running_backwards_and_settings_that_mean_nothing():
    ledger = ScoreLedger()
    ledger.observe(5, [1])
    with pytest.raises(ValueError, match="older"):
        ledger.observe(4, [1])
    ledger.observe(5, [1])                                       # the same action twice is the same action
    for bad in (dict(settle_steps=0), dict(settle_steps=1.5), dict(min_age_steps=-1), dict(min_age_steps=True)):
        with pytest.raises(ValueError):
            ScoreLedger(**bad)
