"""The approach ledger of the target closing: the path walked and the range closed while tracking one candidate."""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.object_search.closing.approach_history import ApproachHistory


def test_history_sums_the_walked_path_and_the_range_closed_once_per_step():
    history = ApproachHistory((0.0, 0.0), 4.0, step=3)
    history.record((0.25, 0.0), 3.75, step=4)
    history.record((0.25, 0.0), 3.75, step=4)               # the same action twice: ignored
    history.record((0.5, 0.0), 3.5, step=5)
    history.record((0.5, 0.25), 3.6, step=6)               # a side step does not undo the range closed
    assert history.path_m == pytest.approx(0.75) and history.closed_m == pytest.approx(0.5)
    assert history.actions == 3 and history.min_range_m == 3.5
    assert not history.qualifies(3.0, 1.0)
    for step in range(7, 20):
        history.record((0.5 + 0.25 * (step - 6), 0.25), max(0.5, 3.5 - 0.25 * (step - 6)), step)
    assert history.path_m > 3.0 and history.closed_m >= 1.0 and history.qualifies(3.0, 1.0)
    assert history.diagnostics()["start_step"] == 3


def test_a_long_walk_that_never_got_nearer_is_not_an_approach():
    history = ApproachHistory((0.0, 0.0), 2.0, step=0)
    for step in range(1, 20):
        history.record((0.25 * step, 0.0), 2.0 + 0.25 * step, step)   # walking AWAY
    assert history.path_m > 4.0 and history.closed_m == 0.0 and not history.qualifies(3.0, 1.0)


def test_a_history_can_be_disabled_with_a_zero_path_requirement_only_by_the_caller():
    history = ApproachHistory((0.0, 0.0), 2.0, step=0)
    assert history.qualifies(0.0, 0.0), "the caller decides what counts; zero asks nothing"

