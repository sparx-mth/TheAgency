"""The counter that notices InternVLA-N1 has stopped listening.

The numbers in these tests are measurements, not choices. See the module
docstring of ``stop_latch.py`` for where they come from.
"""
import pytest

from sparx_agency.core.planning.vlas.internvla_n1.stop_latch import StopLatch


def test_a_short_run_of_stops_is_left_alone():
    """Up to four in a row is a policy that arrives and carries on.

    Runs of 1, 4 and 6 were all recorded and all recovered without help. Only
    runs of 34 and up never did.
    """
    latch = StopLatch(after=5)
    for _ in range(4):
        assert latch.record(True) is False
    assert latch.run == 4


def test_the_fifth_stop_in_a_row_calls_it():
    latch = StopLatch(after=5)
    for _ in range(4):
        assert latch.record(True) is False
    assert latch.record(True) is True
    assert latch.latches == 1


def test_it_fires_once_not_on_every_frame_after():
    """A restart per frame would hammer the server and never let S2 finish.

    The worst measured run was 406 STOPs. Firing on each of them past the
    threshold would be 402 restarts.
    """
    latch = StopLatch(after=5)
    fired = sum(1 for _ in range(406) if latch.record(True))
    assert fired == 406 // 5, "one restart per five refusals, not one per frame"


def test_anything_the_aircraft_can_act_on_clears_the_run():
    """A policy that has just moved is plainly still listening."""
    latch = StopLatch(after=5)
    for _ in range(4):
        latch.record(True)
    assert latch.run == 4
    assert latch.record(False) is False
    assert latch.run == 0
    for _ in range(4):
        assert latch.record(True) is False, "the count restarted, so four is safe again"


def test_clearing_by_hand_does_not_count_a_latch():
    """For a caller that has just changed the view itself, e.g. a nudge back.

    The next STOP is then the first of a new run, because it is an answer to a
    genuinely different picture.
    """
    latch = StopLatch(after=5)
    for _ in range(4):
        latch.record(True)
    latch.clear()
    assert latch.run == 0
    assert latch.latches == 0
    for _ in range(4):
        assert latch.record(True) is False


def test_the_threshold_is_the_callers():
    latch = StopLatch(after=2)
    assert latch.record(True) is False
    assert latch.record(True) is True


def test_a_threshold_below_one_is_refused():
    """It would restart the agent on the first STOP of every genuine arrival."""
    with pytest.raises(ValueError):
        StopLatch(after=0)


def test_it_runs_on_python_38():
    """core/ is imported by the Noetic containers -- no walrus, no PEP 604."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).with_name("..").resolve() / "stop_latch.py"
    tree = ast.parse(src.read_text())
    banned = (ast.NamedExpr, ast.Match) if hasattr(ast, "Match") else (ast.NamedExpr,)
    assert not [n for n in ast.walk(tree) if isinstance(n, banned)]
