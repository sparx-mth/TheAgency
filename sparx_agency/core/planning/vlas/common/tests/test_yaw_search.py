"""Looking around when the policy declines to move."""
import math

import pytest

from sparx_agency.core.planning.vlas.common.yaw_search import (
    YawSearch,
    YawSearchSpec,
)


def test_it_looks_at_the_goal_first():
    """On an open route this is the only heading that matters."""
    search = YawSearch()
    assert search.heading(1.0, 1.0, 0.0) == pytest.approx(1.0)
    assert search.offset == pytest.approx(0.0)


def test_it_holds_a_heading_long_enough_to_ask_from_it():
    """A sweep that outruns the policy visits headings without ever asking."""
    search = YawSearch(YawSearchSpec(dwell_s=1.5))
    assert search.heading(0.0, 0.0, 10.0) == pytest.approx(0.0)
    assert search.heading(0.0, 0.0, 11.0) == pytest.approx(0.0)   # not yet
    assert search.heading(0.0, 0.0, 11.6) != pytest.approx(0.0)   # moved on


def test_the_dwell_starts_only_once_the_aircraft_has_arrived():
    """Otherwise a slow turn eats the time meant for asking."""
    search = YawSearch(YawSearchSpec(dwell_s=1.0))
    for t in (0.0, 1.0, 2.0, 3.0):                 # still 90 deg away throughout
        assert search.heading(0.0, math.pi / 2, t) == pytest.approx(0.0)
    assert search.offset == pytest.approx(0.0)     # never advanced


def sweep(search, steps, bearing=0.0, dwell=1.0):
    """Drive the search with an aircraft that turns to each heading instantly.

    The heading it is holding has to be the one the search last asked for, or
    the dwell clock keeps restarting and the sweep never advances -- which is
    also true in the air, and is why the dwell only starts on arrival.
    """
    seen, clock, yaw = [], 0.0, bearing
    for _ in range(steps):
        yaw = search.heading(bearing, yaw, clock)
        seen.append(round(math.degrees(yaw)))
        clock += dwell + 0.1
    return seen


def test_the_search_widens_either_side_of_the_goal():
    """Alternating keeps it near the goal direction as long as possible."""
    ordered = []
    for heading in sweep(YawSearch(YawSearchSpec(dwell_s=1.0)), steps=8):
        if not ordered or ordered[-1] != heading:
            ordered.append(heading)
    assert ordered[:5] == [0, 45, -45, 90, -90]


def test_a_full_sweep_wraps_and_is_counted():
    search = YawSearch(YawSearchSpec(offsets=(0.0, 1.0), dwell_s=0.5))
    headings = sweep(search, steps=6, dwell=0.5)
    assert search.sweeps >= 1
    assert set(headings) == {0, 57}                 # both offsets were tried


def test_finding_a_route_puts_the_search_back_on_the_goal():
    search = YawSearch(YawSearchSpec(dwell_s=0.5))
    search.heading(0.0, 0.0, 0.0)
    search.heading(0.0, 0.0, 1.0)
    assert search.offset != pytest.approx(0.0)
    search.reset()
    assert search.offset == pytest.approx(0.0)
    assert search.heading(2.0, 2.0, 5.0) == pytest.approx(2.0)


def test_the_heading_is_relative_to_the_goal_and_wraps():
    search = YawSearch()
    assert search.heading(math.pi - 0.1, 0.0, 0.0) == pytest.approx(math.pi - 0.1)


def test_a_search_with_no_headings_is_rejected():
    with pytest.raises(ValueError):
        YawSearchSpec(offsets=())


def test_a_non_positive_dwell_is_rejected():
    """Zero dwell advances every tick and asks the policy from nowhere."""
    with pytest.raises(ValueError):
        YawSearchSpec(dwell_s=0.0)
