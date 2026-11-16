"""The stop criterion's two measures: what range each reads off the agent's own measurements."""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.object_search.closing.stop_criterion import EUCLIDEAN, MEASURES, PLANAR, stop_range_m


def test_the_planar_measure_reads_the_horizontal_range_whatever_the_height():
    for elevation in (0.0, 0.4, 1.4, -0.2):
        assert stop_range_m(PLANAR, 0.95, elevation) == 0.95


def test_the_euclidean_measure_lifts_the_range_by_the_height_of_the_targets_underside():
    assert stop_range_m(EUCLIDEAN, 0.6, 0.8) == pytest.approx(1.0)
    assert stop_range_m(EUCLIDEAN, 0.0, 0.7) == pytest.approx(0.7), "standing under it: the lift alone"
    # A picture whose underside hangs 0.75 m up is a metre away from the floor 0.66 m out, not a metre out.
    assert stop_range_m(EUCLIDEAN, 0.66, 0.75) == pytest.approx(1.0, abs=0.002)
    assert stop_range_m(EUCLIDEAN, 1.0, 0.75) == pytest.approx(1.25)


def test_an_object_on_the_floor_or_below_it_reads_the_same_by_both_measures():
    for elevation in (0.0, -0.3):
        assert stop_range_m(EUCLIDEAN, 0.9, elevation) == stop_range_m(PLANAR, 0.9, elevation) == 0.9


def test_the_euclidean_reading_is_never_under_the_planar_one():
    for planar in (0.0, 0.3, 0.9, 1.05):
        for elevation in (-0.5, 0.0, 0.3, 0.9, 1.6):
            assert stop_range_m(EUCLIDEAN, planar, elevation) >= stop_range_m(PLANAR, planar, elevation)


def test_the_measures_are_the_two_names_and_nothing_else_is_read():
    assert MEASURES == (PLANAR, EUCLIDEAN) == ("planar", "euclidean")
    for bad in ("geodesic", "3d", "", None):
        with pytest.raises(ValueError, match="stop measure"):
            stop_range_m(bad, 0.9, 0.0)


@pytest.mark.parametrize("planar, elevation", [(-0.1, 0.0), (math.nan, 0.0), (math.inf, 0.0), (0.9, math.nan), (0.9, math.inf)])
def test_a_range_that_is_no_range_is_refused_rather_than_compared(planar, elevation):
    for measure in MEASURES:
        with pytest.raises(ValueError, match="stop range"):
            stop_range_m(measure, planar, elevation)
