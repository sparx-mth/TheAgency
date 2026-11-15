"""The converter's path geometry, one fact at a time.

Every measure of progress the converter takes reduces to one of these
functions, so each is pinned here against hand-computed values: how long a
path is, where a point projects onto it and that the projection only ever
looks forward, where the point a given distance along the path is, and where
the path leaves a disc round the agent.
"""
from __future__ import annotations

import math
import random

import pytest

from sparx_agency.core.planning.objnav.action_converter.path_geometry import (
    arclength_beyond,
    path_length,
    point_at_arclength,
    project_onto_path,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError

#: An L: 4 m east, then 4 m north.
ELL = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))
#: A hairpin: 4 m east, 1 m north, 4 m back west, 1 m from the way out.
HAIRPIN = ((0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (0.0, 1.0))


# -- lengths and projection ---------------------------------------------------

def test_path_length_sums_the_segments():
    """Remaining-path bookkeeping and the rollout budget both rest on it."""
    assert path_length(((0.0, 0.0), (3.0, 0.0), (3.0, 4.0))) == 7.0


def test_a_one_point_path_has_zero_length():
    """A single goal point is a legal path; its length is not an error."""
    assert path_length(((2.0, 5.0),)) == 0.0


def test_an_empty_path_is_refused_everywhere():
    """There is nothing to project onto, and a silent 0 would read as arrived."""
    for call in (lambda: path_length(()),
                 lambda: project_onto_path((), 0.0, 0.0),
                 lambda: point_at_arclength((), 1.0)):
        with pytest.raises(ObjNavError):
            call()


@pytest.mark.parametrize("points", [
    ((0.0, 0.0), (1.0,)),
    ((0.0, 0.0), (float("nan"), 1.0)),
    ((0.0, 0.0), (True, 1.0)),
])
def test_a_malformed_waypoint_is_refused(points):
    """A NaN would lose every distance comparison and silently skip its segment."""
    with pytest.raises(ObjNavError):
        path_length(points)


def test_a_point_projects_onto_its_nearest_segment():
    """Segment, arc length, cross-track and foot are what every result reports."""
    assert project_onto_path(ELL, 2.0, 1.0) == (0, 2.0, 1.0, (2.0, 0.0))
    assert project_onto_path(ELL, 5.0, 3.0) == (1, 7.0, 1.0, (4.0, 3.0))


def test_a_point_before_the_start_projects_onto_the_first_waypoint():
    """Clamping to the segment keeps the arc length at 0, never negative."""
    assert project_onto_path(ELL, -1.0, 0.0) == (0, 0.0, 1.0, (0.0, 0.0))


def test_a_point_equally_near_two_legs_stays_on_the_earlier_one():
    """A tie going to the later leg would skip the whole hairpin."""
    segment, arc, cross_track, _ = project_onto_path(HAIRPIN, 2.0, 0.5)
    assert (segment, arc, cross_track) == (0, 2.0, 0.5)


def test_segments_before_from_segment_are_never_candidates():
    """On the way back, the nearby outbound leg must not pull progress back."""
    assert project_onto_path(HAIRPIN, 2.0, 0.1, from_segment=2) == (
        2, 7.0, 0.9, (2.0, 1.0))


def test_from_segment_is_clamped_into_the_path():
    """A cursor from a longer, earlier path cannot index past this one."""
    assert project_onto_path(HAIRPIN, 2.0, 0.1, from_segment=-3) == \
        project_onto_path(HAIRPIN, 2.0, 0.1)
    assert project_onto_path(HAIRPIN, 2.0, 0.1, from_segment=99)[0] == 2


def test_the_arc_bound_keeps_the_projection_ahead_of_progress():
    """Walking back along a segment must not run progress backwards."""
    assert project_onto_path(ELL, 1.0, 0.0, from_arc_m=3.0) == (
        0, 3.0, 2.0, (3.0, 0.0))


def test_an_arc_bound_past_the_end_leaves_only_the_last_waypoint():
    """Progress at the goal projects onto the goal, wherever the agent is."""
    segment, arc, cross_track, foot = project_onto_path(ELL, 0.0, 0.0,
                                                        from_arc_m=50.0)
    assert (segment, arc, foot) == (1, 8.0, (4.0, 4.0))
    assert cross_track == pytest.approx(math.hypot(4.0, 4.0))


def test_the_default_bounds_search_the_whole_path():
    """The bounds only ever narrow the search; the defaults narrow nothing."""
    for x, y in ((5.0, 3.0), (2.0, 0.5), (-1.0, -1.0)):
        assert project_onto_path(HAIRPIN, x, y, 0, 0.0) == \
            project_onto_path(HAIRPIN, x, y)


#: An out-and-back whose return runs 5 cm beside its way out.
BESIDE = ((0.0, 0.0), (5.0, 0.0), (5.0, 0.05), (0.0, 0.05))


def test_within_the_band_an_earlier_leg_keeps_a_point_a_later_leg_passes_nearer():
    """Nearest-wins hands the point 4 cm off the way out to the return leg 1 cm away, skipping the walk to the far end."""
    assert project_onto_path(BESIDE, 2.0, 0.04)[0] == 2
    segment, arc, cross_track, foot = project_onto_path(
        BESIDE, 2.0, 0.04, keep_within_m=0.134)
    assert (segment, arc, foot) == (0, 2.0, (2.0, 0.0))
    assert cross_track == pytest.approx(0.04)


def test_within_the_band_a_leg_walked_past_its_end_gives_way_to_the_next():
    """A leg the point has walked off the end of must not hold the progress at its end vertex."""
    folds = ((0.0, 0.0), (-0.5, 0.0), (-0.75, 0.0), (-0.5, 0.0), (-0.75, 0.0))
    # Segment 0 ends 0.126 m away, inside the band; the point stands on 1.
    segment, _, cross_track, foot = project_onto_path(
        folds, -0.62, -0.04, keep_within_m=0.134)
    assert (segment, foot) == (1, (-0.62, 0.0))
    assert cross_track == pytest.approx(0.04)


def test_beyond_the_band_the_nearest_leg_still_wins():
    """A point standing on the return leg, a metre from the way out, is on the return leg."""
    assert project_onto_path(HAIRPIN, 2.0, 1.0, keep_within_m=0.134) == (
        2, 7.0, 0.0, (2.0, 1.0))
    assert project_onto_path(HAIRPIN, 2.0, 0.5, keep_within_m=0.134) == \
        project_onto_path(HAIRPIN, 2.0, 0.5)


def test_a_one_point_path_projects_onto_that_point():
    """A lone goal has one segment index, 0, and no arc length."""
    assert project_onto_path(((1.0, 1.0),), 4.0, 5.0) == (
        0, 0.0, 5.0, (1.0, 1.0))


def test_zero_length_segments_do_not_break_projection_or_interpolation():
    """Planners emit duplicate points; a division by zero there would crash a run."""
    points = ((0.0, 0.0), (0.0, 0.0), (2.0, 0.0))
    assert project_onto_path(points, 1.0, 1.0) == (1, 1.0, 1.0, (1.0, 0.0))
    assert point_at_arclength(points, 1.0) == (1.0, 0.0)


def test_the_projection_arc_never_exceeds_the_path_length():
    """The remaining path is length minus arc, and must never go negative."""
    rng = random.Random(7)
    points = tuple((rng.uniform(0.0, 10.0), rng.uniform(0.0, 10.0))
                   for _ in range(8))
    total = path_length(points)
    for _ in range(500):
        _, arc, _, _ = project_onto_path(
            points, rng.uniform(-2.0, 12.0), rng.uniform(-2.0, 12.0),
            from_segment=rng.randint(0, 7),
            from_arc_m=rng.uniform(0.0, total + 1.0))
        assert 0.0 <= arc <= total


# -- arc length -------------------------------------------------------------

def test_point_at_arclength_interpolates_along_the_path():
    """The lookahead point is found this way."""
    assert point_at_arclength(ELL, 2.0) == (2.0, 0.0)
    assert point_at_arclength(ELL, 6.0) == (4.0, 2.0)


def test_point_at_arclength_clamps_to_the_ends():
    """A lookahead past the goal aims at the goal, not beyond it."""
    assert point_at_arclength(ELL, -1.0) == (0.0, 0.0)
    assert point_at_arclength(ELL, 100.0) == (4.0, 4.0)


def test_the_end_of_the_path_is_exactly_the_last_waypoint():
    """The converter knows it aims at the goal only if this is exact."""
    points = ((0.1, 0.2), (0.7, 0.3), (1.3, 2.9))
    assert point_at_arclength(points, path_length(points)) == points[-1]


def test_a_nan_arclength_is_refused():
    """Clamping NaN would silently aim at the start of the path."""
    with pytest.raises(ObjNavError):
        point_at_arclength(ELL, float("nan"))


@pytest.mark.parametrize("call", [
    lambda: project_onto_path(ELL, float("inf"), 0.0),
    lambda: project_onto_path(ELL, 0.0, 0.0, from_segment=1.5),
    lambda: project_onto_path(ELL, 0.0, 0.0, from_segment=True),
    lambda: project_onto_path(ELL, 0.0, 0.0, from_arc_m=float("nan")),
    lambda: project_onto_path(ELL, 0.0, 0.0, keep_within_m=float("nan")),
    lambda: project_onto_path(ELL, 0.0, 0.0, keep_within_m=-0.1),
])
def test_invalid_numbers_are_refused(call):
    """A NaN compares false with everything and would pick a segment silently."""
    with pytest.raises(ObjNavError):
        call()


# -- leaving a disc -----------------------------------------------------------

def test_arclength_beyond_finds_where_the_path_leaves_the_disc():
    """A fold's aim is moved exactly this far; farther would cut the fold's corner."""
    assert arclength_beyond(ELL, 2.0, 0.0, 0.5, from_arc_m=2.0) == \
        pytest.approx(2.5)
    # From 3.8 m the corner is still inside; the path leaves on the next leg.
    assert arclength_beyond(ELL, 4.0, 0.0, 0.5, from_arc_m=3.8) == \
        pytest.approx(4.5)


def test_arclength_beyond_returns_the_start_when_it_is_already_outside():
    """Only an aim inside the disc is moved."""
    assert arclength_beyond(ELL, 0.0, 0.0, 0.5, from_arc_m=1.0) == 1.0


def test_arclength_beyond_ends_at_the_goal_when_the_rest_stays_inside():
    """The aim never runs past the last waypoint."""
    assert arclength_beyond(ELL, 4.0, 3.5, 1.0, from_arc_m=7.5) == \
        path_length(ELL)


def test_arclength_beyond_walks_an_exact_fold_along_its_return_leg():
    """The aim leaves the agent's own foot along the way back, not across it."""
    fold = ((0.0, 0.0), (0.0, 2.0), (0.0, -1.0))
    assert arclength_beyond(fold, 0.0, 1.75, 0.125, from_arc_m=2.25) == \
        pytest.approx(2.375)


@pytest.mark.parametrize("call", [
    lambda: arclength_beyond(ELL, 0.0, 0.0, 0.0),
    lambda: arclength_beyond(ELL, 0.0, 0.0, -1.0),
    lambda: arclength_beyond(ELL, float("nan"), 0.0, 0.5),
    lambda: arclength_beyond(ELL, 0.0, 0.0, 0.5, from_arc_m=float("inf")),
    lambda: arclength_beyond((), 0.0, 0.0, 0.5),
], ids=["zero-radius", "negative-radius", "nan-centre", "inf-start",
        "empty-path"])
def test_arclength_beyond_refuses_invalid_numbers(call):
    """A zero radius would switch the fold guard off; a NaN would compare false everywhere."""
    with pytest.raises(ObjNavError):
        call()
