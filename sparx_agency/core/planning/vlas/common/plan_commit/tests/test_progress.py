"""Arc length along a polyline, and distance from it."""
import numpy as np
import pytest

from sparx_agency.core.planning.vlas.common.plan_commit.progress import (
    cumulative_arc,
    project,
)

STRAIGHT = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
CORNER = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])


def test_cumulative_arc_counts_metres():
    assert cumulative_arc(STRAIGHT) == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_cumulative_arc_of_a_single_point_is_zero():
    assert cumulative_arc(np.array([[1.0, 2.0]])) == pytest.approx([0.0])


def test_project_onto_a_segment_gives_arc_and_offset():
    arc, lateral, segment = project(STRAIGHT, 1.5, 0.4)
    assert arc == pytest.approx(1.5)
    assert lateral == pytest.approx(0.4)
    assert segment == 1


def test_project_clamps_before_the_start():
    arc, lateral, _ = project(STRAIGHT, -1.0, 0.0)
    assert arc == pytest.approx(0.0)
    assert lateral == pytest.approx(1.0)


def test_project_clamps_past_the_end():
    arc, lateral, _ = project(STRAIGHT, 5.0, 0.0)
    assert arc == pytest.approx(3.0)
    assert lateral == pytest.approx(2.0)


def test_project_rounds_a_corner_on_the_nearer_leg():
    arc, lateral, segment = project(CORNER, 2.3, 1.0)
    assert segment == 1
    assert arc == pytest.approx(3.0)
    assert lateral == pytest.approx(0.3)


def test_project_ignores_zero_length_segments():
    """A repeated vertex projects onto itself from every direction and would
    otherwise win ties against the real geometry sitting on top of it."""
    doubled = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    arc, lateral, _ = project(doubled, 1.5, 0.2)
    assert arc == pytest.approx(1.5)
    assert lateral == pytest.approx(0.2)


def test_project_onto_a_degenerate_polyline_measures_from_its_only_point():
    arc, lateral, _ = project(np.array([[1.0, 1.0], [1.0, 1.0]]), 1.0, 4.0)
    assert arc == pytest.approx(0.0)
    assert lateral == pytest.approx(3.0)


# --- the forward cursor -----------------------------------------------------

HAIRPIN = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.05], [0.0, 0.05]])


def test_a_hairpin_reads_as_finished_from_the_start_without_a_cursor():
    """Why `from_segment` exists. The return leg passes 5 cm from the start, so
    a point at the start is nearest to it and projects 4 m along."""
    arc, _, _ = project(HAIRPIN, 0.02, 0.04)
    assert arc > 3.9


def test_the_cursor_refuses_to_look_behind_itself():
    arc, _, segment = project(HAIRPIN, 0.02, 0.04, from_segment=0)
    assert arc > 3.9 and segment == 2
    arc, lateral, segment = project(HAIRPIN, 0.02, 0.01, from_segment=0)
    assert segment == 0                       # nearest is genuinely the start
    assert project(HAIRPIN, 1.0, 0.02, from_segment=2)[0] > 2.0


def test_a_cursor_past_the_end_clamps_to_the_last_segment():
    """A cursor kept across a re-plan must not index off a shorter route."""
    arc, _, segment = project(HAIRPIN, 0.0, 0.05, from_segment=99)
    assert segment == 2
    assert arc == pytest.approx(4.05)


def test_the_cursor_advances_at_most_a_window_per_call():
    """Progress is earned a window of segments at a time, wherever the query
    point lies. Without the upper bound a single tick could hand back the far
    end of a route the aircraft has not flown, and the commitment would be
    retired on the spot."""
    straight = np.stack([np.arange(40) * 0.2, np.zeros(40)], axis=1)
    queries = ((7.0, 0.0),        # far down the route
               (0.0, 0.0),        # behind the cursor
               (39.0, -5.0))      # long past the end and well off to one side
    for window in (1, 2, 4, 8):
        for cursor in (0, 10, 25):
            for x, y in queries:
                _, _, segment = project(straight, x, y, cursor, window=window)
                assert cursor <= segment <= cursor + window, (
                    f"cursor {cursor} jumped to segment {segment} "
                    f"with window {window}")


def test_a_figure_eight_walked_with_the_cursor_never_reads_backwards():
    """Arc never decreases on a route that crosses itself, as long as the
    segment that comes back is fed in as the next cursor. The two branches of a
    lemniscate meet at zero separation and cross at a right angle, so an
    aircraft 3 cm off the branch it is flying sits almost exactly on the other
    one; an unwindowed nearest-point search reads 2.9 m less arc there, and the
    executor would set about flying the first loop a second time."""
    t = np.linspace(0.0, 2.0 * np.pi, 60)
    lemniscate = np.stack([np.sin(t), np.sin(t) * np.cos(t)], axis=1)
    cursor, arcs = 0, []
    for k in range(1, lemniscate.shape[0]):
        step = lemniscate[k] - lemniscate[k - 1]
        left = np.array([-step[1], step[0]]) / np.linalg.norm(step)
        # Mid-segment, 3 cm to the left of it: where the aircraft really is.
        query = 0.5 * (lemniscate[k - 1] + lemniscate[k]) + 0.03 * left
        arc, _, cursor = project(lemniscate, query[0], query[1], cursor)
        arcs.append(arc)
    assert np.all(np.diff(arcs) >= 0.0), "arc went backwards at the crossing"
    # And it walked the whole figure rather than stalling on one branch.
    assert arcs[-1] > 0.95 * cumulative_arc(lemniscate)[-1]
