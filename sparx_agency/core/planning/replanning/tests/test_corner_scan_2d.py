"""Unit tests for the hard-turn corner scan (the hybrid A*->NavDP engage signal).

Locks in the behaviour the FALCON ``hybrid`` mode relies on:
  * a straight route has no hard turn;
  * a genuine sharp corner is reported at its along-route distance and full angle;
  * the distance shrinks MONOTONICALLY as the drone approaches and the turn stays
    flagged for the whole approach within the engage range (the robustness the
    redesign buys: a reliable multi-tick confirm, not a razor-thin band);
  * a corner past the engage range or already behind the drone is NOT flagged;
  * the span-based measure catches a corner the planner split into two vertices and
    ignores a grid A* weave (jitter cancels between entry and exit).
"""
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.replanning.corner_scan_2d import (
    net_turn_at_arclength_2d,
    scan_hard_turn_ahead,
)

INF = float("inf")


def _pts(*xy):
    return [Pose2D(float(x), float(y)) for x, y in xy]


# ─── net_turn_at_arclength_2d ────────────────────────────────────────────────
def test_net_turn_at_vertex_right_angle():
    pts = _pts((0, 0), (2, 0), (2, 2))
    cum = [0.0, 2.0, 4.0]
    assert abs(net_turn_at_arclength_2d(pts, cum, 1, span_m=0.7, total_m=4.0) - 90.0) < 1.0


def test_net_turn_at_vertex_straight_is_zero():
    pts = _pts((0, 0), (1, 0), (2, 0))
    cum = [0.0, 1.0, 2.0]
    assert net_turn_at_arclength_2d(pts, cum, 1, span_m=0.7, total_m=2.0) < 1e-6


# ─── scan_hard_turn_ahead: basic detection ───────────────────────────────────
def test_straight_route_has_no_hard_turn():
    dist, deg, mx, _ = scan_hard_turn_ahead(
        _pts((0, 0), (5, 0)), Pose2D(0, 0), turn_thresh_deg=70.0)
    assert dist == INF and deg == 0.0 and mx == 0.0


def test_sharp_corner_reported_at_its_distance():
    dist, deg, mx, _ = scan_hard_turn_ahead(
        _pts((0, 0), (2, 0), (2, 3)), Pose2D(0, 0), turn_thresh_deg=70.0,
        max_scan_m=4.0)
    assert abs(dist - 2.0) < 1e-6          # the corner is 2 m along the route
    assert abs(deg - 90.0) < 1.0
    assert mx >= deg


def test_max_turn_deg_spans_full_range_not_just_first_hard_corner():
    # Two hard corners: a 90 deg at 1 m and a sharp ~174 deg reversal at 2 m. hard_dist
    # / hard_deg report the NEAREST hard corner; max_turn_deg is the sharpest in range
    # (regression: an early break understated it as 90 rather than ~174).
    route = _pts((0, 0), (1, 0), (1, 1), (0.9, 0))
    dist, deg, mx, _ = scan_hard_turn_ahead(
        route, Pose2D(0, 0), turn_thresh_deg=70.0, span_m=0.3, max_scan_m=8.0)
    assert abs(dist - 1.0) < 1e-6 and abs(deg - 90.0) < 2.0
    assert mx > 150.0          # the far reversal, not just the nearer 90 deg corner


def test_corner_beyond_engage_range_is_not_flagged():
    # Corner 3 m ahead, engage range only 2 m: A* keeps flying (not close enough).
    dist, deg, _, _ = scan_hard_turn_ahead(
        _pts((0, 0), (3, 0), (3, 3)), Pose2D(0, 0), turn_thresh_deg=70.0,
        max_scan_m=2.0)
    assert dist == INF and deg == 0.0


def test_corner_behind_the_drone_is_not_flagged():
    # Drone has flown past the corner: nothing hard ahead (the "return to A*" side).
    dist, _, _, _ = scan_hard_turn_ahead(
        _pts((0, 0), (2, 0), (2, 3)), Pose2D(2, 2), turn_thresh_deg=70.0,
        max_scan_m=4.0)
    assert dist == INF


def test_corner_within_skip_is_ignored():
    # A corner right under the drone (< skip_m ahead) must not read as "ahead".
    dist, _, _, _ = scan_hard_turn_ahead(
        _pts((0, 0), (0.2, 0), (0.2, 3)), Pose2D(0, 0), turn_thresh_deg=70.0,
        skip_m=0.5, max_scan_m=4.0)
    assert dist == INF


# ─── the redesign's core property: a monotone, wide approach band ────────────
def test_distance_is_monotone_and_band_is_wide_on_approach():
    route = _pts((0, 0), (3, 0), (3, 3))          # corner at (3, 0)
    dists = []
    x = 0.0
    while x <= 3.0 + 1e-9:
        d, _, _, _ = scan_hard_turn_ahead(
            route, Pose2D(x, 0.0), turn_thresh_deg=70.0, skip_m=0.3, max_scan_m=2.0)
        dists.append(d)
        x += 0.25
    finite = [d for d in dists if d != INF]
    # Flagged for a wide band of positions (not one tick) -> the confirm streak fires.
    assert len(finite) >= 6
    # And the reported distance-to-turn shrinks monotonically as the drone closes in.
    for a, b in zip(finite[:-1], finite[1:]):
        assert b <= a + 1e-9


# ─── robustness: split corner caught, weave ignored ──────────────────────────
def test_span_catches_a_corner_split_into_two_vertices():
    # A 90 deg turn the planner discretized as two ~45 deg vertices: each vertex
    # angle is below 70, but the net turn across the span is the full ~90.
    route = _pts((0, 0), (2, 0), (2.2, 0.2), (2.2, 3))
    dist, deg, _, _ = scan_hard_turn_ahead(
        route, Pose2D(0, 0), turn_thresh_deg=70.0, span_m=0.7, max_scan_m=4.0)
    assert dist != INF and deg >= 70.0


def test_weaving_corridor_is_not_a_hard_turn():
    weave = _pts((0, 0), (1, 0.4), (2, 0), (3, 0.4), (4, 0), (5, 0.4), (6, 0))
    dist, _, mx, _ = scan_hard_turn_ahead(
        weave, Pose2D(0, 0), turn_thresh_deg=70.0, max_scan_m=8.0)
    assert dist == INF          # no single corner nets past 70 deg
    assert mx < 70.0


# ─── forward-monotone projection hint ────────────────────────────────────────
def test_forward_monotone_via_min_index():
    # A route that doubles back near itself: feeding min_index keeps the projection
    # forward so the corner distance is measured from where the drone actually is.
    route = _pts((0, 0), (4, 0), (4, 4))
    _, _, _, seg = scan_hard_turn_ahead(
        route, Pose2D(3.9, 0.1), turn_thresh_deg=70.0, min_index=0, max_scan_m=5.0)
    assert seg in (0, 1)


def test_merge_collinear_denoises_before_scanning():
    # Tiny staircase jitter on an otherwise straight run must not read as a corner.
    stair = _pts((0, 0), (1, 0.02), (2, 0), (3, 0.02), (4, 0))
    dist, _, _, _ = scan_hard_turn_ahead(
        stair, Pose2D(0, 0), turn_thresh_deg=70.0, merge_collinear_deg=15.0,
        max_scan_m=6.0)
    assert dist == INF
