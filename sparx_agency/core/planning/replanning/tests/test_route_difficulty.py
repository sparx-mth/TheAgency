"""Unit tests for route-difficulty detection (pure geometry + a map query).

Locks in the two signals the FALCON ``hybrid`` nav mode switches on:
  * a hard turn ahead accumulates its heading change (a straight run is 0);
  * a doorway is narrow on BOTH sides, while a route that merely clips one
    convex corner in an open room is NOT narrow (only nearest-wall clearance
    would wrongly flag it) -- the perpendicular free-width test separates them;
  * the forward window is taken ahead of the drone and honours skip/lookahead;
  * with no occupancy query the narrowness test is skipped (turn-only mode).
"""
import math

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.replanning.route_difficulty import (
    assess_route_difficulty,
    forward_window_2d,
    net_turn_deg,
    passage_free_width_2d,
    windowed_turn_deg,
)


def _pts(*xy):
    return [Pose2D(float(x), float(y)) for x, y in xy]


# ─── windowed_turn_deg ───────────────────────────────────────────────────────
def test_turn_zero_on_straight_run():
    assert windowed_turn_deg(_pts((0, 0), (1, 0), (2, 0), (3, 0))) == 0.0


def test_turn_measures_a_right_angle_corner():
    turn = windowed_turn_deg(_pts((0, 0), (2, 0), (2, 2)))
    assert abs(turn - 90.0) < 1e-6


def test_turn_accumulates_a_distributed_bend():
    # Three 30 deg steps -> a 90 deg sweep, even though no single vertex is sharp.
    a = math.radians(30.0)
    pts = [Pose2D(0, 0)]
    x = y = 0.0
    ang = 0.0
    for _ in range(3):
        x += math.cos(ang)
        y += math.sin(ang)
        pts.append(Pose2D(x, y))
        ang += a
    x += math.cos(ang); y += math.sin(ang)
    pts.append(Pose2D(x, y))
    assert abs(windowed_turn_deg(pts) - 90.0) < 1e-6


# ─── net_turn_deg (window-level net turn; the per-corner scan is now the gate) ─
def test_net_turn_zero_on_straight_run():
    assert net_turn_deg(_pts((0, 0), (1, 0), (2, 0), (3, 0))) < 1e-6


def test_net_turn_measures_a_right_angle_corner():
    assert abs(net_turn_deg(_pts((0, 0), (2, 0), (2, 2))) - 90.0) < 1.0


def test_net_turn_ignores_a_weaving_corridor():
    # A straight corridor the grid A* weaved down (alternating jogs): the cumulative
    # SUM balloons into a big false turn, but the NET entry-vs-exit turn stays well
    # below the ~75 deg gate -- the "NavDP takes over and writes a turn" fix.
    weave = _pts((0, 0), (1, 0.4), (2, 0), (3, 0.4), (4, 0), (5, 0.4), (6, 0))
    total_sum = windowed_turn_deg(weave)
    net = net_turn_deg(weave)
    assert total_sum > 150.0                   # the raw sum screams "hard turn"
    assert net < 60.0 and net < 0.3 * total_sum  # ...net says essentially straight


# ─── passage_free_width_2d ───────────────────────────────────────────────────
def _doorway_occ(x, y):
    """Solid walls at |y| >= 0.4 for x in [1, 3]: a 0.8 m gap along the x axis."""
    return 1.0 <= x <= 3.0 and abs(y) >= 0.4


def _one_wall_occ(x, y):
    """A wall only on the +y side (a convex corner clipped by the route)."""
    return 1.0 <= x <= 3.0 and y >= 0.4


def test_open_space_reads_fully_open():
    win = _pts((0, 0), (5, 0))
    w = passage_free_width_2d(win, lambda x, y: False, step_m=0.1, max_half_m=1.0)
    assert abs(w - 2.0) < 1e-9  # 2 * max_half_m


def test_doorway_is_narrow_on_both_sides():
    win = _pts((0, 0), (4, 0))
    w = passage_free_width_2d(win, _doorway_occ, step_m=0.1, max_half_m=1.0)
    assert 0.6 <= w <= 0.8  # ~0.35 + 0.35 (midpoint estimate of a 0.8 m gap)


def test_off_grid_doorway_not_over_reported():
    # Regression: walls at |y| >= 0.45 (a 0.9 m gap, < the 1.0 m threshold) must
    # read narrow. Returning the first-OCCUPIED distance (0.5) would sum to 1.0 and
    # miss it; the midpoint estimate (0.45) sums to 0.9 and correctly flags narrow.
    def occ(x, y):
        return 1.0 <= x <= 3.0 and abs(y) >= 0.45
    w = passage_free_width_2d(_pts((0, 0), (4, 0)), occ, step_m=0.1, max_half_m=1.0)
    assert w < 1.0, "a 0.9 m doorway must not read as passable (got %.3f)" % w


def test_corner_clip_is_not_narrow():
    # One-sided wall: near side ~0.4, far side open (capped) -> wide sum.
    win = _pts((0, 0), (4, 0))
    w = passage_free_width_2d(win, _one_wall_occ, step_m=0.1, max_half_m=1.0)
    assert w > 1.2  # ~0.4 + 1.0, clearly not a doorway


def test_width_is_orientation_independent():
    # Same doorway, route flown in -x: width must match (+x) direction.
    fwd = passage_free_width_2d(_pts((0, 0), (4, 0)), _doorway_occ, 0.1, 1.0)
    bwd = passage_free_width_2d(_pts((4, 0), (0, 0)), _doorway_occ, 0.1, 1.0)
    assert abs(fwd - bwd) < 1e-9


# ─── forward_window_2d ───────────────────────────────────────────────────────
def test_window_honours_skip_and_lookahead():
    route = _pts((0, 0), (10, 0))
    win, _ = forward_window_2d(route, Pose2D(0, 0), lookahead_m=3.0, skip_m=1.0)
    assert abs(win[0].x - 1.0) < 1e-9
    assert abs(win[-1].x - 4.0) < 1e-9


def test_window_projection_is_forward_monotone():
    route = _pts((0, 0), (10, 0))
    # Drone 5 m along; window must start at ~5, not snap back to the origin.
    win, seg = forward_window_2d(route, Pose2D(5, 0), lookahead_m=2.0, skip_m=0.0)
    assert abs(win[0].x - 5.0) < 1e-6


def test_window_keeps_interior_corner_vertex():
    route = _pts((0, 0), (2, 0), (2, 2))
    win, _ = forward_window_2d(route, Pose2D(0, 0), lookahead_m=10.0, skip_m=0.0)
    assert any(abs(p.x - 2.0) < 1e-9 and abs(p.y) < 1e-9 for p in win)  # the corner


def test_window_no_duplicate_when_skip_lands_on_vertex():
    # Regression: skip_m landing exactly on a vertex must not emit a duplicate head
    # point (a degenerate zero-length leg). Route corner at x=1, skip lands there.
    route = _pts((0, 0), (1, 0), (1, 5))
    win, _ = forward_window_2d(route, Pose2D(0, 0), lookahead_m=3.0, skip_m=1.0)
    for a, b in zip(win[:-1], win[1:]):
        assert (abs(a.x - b.x) > 1e-9 or abs(a.y - b.y) > 1e-9), "duplicate vertex in window"
    assert abs(win[0].x - 1.0) < 1e-9 and abs(win[0].y) < 1e-9


# ─── assess_route_difficulty (composition) ───────────────────────────────────
_KW = dict(lookahead_m=4.0, turn_thresh_deg=45.0, passage_width_thresh_m=1.0)


def test_clear_straight_open_route_is_not_difficult():
    d, _ = assess_route_difficulty(
        _pts((0, 0), (5, 0)), Pose2D(0, 0), occupied=lambda x, y: False, **_KW)
    assert not d.is_difficult and d.reason == "clear"


def test_hard_turn_is_flagged():
    d, _ = assess_route_difficulty(
        _pts((0, 0), (2, 0), (2, 3)), Pose2D(0, 0),
        occupied=lambda x, y: False, **_KW)
    assert d.hard_turn and not d.narrow and d.reason == "turn"


def test_doorway_is_flagged_narrow():
    d, _ = assess_route_difficulty(
        _pts((0, 0), (4, 0)), Pose2D(0, 0), occupied=_doorway_occ, **_KW)
    assert d.narrow and not d.hard_turn and d.reason == "narrow"


def test_weaving_corridor_is_not_a_hard_turn():
    # A generally-straight corridor the grid A* weaved down: at the real 75 deg gate
    # the NET turn does NOT flag it as hard -- the "NavDP takes over immediately and
    # writes a turn" fix. (The cumulative-sum metric would have flagged it.)
    weave = _pts((0, 0), (1, 0.4), (2, 0), (3, 0.4), (4, 0), (5, 0.4), (6, 0))
    d, _ = assess_route_difficulty(
        weave, Pose2D(0, 0), lookahead_m=8.0, turn_thresh_deg=75.0,
        passage_width_thresh_m=1.0)
    assert not d.hard_turn


def test_real_corner_is_a_hard_turn():
    # A genuine 90 deg corner (corridor -> room) is flagged.
    d, _ = assess_route_difficulty(
        _pts((0, 0), (2, 0), (2, 3)), Pose2D(0, 0), lookahead_m=6.0,
        turn_thresh_deg=75.0, passage_width_thresh_m=1.0)
    assert d.hard_turn and abs(d.turn_deg - 90.0) < 2.0


def test_turn_dist_reports_distance_to_the_corner():
    # The engage signal is the along-route distance to the next hard turn.
    d, _ = assess_route_difficulty(
        _pts((0, 0), (2, 0), (2, 3)), Pose2D(0, 0), lookahead_m=4.0,
        turn_thresh_deg=70.0, passage_width_thresh_m=1.0)
    assert d.hard_turn and abs(d.turn_dist_m - 2.0) < 1e-6


def test_turn_scan_range_gates_a_far_corner():
    # A corner within the (doorway) lookahead but beyond the turn engage range must
    # NOT be hard yet -- A* flies the approach until the drone is close enough.
    route = _pts((0, 0), (3, 0), (3, 3))          # corner 3 m ahead
    far, _ = assess_route_difficulty(
        route, Pose2D(0, 0), lookahead_m=6.0, turn_thresh_deg=70.0,
        passage_width_thresh_m=1.0, turn_scan_m=2.0)
    near, _ = assess_route_difficulty(
        route, Pose2D(1.5, 0), lookahead_m=6.0, turn_thresh_deg=70.0,
        passage_width_thresh_m=1.0, turn_scan_m=2.0)
    assert not far.hard_turn and far.turn_dist_m == float("inf")
    assert near.hard_turn and abs(near.turn_dist_m - 1.5) < 1e-6


def test_no_hard_turn_has_infinite_turn_dist():
    d, _ = assess_route_difficulty(
        _pts((0, 0), (5, 0)), Pose2D(0, 0), lookahead_m=4.0, turn_thresh_deg=70.0,
        passage_width_thresh_m=1.0)
    assert not d.hard_turn and d.turn_dist_m == float("inf")


def test_corner_flagged_across_an_approach_band():
    # As the drone approaches a 90 deg corner, net_turn must clear the 75 deg gate
    # for a BAND of positions (not a single tick), or a fast drone slips past the
    # peak before the node's consecutive-confirm streak fires (the razor-thin-band
    # concern at the default 2.0 m lookahead).
    route = _pts((0, 0), (3, 0), (3, 3))                 # corner at (3, 0)
    hard = 0
    x = 0.0
    while x <= 3.0:
        d, _ = assess_route_difficulty(
            route, Pose2D(x, 0.0), lookahead_m=2.0, turn_thresh_deg=75.0,
            passage_width_thresh_m=1.0, skip_m=0.3)
        hard += 1 if d.hard_turn else 0
        x += 0.3
    assert hard >= 3


def test_sustained_span_rejects_single_speckle():
    # A ~1-sample two-sided pinch (like one BEV occupancy speckle cell) trips the
    # single-sample minimum but must NOT count as a doorway once a sustained span is
    # required -- this is the "NavDP everywhere on a noisy corridor" fix.
    def speckle(x, y):
        return 1.95 <= x <= 2.05 and abs(y) >= 0.3
    pts = _pts((0, 0), (4, 0))
    lax, _ = assess_route_difficulty(pts, Pose2D(0, 0), occupied=speckle,
                                     min_narrow_span_m=0.0, **_KW)
    strict, _ = assess_route_difficulty(pts, Pose2D(0, 0), occupied=speckle,
                                        min_narrow_span_m=0.3, **_KW)
    assert lax.narrow and not strict.narrow


def test_sustained_span_keeps_real_doorway():
    d, _ = assess_route_difficulty(_pts((0, 0), (4, 0)), Pose2D(0, 0),
                                   occupied=_doorway_occ, min_narrow_span_m=0.3, **_KW)
    assert d.narrow  # a real doorway is narrow over ~2 m, well past the 0.3 m span


def test_narrowness_skipped_without_occupancy_query():
    d, _ = assess_route_difficulty(
        _pts((0, 0), (4, 0)), Pose2D(0, 0), occupied=None, **_KW)
    assert not d.narrow and d.passage_width_m == float("inf")


def test_out_of_bounds_query_never_false_positives():
    # A query that raises off-map would break a naive scan; returning False
    # (unknown = not a wall) must read as open, never narrow.
    d, _ = assess_route_difficulty(
        _pts((0, 0), (4, 0)), Pose2D(0, 0),
        occupied=lambda x, y: False, **_KW)
    assert not d.narrow
