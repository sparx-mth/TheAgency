"""Corridor centering must thread openings and stay out of the way everywhere else.

The clearance field is analytic in every test here, so the expected answer is
arithmetic rather than a recorded number: a straight corridor of half width ``h``
has clearance ``h - |offset|`` on its centre line, and the peak the class is
looking for is at offset zero by construction.
"""
import math

import pytest

from sparx_agency.core.planning.local_planners.corridor_centering import (
    CorridorCentering, CorridorCenteringConfig,
)


def _corridor(half_width, axis="x", centre=0.0):
    """Clearance field of an infinite straight corridor along ``axis``.

    Returns ``f(x, y, z)``: distance to the nearer of the two walls, and None
    once the point is outside them (the caller's search radius found nothing,
    which is how the gate reports open space).
    """
    def clearance(x, y, _z):
        across = (y - centre) if axis == "x" else (x - centre)
        room = half_width - abs(across)
        return room if room > 0.0 else None
    return clearance


def _single_wall(at_y, search_m=1.0):
    """One wall, no corridor: the convex case the parabolic fit must not trust.

    Clamped at ``search_m`` and returning None beyond it, because that is what a
    real occupancy query does — an unbounded distance would let the far side of
    a single wall masquerade as the far side of a corridor.
    """
    def clearance(_x, y, _z):
        d = abs(y - at_y)
        return d if d <= search_m else None
    return clearance


def _open(_x, _y, _z):
    return None


# ── engagement: only in tight places ─────────────────────────────────────

def test_open_space_produces_no_bias_at_all():
    centering = CorridorCentering()
    bias = centering.bias(_open, (0.0, 0.0, 1.2), (1.0, 0.0))
    assert not bias.engaged
    assert bias.speed == 0.0


def test_a_warehouse_aisle_flown_down_the_middle_is_left_alone():
    """0.70 m of room is above the engage threshold: no interference at all."""
    centering = CorridorCentering(CorridorCenteringConfig(engage_clearance_m=0.60))
    aisle = _corridor(0.70)
    bias = centering.bias(aisle, (0.0, 0.0, 1.2), (1.0, 0.0))
    assert not bias.engaged


def test_engagement_follows_the_aircraft_s_own_room_not_the_corridor_s_width():
    """Off to one side of a wide aisle IS a tight place, and is corrected.

    The threshold is deliberately read at the aircraft rather than from the
    corridor: an aircraft 0.20 m off the centre line of a 1.4 m aisle has only
    0.50 m of room, which is less than a doorway offers, and nudging it back to
    the middle is the whole intent. What "wide aisle" buys is that a centred
    aircraft is never touched, which the previous test pins.
    """
    centering = CorridorCentering(CorridorCenteringConfig(engage_clearance_m=0.60))
    bias = centering.bias(_corridor(0.70), (0.0, 0.20, 1.2), (1.0, 0.0))
    assert bias.engaged
    assert bias.world_vy < 0.0
    assert bias.offset_m == pytest.approx(-0.20, abs=0.01)


def test_a_doorway_engages():
    centering = CorridorCentering()
    door = _corridor(0.45)
    bias = centering.bias(door, (0.0, 0.15, 1.2), (1.0, 0.0))
    assert bias.engaged
    assert bias.speed > 0.0


def test_a_stationary_aircraft_is_never_pushed_sideways():
    """There is no 'across' without a direction of travel."""
    centering = CorridorCentering()
    bias = centering.bias(_corridor(0.45), (0.0, 0.2, 1.2), (0.0, 0.0))
    assert not bias.engaged


# ── direction and magnitude ──────────────────────────────────────────────

def test_the_bias_points_back_toward_the_centre_line():
    """Off to the left of a doorway, the push is to the right, and vice versa."""
    centering = CorridorCentering()
    door = _corridor(0.45)
    left_of_centre = centering.bias(door, (0.0, 0.15, 1.2), (1.0, 0.0))
    right_of_centre = centering.bias(door, (0.0, -0.15, 1.2), (1.0, 0.0))
    assert left_of_centre.world_vy < 0.0
    assert right_of_centre.world_vy > 0.0
    assert left_of_centre.world_vx == pytest.approx(0.0, abs=1e-9)


def test_an_aircraft_already_centred_is_not_pushed():
    centering = CorridorCentering()
    bias = centering.bias(_corridor(0.45), (0.0, 0.0, 1.2), (1.0, 0.0))
    assert not bias.engaged


def test_the_estimated_offset_is_exact_for_a_corridor():
    """Half the probe difference recovers the offset exactly, not approximately.

    A clearance field is a distance field, so across a corridor it falls 1:1
    with lateral offset and the half difference of two symmetric probes IS the
    error. This is the property a parabolic peak fit does not have: see
    ``_peak_offset``.
    """
    centering = CorridorCentering(CorridorCenteringConfig(probe_m=0.15))
    door = _corridor(0.45)
    for true_offset in (0.05, 0.10, -0.08, 0.14):
        bias = centering.bias(door, (0.0, true_offset, 1.2), (1.0, 0.0))
        assert bias.offset_m == pytest.approx(-true_offset, abs=1e-9)


def test_the_estimate_does_not_overshoot_as_the_error_approaches_the_probe():
    """The failure mode of a parabolic fit, pinned as a regression.

    A vertex fit returns ``-d*e/(d - e)``, which diverges as ``e`` approaches
    ``d`` -- a 0.10 m error asking for 0.30 m of correction. The estimator must
    stay proportional.
    """
    centering = CorridorCentering(CorridorCenteringConfig(probe_m=0.15))
    door = _corridor(0.45)
    for true_offset in (0.02, 0.06, 0.10, 0.14):
        bias = centering.bias(door, (0.0, true_offset, 1.2), (1.0, 0.0))
        assert abs(bias.offset_m) <= true_offset + 1e-9


def test_the_bias_is_capped():
    centering = CorridorCentering(CorridorCenteringConfig(max_speed=0.15, gain=5.0))
    bias = centering.bias(_corridor(0.45), (0.0, 0.20, 1.2), (1.0, 0.0))
    assert bias.speed <= 0.15 + 1e-9


def test_the_bias_is_perpendicular_to_travel_on_any_heading():
    """Along-track schedule is the servo's; this class only moves across it."""
    centering = CorridorCentering()
    door = _corridor(0.45, axis="x")
    for heading in (0.0, 0.4, math.pi / 3.0):
        ux, uy = math.cos(heading), math.sin(heading)
        bias = centering.bias(door, (0.0, 0.15, 1.2), (ux, uy))
        if not bias.engaged:
            continue
        along = bias.world_vx * ux + bias.world_vy * uy
        assert along == pytest.approx(0.0, abs=1e-9)


def test_a_corridor_along_y_is_centred_across_x():
    centering = CorridorCentering()
    corridor = _corridor(0.45, axis="y")
    bias = centering.bias(corridor, (0.15, 0.0, 1.2), (0.0, 1.0))
    assert bias.engaged
    assert bias.world_vx < 0.0
    assert bias.world_vy == pytest.approx(0.0, abs=1e-9)


# ── the degenerate fields, where a naive fit misbehaves ──────────────────

def test_a_single_wall_pushes_away_from_it_rather_than_toward_a_fake_peak():
    """Convex triple: there is no valley, so the parabola must not be trusted.

    Beside one obstacle the clearance field is monotonic, and a vertex fit on a
    monotonic triple returns a point on the WRONG side -- straight into the wall.
    """
    centering = CorridorCentering(CorridorCenteringConfig(engage_clearance_m=1.0))
    bias = centering.bias(_single_wall(at_y=0.5), (0.0, 0.0, 1.2), (1.0, 0.0))
    assert bias.engaged
    assert bias.world_vy < 0.0          # away from the wall at +y


def test_a_flat_field_produces_no_bias():
    centering = CorridorCentering()
    bias = centering.bias(lambda *_: 0.4, (0.0, 0.0, 1.2), (1.0, 0.0))
    assert not bias.engaged


def test_the_estimate_never_claims_a_peak_beyond_the_probes():
    """Past the probes the fit is extrapolation into geometry it never sampled."""
    centering = CorridorCentering(CorridorCenteringConfig(probe_m=0.20, gain=1.0,
                                                          max_speed=10.0))
    # A very shallow valley: the vertex formula wants to run far outside.
    def shallow(_x, y, _z):
        return 0.50 - 0.001 * abs(y - 0.4)
    bias = centering.bias(shallow, (0.0, 0.0, 1.2), (1.0, 0.0))
    assert abs(bias.offset_m) <= 0.20 + 1e-9


# ── across_width: telling a corridor from a wall ─────────────────────────

def _rays_corridor(half_width, axis="x", centre=0.0):
    """Ray caster for an infinite straight corridor: `f(pos, dir, max) -> d|None`."""
    def first_block(pos, direction, max_dist):
        across = (pos[1] - centre) if axis == "x" else (pos[0] - centre)
        comp = direction[1] if axis == "x" else direction[0]
        if abs(comp) < 1e-9:
            return None                 # parallel to the walls
        wall = half_width if comp > 0 else -half_width
        d = (wall - across) / comp
        return d if 0.0 <= d <= max_dist else None
    return first_block


def _rays_single_wall(at_y):
    """One wall at ``y = at_y``, nothing on the other side."""
    def first_block(pos, direction, max_dist):
        if abs(direction[1]) < 1e-9:
            return None
        d = (at_y - pos[1]) / direction[1]
        return d if 0.0 <= d <= max_dist else None
    return first_block


def _rays_open(_pos, _direction, _max):
    return None


def test_across_width_measures_a_corridor():
    """A 0.90 m corridor reads as 0.90 m wide from anywhere inside it."""
    centering = CorridorCentering()
    door = _rays_corridor(0.45)
    for offset in (0.0, 0.10, -0.15):
        width = centering.across_width(door, (0.0, offset, 1.2), (1.0, 0.0))
        assert width == pytest.approx(0.90, abs=1e-9)


def test_a_single_wall_never_reads_as_a_passage():
    """The discrimination the whole method exists for.

    Beside one wall the clearance is small and the correct response is to back
    away. Inside a doorway the clearance is equally small and backing away only
    replays the approach.

    This is also why the method takes a RAY rather than the clearance field:
    probed 0.25 m either side of an aircraft 0.30 m from a single wall, both
    clearance probes return distances to that same wall and their sum reads as
    a 1.10 m corridor that is not there. A ray asks the right question.
    """
    centering = CorridorCentering()
    for standoff in (0.30, 0.40, 0.60, 0.90, 1.50):
        assert centering.across_width(_rays_single_wall(at_y=standoff),
                                      (0.0, 0.0, 1.2), (1.0, 0.0)) is None


def test_across_width_is_none_in_open_space():
    centering = CorridorCentering()
    assert centering.across_width(_rays_open, (0.0, 0.0, 1.2), (1.0, 0.0)) is None


def test_across_width_is_none_without_a_direction_of_travel():
    centering = CorridorCentering()
    assert centering.across_width(_rays_corridor(0.45), (0.0, 0.0, 1.2),
                                  (0.0, 0.0)) is None


def test_across_width_separates_the_two_worlds_openings_from_a_room():
    """A hospital doorway and a warehouse aisle are passages; a room is not."""
    centering = CorridorCentering()
    door = centering.across_width(_rays_corridor(0.465), (0.0, 0.0, 1.2), (1.0, 0.0))
    aisle = centering.across_width(_rays_corridor(0.52), (0.0, 0.0, 1.2), (1.0, 0.0))
    room = centering.across_width(_rays_corridor(2.5), (0.0, 0.0, 1.2), (1.0, 0.0))
    assert door == pytest.approx(0.93, abs=1e-9)
    assert aisle == pytest.approx(1.04, abs=1e-9)
    assert room is None                 # both walls beyond the 1.5 m look
    assert door < 1.20 and aisle < 1.20


def test_across_width_looks_no_further_than_asked():
    centering = CorridorCentering()
    corridor = _rays_corridor(0.9)      # 1.8 m wide
    assert centering.across_width(corridor, (0.0, 0.0, 1.2), (1.0, 0.0),
                                  max_m=0.5) is None
    assert centering.across_width(corridor, (0.0, 0.0, 1.2), (1.0, 0.0),
                                  max_m=1.5) == pytest.approx(1.8, abs=1e-9)


def test_a_probe_landing_outside_the_search_radius_reads_as_roomier():
    """Half in a doorway, half out: the open side must win, not be ignored."""
    centering = CorridorCentering(CorridorCenteringConfig(probe_m=0.25))

    def half_open(_x, y, _z):
        # wall at y = -0.3 only; everything above y = 0.2 is out of range
        if y > 0.2:
            return None
        return abs(y + 0.3)

    bias = centering.bias(half_open, (0.0, 0.0, 1.2), (1.0, 0.0))
    assert bias.engaged
    assert bias.world_vy > 0.0          # toward the open side
