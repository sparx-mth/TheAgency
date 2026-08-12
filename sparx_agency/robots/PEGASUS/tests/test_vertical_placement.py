"""Where a duplicated obstacle ends up vertically, which decides whether a
drone has to go around it or can simply fly over it.

The arithmetic is separated from the USD work precisely so it can be checked
here, without a simulator: the first augmented office and warehouse were built
with an obstacle top of exactly the cruise altitude, and the resulting scenes
looked cluttered while leaving the flight plane itself comparatively clear.

Run:
    venv/bin/python -m pytest sparx_agency/robots/PEGASUS/tests/test_vertical_placement.py
"""
from __future__ import annotations

from sparx_agency.robots.PEGASUS.adapters.scene_augment import vertical_placement

CRUISE_M = 1.5


def span(base, top, scale, dz):
    """The world band a copy occupies, given a base-anchored scale and an offset."""
    return base + dz, base + scale * (top - base) + dz


def test_no_treatment_leaves_the_copy_exactly_where_the_source_was():
    scale, dz = vertical_placement(0.0, 0.8)
    assert (scale, dz) == (1.0, 0.0)


def test_stretch_top_grows_a_floor_standing_prop_to_that_height():
    scale, dz = vertical_placement(0.0, 0.8, stretch_top_m=2.4)
    assert dz == 0.0                                  # still on the floor
    assert span(0.0, 0.8, scale, dz) == (0.0, 2.4)


def test_stretch_top_never_shrinks_something_already_tall_enough():
    scale, dz = vertical_placement(0.0, 3.0, stretch_top_m=2.4)
    assert (scale, dz) == (1.0, 0.0)


def test_stretching_to_the_cruise_altitude_stops_at_the_flight_plane():
    # The bug this whole helper exists to make visible: an obstacle stretched to
    # exactly the cruise height tops out in the aircraft's own plane, so there is
    # clear air immediately above it.
    _lo, hi = span(0.0, 0.8, *vertical_placement(0.0, 0.8, stretch_top_m=CRUISE_M))
    assert hi == CRUISE_M
    # Stretched past it instead, the obstacle covers the plane with room to spare.
    _lo, hi = span(0.0, 0.8, *vertical_placement(0.0, 0.8, stretch_top_m=2.4))
    assert hi > CRUISE_M + 0.5


def test_floating_obstacle_is_centred_on_the_cruise_height_and_off_the_floor():
    scale, dz = vertical_placement(0.0, 0.8, span_m=1.2, centre_m=CRUISE_M)
    lo, hi = span(0.0, 0.8, scale, dz)
    assert lo > 0.0, "a floating obstacle must not touch the floor"
    assert abs((lo + hi) / 2.0 - CRUISE_M) < 1e-9
    assert abs((hi - lo) - 1.2) < 1e-9
    assert lo < CRUISE_M < hi, "the cruise plane must be inside the obstacle"


def test_a_floating_obstacle_blocks_the_whole_airframe_band():
    # Half a metre either side of the cruise plane is more than the airframe,
    # so no altitude the follower holds gets past it.
    scale, dz = vertical_placement(0.0, 0.5, span_m=1.2, centre_m=CRUISE_M)
    lo, hi = span(0.0, 0.5, scale, dz)
    assert lo <= CRUISE_M - 0.5 and hi >= CRUISE_M + 0.5


def test_a_tall_source_is_not_shrunk_to_the_span_but_is_still_centred():
    scale, dz = vertical_placement(0.0, 4.0, span_m=1.2, centre_m=CRUISE_M)
    lo, hi = span(0.0, 4.0, scale, dz)
    assert scale == 1.0                               # span never shrinks
    assert abs((lo + hi) / 2.0 - CRUISE_M) < 1e-9
    assert lo < CRUISE_M < hi


def test_a_source_already_off_the_floor_is_measured_from_its_own_base():
    # A prim whose local origin is not its floor contact must still come out
    # centred where it was asked for, not offset by however high it started.
    scale, dz = vertical_placement(2.0, 2.6, span_m=1.2, centre_m=CRUISE_M)
    lo, hi = span(2.0, 2.6, scale, dz)
    assert abs((lo + hi) / 2.0 - CRUISE_M) < 1e-9
    assert abs((hi - lo) - 1.2) < 1e-9


def test_a_degenerate_flat_source_does_not_divide_by_zero():
    scale, dz = vertical_placement(1.0, 1.0, span_m=1.2, centre_m=CRUISE_M)
    assert scale > 0.0 and abs(dz) < 1e3
