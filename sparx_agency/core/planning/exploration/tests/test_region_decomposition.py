"""Two height bands in, a building out."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.exploration.region_decomposition import (
    decompose_regions,
    split_region_by_bands,
)
from sparx_agency.core.planning.exploration.region_map import NO_REGION

from sparx_agency.core.planning.environment.grid_regions import flood_region

from .conftest import RES


def flood(mask, seed):
    """The cells reachable from a seed cell, for asserting about coverage."""
    return flood_region(mask, seed[0], seed[1], connectivity=4)


def test_the_rooms_are_the_components_the_wall_band_separates(region_map):
    assert len(region_map.rooms()) == 5
    assert len(region_map.corridors()) == 1
    for room in region_map.rooms():
        assert 13.0 < room.area_m2 < 15.0


def test_one_doorway_per_room_at_its_real_width(region_map):
    assert len(region_map.portals) == 5
    for portal in region_map.portals.values():
        assert portal.width_m == pytest.approx(1.0, abs=0.05)
    xs = sorted(round(p.centre[0], 1) for p in region_map.portals.values())
    assert xs == [2.0, 5.0, 8.0, 11.0, 14.0]


def test_every_room_is_joined_to_the_corridor_and_to_nothing_else(region_map):
    corridor = region_map.corridors()[0]
    for room in region_map.rooms():
        neighbours = region_map.neighbours(room.id)
        assert [r.id for r, _ in neighbours] == [corridor.id]


def test_every_reachable_cell_is_assigned_and_no_wall_is(region_map, flight_band):
    """Every cell the aircraft can reach belongs to exactly one region.

    A reachable cell left unlabelled is a place the aircraft can be where the
    supervisor does not know what room it is in, and it will find them all.
    """
    labelled = region_map.labels != NO_REGION
    assert labelled.sum() > 0
    assert not (labelled & flight_band).any(), "no wall may carry a region"
    reachable = flood(~flight_band, seed=(10, 50))       # from the corridor
    assert (labelled | ~reachable).all(), "a reachable cell has no region"


def test_a_sealed_room_is_not_on_the_checklist(region_map, flight_band):
    """The store has no door in either band, so no flight can ever clear it."""
    sealed = region_map.region_at(16.5, 4.5)
    assert sealed is None, "an unreachable room must not become a region"
    assert len(region_map.rooms()) == 5


def test_the_corridor_seed_decides_which_component_is_circulation(flight_band,
                                                                  wall_band):
    seeded = decompose_regions(flight_band, wall_band, RES, 0.0, 0.0,
                               corridor_seed=(7.5, 1.0))
    assert seeded.corridors()[0].centre[1] < 2.0, "the corridor is the low strip"
    # Naming one of the rooms instead makes THAT the corridor -- the function
    # takes the caller's word for it rather than second-guessing the building.
    other = decompose_regions(flight_band, wall_band, RES, 0.0, 0.0,
                              corridor_seed=(1.5, 4.5))
    assert other.corridors()[0].centre[1] > 2.0


def test_without_a_seed_the_largest_component_is_the_corridor(flight_band, wall_band):
    guessed = decompose_regions(flight_band, wall_band, RES, 0.0, 0.0)
    assert guessed.corridors()[0].area_m2 == max(
        r.area_m2 for r in guessed.regions.values())


def test_a_room_below_the_size_floor_is_absorbed_not_kept(flight_band, wall_band):
    big = decompose_regions(flight_band, wall_band, RES, 0.0, 0.0,
                            min_room_m2=20.0, corridor_seed=(7.5, 1.0))
    assert big.rooms() == [], "13.9 m2 rooms are under a 20 m2 floor"
    assert (big.labels != NO_REGION).sum() > 0, "their cells still belong somewhere"


def test_bands_split_the_corridor_and_the_junctions_become_portals(region_map):
    corridor = region_map.corridors()[0]
    split = split_region_by_bands(region_map, corridor.id, [])
    assert len(split.corridors()) == 1, "no bands, no change"

    # Cut the corridor across, which is not how this fixture is laid out, but
    # exercises the mechanism: two slices, joined where they meet.
    two = split_region_by_bands(region_map, corridor.id,
                                [("low", -1.0, 1.0), ("high", 1.0, 3.0)])
    assert len(two.corridors()) == 2
    names = sorted(r.name for r in two.corridors())
    assert names == ["high", "low"]
    junction = two.portal_between(*[r.id for r in two.corridors()])
    assert junction is not None and junction.width_m > 10.0


def test_overlapping_bands_are_refused(region_map):
    corridor = region_map.corridors()[0]
    with pytest.raises(ValueError):
        split_region_by_bands(region_map, corridor.id,
                              [("a", 0.0, 1.5), ("b", 1.0, 2.0)])


def test_splitting_a_region_that_is_not_there_is_refused(region_map):
    with pytest.raises(ValueError):
        split_region_by_bands(region_map, 999, [("a", 0.0, 1.0)])


def test_mismatched_bands_are_refused(flight_band):
    with pytest.raises(ValueError):
        decompose_regions(flight_band, np.zeros((4, 4), bool), RES, 0.0, 0.0)


def test_a_world_with_no_enclosed_space_is_refused():
    open_field = np.zeros((30, 30), dtype=bool)
    with pytest.raises(ValueError):
        decompose_regions(open_field, open_field, RES, 0.0, 0.0)
