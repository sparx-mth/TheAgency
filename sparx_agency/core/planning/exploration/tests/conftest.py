"""A small building, in both height bands, for the exploration tests.

    y
    7  #####################################
       #     #     #     #     #     #  S  #
       #  A  #  B  #  C  #  D  #  E  #     #   five rooms, and one sealed
       #     #     #     #     #     #     #   store with no door at all
    2  ## ####  ####  ####  ####  ##########   a 1 m door under each of A-E
       #                                   #
       #           the corridor            #   2 m wide, 17 m long
    0  #####################################
       0                                  17

At flight height the doors stand open, so the free space is one blob. At lintel
height they are closed, so each room is its own component -- which is exactly the
pair of facts the decomposition is built on, at a size a test can assert about.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.exploration.region_coverage import RegionCoverage
from sparx_agency.core.planning.exploration.region_decomposition import (
    decompose_regions,
)

RES = 0.1
#: Cell coordinates of the five doors, as (column of the door's left edge).
_DOOR_COLUMNS = [15, 45, 75, 105, 135]


def _blank():
    """An 18 x 7 m grid with an outer wall, row 0 at minimum y."""
    occupied = np.zeros((70, 180), dtype=bool)
    occupied[0, :] = occupied[-1, :] = True
    occupied[:, 0] = occupied[:, -1] = True
    return occupied


def _with_rooms(doors_open):
    occupied = _blank()
    occupied[20, :] = True                       # the wall between corridor and rooms
    for column in range(30, 180, 30):            # partitions between the rooms
        occupied[20:70, column] = True
    if doors_open:
        for column in _DOOR_COLUMNS:
            occupied[20, column:column + 10] = False    # a 1 m opening
    return occupied


@pytest.fixture
def flight_band():
    """The band the aircraft flies in: every door open."""
    return _with_rooms(doors_open=True)


@pytest.fixture
def wall_band():
    """The band under the ceiling: every door closed by its lintel."""
    return _with_rooms(doors_open=False)


@pytest.fixture
def region_map(flight_band, wall_band):
    """Five reachable rooms off one corridor. The sealed store is not one.

    It has no door in either band, so it is its own enclosed free component and
    never joins the floor -- which is the right answer and worth a fixture: a
    survey that put an unreachable room on its checklist would never finish.
    """
    return decompose_regions(flight_band, wall_band, RES, 0.0, 0.0,
                             corridor_seed=(7.5, 1.0))


@pytest.fixture
def nothing_seen(region_map):
    """A seen-mask with nothing in it, shaped like the region grid."""
    return np.zeros(region_map.labels.shape, dtype=bool)


@pytest.fixture
def coverage(region_map):
    """Per-region progress over the whole floor."""
    return RegionCoverage(region_map, region_map.labels != 0,
                          scanned_fraction=0.60)


def see(region_map, mask, region_id, fraction):
    """Mark ``fraction`` of one region's cells as seen, in place."""
    rows, cols = np.nonzero(region_map.labels == int(region_id))
    take = max(0, min(rows.size, int(round(fraction * rows.size))))
    mask[rows[:take], cols[:take]] = True
    return mask
