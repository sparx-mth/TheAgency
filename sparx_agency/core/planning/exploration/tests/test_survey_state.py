"""Carrying a survey across flights, and refusing to carry it into the wrong one."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment.occupancy_io import occupancy_from_mask
from sparx_agency.core.planning.exploration.mission import (
    ExplorationSupervisor,
    SupervisorParams,
)
from sparx_agency.core.planning.exploration.region_coverage import RegionCoverage
from sparx_agency.core.planning.exploration.survey_state import (
    load_survey,
    save_survey,
)
from sparx_agency.core.planning.exploration.visibility_coverage import (
    SensorCone,
    VisibilityCoverage,
)

from .conftest import RES, see

CORRIDOR = (6.5, 1.0)
NORTH = math.radians(90)


def _cone():
    return SensorCone(half_fov_rad=math.radians(37.5), max_range_m=8.0)


def _pair(region_map, flight_band, scanned_fraction=0.60):
    """A fresh coverage tracker and supervisor over the fixture building."""
    grid = occupancy_from_mask(flight_band, RES, 0.0, 0.0)
    coverage = VisibilityCoverage(grid, _cone(),
                                  countable=region_map.labels != 0)
    region_coverage = RegionCoverage(region_map, coverage.countable_mask,
                                     scanned_fraction=scanned_fraction)
    return coverage, ExplorationSupervisor(
        region_map, region_coverage,
        SupervisorParams(scanned_fraction=scanned_fraction))


def test_a_survey_survives_being_written_and_read(tmp_path, region_map,
                                                   flight_band):
    coverage, sup = _pair(region_map, flight_band)
    for degrees in range(0, 360, 30):
        coverage.observe(6.5, 1.0, math.radians(degrees))
    before = coverage.cells_seen
    assert before > 0

    path = str(tmp_path / "survey.npz")
    save_survey(path, coverage, sup)

    fresh_cov, fresh_sup = _pair(region_map, flight_band)
    assert fresh_cov.cells_seen == 0
    assert load_survey(path, fresh_cov, fresh_sup) is True
    assert fresh_cov.cells_seen == before
    assert np.array_equal(fresh_cov.seen_mask, coverage.seen_mask)


def test_what_is_finished_and_what_is_retired_come_back_too(tmp_path, region_map,
                                                             flight_band):
    """Without the bookkeeping a resumed run re-tries every door it gave up on."""
    coverage, sup = _pair(region_map, flight_band)
    sup._accepted.add(3)
    sup._exhausted.add(("enter_room", 4))
    sup._attempts[("enter_room", 4)] = 5
    sup._issues[("approach_door", 4)] = 9
    path = str(tmp_path / "survey.npz")
    save_survey(path, coverage, sup)

    _, fresh = _pair(region_map, flight_band)
    load_survey(path, *_reuse(coverage, fresh))
    assert 3 in fresh._accepted
    assert ("enter_room", 4) in fresh._exhausted
    assert fresh._attempts[("enter_room", 4)] == 5
    assert fresh._issues[("approach_door", 4)] == 9


def _reuse(coverage, supervisor):
    """load_survey wants (coverage, supervisor); the mask half is not the point here."""
    return (coverage, supervisor)


def test_nothing_to_resume_is_not_an_error(tmp_path, region_map, flight_band):
    coverage, sup = _pair(region_map, flight_band)
    assert load_survey(str(tmp_path / "absent.npz"), coverage, sup) is False
    assert load_survey("", coverage, sup) is False
    assert coverage.cells_seen == 0


def test_a_survey_of_a_different_building_is_refused(tmp_path, region_map,
                                                      flight_band):
    """A seen-mask over the wrong map is confidently wrong, and so is every
    number computed from it -- so this raises rather than starting again."""
    coverage, sup = _pair(region_map, flight_band)
    coverage.observe(6.5, 1.0, NORTH)
    path = str(tmp_path / "survey.npz")
    save_survey(path, coverage, sup)

    smaller = np.zeros((20, 20), dtype=bool)
    smaller[0, :] = smaller[-1, :] = smaller[:, 0] = smaller[:, -1] = True
    other = VisibilityCoverage(occupancy_from_mask(smaller, RES, 0.0, 0.0), _cone())
    with pytest.raises(ValueError):
        load_survey(path, other, sup)


def test_a_survey_at_a_different_resolution_is_refused(tmp_path, region_map,
                                                        flight_band):
    coverage, sup = _pair(region_map, flight_band)
    path = str(tmp_path / "survey.npz")
    save_survey(path, coverage, sup)
    grid = occupancy_from_mask(flight_band, RES * 2.0, 0.0, 0.0)
    other = VisibilityCoverage(grid, _cone(), countable=region_map.labels != 0)
    with pytest.raises(ValueError):
        load_survey(path, other, sup)


def test_a_survey_at_a_different_origin_is_refused(tmp_path, region_map,
                                                    flight_band):
    coverage, sup = _pair(region_map, flight_band)
    path = str(tmp_path / "survey.npz")
    save_survey(path, coverage, sup)
    grid = occupancy_from_mask(flight_band, RES, 100.0, 0.0)
    other = VisibilityCoverage(grid, _cone(), countable=region_map.labels != 0)
    with pytest.raises(ValueError):
        load_survey(path, other, sup)


def test_a_resumed_survey_does_not_re_order_what_is_already_done(
        tmp_path, region_map, flight_band, nothing_seen):
    """The point of the whole thing: run n+1 continues rather than restarts."""
    coverage, sup = _pair(region_map, flight_band)
    corridor = region_map.corridors()[0]
    seen = see(region_map, nothing_seen, corridor.id, 1.0)
    coverage.restore_seen(seen)
    first = sup.update(*CORRIDOR, NORTH, coverage.seen_mask, 0.0).mission
    assert first is not None

    path = str(tmp_path / "survey.npz")
    # Pretend the corridor scan finished and one room was given up on.
    sup._accepted.add(corridor.id)
    sup._exhausted.add(("approach_door", first.target_id))
    sup._exhausted.add(("enter_room", first.target_id))
    save_survey(path, coverage, sup)

    fresh_cov, fresh_sup = _pair(region_map, flight_band)
    load_survey(path, fresh_cov, fresh_sup)
    resumed = fresh_sup.update(*CORRIDOR, NORTH, fresh_cov.seen_mask, 0.0).mission
    assert resumed is not None
    assert resumed.target_id != first.target_id, "it went back to a finished room"


def test_the_mask_is_clipped_to_this_buildings_floor_on_the_way_in(
        tmp_path, region_map, flight_band):
    """A mask saved against a wider denominator cannot inflate the percentage."""
    coverage, sup = _pair(region_map, flight_band)
    everything = np.ones(coverage.seen_mask.shape, dtype=bool)
    coverage.restore_seen(everything)
    assert coverage.cells_seen == coverage.cells_total
    assert coverage.fraction_seen == pytest.approx(1.0)
