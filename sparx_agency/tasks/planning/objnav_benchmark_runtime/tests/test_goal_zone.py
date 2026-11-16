"""A dataset's goal zone, and the STOP criterion each benchmark's adapter derives from it."""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.closing.stop_criterion import EUCLIDEAN, PLANAR, stop_range_m
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import protocol as gibson_protocol
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import run as gibson_run
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.goal_zone import STOP_MARGIN_M, GoalZone, stop_criterion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d import protocol as hm3d_protocol
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import protocol as mp3d_protocol


def test_each_habitat_dataset_states_its_own_zone_and_they_differ_only_in_the_measure():
    """Both generators drew view points out to a metre from the object's box on a grid of half the radius of the
    agent the dataset was made for (0.18 m; 0.17 m for HM3D v2, whatever body a protocol evaluates with since);
    HM3D's metre is horizontal, MP3D's is 3-D from the floor (measured on the validation goals, 2026-11-16)."""
    assert hm3d_protocol.GOAL_ZONES == {"v1": GoalZone(1.0, PLANAR, 0.09), "v2": GoalZone(1.0, PLANAR, 0.085)}
    assert mp3d_protocol.GOAL_ZONES == {"v1": GoalZone(1.0, EUCLIDEAN, 0.09)}
    for module in (hm3d_protocol, mp3d_protocol):
        assert set(module.GOAL_ZONES) == set(module.PROTOCOLS), "every protocol version has a zone"
        for version, protocol in module.PROTOCOLS.items():
            zone = protocol.goal_zone
            assert zone is module.GOAL_ZONES[version]
            assert zone.grid_m * math.sqrt(0.5) < protocol.success_distance_m, "no place is a success radius from every candidate"
            assert "goal_zone" not in asdict(protocol), "a property, not a field: the frozen protocol record is unchanged"


def test_the_stop_distance_is_the_zone_less_a_decimetre_by_the_zones_own_measure():
    assert STOP_MARGIN_M == 0.1
    assert stop_criterion(GoalZone(1.0, PLANAR, 0.09), 0.1) == {"stop_within_m": 0.9, "stop_measure": PLANAR}
    assert stop_criterion(GoalZone(1.0, EUCLIDEAN, 0.085), 0.1) == {"stop_within_m": 0.9, "stop_measure": EUCLIDEAN}
    assert stop_criterion(GoalZone(1.5, PLANAR, 0.09), 0.1) == {"stop_within_m": 1.4, "stop_measure": PLANAR}
    # The success distance is checked, never subtracted: a run scored at ApexNav's wider radius stops where the
    # official one does (hm3d/protocol.py, ``with_success_distance``).
    loose = hm3d_protocol.PROTOCOLS["v1"].with_success_distance(hm3d_protocol.APEXNAV_SUCCESS_DISTANCE_M)
    assert stop_criterion(loose.goal_zone, loose.success_distance_m) == {"stop_within_m": 0.9, "stop_measure": PLANAR}
    for module in (hm3d_protocol, mp3d_protocol):
        for protocol in module.PROTOCOLS.values():
            criterion = stop_criterion(protocol.goal_zone, protocol.success_distance_m)
            assert criterion == {"stop_within_m": 0.9, "stop_measure": protocol.goal_zone.measure}
            closing = RPTSettings(target_closing=criterion).target_closing
            assert closing.stop_within_m <= closing.terminal_distance_m, "inside the range the STOP is decided in"


@pytest.mark.parametrize("zone, measure", [(hm3d_protocol.GOAL_ZONES["v1"], PLANAR), (hm3d_protocol.GOAL_ZONES["v2"], PLANAR),
                                           (mp3d_protocol.GOAL_ZONES["v1"], EUCLIDEAN)])
def test_a_stop_at_the_stop_distance_has_a_view_point_within_the_success_radius_however_the_grid_lies(zone, measure):
    """The margin's argument, checked on the geometry rather than taken on trust: for an object's box at any
    height and a candidate grid at any offset, every place whose range by the zone's measure reads within the stop
    distance has a grid candidate inside the zone within the success radius of it."""
    success = 0.1
    stop = stop_criterion(zone, success)["stop_within_m"]
    rng = np.random.default_rng(20261116)

    def box_range(points, half, underside):
        """The zone's measure from floor points to a box of half-extent ``half`` whose underside is ``underside`` up."""
        planar = np.hypot(np.maximum(np.abs(points[:, 0]) - half[0], 0.0), np.maximum(np.abs(points[:, 1]) - half[1], 0.0))
        return np.array([stop_range_m(measure, float(value), underside) for value in planar])

    for _ in range(40):
        half = rng.uniform(0.05, 0.9, 2)
        underside = float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.85)]))
        offset = rng.uniform(0.0, zone.grid_m, 2)
        axis = np.arange(-3.0, 3.0, zone.grid_m)
        grid = np.stack(np.meshgrid(axis + offset[0], axis + offset[1]), axis=-1).reshape(-1, 2)
        candidates = grid[box_range(grid, half, underside) <= zone.distance_m]
        places = rng.uniform(-2.2, 2.2, (300, 2))
        places = places[box_range(places, half, underside) <= stop]
        if not len(places):
            continue
        nearest = np.min(np.linalg.norm(places[:, None, :] - candidates[None, :, :], axis=2), axis=1)
        assert nearest.max() < success, "a candidate of the zone within the success radius of every such place"
        assert nearest.max() <= zone.grid_m * math.sqrt(0.5) + 1e-9


def test_a_zone_or_a_radius_the_margin_cannot_be_argued_for_is_refused():
    with pytest.raises(ValueError, match="beyond the 0.10 m success distance"):
        stop_criterion(GoalZone(1.0, PLANAR, 0.2), 0.1)                 # candidates sparser than the radius reaches
    with pytest.raises(ValueError, match="more than the 0.10 m STOP margin"):
        stop_criterion(GoalZone(1.0, PLANAR, 0.2), 0.5)                 # ... or than the margin keeps inside the zone
    with pytest.raises(ValueError, match="leaves no STOP distance"):
        stop_criterion(GoalZone(0.1, PLANAR, 0.09), 0.1)
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="success distance"):
            stop_criterion(GoalZone(1.0, PLANAR, 0.09), bad)
    for fields in ((0.0, PLANAR, 0.09), (1.0, PLANAR, 0.0), (True, PLANAR, 0.09), (1.0, PLANAR, "0.09")):
        with pytest.raises(ValueError, match="positive number"):
            GoalZone(*fields)
    for measure in ("geodesic", "view_points", ""):
        with pytest.raises(ValueError, match="measure"):
            GoalZone(1.0, measure, 0.09)


def test_gibson_states_its_footprint_radius_in_the_plane_and_the_development_defaults_keep_it():
    """SemExp's success is the agent inside the goal footprint dilated by a metre on the 5 cm map: planar. The STOP
    is taken a decimetre -- two map cells -- inside it."""
    assert gibson_protocol.PROTOCOL.success_radius_m == 1.0 and gibson_protocol.PROTOCOL.map_resolution_m == 0.05
    assert STOP_MARGIN_M == pytest.approx(2 * gibson_protocol.PROTOCOL.map_resolution_m), "the one margin, shared"
    assert gibson_run.method_defaults() == {"target_closing": {"stop_within_m": 0.9, "stop_measure": PLANAR}}
    seen = {}

    def build(args, mapper, defaults=None):
        seen["defaults"] = defaults
        return "policy", "method"

    original = gibson_run.build_policy
    gibson_run.build_policy = build
    try:
        assert gibson_run._method(None) == ("policy", "method")
        assert seen["defaults"] == gibson_run.method_defaults()
        gibson_run._method(None, defaults={"allow_stair_traversal": True})
        assert seen["defaults"] == dict(gibson_run.method_defaults(), allow_stair_traversal=True)
    finally:
        gibson_run.build_policy = original
