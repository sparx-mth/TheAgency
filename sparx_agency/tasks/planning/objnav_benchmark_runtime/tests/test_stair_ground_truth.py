"""Ground-truth storeys and stair connectors read from a navmesh, and the policy's view of them."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.stair_connectors import (
    enu_to_habitat, floor_levels, habitat_to_enu, scene_structure, stair_connectors, triangles)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import GroundTruthStairs


def quad(x0, x1, z0, z1, y0, y1=None):
    """Two triangles over the rectangle x0..x1 by z0..z1, at height y0 (or sloping to y1 along x)."""
    y1 = y0 if y1 is None else y1
    a, b, c, d = (x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)
    return [a, b, c, a, c, d]


def two_storey_house(with_stairs=True):
    """Habitat frame (+Y up): floor A at y=0, floor B at y=2.7, a straight flight between them."""
    vertices = quad(0, 6, 0, 6, 0.0) + quad(8, 14, 0, 6, 2.7)
    if with_stairs:
        for i in range(4):
            vertices += quad(6 + 0.5 * i, 6.5 + 0.5 * i, 2, 4, 2.7 * i / 4, 2.7 * (i + 1) / 4)
    return np.asarray(vertices, dtype=float).reshape(-1)


def test_frame_conversion_matches_the_simulator_bridge_and_inverts():
    assert habitat_to_enu((2, 3, -4)) == [4, -2, 3]
    assert enu_to_habitat(habitat_to_enu((0.5, 1.5, -2.5))) == [0.5, 1.5, -2.5]


def test_storeys_are_the_area_supported_heights_not_the_treads():
    levels = floor_levels(triangles(two_storey_house()))
    assert [level["height_m"] for level in levels] == pytest.approx([0.0, 2.7], abs=0.05)
    assert all(level["area_m2"] >= 30 for level in levels)


def test_one_flight_becomes_one_connector_from_bottom_floor_to_top_floor():
    tris = triangles(two_storey_house())
    connectors = stair_connectors(tris, floor_levels(tris))
    assert len(connectors) == 1
    c = connectors[0]
    assert c["id"] == 0
    assert c["bottom_z"] == pytest.approx(0.0, abs=0.05) and c["top_z"] == pytest.approx(2.7, abs=0.05)
    assert c["rise_m"] == pytest.approx(2.7, abs=0.1)
    assert c["bottom_xyz"][2] == pytest.approx(0.0, abs=0.2) and c["top_xyz"][2] == pytest.approx(2.7, abs=0.2)
    polyline = c["polyline_xyz"]
    assert polyline[0] == c["bottom_xyz"] and polyline[-1] == c["top_xyz"]
    heights = [p[2] for p in polyline]
    assert all(b >= a - 0.05 for a, b in zip(heights, heights[1:])), "bottom to top, never back down"
    # The flight runs along Habitat +x at z ~ 3, which is ENU -y at x ~ -3.
    assert c["bottom_xyz"][1] > c["top_xyz"][1]
    assert all(abs(p[0] + 3.0) < 1.5 for p in polyline)
    assert c["length_m"] > 2.7


def test_a_single_storey_has_no_connectors_and_the_block_is_json():
    structure = scene_structure(two_storey_house(with_stairs=False)[: 6 * 3])
    assert [round(level["height_m"], 2) for level in structure["floor_levels"]] == [0.0]
    assert structure["stair_connectors"] == []
    full = scene_structure(two_storey_house())
    json.dumps(full, allow_nan=False)
    assert full["stair_source"] == "navmesh" and len(full["stair_connectors"]) == 1


def test_the_route_is_the_centreline_and_the_pathfinder_only_repairs_a_leg_the_agent_cannot_walk():
    """A navmesh shortest path hugs the walls; the route is the middle of the flight, band by band.

    The pathfinder's path is asked for only when the simulator's own stride
    check says a leg cannot be walked -- and then only for that leg.
    """
    tris = triangles(two_storey_house())
    levels = floor_levels(tris)
    asked = []

    def path_between(start, end):
        asked.append((start, end))
        return [start, [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 2.5], end]

    walkable = lambda start, end: end                      # every stride lands where it aimed
    c = stair_connectors(tris, levels, path_between, None, walkable)[0]
    assert asked == [], "every leg walkable: the pathfinder is never consulted"
    # One vertex per 0.2 m height band of the flight, plus the two floor anchors.
    inner = c["polyline_xyz"][1:-1]
    assert len(inner) >= 10 and all(abs(p[0] + 3.0) < 0.15 for p in inner), "the middle of a 2 m wide flight is x = -3"
    heights = [p[2] for p in c["polyline_xyz"]]
    assert all(b >= a - 0.05 for a, b in zip(heights, heights[1:]))
    assert c["walkability"]["checked"] > 0 and c["walkability"]["repaired"] == 0

    def blocked_midway(start, end):
        # A stride that crosses z = 3.0 (Habitat) is stopped short: a banister in the way.
        if (start[2] - 3.0) * (end[2] - 3.0) < 0 and abs(end[2] - start[2]) > 0.02:
            return [end[0], end[1], start[2]]
        return end

    asked.clear()
    d = stair_connectors(tris, levels, path_between, None, blocked_midway)[0]
    assert d["walkability"]["repaired"] >= 1 and len(asked) == d["walkability"]["repaired"], (
        "only the legs the stride check refused were handed to the pathfinder")
    assert any(abs(p[0] + 2.5) < 1e-6 for p in d["polyline_xyz"]), "the repair's own vertex (Habitat z=2.5: ENU x=-2.5) is in the route"


def test_a_connector_carries_a_sample_of_its_surface_for_the_detector():
    tris = triangles(two_storey_house())
    c = stair_connectors(tris, floor_levels(tris))[0]
    surface = np.asarray(c["surface_xyz"], dtype=float)
    assert len(surface) >= 20
    assert surface[:, 2].min() > 0.3 and surface[:, 2].max() < 2.4, "the treads, not the floors"
    assert np.all(np.abs(surface[:, 0] + 3.0) < 1.2), "on the flight (Habitat z 2..4 is ENU x -4..-2)"


def three_storey_house():
    """Habitat frame: floors at y=0, 2.7 and 5.4 with ONE stairwell (same XZ footprint) serving both intervals."""
    vertices = quad(0, 6, 0, 6, 0.0) + quad(8, 14, 0, 6, 2.7) + quad(0, 6, 8, 14, 5.4)
    for i in range(4):
        vertices += quad(6 + 0.5 * i, 6.5 + 0.5 * i, 2, 4, 2.7 * i / 4, 2.7 * (i + 1) / 4)
    for i in range(4):
        vertices += quad(6.5 - 0.5 * i, 7.0 - 0.5 * i, 6, 8, 2.7 + 2.7 * i / 4, 2.7 + 2.7 * (i + 1) / 4)
    return np.asarray(vertices, dtype=float).reshape(-1)


def test_a_stairwell_serving_three_storeys_is_two_connectors_between_adjacent_storeys():
    """Pomaria's stairwell read as ONE 5.4 m connector from the basement straight to the top storey."""
    tris = triangles(three_storey_house())
    levels = floor_levels(tris)
    assert [round(level["height_m"], 1) for level in levels] == [0.0, 2.7, 5.4]
    connectors = stair_connectors(tris, levels)
    assert [(round(c["bottom_z"], 1), round(c["top_z"], 1)) for c in connectors] == [(0.0, 2.7), (2.7, 5.4)]
    assert all(c["rise_m"] == pytest.approx(2.7, abs=0.1) for c in connectors)
    for c in connectors:
        heights = [p[2] for p in c["polyline_xyz"]]
        assert heights[0] == pytest.approx(c["bottom_z"], abs=0.2) and heights[-1] == pytest.approx(c["top_z"], abs=0.2)


# -- the policy's reading -------------------------------------------------------
def metadata(with_stairs=True):
    return scene_structure(two_storey_house(with_stairs))


def test_missing_or_single_storey_structure_means_nothing_to_climb():
    assert not GroundTruthStairs.from_metadata(None).available
    assert not GroundTruthStairs.from_metadata({}).available
    single = GroundTruthStairs.from_metadata({"stair_connectors": [], "floor_levels": [{"height_m": 0.0}]})
    assert single.available and single.connectors == [] and single.levels == [0.0]


def test_connectors_are_oriented_from_the_storey_the_agent_stands_on():
    stairs = GroundTruthStairs.from_metadata(metadata())
    assert stairs.levels == pytest.approx([0.0, 2.7], abs=0.05)
    c = stairs.connectors[0]
    assert stairs.touching(0.0, 0.3) == [c] and stairs.touching(2.7, 0.3) == [c]
    assert stairs.touching(1.35, 0.3) == []
    up, entry, exit_, destination_z, polyline = c.oriented(0.0, 0.3)
    assert up == 1 and entry == c.bottom and exit_ == c.top and destination_z == pytest.approx(2.7, abs=0.05)
    assert polyline[0] == c.bottom
    down, entry, exit_, destination_z, polyline = c.oriented(2.7, 0.3)
    assert down == -1 and entry == c.top and exit_ == c.bottom and destination_z == pytest.approx(0.0, abs=0.05)
    assert polyline[0] == c.top and polyline[-1] == c.bottom
    with pytest.raises(ValueError):
        c.oriented(1.35, 0.3)


def test_nearest_connector_is_by_xy_distance_to_the_flight_within_a_radius():
    stairs = GroundTruthStairs.from_metadata(metadata())
    c = stairs.connectors[0]
    on_the_flight = c.polyline[len(c.polyline) // 2]
    assert c.distance_xy(on_the_flight[0], on_the_flight[1]) < 0.05
    assert stairs.nearest(on_the_flight[0], on_the_flight[1], 0.0, 0.3, within_m=0.75) is c
    far_x, far_y = c.bottom[0] + 5.0, c.bottom[1] + 5.0
    assert c.distance_xy(far_x, far_y) > 4.0
    assert stairs.nearest(far_x, far_y, 0.0, 0.3, within_m=0.75) is None
    assert stairs.nearest(on_the_flight[0], on_the_flight[1], 1.35, 0.3, within_m=0.75) is None, "not from mid-flight"


def test_diagnostics_are_plain_json():
    json.dumps(GroundTruthStairs.from_metadata(metadata()).diagnostics(), allow_nan=False)
    assert math.isfinite(GroundTruthStairs.from_metadata(metadata()).connectors[0].length_m)



