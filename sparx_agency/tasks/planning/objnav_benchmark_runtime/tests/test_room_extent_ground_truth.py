"""The room extents against Matterport's room annotations (skipped without the MP3D scenes).

The ``.house`` files are the only room annotations the benchmarks ship. The
harness (``mp3d/room_extent_ground_truth.py``) rasterises their floor
polygons, casts the benchmark camera over them from the panorama positions,
and reads the extents; these tests hold the mechanism to its two promises on
that ground truth -- a size called FINAL is the room's, a size called a LOWER
BOUND never overstates it -- and to the accuracy of the final sizes.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import room_extent_ground_truth as gt

SCENES = ("17DRP5sb8fy", "zsNo4HB9uLZ", "8194nk5LbLH")


@pytest.fixture(scope="module")
def records():
    missing = [scene for scene in SCENES if not gt.house_path(scene).is_file()]
    if missing:
        pytest.skip("MP3D house annotations not installed: %s" % ", ".join(missing))
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PROTOCOLS
    camera = next(iter(PROTOCOLS.values())).camera()
    out = []
    for scene in SCENES:
        out += gt.evaluate_scene(scene, camera)
    assert len(out) >= 20, "the three scenes hold a few dozen annotated rooms"
    return out


def test_the_house_file_gives_one_floor_polygon_per_room_and_the_panoramas_in_it():
    if not gt.house_path(SCENES[0]).is_file():
        pytest.skip("MP3D house annotations not installed")
    regions, panoramas = gt.parse_house(gt.house_path(SCENES[0]))
    assert len(regions) == 10 and all(len(r["polygon"]) >= 3 for r in regions.values())
    assert gt.polygon_area(regions[0]["polygon"]) == pytest.approx(17.2, abs=0.3), "the bedroom's annotated floor"
    assert panoramas and all(r in regions for r, _, _ in panoramas)
    plan = gt.FloorPlan(regions, panoramas, 0)
    assert (plan.truth == gt.VALUES.free).sum() * plan.resolution ** 2 == pytest.approx(
        sum(gt.polygon_area(r["polygon"]) for r in plan.regions.values()), rel=0.15)
    assert plan.panoramas, "panoramas standing on true floor"


def test_a_room_as_first_seen_from_its_door_is_a_lower_bound_that_never_overstates_the_true_size(records):
    first = [r for r in records if r["first"] is not None]
    assert first
    assert not any(r["first"]["bounded"] for r in first), "one frame from the door never shows every wall"
    assert all(r["first"]["area_m2"] <= 1.05 * r["truth_area_m2"] for r in first), "a lower bound stays under the truth"
    assert np.median([r["first"]["area_m2"] / r["truth_area_m2"] for r in first]) < 0.8, "and is a real lower bound"


def test_a_scanned_rooms_size_is_final_and_within_a_tenth_of_the_annotated_area(records):
    scanned = [r for r in records if r["scanned"] is not None]
    final = [r for r in scanned if r["scanned"]["bounded"]]
    assert len(final) >= 0.75 * len(scanned), "most rooms scanned from inside have every wall located"
    errors = [abs(r["scanned"]["area_m2"] - r["truth_area_m2"]) / r["truth_area_m2"] for r in final]
    assert float(np.median(errors)) <= 0.10, "the wall's half cell per side is the bias"
    assert all(r["scanned"]["area_m2"] <= 1.05 * r["truth_area_m2"] for r in final), "never over the truth"
    dims = [max(abs(r["scanned"]["length_m"] - r["truth_length_m"]), abs(r["scanned"]["width_m"] - r["truth_width_m"]))
            for r in final]
    assert float(np.median(dims)) <= 0.3
    lower = [r for r in scanned if not r["scanned"]["bounded"]]
    assert all(r["scanned"]["area_m2"] <= 1.05 * r["truth_area_m2"] for r in lower)
    assert all(r["scanned"]["observed_fraction"] >= 0.9 for r in scanned), "a full rotation at every panorama observes the room"


def test_the_accuracy_summary_reports_both_stages(records):
    summary = gt.accuracy(records)
    assert summary["rooms"] == len(records)
    assert summary["first"]["final"] == 0 and summary["first"]["lower_bounds_overstated"] == 0
    assert summary["scanned"]["final_area_error_median"] is not None
    assert summary["scanned"]["lower_bounds_overstated"] == 0
