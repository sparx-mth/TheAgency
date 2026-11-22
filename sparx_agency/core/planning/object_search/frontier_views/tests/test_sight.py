"""A frame's sight of map cells: range, the picture, the sight line, and seeing past a cell.

Each test draws a small plan, stands a camera on it and asks about named
cells; the expected stage and pixel follow from the plan by hand. One test
does not: the share of a cell's height band inside the picture is checked
against the repository's own projection (``camera_geometry.project_to_image``)
sampled along the band, for poses and pitches drawn at random -- the
analytic clip must agree with the brute-force count wherever it is asked.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.frontier_views.sight import (
    OCCLUDED, OUT_OF_FRAME, OUT_OF_RANGE, SEEN, SightScene)
from sparx_agency.core.planning.object_search.frontier_views.tests.rig import (
    ORIGIN, RESOLUTION, Plan, camera, cell, pose, room_with_doorway)
from sparx_agency.core.planning.objnav.camera_geometry import project_to_image

CAMERA = camera()


def sight_of(plan, points, at, settings=None, settled=None):
    """The sight from ``at`` of the cells holding the world ``points``, in order."""
    rows, cols = zip(*(cell(x, y) for x, y in points))
    scene = SightScene(plan.world(), rows, cols, CAMERA, settings, plan.settled if settled is None else settled)
    return scene.sight(at)


def test_a_cell_ahead_lands_on_the_centre_column_left_is_left_and_the_floor_is_below_the_horizon():
    plan = Plan().free(0.0, -3.0, 6.0, 3.0)
    sight = sight_of(plan, [(3.05, 0.05), (3.05, 1.05), (3.05, -0.95)], pose(0.05, 0.05))
    k = CAMERA.intrinsics
    assert list(sight.stage) == [SEEN, SEEN, SEEN]
    assert sight.u[0] == pytest.approx(k.cx, abs=0.5)
    assert sight.u[1] < k.cx - 100 and sight.u[2] > k.cx + 100          # +y is the camera's left: a smaller column
    assert sight.u[1] + sight.u[2] == pytest.approx(2 * k.cx, abs=0.5)
    # Three metres ahead of a level camera 0.88 m up: the floor 0.88 m below the axis, the band's top 0.62 m above it.
    assert sight.v_bottom[0] == pytest.approx(k.cy + k.fy * 0.88 / 3.0, abs=0.5)
    assert sight.v_top[0] == pytest.approx(k.cy - k.fy * 0.62 / 3.0, abs=0.5)
    assert sight.range_m[0] == pytest.approx(3.0, abs=1e-6)


def test_the_camera_turns_with_its_yaw():
    plan = Plan().free(-6.0, -6.0, 6.0, 6.0)
    north = sight_of(plan, [(0.05, 3.05), (3.05, 0.05)], pose(0.05, 0.05, yaw_deg=90.0))
    assert list(north.stage) == [SEEN, OUT_OF_FRAME]
    assert north.u[0] == pytest.approx(CAMERA.intrinsics.cx, abs=0.5)


def test_range_field_of_view_and_the_border_margin_are_gates_in_that_order():
    plan = Plan().free(-9.0, -9.0, 9.0, 9.0)
    here = pose(0.05, 0.05)
    half = math.atan(0.5 * CAMERA.intrinsics.width / CAMERA.intrinsics.fx)
    just_inside = (3.05, 0.05 + 3.0 * math.tan(half - math.radians(1.0)))     # a degree inside the edge of the picture
    sight = sight_of(plan, [(8.55, 0.05), (-2.95, 0.05), (2.05, 2.55), just_inside], here)
    assert list(sight.stage) == [OUT_OF_RANGE, OUT_OF_FRAME, OUT_OF_FRAME, OUT_OF_FRAME]
    assert np.isnan(sight.u[:3]).all() and (sight.quality == 0).all()
    # The last cell is in the image and inside the margin: without the margin it counts.
    assert list(sight_of(plan, [just_inside], here, ViewSettings(border_fraction=0.0)).stage) == [SEEN]
    with pytest.raises(ValueError, match="max_range_m"):
        ViewSettings(max_range_m=2.0)                                          # under good_range_m


def test_a_wall_hides_a_cell_and_the_map_of_now_decides():
    plan = Plan().free(0.0, -2.0, 6.0, 2.0)
    target, here = [(5.05, 0.05)], pose(0.55, 0.05)
    assert list(sight_of(plan, target, here).stage) == [SEEN]
    plan.wall(3.0, -2.0, 3.1, 2.0)                    # mapped after the frame was taken: it stood then too
    walled = sight_of(plan, target, here)
    assert list(walled.stage) == [OCCLUDED] and walled.quality[0] == 0.0 and not walled.through[0]
    assert not np.isnan(walled.u[0])                  # in the picture, behind the wall: the pixel is still known


def test_a_wall_is_not_got_round_by_the_corner_slack_and_a_corner_is():
    """The edge of what a camera sees past a jamb is its own tangent to the jamb: on a grid the tangent clips the corner."""
    plan = Plan().free(0.0, -3.0, 8.0, 3.0)
    plan.wall(3.0, -3.0, 3.1, 0.0)                    # a wall from the south side up to y = 0: its end is a jamb
    here = pose(0.05, -1.05)
    # The line from the camera through the jamb's corner cell (3.05, -0.05) reaches x = 6.05 at y = 0.95.
    tangent = [(6.05, 0.95)]
    assert list(sight_of(plan, tangent, here, ViewSettings(corner_slack_cells=0)).stage) == [OCCLUDED]
    assert list(sight_of(plan, tangent, here, ViewSettings(corner_slack_cells=1)).stage) == [SEEN]
    behind = [(6.05, -1.05)]                          # squarely behind the wall
    for slack in (0, 1, 3):
        assert list(sight_of(plan, behind, here, ViewSettings(corner_slack_cells=slack)).stage) == [OCCLUDED]
    with pytest.raises(ValueError, match="corner_slack_cells"):
        ViewSettings(corner_slack_cells=4)


def test_seen_through_means_the_line_runs_on_into_unsettled_unknown():
    plan = room_with_doorway()
    here = pose(1.05, 0.05)
    end, side, inside = (6.95, 0.05), (6.05, 0.35), (4.05, 0.05)
    sight = sight_of(plan, [end, side, inside], here)
    assert list(sight.stage) == [SEEN, SEEN, SEEN]
    # The corridor's far end, square on: past it is unknown. Its side, along the boundary: the line runs on over
    # the corridor's floor. A cell of the room: floor beyond.
    assert list(sight.through) == [True, False, False]
    settings = ViewSettings()
    assert sight.quality[0] == pytest.approx(settings.good_range_m / 5.9, rel=1e-3)                    # 1 x the falloff
    assert sight.quality[2] == pytest.approx(settings.at_weight, rel=1e-3)                             # at 3 m: no falloff
    # Settled unknown (a window, a railing) is not seen through: nothing lies behind it to be shown.
    plan.settle(7.0, -0.4, 7.5, 0.4)
    assert list(sight_of(plan, [end], here).through) == [False]
    # A wall right behind the cell is not unknown either.
    plan = room_with_doorway().wall(7.0, -0.4, 7.1, 0.4)
    assert list(sight_of(plan, [end], here).through) == [False]


def test_seen_through_is_asked_of_the_direct_line_alone():
    """The corner slack finds a clear line to a cell; what lies PAST the cell is where the camera looks, not a cell aside."""
    here, target = pose(0.05, 0.05), [(4.05, 0.05)]
    for low, high in ((-0.1, 0.0), (0.1, 0.2)):            # unknown past the cell, in the row to either side of the line
        plan = Plan().free(0.0, -1.0, 6.0, 1.0).unknown(4.1, low, 4.6, high)
        sight = sight_of(plan, target, here)
        assert list(sight.stage) == [SEEN] and not sight.through[0]
    plan = Plan().free(0.0, -1.0, 6.0, 1.0).unknown(4.1, 0.0, 4.6, 0.1)     # ... and on the line itself
    assert list(sight_of(plan, target, here).through) == [True]


def test_the_strict_rule_gives_a_cell_seen_but_not_through_no_worth():
    plan = room_with_doorway()
    sight = sight_of(plan, [(6.95, 0.05), (6.05, 0.35)], pose(1.05, 0.05), ViewSettings(at_weight=0.0))
    assert list(sight.stage) == [SEEN, SEEN]
    assert sight.quality[0] > 0.0 and sight.quality[1] == 0.0


def test_quality_falls_with_range_only_past_the_good_range():
    plan = Plan().free(0.0, -1.0, 9.0, 1.0)
    sight = sight_of(plan, [(1.55, 0.05), (3.05, 0.05), (6.05, 0.05)], pose(0.05, 0.05))
    at = ViewSettings().at_weight
    assert sight.quality[1] == pytest.approx(at, rel=1e-3)
    assert sight.quality[2] == pytest.approx(0.5 * at, rel=1e-3)
    assert sight.quality[0] == pytest.approx(at, rel=1e-3)                    # nearer than good: full, the band is all in


def test_deep_unknown_hides_a_cell_the_fringe_and_the_blind_radius_do_not():
    here = pose(0.05, 0.05)
    target = [(6.05, 0.05)]
    # Two known floors with a metre of unknown between them: something may stand in it.
    gap = Plan().free(0.0, -1.0, 3.0, 1.0).free(4.0, -1.0, 7.0, 1.0)
    assert list(sight_of(gap, target, here).stage) == [OCCLUDED]
    # Two cells of unknown: both touch known floor -- the fringe of the map, a sampling hole.
    hole = Plan().free(0.0, -1.0, 7.0, 1.0).unknown(3.0, -1.0, 3.2, 1.0)
    assert list(sight_of(hole, target, here).stage) == [SEEN]
    # Four cells: the middle two touch no floor. A cell and a half of deep unknown is the slack; this is two.
    wide = Plan().free(0.0, -1.0, 7.0, 1.0).unknown(3.0, -1.0, 3.4, 1.0)
    assert list(sight_of(wide, target, here).stage) == [OCCLUDED]
    assert list(sight_of(wide, target, here, ViewSettings(unknown_slack_m=0.25)).stage) == [SEEN]
    # The floor round the camera's own feet is unknown wherever nothing else looked: a level camera 0.88 m up
    # sees no floor nearer than 1.42 m.
    feet = Plan().free(1.3, -1.0, 7.0, 1.0).free(0.0, 0.0, 0.1, 0.1)
    assert list(sight_of(feet, target, here).stage) == [SEEN]
    # A camera tilted 30 degrees down sees its floor from half a metre: the same unknown now hides.
    assert list(sight_of(feet, [(3.05, 0.05)], pose(0.05, 0.05, pitch_deg=30.0)).stage) == [OCCLUDED]


def test_the_pitch_needs_no_rule_of_its_own_the_band_in_the_picture_is_the_rule():
    plan = Plan().free(0.0, -1.0, 7.0, 1.0)
    target = [(3.05, 0.05)]
    level = sight_of(plan, target, pose(0.05, 0.05))
    down = sight_of(plan, target, pose(0.05, 0.05, pitch_deg=30.0))
    steep = sight_of(plan, target, pose(0.05, 0.05, pitch_deg=60.0))
    assert list(level.stage) == [SEEN] and list(down.stage) == [SEEN] and list(steep.stage) == [OUT_OF_FRAME]
    k = CAMERA.intrinsics
    top_ray = math.atan(0.5 * k.height / k.fy) - math.radians(30.0)                    # above the horizon, radians
    share = (0.88 + 3.0 * math.tan(top_ray)) / 1.5                                     # of the 0 - 1.5 m band
    assert down.quality[0] == pytest.approx(level.quality[0] * share, rel=2e-2)
    assert down.v_top[0] == pytest.approx(0.0, abs=0.5) and down.v_bottom[0] < level.v_bottom[0]
    # At its feet the steep camera does see the floor.
    assert list(sight_of(plan, [(0.85, 0.05)], pose(0.05, 0.05, pitch_deg=60.0)).stage) == [SEEN]


def test_the_band_in_the_picture_agrees_with_the_projection_sampled_along_it():
    """The analytic clip of the height band against the image, against a brute-force count over the same band."""
    rng = np.random.RandomState(11)
    plan = Plan().free(-9.0, -9.0, 9.0, 9.0)
    world = plan.world()
    settings = ViewSettings()
    k = CAMERA.intrinsics
    margin = settings.border_fraction * k.width
    heights = np.linspace(settings.band_m[0], settings.band_m[1], 601)
    checked = 0
    for _ in range(40):
        at = pose(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-180, 180), rng.choice([-30.0, 0.0, 30.0, 60.0]),
                  z=rng.uniform(-0.2, 0.2))
        points = [(at.x + rng.uniform(-6, 6), at.y + rng.uniform(-6, 6)) for _ in range(25)]
        rows, cols = zip(*(cell(x, y) for x, y in points))
        scene = SightScene(world, rows, cols, CAMERA, settings)
        sight = scene.sight(at)
        for index, (row, col) in enumerate(zip(rows, cols)):
            if sight.stage[index] == OUT_OF_RANGE:
                continue
            x, y = (col + 0.5) * RESOLUTION + ORIGIN, (row + 0.5) * RESOLUTION + ORIGIN
            band = np.stack([np.full(len(heights), x), np.full(len(heights), y), at.z + heights], axis=1)
            pixels, depth = project_to_image(band, CAMERA, at)
            inside = ((depth > 0.05) & (pixels[:, 0] >= margin) & (pixels[:, 0] <= k.width - 1 - margin)
                      & (pixels[:, 1] >= 0) & (pixels[:, 1] <= k.height - 1))
            expected = float(inside.mean())
            framed = sight.stage[index] >= OCCLUDED
            assert framed == (expected >= settings.min_band_visible - 0.01) or abs(expected - settings.min_band_visible) < 0.01
            if framed and sight.stage[index] == SEEN:
                falloff = min(1.0, settings.good_range_m / max(float(sight.range_m[index]), 1e-9))
                worth = 1.0 if sight.through[index] else settings.at_weight
                assert sight.quality[index] / (falloff * worth) == pytest.approx(expected, abs=0.01)
                # The sampled band's ends lie within one sample's worth of rows of the exact ones.
                one_sample = float(np.abs(np.diff(pixels[inside, 1])).max()) + 1.0 if inside.sum() > 1 else 3.0
                assert pixels[inside, 1].min() == pytest.approx(sight.v_top[index], abs=one_sample)
                assert pixels[inside, 1].max() == pytest.approx(sight.v_bottom[index], abs=one_sample)
                assert pixels[inside, 0].min() - 1.0 <= sight.u[index] <= pixels[inside, 0].max() + 1.0
                checked += 1
    assert checked > 100


def test_a_scene_refuses_cells_off_the_grid_and_masks_of_another_shape():
    plan = Plan().free(0.0, -1.0, 3.0, 1.0)
    world = plan.world()
    with pytest.raises(ValueError, match="off the"):
        SightScene(world, [0, 999], [0, 5], CAMERA)
    with pytest.raises(ValueError, match="same length"):
        SightScene(world, [1, 2], [3], CAMERA)
    with pytest.raises(ValueError, match="settled"):
        SightScene(world, [1], [3], CAMERA, settled=np.zeros((3, 3), bool))
    empty = SightScene(world, [], [], CAMERA).sight(pose(0.05, 0.05))
    assert len(empty.stage) == 0 and len(empty.quality) == 0


@pytest.mark.parametrize("override", [{"band_m": (1.0, 0.5)}, {"band_m": (0.0, 1.5, 2.0)}, {"at_weight": 1.5},
                                      {"min_band_visible": 0.0}, {"border_fraction": 0.5}, {"max_frames": 0},
                                      {"min_seen_cells": 0}, {"unknown_slack_m": -0.1}, {"through_run_m": 0.0},
                                      {"storey_tolerance_m": float("nan")}, {"max_frames": True}])
def test_settings_refuse_values_outside_their_range(override):
    with pytest.raises(ValueError):
        ViewSettings(**override)
