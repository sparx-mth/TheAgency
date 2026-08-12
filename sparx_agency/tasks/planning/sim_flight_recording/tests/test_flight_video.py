"""Composing a flight into a watchable frame: the geometry, not the encoding.

The claim this whole video rests on is that the camera half and the map half
describe the same instant. That holds because ``poses.npy`` has one row per
recorded frame and both halves are indexed by it -- so what needs testing is that
nothing along the way rescales, re-times or re-indexes one of them. ffmpeg is not
exercised here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment.occupancy_grid2d import (
    OccupancyGrid2D, OccupancyGrid2DParams,
)
from sparx_agency.tasks.planning.sim_flight_recording import flight_video
from sparx_agency.tasks.planning.sim_flight_recording.flight_map_panel import (
    MIN_PANEL_PX, MapPanel, route_points,
)


def _grid(rows=80, columns=40, resolution=0.1, origin=(-2.0, -4.0)):
    """A small empty map, taller than it is wide, like the office scene."""
    values = np.zeros((rows, columns), dtype=np.int8)
    return OccupancyGrid2D(values, OccupancyGrid2DParams(
        resolution=resolution, origin_x=origin[0], origin_y=origin[1]))


# --- the map panel ----------------------------------------------------------

def test_the_panel_keeps_the_buildings_shape():
    """A squashed map is misleading about the building, so the aspect is kept."""
    panel = MapPanel(_grid(rows=80, columns=40), size_px=400)
    assert panel.height == 400
    assert panel.width == 200


def test_the_panel_refuses_to_be_drawn_too_small():
    with pytest.raises(ValueError, match="at least"):
        MapPanel(_grid(), size_px=MIN_PANEL_PX - 1)


def test_the_map_origin_is_the_panels_origin():
    grid = _grid(origin=(-2.0, -4.0))
    panel = MapPanel(grid, size_px=400)
    assert panel.to_pixel(-2.0, -4.0) == (0, 0)


def test_pixels_advance_with_world_metres():
    panel = MapPanel(_grid(rows=80, columns=40, resolution=0.1), size_px=400)
    left = panel.to_pixel(0.0, 0.0)
    right = panel.to_pixel(1.0, 0.0)
    assert right[0] > left[0]
    assert right[1] == left[1]


def test_a_drawn_panel_is_the_declared_size():
    panel = MapPanel(_grid(), size_px=400)
    frame = panel.draw(np.zeros((5, 2)), 1, (0.0, 0.0, 0.0))
    assert frame.shape == (panel.height, panel.width, 3)


def test_north_is_up():
    """The map is +y up and a numpy row index counts down, so the panel is flipped."""
    grid = _grid()
    panel = MapPanel(grid, size_px=400)
    high = panel.draw(np.array([[0.0, 3.0]]), 1, (0.0, 0.0, 0.0))
    low = panel.draw(np.array([[0.0, -3.0]]), 1, (0.0, 0.0, 0.0))

    def marker_row(frame):
        rows = np.where((frame != panel._base[::-1]).any(axis=(1, 2)))[0]
        return rows.mean()

    assert marker_row(high) < marker_row(low)


def test_the_trail_grows_with_the_frame_index():
    panel = MapPanel(_grid(), size_px=400)
    flown = np.stack([np.linspace(-1.0, 1.0, 40), np.zeros(40)], axis=-1)
    early = panel.draw(flown, 3, (0.0, 0.0, 0.0))
    late = panel.draw(flown, 40, (0.0, 0.0, 0.0))
    assert (late != early).any()


def test_an_index_past_the_end_is_clipped_rather_than_raising():
    """A recording with more frames than poses must not kill the render."""
    panel = MapPanel(_grid(), size_px=400)
    flown = np.zeros((4, 2))
    assert panel.draw(flown, 9999, (0.0, 0.0, 0.0)).shape[2] == 3


def test_nothing_is_drawn_for_a_flight_that_has_not_started():
    panel = MapPanel(_grid(), size_px=400)
    frame = panel.draw(np.zeros((4, 2)), 0, (0.0, 0.0, 0.0))
    assert np.array_equal(frame, panel._base[::-1])


def test_the_heading_arrow_points_where_the_aircraft_faces():
    panel = MapPanel(_grid(rows=120, columns=120), size_px=400)
    flown = np.array([[0.0, 0.0]])
    east = panel.draw(flown, 1, (0.0, 0.0, 0.0))
    north = panel.draw(flown, 1, (0.0, 0.0, math.pi / 2))
    assert (east != north).any()


# --- the A* route ------------------------------------------------------------

def test_the_route_starts_where_the_aircraft_did():
    """Drawn from the waypoints alone it begins at the first waypoint and looks
    like it left the aircraft behind."""
    meta = {"start_xy": [1.0, 2.0], "planned_waypoints": [[3.0, 4.0, 1.5, 0.0]]}
    assert route_points(meta, (9.0, 9.0)) == [(1.0, 2.0), (3.0, 4.0)]


def test_the_route_prefers_the_routes_own_start():
    """start_xy is where the *recording* begins; route_start_xy is where this
    route picked up, which is the one the route should be drawn from."""
    meta = {"start_xy": [0.0, 0.0], "route_start_xy": [5.0, 5.0],
            "planned_waypoints": [[6.0, 6.0]]}
    assert route_points(meta, (0.0, 0.0))[0] == (5.0, 5.0)


def test_a_recording_with_no_route_draws_none():
    assert route_points({"start_xy": [0.0, 0.0]}, (0.0, 0.0)) == []
    assert route_points({"planned_waypoints": []}, (0.0, 0.0)) == []


def test_the_first_pose_is_the_last_resort_for_the_routes_start():
    meta = {"planned_waypoints": [[2.0, 2.0]]}
    assert route_points(meta, (7.0, 8.0)) == [(7.0, 8.0), (2.0, 2.0)]


# --- fitting the two halves --------------------------------------------------

def test_a_fitted_image_is_exactly_the_box():
    fitted = flight_video.fit(np.zeros((100, 300, 3), np.uint8), 200, 200)
    assert fitted.shape == (200, 200, 3)


def test_fitting_letterboxes_rather_than_stretching():
    """A stretched camera frame misrepresents what the lens saw."""
    image = np.full((100, 300, 3), 17, np.uint8)
    fitted = flight_video.fit(image, 300, 300)
    assert (fitted[0] == flight_video.BACKGROUND).all()      # padding, not content
    assert (fitted[150] == 17).all()                         # content in the middle


def test_a_tall_image_is_fitted_by_its_height():
    fitted = flight_video.fit(np.full((400, 100, 3), 9, np.uint8), 300, 300)
    assert fitted.shape == (300, 300, 3)
    assert (fitted[:, 0] == flight_video.BACKGROUND).all()


def test_side_by_side_panes_share_a_height_and_keep_their_own_widths():
    """Equal halves would spend most of the map's half on blank margin."""
    (camera, panel), size = flight_video.pane_sizes((392, 504), (540, 220), 540, "side")
    assert camera[1] == panel[1] == 540
    assert camera[0] != panel[0]
    assert size == (camera[0] + panel[0], 540 + flight_video.CAPTION_HEIGHT)


def test_stacked_panes_share_a_width():
    (camera, panel), size = flight_video.pane_sizes((392, 504), (540, 220), 540, "stack")
    assert camera[0] == panel[0] == 540
    assert size == (540, camera[1] + panel[1] + flight_video.CAPTION_HEIGHT)


def test_neither_pane_distorts_its_content():
    """A stretched map misrepresents the building and a stretched frame the lens."""
    (camera, panel), _ = flight_video.pane_sizes((392, 504), (745, 307), 600, "side")
    assert camera[0] / camera[1] == pytest.approx(504 / 392, rel=0.01)
    assert panel[0] / panel[1] == pytest.approx(307 / 745, rel=0.01)


def test_a_composed_frame_matches_the_declared_size():
    """The encoder is told the size up front; a frame of another size corrupts
    the stream rather than failing."""
    camera_shape, panel_shape = (392, 504), (540, 220)
    for layout in ("side", "stack"):
        panes, size = flight_video.pane_sizes(camera_shape, panel_shape, 540, layout)
        frame = flight_video._frame(np.zeros(camera_shape + (3,), np.uint8),
                                    np.zeros(panel_shape + (3,), np.uint8),
                                    panes, layout)
        bar = flight_video.caption_bar(frame.shape[1], "x")
        composed = np.vstack([bar, frame])
        assert (composed.shape[1], composed.shape[0]) == size


def test_a_missing_camera_frame_still_composes():
    """A dropped JPEG must not take the whole flight's video with it."""
    frame = flight_video._frame(None, np.zeros((540, 220, 3), np.uint8),
                                ((694, 540), (220, 540)), "side")
    assert frame.shape == (540, 694 + 220, 3)


def test_the_caption_names_the_flight_and_how_it_ended():
    label = flight_video._label("office_w0_e003",
                                {"outcome": "landed", "goal_error_m": 0.53,
                                 "estimator_drift_m": 0.02})
    assert "office_w0_e003" in label and "landed" in label and "0.53" in label


def test_the_caption_survives_metadata_without_numbers():
    assert "?" in flight_video._label("r", {})


def test_the_caption_bar_is_the_width_it_was_asked_for():
    assert flight_video.caption_bar(640, "hello").shape == (
        flight_video.CAPTION_HEIGHT, 640, 3)


def test_combining_nothing_is_not_an_error(tmp_path):
    assert flight_video.combine([], tmp_path / "out.mp4") is None
    assert flight_video.combine([tmp_path / "missing.mp4"], tmp_path / "o.mp4") is None


def test_every_pane_dimension_is_even():
    """libx264 with yuv420p rejects an odd width, and reports it only as a broken
    pipe from the far end."""
    for layout in ("side", "stack"):
        for pane in (401, 540, 639, 640):
            for camera_shape, panel_shape in (((392, 504), (745, 307)),
                                              ((480, 640), (300, 900))):
                panes, size = flight_video.pane_sizes(camera_shape, panel_shape,
                                                      pane, layout)
                assert all(value % 2 == 0 for pair in panes for value in pair)
                assert size[0] % 2 == 0 and size[1] % 2 == 0


def test_the_caption_height_is_even_too():
    assert flight_video.CAPTION_HEIGHT % 2 == 0


def test_the_caption_says_when_only_part_of_the_route_is_drawn():
    """A replanned flight has only its last route in the metadata; unlabelled,
    the short green line reads as the aircraft ignoring its plan."""
    label = flight_video._label("r", {"outcome": "landed", "replans": 2})
    assert "replanned 2x" in label


def test_the_caption_says_nothing_about_replans_when_there_were_none():
    assert "replan" not in flight_video._label("r", {"outcome": "landed",
                                                     "replans": 0})


def test_gamma_one_means_no_lookup_table_at_all():
    """The recorded frames are what the policy sees; leaving them alone is the
    default, and doing it by skipping the LUT keeps that literal."""
    assert flight_video.gamma_lut(1.0) is None


def test_a_gamma_below_one_brightens_the_midtones():
    table = flight_video.gamma_lut(0.6)
    assert table[128] > 128
    assert table[0] == 0 and table[255] == 255


def test_a_gamma_above_one_darkens_the_midtones():
    assert flight_video.gamma_lut(1.6)[128] < 128


def test_a_non_positive_gamma_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        flight_video.gamma_lut(0.0)


# --- not overwriting a previous campaign -------------------------------------

def test_a_second_run_of_the_same_worker_is_detected(tmp_path):
    """Episode numbering restarts at 000, so this collision is silent and total."""
    from sparx_agency.tasks.planning.sim_flight_recording.collect import (
        existing_recordings,
    )

    for index in range(3):
        (tmp_path / f"office_w0_e{index:03d}").mkdir()
    assert len(existing_recordings(tmp_path, "office", 0)) == 3


def test_another_worker_or_scene_does_not_clash(tmp_path):
    from sparx_agency.tasks.planning.sim_flight_recording.collect import (
        existing_recordings,
    )

    (tmp_path / "office_w0_e000").mkdir()
    assert existing_recordings(tmp_path, "office", 1) == []
    assert existing_recordings(tmp_path, "warehouse_shelves", 0) == []


def test_an_empty_directory_does_not_clash(tmp_path):
    from sparx_agency.tasks.planning.sim_flight_recording.collect import (
        existing_recordings,
    )

    assert existing_recordings(tmp_path, "office", 0) == []


def test_the_caption_stays_ascii():
    """cv2's Hershey fonts have no glyph outside ASCII and draw '??' instead."""
    label = flight_video._label("r", {"outcome": "landed", "replans": 1})
    assert label.isascii()
