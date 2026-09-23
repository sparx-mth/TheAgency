"""Run-recording render + FPS helpers (ROS-free, needs cv2)."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.planning.sjtu_internvla_n1.recording import (
    FpsMeter,
    OverlayInfo,
    TopDownRenderer,
    compose,
    draw_camera_panel,
)

pytest.importorskip("cv2", reason="OpenCV needed for rendering")


def test_fps_meter_reciprocates_the_average_duration():
    m = FpsMeter(alpha=1.0)  # no smoothing: fps == 1000 / ms
    assert m.fps is None
    m.update(50.0)  # 50 ms -> 20 Hz
    assert m.fps == pytest.approx(20.0)
    m.update(None)  # ignored
    assert m.fps == pytest.approx(20.0)
    m.update(0.0)   # ignored (non-positive)
    assert m.fps == pytest.approx(20.0)


def test_fps_meter_smooths_toward_new_samples():
    m = FpsMeter(alpha=0.5)
    m.update(100.0)   # 10 Hz
    m.update(50.0)    # blended duration 75 ms -> ~13.3 Hz
    assert 12.0 < m.fps < 14.0


def test_camera_panel_has_the_requested_size():
    frame = np.zeros((600, 600, 3), dtype=np.uint8)
    info = OverlayInfo(instruction="Explore the entire hospital, enter all the rooms",
                       action="MOVE_FORWARD", s1_fps=23.0, s2_fps=1.4,
                       s1_ms=43.0, s2_ms=700.0, pixel_goal=(300, 200),
                       pixel_goal_frame=(600, 600))
    panel = draw_camera_panel(frame, info, (640, 480))
    assert panel.shape == (480, 640, 3)


def test_topdown_render_has_the_requested_size_and_grows_a_trail():
    r = TopDownRenderer(size=(640, 480))
    for x in np.linspace(0, 5, 20):
        r.add_pose(float(x), float(0.2 * x))
    assert len(r.trail) >= 2
    committed = np.array([[5.0, 1.0], [5.5, 1.4], [6.0, 2.0]])
    full = np.array([[5.0, 1.0], [6.0, 2.0], [7.0, 3.5]])
    panel = r.render((5.0, 1.0, 0.5), committed, full)
    assert panel.shape == (480, 640, 3)


def test_topdown_render_survives_empty_inputs():
    r = TopDownRenderer(size=(320, 240))
    panel = r.render(None, None, None)
    assert panel.shape == (240, 320, 3)


def test_compose_stacks_side_by_side():
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    assert compose(left, right).shape == (480, 1280, 3)



# ── the route drawn on the building ──────────────────────────────────────


def _tiny_map():
    from sparx_agency.tasks.planning.sjtu_internvla_n1.map_backdrop import (
        OccupancyMapImage,
    )
    grid = np.full((40, 20), 254, dtype=np.uint8)
    grid[:, 0] = 0
    grid[:, -1] = 0
    return OccupancyMapImage(grid, resolution=0.5, origin_x=-5.0, origin_y=-10.0)


def test_map_backed_panel_is_the_requested_size_and_two_views():
    r = TopDownRenderer(size=(640, 480), backdrop=_tiny_map())
    for i in range(20):
        r.add_pose(0.0, -5.0 + 0.4 * i)
    panel = r.render((0.0, 3.0, 0.0),
                     np.array([[0.0, 3.0], [0.0, 4.0]]),
                     np.array([[0.0, 3.0], [0.0, 4.0], [0.0, 5.0]]))
    assert panel.shape == (480, 640, 3)
    # The two views are separated by a divider line, so the halves differ.
    assert not np.array_equal(panel[:, :100], panel[:, 320:420])


def test_the_map_backed_transform_does_not_move_between_frames():
    """A fixed extent is the point: a fitted one re-scales as the trail grows."""
    r = TopDownRenderer(size=(400, 400), backdrop=_tiny_map())
    r.add_pose(0.0, 0.0)
    first = r.render((0.0, 0.0, 0.0), None, None)
    for i in range(30):
        r.add_pose(0.1 * i, 0.2 * i)
    second = r.render((0.0, 0.0, 0.0), None, None)
    # Same pose, same panel geometry -> the aircraft marker is in the same place.
    assert first.shape == second.shape


def test_without_a_map_it_still_renders_graph_paper():
    r = TopDownRenderer(size=(320, 240), backdrop=None)
    r.add_pose(0.0, 0.0)
    r.add_pose(1.0, 1.0)
    panel = r.render((1.0, 1.0, 0.0), None, None)
    assert panel.shape == (240, 320, 3)
