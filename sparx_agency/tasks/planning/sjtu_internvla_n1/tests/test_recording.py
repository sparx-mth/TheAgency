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

