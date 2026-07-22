"""The renderer produces a valid image for full, partial and empty frames."""
import numpy as np

from sparx_agency.tasks.planning.nav_debug.frame import (
    BevMap, Drift, GaugeScales, NavFrame, Quality, ReplanEvent, Routes,
)
from sparx_agency.tasks.planning.nav_debug.render import render


def _bev():
    grid = np.full((30, 40), 0, np.int8)
    grid[0, :] = 100
    return BevMap(grid=grid, resolution=0.1, origin_x=-2.0, origin_y=-1.0)


def _full_frame():
    return NavFrame(
        stamp=1.0, x=0.2, y=0.3, yaw=1.2, z=1.0,
        trail=[(0.0, 0.0), (0.1, 0.15)],
        our_cmd=(0.30, 0.05, 0.08, -0.20), drone_cmd=(400, -80, 60, 320),
        quality=Quality(0.62, 0.18, 0.8, False, 0.3, "TRACK"),
        drift=Drift(0.0, 0.05, 0.0, 0.04, 0.0, 3.0, 0.5, 1.0,
                    "holding roll", "TRACK", "IDLE", ""),
        target=(2, 5, 2.0, 0.0), advanced=True,
        bev=_bev(), routes=Routes(astar=[(0, 0), (1, 0)], safe=[(0, 0), (1, 0)],
                                  final=[(0, 0), (1, 0), (2, 0)], goal=(2.0, 0.0),
                                  lookahead=(0.5, 0.0)),
        replan=ReplanEvent(0.9, "rotation", "REPLAN: rotated 34 deg", 0.1),
        cmd_history=[0.1, 0.2, 0.3, 0.3], conf_history=[0.6, 0.62, 0.62],
        why="holding 5cm/s right roll vs drift")


def test_render_full_frame():
    img = render(_full_frame(), GaugeScales())
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 100 and img.shape[1] > 400   # map + panel side by side


def test_render_without_map_or_commands():
    fr = NavFrame(stamp=0.0, x=0.0, y=0.0, yaw=0.0)     # nothing but a pose
    img = render(fr)
    assert img.ndim == 3 and img.shape[2] == 3          # must not raise


def test_render_handles_extreme_route_coords():
    fr = _full_frame()
    fr.routes.final = [(0, 0), (1e6, -1e6)]             # far off the map -> clipped
    img = render(fr)
    assert img.shape[2] == 3
