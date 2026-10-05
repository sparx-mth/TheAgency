"""The gain of a glance: unknown floor a look would reveal, in plan view (2026-10-05)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.view_gain import UnknownView, ViewCone, cone_from_camera, glance_gains

RES = 0.1
FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)


def corridor_with_a_door():
    """A corridor along +x, walls above and below, a 0.8 m doorway in the upper wall at x ~ 5.0-5.8 m,
    and an unknown room beyond it. The corridor's floor is known free; the room is unknown."""
    g = np.full((100, 120), UNK, np.int8)
    g[20:30, 0:120] = FREE                 # the corridor: y 2.0-3.0 m
    g[30, 0:120] = OCC                     # the upper wall ...
    g[30, 50:58] = FREE                    # ... with a doorway at x 5.0-5.8 m
    g[19, 0:120] = OCC                     # the lower wall
    g[0:19, :] = OCC                       # solid below
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def cone(max_range_m=5.0, blind_m=0.0):
    return ViewCone(half_fov_rad=math.radians(39.5), max_range_m=max_range_m, min_range_m=blind_m)


def test_a_cone_from_a_pinhole_has_the_right_half_fov():
    c = cone_from_camera(640, 383.0, 5.0, 0.5)
    assert c.half_fov_rad == pytest.approx(math.atan(320 / 383.0))
    assert c.max_range_m == 5.0 and c.min_range_m == 0.5
    for bad in (dict(half_fov_rad=0.0, max_range_m=5.0), dict(half_fov_rad=1.0, max_range_m=0.0),
                dict(half_fov_rad=1.0, max_range_m=5.0, min_range_m=6.0)):
        with pytest.raises(ValueError):
            ViewCone(**bad)


def test_rays_pass_through_the_unknown_and_stop_at_walls():
    view = UnknownView(corridor_with_a_door(), cone())
    abreast = (5.4, 2.5)
    through_the_door = view.unknown_in_sector(abreast, math.pi / 2, math.radians(39.5))
    assert through_the_door.sum() > 200, "the room beyond the doorway, optimistically to the range limit"
    ys, xs = np.nonzero(through_the_door)
    assert ys.min() >= 31, "nothing on the corridor's side of the wall is unknown"
    into_the_wall = view.unknown_in_sector((2.0, 2.5), math.pi / 2, math.radians(20.0))
    assert into_the_wall.sum() == 0, "a wall stops every ray"
    along = view.unknown_ahead(abreast, 0.0)
    assert along.sum() == 0, "the corridor ahead is known floor"
    assert view.unknown_in_sector((float("nan"), 2.5), 0.0, 1.0).sum() == 0
    assert view.unknown_in_sector((-5.0, -5.0), 0.0, math.pi).sum() == 0, "off the map"


def test_the_blind_radius_is_not_a_glances_to_reveal():
    g = np.full((60, 60), UNK, np.int8)
    g[25:35, 25:35] = FREE                 # a metre of known floor around the agent
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    here = (3.0, 3.0)
    everything = UnknownView(world, cone(max_range_m=2.0)).unknown_in_sector(here, 0.0, math.pi)
    beyond = UnknownView(world, cone(max_range_m=2.0, blind_m=1.2)).unknown_in_sector(here, 0.0, math.pi)
    assert 0 < beyond.sum() < everything.sum()
    ys, xs = np.nonzero(beyond)
    assert np.hypot((xs + 0.5) * RES - here[0], (ys + 0.5) * RES - here[1]).min() >= 1.1


def test_the_gain_peaks_abreast_of_the_doorway_and_discounts_what_the_walk_reveals():
    """The user's example: three unknown areas to the left; a step too early or too late and the
    frame hides some of them. Along a corridor the left gain is largest abreast of the door."""
    view = UnknownView(corridor_with_a_door(), cone())
    gains = {x: glance_gains(view, (x, 2.5), 0.0, math.pi / 2) for x in (2.0, 3.5, 4.5, 5.4, 6.5, 8.0)}
    assert gains[5.4].left_m2 > gains[4.5].left_m2 > gains[2.0].left_m2
    assert gains[6.5].left_m2 == 0.0 and gains[8.0].left_m2 == 0.0, (
        "a side glance sweeps from the heading's cone out to the side: a door already behind the shoulder is not in it")
    assert gains[5.4].full_m2 > gains[6.5].full_m2 > gains[8.0].full_m2 >= 0.0, "a full circle still sees it, less and less"
    assert gains[5.4].right_m2 == 0.0, "the lower wall is solid"
    assert gains[5.4].full_m2 >= gains[5.4].left_m2
    # What the walk reveals anyway is not the glance's gain.
    walk = view.unknown_in_sector((5.4, 2.5), math.pi / 2, math.radians(39.5))
    assert glance_gains(view, (5.4, 2.5), 0.0, math.pi / 2, already=walk).left_m2 < gains[5.4].left_m2 * 0.5
    assert glance_gains(view, (5.4, 2.5), 0.0, math.pi / 2, already=np.ones(view._unknown.shape, bool)).full_m2 == 0.0
