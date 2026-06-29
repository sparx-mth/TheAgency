"""Tests for the scipy-free Euclidean clearance field.

Run:
    .venv/bin/python -m pytest \
        sparx_agency/core/planning/planners/common/tests/test_clearance_2d.py
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.planning.planners.common.clearance_2d import clearance_field


def test_obstacle_cell_is_zero_clearance():
    occ = np.zeros((5, 5), dtype=bool)
    occ[2, 2] = True
    cl = clearance_field(occ, resolution=1.0, max_clearance_m=10.0)
    assert cl[2, 2] == 0.0


def test_exact_euclidean_distance():
    occ = np.zeros((4, 4), dtype=bool)
    occ[0, 0] = True
    cl = clearance_field(occ, resolution=1.0, max_clearance_m=10.0)
    # Distance from (y,x) to the obstacle at (0,0) is sqrt(x^2 + y^2).
    for y in range(4):
        for x in range(4):
            assert math.isclose(cl[y, x], math.hypot(x, y), rel_tol=1e-6, abs_tol=1e-6)


def test_resolution_scales_meters():
    occ = np.zeros((1, 5), dtype=bool)
    occ[0, 0] = True
    cl = clearance_field(occ, resolution=0.25, max_clearance_m=10.0)
    # cell (0, k) is k cells == 0.25*k meters from the obstacle.
    assert math.isclose(cl[0, 3], 0.75, rel_tol=1e-6)


def test_cap_clamps_far_cells_but_keeps_near_exact():
    occ = np.zeros((1, 30), dtype=bool)
    occ[0, 0] = True
    cap = 1.0  # meters
    cl = clearance_field(occ, resolution=0.1, max_clearance_m=cap)
    assert math.isclose(cl[0, 5], 0.5, rel_tol=1e-6)   # within the band -> exact
    assert cl[0, 29] >= cap                            # beyond the band -> clamped


def test_centre_of_corridor_has_max_clearance():
    # Horizontal corridor: walls on the top and bottom rows, free in between.
    occ = np.zeros((7, 9), dtype=bool)
    occ[0, :] = True
    occ[6, :] = True
    cl = clearance_field(occ, resolution=1.0, max_clearance_m=10.0)
    col = cl[:, 4]
    # The centre row (y=3) is the unique maximum-clearance cell in the column.
    assert int(np.argmax(col)) == 3
