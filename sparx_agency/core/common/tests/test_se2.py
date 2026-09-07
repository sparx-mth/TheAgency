"""The shared SE(2) pair, checked against the copies it replaced.

Two independent implementations of this lived in ``vlas`` -- a Python loop in
``navdp/geometry`` and a numpy stack in ``plan_commit`` -- so the assertions that
matter are that scalar and vectorised forms agree with each other and with the
literal formulas those copies used.
"""
import math

import numpy as np
import pytest

from sparx_agency.core.common.math.se2 import (
    body_to_world_2d,
    body_to_world_xy,
    rotate_2d,
    world_to_body_2d,
)

POSES = [(0.0, 0.0, 0.0), (1.0, -2.0, 0.7), (-3.5, 4.25, -2.9),
         (10.0, 10.0, math.pi), (0.0, 0.0, 2 * math.pi)]


def _reference_body_to_world(fwd, left, rx, ry, yaw):
    """The formula the NavDP loop and the plan_commit stack both spelled out."""
    c, s = math.cos(yaw), math.sin(yaw)
    return rx + fwd * c - left * s, ry + fwd * s + left * c


@pytest.mark.parametrize("pose", POSES)
def test_matches_the_formula_it_replaced(pose):
    for fwd, left in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.5, -1.25)]:
        assert body_to_world_2d(fwd, left, *pose) == pytest.approx(
            _reference_body_to_world(fwd, left, *pose), abs=1e-12)


@pytest.mark.parametrize("pose", POSES)
def test_world_to_body_is_the_exact_inverse(pose):
    for fwd, left in [(0.0, 0.0), (3.0, 0.5), (-1.0, -4.0)]:
        wx, wy = body_to_world_2d(fwd, left, *pose)
        back = world_to_body_2d(wx, wy, *pose)
        assert back == pytest.approx((fwd, left), abs=1e-9)


@pytest.mark.parametrize("pose", POSES)
def test_vectorised_agrees_with_scalar(pose):
    body = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, -1.5]])
    got = body_to_world_xy(body, *pose)
    want = np.array([body_to_world_2d(f, l, *pose) for f, l in body])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_extra_columns_are_ignored_not_rotated():
    # A policy's yaw channel is a heading, not a position: rotating it would be
    # wrong, and carrying it through would make the result (N, 3).
    body = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, -99.0]])
    got = body_to_world_xy(body, 0.0, 0.0, 0.5)
    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, body_to_world_xy(body[:, :2], 0.0, 0.0, 0.5))


def test_a_single_row_is_accepted():
    assert body_to_world_xy([1.0, 0.0], 0.0, 0.0, 0.0).shape == (1, 2)


def test_rotate_is_pure_rotation():
    x, y = rotate_2d(1.0, 0.0, math.pi / 2)
    assert (x, y) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert math.hypot(*rotate_2d(3.0, 4.0, 1.234)) == pytest.approx(5.0, abs=1e-12)


def test_a_bare_row_is_one_waypoint_even_with_a_yaw_column():
    # (3,) is a single (forward, left, yaw); the yaw is ignored like any extra
    # column, which is why this is not a shape error.
    got = body_to_world_xy(np.array([1.0, 2.0, 0.9]), 0.0, 0.0, 0.0)
    assert got.shape == (1, 2)
    np.testing.assert_allclose(got, [[1.0, 2.0]])


def test_an_empty_path_stays_empty_rather_than_raising():
    assert body_to_world_xy(np.zeros((0, 2)), 1.0, 2.0, 0.3).shape == (0, 2)


@pytest.mark.parametrize("bad", [np.zeros((1,)), np.zeros((2, 1))])
def test_fewer_than_two_columns_raises(bad):
    with pytest.raises(ValueError):
        body_to_world_xy(bad, 0.0, 0.0, 0.0)
