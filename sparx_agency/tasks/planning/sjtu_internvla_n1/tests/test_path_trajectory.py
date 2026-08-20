"""World polyline -> constant-cruise trajectory (ROS-free)."""
from __future__ import annotations

import math

import pytest

from sparx_agency.tasks.planning.sjtu_internvla_n1.path_trajectory import (
    trajectory_from_points,
)


def test_time_is_monotonic_and_scaled_by_cruise_speed():
    pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
    traj = trajectory_from_points(pts, cruise_speed=0.5)
    samples = traj.sample_by_time(0.1)
    times = [p.t for p in samples]
    assert times == sorted(times)
    # 2 m at 0.5 m/s spans 4 s
    assert traj.total_time == pytest.approx(4.0)


def test_yaw_follows_the_route_tangent():
    pts = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    traj = trajectory_from_points(pts, cruise_speed=1.0)
    first = traj.sample(0.0)
    assert first.yaw == pytest.approx(0.0)          # heading +x
    # near the corner the route turns to +y (+pi/2)
    mid = traj.sample(1.0)
    assert mid.yaw == pytest.approx(math.pi / 2)


def test_arc_length_is_carried_as_s():
    pts = [(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)]  # 5 m segment
    traj = trajectory_from_points(pts, cruise_speed=1.0)
    assert traj.sample(traj.total_time).s == pytest.approx(5.0)


def test_duplicate_points_are_collapsed():
    pts = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]
    traj = trajectory_from_points(pts, cruise_speed=1.0)
    assert traj.total_time == pytest.approx(1.0)


def test_too_few_distinct_points_raises():
    with pytest.raises(ValueError):
        trajectory_from_points([(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)], cruise_speed=1.0)


def test_non_positive_cruise_speed_raises():
    with pytest.raises(ValueError):
        trajectory_from_points([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], cruise_speed=0.0)

