"""Physical constants shared by the control chain.

One definition, because a controller that disagrees with its own thrust model
about the strength of gravity holds the wrong hover throttle and blames the
gains.
"""
from __future__ import annotations

GRAVITY_MPS2 = 9.80665
"""Standard gravity, m/s^2.

The CODATA standard value rather than a rounded 9.81. The difference is 0.05%,
which matters nowhere in flight -- but it is the value PX4 and every IMU
datasheet use, and matching them keeps a measured hover throttle comparable with
a computed one.
"""
