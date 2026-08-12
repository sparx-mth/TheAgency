"""A stand-in for an autopilot that closes its own velocity loop.

Exactly the model :mod:`~sparx_agency.core.control.velocity_servo.plant`
describes -- a first-order lag behind a pure transport delay, per axis -- so a
closed-loop test measures whether inverting that model actually works rather
than whether the controller agrees with itself. The numbers default to the ones
measured on the reference Gazebo airframe.

Deliberately *not* a perfect follower. A test flown against an airframe whose
velocity equals its command proves nothing about a controller whose entire
purpose is to compensate for an airframe whose velocity does not.
"""
from __future__ import annotations

import math

import numpy as np


class LaggingAirframe:
    """Velocity-controlled airframe with delay, lag and optional disturbance.

    Args:
        tau_xy: Horizontal velocity time constant, seconds.
        tau_z: Vertical velocity time constant, seconds.
        delay_s: Transport delay on every axis, seconds.
        yaw_tau: Heading-rate time constant, seconds.
        drift: Constant world-frame velocity disturbance, m/s, standing in for
            wind or a trim error. The integrator is what should absorb it.
        dc_gain: Steady-state ratio of achieved to commanded velocity.
    """

    def __init__(self, tau_xy=0.51, tau_z=0.41, delay_s=0.18, yaw_tau=0.48,
                 drift=(0.0, 0.0, 0.0), dc_gain=1.0):
        # type: (float, float, float, float, object, float) -> None
        self.tau = np.array([tau_xy, tau_xy, tau_z], dtype=float)
        self.delay_s = float(delay_s)
        self.yaw_tau = float(yaw_tau)
        self.drift = np.asarray(drift, dtype=float).reshape(3)
        self.dc_gain = float(dc_gain)
        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self._queue = []            # type: list

    def place(self, position, yaw=0.0, velocity=(0.0, 0.0, 0.0), dt=0.02):
        # type: (object, float, object, float) -> None
        """Teleport the airframe, for setting up an initial condition.

        Seeding ``velocity`` also primes the delay queue with the command that
        would have produced it, so the aircraft starts genuinely in trim rather
        than with an empty pipeline that reads as a stall for one delay. Without
        that, every measurement is dominated by an acquisition transient that
        says nothing about tracking.
        """
        self.position = np.asarray(position, dtype=float).reshape(3).copy()
        self.velocity = np.asarray(velocity, dtype=float).reshape(3).copy()
        self.yaw = float(yaw)
        self.yaw_rate = 0.0
        held = max(int(round(self.delay_s / dt)), 0)
        trim = (self.velocity - self.drift) / max(self.dc_gain, 1e-9)
        self._queue = [(trim.copy(), 0.0) for _ in range(held)]

    def step(self, body_command, yaw_rate_command, dt):
        # type: (object, float, float) -> None
        """Advance one tick under a body-frame velocity command.

        The command is rotated back into the world using the airframe's own
        heading, which is what makes a controller that rotated with a stale yaw
        visibly wrong in a turn.
        """
        body = np.asarray(body_command, dtype=float).reshape(3)
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        world = np.array([cos * body[0] - sin * body[1],
                          sin * body[0] + cos * body[1],
                          body[2]], dtype=float)

        self._queue.append((world, float(yaw_rate_command)))
        held = int(round(self.delay_s / dt))
        if len(self._queue) > held:
            applied, applied_yaw_rate = self._queue.pop(0)
        else:
            applied, applied_yaw_rate = np.zeros(3, dtype=float), 0.0

        alpha = np.clip(dt / np.maximum(self.tau, 1e-9), 0.0, 1.0)
        self.velocity += alpha * (self.dc_gain * applied - self.velocity)
        self.position += (self.velocity + self.drift) * dt

        yaw_alpha = min(dt / max(self.yaw_tau, 1e-9), 1.0)
        self.yaw_rate += yaw_alpha * (applied_yaw_rate - self.yaw_rate)
        self.yaw += self.yaw_rate * dt
