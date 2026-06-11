"""
Simple Drone Simulator with Noise Model.

Environment-only: physics + noise + collision callback.
No planning, no replanning, no policy logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import numpy as np


@dataclass
class DroneSimParams:
    dt: float = 0.02

    tau_velocity: float = 0.15
    tau_yaw: float = 0.1

    max_speed_xy: float = 1.0
    max_speed_z: float = 0.5
    max_yaw_rate: float = 1.0

    wind_enabled: bool = True
    wind_mean: Tuple[float, float, float] = (0.05, 0.0, 0.0)
    wind_std: float = 0.1
    wind_tau: float = 2.0

    gust_enabled: bool = True
    gust_probability: float = 0.01
    gust_magnitude: float = 0.3
    gust_duration: float = 0.5

    process_noise_std: float = 0.02

    position_noise_std: float = 0.01
    velocity_noise_std: float = 0.02
    yaw_noise_std: float = 0.01

    command_delay: float = 0.02

    collision_radius: float = 0.10


@dataclass
class DroneState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    yaw: float = 0.0
    yaw_rate: float = 0.0

    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0

    gust_remaining: float = 0.0
    gust_vx: float = 0.0
    gust_vy: float = 0.0

    t: float = 0.0


class DroneSimulator:
    def __init__(
        self,
        params: Optional[DroneSimParams] = None,
        initial_state: Optional[DroneState] = None,
        obstacle_fn: Optional[Callable[[float, float], bool]] = None,
        seed: Optional[int] = None,
    ):
        self.params = params or DroneSimParams()
        self.state = initial_state or DroneState()
        self.obstacle_fn = obstacle_fn

        self._rng = np.random.default_rng(seed)
        self._command_buffer: List[Tuple[float, float, float, float, float]] = []

        if self.params.wind_enabled:
            self.state.wind_x = self.params.wind_mean[0]
            self.state.wind_y = self.params.wind_mean[1]
            self.state.wind_z = self.params.wind_mean[2]

    def reset(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> DroneState:
        self.state = DroneState(x=x, y=y, z=z, yaw=yaw)
        if self.params.wind_enabled:
            self.state.wind_x = self.params.wind_mean[0]
            self.state.wind_y = self.params.wind_mean[1]
            self.state.wind_z = self.params.wind_mean[2]
        self._command_buffer.clear()
        return self.state

    def step(self, vx_cmd: float, vy_cmd: float, vz_cmd: float = 0.0, yaw_rate_cmd: float = 0.0) -> Tuple[DroneState, dict]:
        p = self.params
        s = self.state
        dt = p.dt

        cos_yaw = float(np.cos(s.yaw))
        sin_yaw = float(np.sin(s.yaw))

        vx_world_cmd = vx_cmd * cos_yaw - vy_cmd * sin_yaw
        vy_world_cmd = vx_cmd * sin_yaw + vy_cmd * cos_yaw
        vz_world_cmd = vz_cmd

        speed_xy = float(np.sqrt(vx_world_cmd**2 + vy_world_cmd**2))
        if speed_xy > p.max_speed_xy and speed_xy > 1e-9:
            scale = p.max_speed_xy / speed_xy
            vx_world_cmd *= scale
            vy_world_cmd *= scale

        vz_world_cmd = float(np.clip(vz_world_cmd, -p.max_speed_z, p.max_speed_z))
        yaw_rate_cmd = float(np.clip(yaw_rate_cmd, -p.max_yaw_rate, p.max_yaw_rate))

        if p.wind_enabled:
            wind_noise = self._rng.normal(0.0, p.wind_std * np.sqrt(dt), 3)
            alpha = float(np.exp(-dt / p.wind_tau))
            s.wind_x = alpha * s.wind_x + (1 - alpha) * p.wind_mean[0] + float(wind_noise[0])
            s.wind_y = alpha * s.wind_y + (1 - alpha) * p.wind_mean[1] + float(wind_noise[1])
            s.wind_z = alpha * s.wind_z + (1 - alpha) * p.wind_mean[2] + float(wind_noise[2])

        if p.gust_enabled:
            if s.gust_remaining > 0.0:
                s.gust_remaining -= dt
            elif float(self._rng.random()) < p.gust_probability:
                gust_angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
                s.gust_vx = p.gust_magnitude * float(np.cos(gust_angle))
                s.gust_vy = p.gust_magnitude * float(np.sin(gust_angle))
                s.gust_remaining = p.gust_duration

        dist_vx = s.wind_x + (s.gust_vx if s.gust_remaining > 0.0 else 0.0)
        dist_vy = s.wind_y + (s.gust_vy if s.gust_remaining > 0.0 else 0.0)
        dist_vz = s.wind_z

        proc_noise = self._rng.normal(0.0, p.process_noise_std * np.sqrt(dt), 3)

        decay = float(1.0 - np.exp(-dt / p.tau_velocity))
        s.vx = s.vx + (vx_world_cmd - s.vx) * decay + dist_vx * dt + float(proc_noise[0])
        s.vy = s.vy + (vy_world_cmd - s.vy) * decay + dist_vy * dt + float(proc_noise[1])
        s.vz = s.vz + (vz_world_cmd - s.vz) * decay + dist_vz * dt + float(proc_noise[2])

        yaw_decay = float(1.0 - np.exp(-dt / p.tau_yaw))
        s.yaw_rate = s.yaw_rate + (yaw_rate_cmd - s.yaw_rate) * yaw_decay

        old_x, old_y = s.x, s.y

        s.x += s.vx * dt
        s.y += s.vy * dt
        s.z += s.vz * dt

        s.yaw += s.yaw_rate * dt
        if s.yaw > np.pi:
            s.yaw -= 2.0 * np.pi
        if s.yaw < -np.pi:
            s.yaw += 2.0 * np.pi

        collision = False
        if self.obstacle_fn is not None and self.obstacle_fn(s.x, s.y):
            s.x, s.y = old_x, old_y
            s.vx = 0.0
            s.vy = 0.0
            collision = True

        if s.z < 0.0:
            s.z = 0.0
            s.vz = max(0.0, s.vz)

        s.t += dt

        info = {
            "collision": collision,
            "wind": (s.wind_x, s.wind_y, s.wind_z),
            "gust_active": s.gust_remaining > 0.0,
            "disturbance": (dist_vx, dist_vy, dist_vz),
        }
        return s, info

    def get_measured_state(self) -> Tuple[float, float, float, float, float, float, float]:
        p = self.params
        s = self.state
        x = s.x + float(self._rng.normal(0.0, p.position_noise_std))
        y = s.y + float(self._rng.normal(0.0, p.position_noise_std))
        z = s.z + float(self._rng.normal(0.0, p.position_noise_std))
        vx = s.vx + float(self._rng.normal(0.0, p.velocity_noise_std))
        vy = s.vy + float(self._rng.normal(0.0, p.velocity_noise_std))
        vz = s.vz + float(self._rng.normal(0.0, p.velocity_noise_std))
        yaw = s.yaw + float(self._rng.normal(0.0, p.yaw_noise_std))
        return x, y, z, vx, vy, vz, yaw

    @property
    def true_position(self) -> Tuple[float, float, float]:
        return self.state.x, self.state.y, self.state.z

    @property
    def true_velocity(self) -> Tuple[float, float, float]:
        return self.state.vx, self.state.vy, self.state.vz

    @property
    def time(self) -> float:
        return self.state.t
