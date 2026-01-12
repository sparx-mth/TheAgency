"""
Simple Drone Simulator with Noise Model.

Simulates a quadrotor/drone with:
- First-order velocity dynamics (can track velocity commands with some delay)
- Wind disturbance (random walk + gusts)
- Sensor noise on position/velocity feedback
- Optional collision detection with obstacles

The simulator is designed to test trajectory trackers like Pure Pursuit
in a realistic but lightweight environment.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import math


@dataclass
class DroneSimParams:
    """
    Drone simulator configuration.

    Dynamics Model:
        v_dot = (v_cmd - v) / tau + wind + noise
        p_dot = v
        yaw_dot = yaw_rate_cmd / tau_yaw

    Noise Sources:
        - Wind: Slowly varying disturbance (Ornstein-Uhlenbeck process)
        - Process noise: Random acceleration perturbations
        - Sensor noise: Gaussian noise on measurements
    """
    # Time step
    dt: float = 0.02  # 50 Hz

    # Velocity dynamics (first-order lag)
    tau_velocity: float = 0.15  # Time constant for velocity response (s)
    tau_yaw: float = 0.1  # Time constant for yaw rate response (s)

    # Kinematic limits
    max_speed_xy: float = 1.0  # m/s
    max_speed_z: float = 0.5  # m/s
    max_yaw_rate: float = 1.0  # rad/s

    # Wind model (Ornstein-Uhlenbeck process)
    wind_enabled: bool = True
    wind_mean: Tuple[float, float, float] = (0.05, 0.0, 0.0)  # Mean wind (m/s)
    wind_std: float = 0.1  # Wind standard deviation (m/s)
    wind_tau: float = 2.0  # Wind correlation time (s)

    # Gust model (occasional larger disturbances)
    gust_enabled: bool = True
    gust_probability: float = 0.01  # Per timestep probability
    gust_magnitude: float = 0.3  # m/s
    gust_duration: float = 0.5  # s

    # Process noise (random accelerations)
    process_noise_std: float = 0.02  # m/s² per √Hz

    # Sensor noise
    position_noise_std: float = 0.01  # m
    velocity_noise_std: float = 0.02  # m/s
    yaw_noise_std: float = 0.01  # rad

    # Command delay (latency)
    command_delay: float = 0.02  # s

    # Collision detection
    collision_radius: float = 0.10  # m


@dataclass
class DroneState:
    """Internal drone state."""
    # Position
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity (world frame)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Heading
    yaw: float = 0.0
    yaw_rate: float = 0.0

    # Wind state
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0

    # Gust state
    gust_remaining: float = 0.0
    gust_vx: float = 0.0
    gust_vy: float = 0.0

    # Time
    t: float = 0.0


class DroneSimulator:
    """
    Simple drone dynamics simulator.

    Integrates velocity commands with first-order dynamics,
    adds wind disturbances, and simulates sensor noise.
    """

    def __init__(
        self,
        params: Optional[DroneSimParams] = None,
        initial_state: Optional[DroneState] = None,
        obstacle_fn: Optional[Callable[[float, float], bool]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize simulator.

        Args:
            params: Simulator configuration
            initial_state: Starting state (defaults to origin)
            obstacle_fn: Function (x, y) -> bool, True if in collision
            seed: Random seed for reproducibility
        """
        self.params = params or DroneSimParams()
        self.state = initial_state or DroneState()
        self.obstacle_fn = obstacle_fn

        self._rng = np.random.default_rng(seed)
        self._command_buffer: List[Tuple[float, float, float, float, float]] = []

        # Initialize wind to mean
        if self.params.wind_enabled:
            self.state.wind_x = self.params.wind_mean[0]
            self.state.wind_y = self.params.wind_mean[1]
            self.state.wind_z = self.params.wind_mean[2]

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        yaw: float = 0.0,
    ) -> DroneState:
        """Reset simulator to specified state."""
        self.state = DroneState(x=x, y=y, z=z, yaw=yaw)
        if self.params.wind_enabled:
            self.state.wind_x = self.params.wind_mean[0]
            self.state.wind_y = self.params.wind_mean[1]
            self.state.wind_z = self.params.wind_mean[2]
        self._command_buffer.clear()
        return self.state

    def step(
        self,
        vx_cmd: float,
        vy_cmd: float,
        vz_cmd: float = 0.0,
        yaw_rate_cmd: float = 0.0,
    ) -> Tuple[DroneState, dict]:
        """
        Step simulation forward by dt.

        Args:
            vx_cmd: Commanded X velocity (body frame for holonomic)
            vy_cmd: Commanded Y velocity
            vz_cmd: Commanded Z velocity
            yaw_rate_cmd: Commanded yaw rate

        Returns:
            (state, info) where info contains debug data
        """
        p = self.params
        s = self.state
        dt = p.dt

        # Transform body-frame commands to world frame
        cos_yaw = np.cos(s.yaw)
        sin_yaw = np.sin(s.yaw)

        # Body to world transformation
        vx_world_cmd = vx_cmd * cos_yaw - vy_cmd * sin_yaw
        vy_world_cmd = vx_cmd * sin_yaw + vy_cmd * cos_yaw
        vz_world_cmd = vz_cmd

        # Clip commands to limits
        speed_xy = np.sqrt(vx_world_cmd**2 + vy_world_cmd**2)
        if speed_xy > p.max_speed_xy:
            scale = p.max_speed_xy / speed_xy
            vx_world_cmd *= scale
            vy_world_cmd *= scale
        vz_world_cmd = np.clip(vz_world_cmd, -p.max_speed_z, p.max_speed_z)
        yaw_rate_cmd = np.clip(yaw_rate_cmd, -p.max_yaw_rate, p.max_yaw_rate)

        # Update wind (Ornstein-Uhlenbeck process)
        if p.wind_enabled:
            wind_noise = self._rng.normal(0, p.wind_std * np.sqrt(dt), 3)
            alpha = np.exp(-dt / p.wind_tau)
            s.wind_x = alpha * s.wind_x + (1 - alpha) * p.wind_mean[0] + wind_noise[0]
            s.wind_y = alpha * s.wind_y + (1 - alpha) * p.wind_mean[1] + wind_noise[1]
            s.wind_z = alpha * s.wind_z + (1 - alpha) * p.wind_mean[2] + wind_noise[2]

        # Update gusts
        if p.gust_enabled:
            if s.gust_remaining > 0:
                s.gust_remaining -= dt
            elif self._rng.random() < p.gust_probability:
                # Start new gust
                gust_angle = self._rng.uniform(0, 2 * np.pi)
                s.gust_vx = p.gust_magnitude * np.cos(gust_angle)
                s.gust_vy = p.gust_magnitude * np.sin(gust_angle)
                s.gust_remaining = p.gust_duration

        # Total disturbance
        dist_vx = s.wind_x + (s.gust_vx if s.gust_remaining > 0 else 0)
        dist_vy = s.wind_y + (s.gust_vy if s.gust_remaining > 0 else 0)
        dist_vz = s.wind_z

        # Process noise
        proc_noise = self._rng.normal(0, p.process_noise_std * np.sqrt(dt), 3)

        # First-order velocity dynamics
        decay = 1 - np.exp(-dt / p.tau_velocity)
        s.vx = s.vx + (vx_world_cmd - s.vx) * decay + dist_vx * dt + proc_noise[0]
        s.vy = s.vy + (vy_world_cmd - s.vy) * decay + dist_vy * dt + proc_noise[1]
        s.vz = s.vz + (vz_world_cmd - s.vz) * decay + dist_vz * dt + proc_noise[2]

        # Yaw dynamics
        yaw_decay = 1 - np.exp(-dt / p.tau_yaw)
        s.yaw_rate = s.yaw_rate + (yaw_rate_cmd - s.yaw_rate) * yaw_decay

        # Save old position for collision check
        old_x, old_y = s.x, s.y

        # Integrate position
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.z += s.vz * dt

        # Integrate yaw
        s.yaw += s.yaw_rate * dt
        # Normalize yaw to [-π, π]
        while s.yaw > np.pi:
            s.yaw -= 2 * np.pi
        while s.yaw < -np.pi:
            s.yaw += 2 * np.pi

        # Collision detection
        collision = False
        if self.obstacle_fn is not None:
            if self.obstacle_fn(s.x, s.y):
                # Collision! Revert position and zero velocity
                s.x = old_x
                s.y = old_y
                s.vx = 0.0
                s.vy = 0.0
                collision = True

        # Altitude limits (ground)
        if s.z < 0:
            s.z = 0
            s.vz = max(0, s.vz)

        # Update time
        s.t += dt

        info = {
            "collision": collision,
            "wind": (s.wind_x, s.wind_y, s.wind_z),
            "gust_active": s.gust_remaining > 0,
            "disturbance": (dist_vx, dist_vy, dist_vz),
        }

        return s, info

    def get_measured_state(self) -> Tuple[float, float, float, float, float, float, float]:
        """
        Get noisy sensor measurements.

        Returns:
            (x, y, z, vx, vy, vz, yaw) with sensor noise
        """
        p = self.params
        s = self.state

        x = s.x + self._rng.normal(0, p.position_noise_std)
        y = s.y + self._rng.normal(0, p.position_noise_std)
        z = s.z + self._rng.normal(0, p.position_noise_std)
        vx = s.vx + self._rng.normal(0, p.velocity_noise_std)
        vy = s.vy + self._rng.normal(0, p.velocity_noise_std)
        vz = s.vz + self._rng.normal(0, p.velocity_noise_std)
        yaw = s.yaw + self._rng.normal(0, p.yaw_noise_std)

        return x, y, z, vx, vy, vz, yaw

    @property
    def true_position(self) -> Tuple[float, float, float]:
        """Get true position (without noise)."""
        return self.state.x, self.state.y, self.state.z

    @property
    def true_velocity(self) -> Tuple[float, float, float]:
        """Get true velocity (without noise)."""
        return self.state.vx, self.state.vy, self.state.vz

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self.state.t


@dataclass
class SimulationResult:
    """Results from a complete simulation run."""
    # Trajectory data
    times: List[float] = field(default_factory=list)
    true_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    measured_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    velocities: List[Tuple[float, float, float]] = field(default_factory=list)
    commands: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # Reference trajectory
    reference_positions: List[Tuple[float, float, float]] = field(default_factory=list)

    # Tracker metadata
    cross_track_errors: List[float] = field(default_factory=list)
    lookahead_distances: List[float] = field(default_factory=list)

    # Events
    collisions: List[float] = field(default_factory=list)  # Times of collisions
    gusts: List[Tuple[float, float]] = field(default_factory=list)  # (start_time, end_time)

    # Final status
    success: bool = False
    final_distance_to_goal: float = float('inf')
    total_time: float = 0.0
    total_path_length: float = 0.0