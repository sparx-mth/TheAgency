"""
environments/single_agent_wrapper.py

This file provides a single-agent wrapper for the multi-agent SLAM environment.
It simplifies the interface for single-agent reinforcement learning scenarios,
making it easier to use with libraries like Stable Baselines3.

The wrapper automatically configures the environment for one agent and handles
the conversion between single-agent and multi-agent interfaces transparently.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from .slam_env import MultiAgentSLAMEnv
from sensors.base_sensor import BaseSensor
from communication.comm_interface import CommunicationInterface


class SingleAgentSLAMEnv(MultiAgentSLAMEnv):
    """
    Single-agent wrapper for the SLAM environment.

    This wrapper simplifies the multi-agent environment for single-agent
    reinforcement learning. It:
    - Forces num_agents=1
    - Simplifies action and observation interfaces
    - Maintains compatibility with all features (sensors, communication, etc.)

    This is the recommended environment for training single RL agents.
    """

    def __init__(
            self,
            width: int = 32,
            height: int = 32,
            max_steps: int = 1000,
            map_path: Optional[str] = None,
            randomize: bool = True,
            render_mode: Optional[str] = None,
            # Sensor configuration
            sensor: Optional[BaseSensor] = None,
            sensor_params: Optional[Dict[str, Any]] = None,
            # Communication
            communication: Optional[CommunicationInterface] = None,
            # Reward parameters
            discovery_reward: float = 0.1,
            collision_penalty: float = -1.0,
            step_penalty: float = -0.001,
            completion_bonus: float = 10.0,
    ):
        """
        Initialize single-agent SLAM environment.

        Args:
            width: Width of the grid map
            height: Height of the grid map
            max_steps: Maximum steps per episode
            map_path: Path to load a pre-defined map
            randomize: Whether to generate random maps
            render_mode: 'human' or 'rgb_array'
            sensor: Specific sensor instance for the agent
            sensor_params: Parameters for creating default sensor
            communication: Communication interface
            discovery_reward: Reward per newly discovered cell
            collision_penalty: Penalty for colliding
            step_penalty: Penalty per step
            completion_bonus: Bonus for completing exploration
        """
        # Configure sensor for single agent
        sensor_config = None
        if sensor is not None:
            sensor_config = {0: sensor}

        # Force single agent
        super().__init__(
            width=width,
            height=height,
            num_agents=1,  # Always single agent
            max_steps=max_steps,
            map_path=map_path,
            randomize=randomize,
            render_mode=render_mode,
            sensor_config=sensor_config,
            default_sensor_params=sensor_params,
            communication=communication,
            discovery_reward=discovery_reward,
            collision_penalty=collision_penalty,
            step_penalty=step_penalty,
            completion_bonus=completion_bonus,
        )

    @property
    def name(self) -> str:
        """Get environment name."""
        return "SingleAgentSLAMEnv"

    def get_drone_sensor(self) -> BaseSensor:
        """
        Get the sensor attached to the single drone.

        Returns:
            The drone's sensor instance
        """
        if self.drones:
            return self.drones[0].sensor
        return None

    def get_drone_state(self) -> Dict[str, Any]:
        """
        Get the current state of the single drone.

        Returns:
            Dictionary with drone state information
        """
        if self.drones:
            return self.drones[0].to_dict()
        return {}