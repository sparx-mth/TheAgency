"""
Custom DQN Agent with Fixed 44x44 Input

This agent uses:
- 44x44 input matrix (with padding for smaller maps)
- 2 other robots' poses
- Current robot's pose
- Standard DQN output (Q-values for actions)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any
from .base_slam_agent import BaseSLAMAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class FixedSizeDQNetwork(nn.Module):
    """
    DQN Network with fixed 44x44 input size.

    Input:
    - 44x44 map (padded if necessary)
    - 2 other robots' poses (x, y, orientation for each)
    - Current robot's pose (x, y, orientation)

    Output:
    - Q-values for 4 actions [FORWARD, TURN_LEFT, TURN_RIGHT, STAY]
    """

    def __init__(self, num_actions: int = 4):
        super(FixedSizeDQNetwork, self).__init__()

        # CNN for 44x44 map processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # 44x44 -> 22x22
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 22x22 -> 11x11
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 11x11 -> 11x11
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 11x11 -> 11x11

        # Calculate flattened size: 11 * 11 * 128 = 15,488
        conv_output_size = 11 * 11 * 128

        # Pose features: 2 other robots (x,y,θ each) + current robot (x,y,θ) = 9 features
        pose_features_size = 9

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + pose_features_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_actions)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, map_input: torch.Tensor, pose_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            map_input: Tensor of shape (batch, 1, 44, 44)
            pose_input: Tensor of shape (batch, 9) containing:
                       [other_robot1_x, other_robot1_y, other_robot1_theta,
                        other_robot2_x, other_robot2_y, other_robot2_theta,
                        current_robot_x, current_robot_y, current_robot_theta]

        Returns:
            Q-values of shape (batch, num_actions)
        """
        # Process map through CNN
        x = F.relu(self.conv1(map_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten CNN output
        x = x.view(x.size(0), -1)

        # Concatenate with pose features
        x = torch.cat([x, pose_input], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)

        return q_values


class CustomDQNAgent(BaseSLAMAgent):
    """
    Custom DQN Agent with fixed 44x44 input and robot pose information.

    This agent:
    - Pads maps smaller than 44x44 to fit the fixed input size
    - Tracks poses of 2 other robots
    - Uses current robot's pose
    - Outputs standard DQN actions
    """

    def __init__(
        self,
        num_agents: int = 3,  # Total number of robots (including this one)
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        update_frequency: int = 4,
        target_update_frequency: int = 100
    ):
        super().__init__(num_agents)

        assert num_agents >= 3, "This agent requires at least 3 robots (current + 2 others)"

        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency

        # Fixed input size
        self.input_size = 44

        # Networks for each robot
        self.q_networks = {}
        self.target_networks = {}
        self.optimizers = {}

        for i in range(num_agents):
            self.q_networks[i] = FixedSizeDQNetwork().to(device)
            self.target_networks[i] = FixedSizeDQNetwork().to(device)
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.optimizers[i] = optim.Adam(self.q_networks[i].parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Tracking
        self.steps = 0
        self.robot_poses = {}  # Store all robot poses

    def reset(self):
        """Reset agent state for new episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.robot_poses = {}

    def get_actions(self, observations: Dict[int, Any], info: Dict[str, Any]) -> Dict[int, int]:
        """Get actions for all agents."""
        # Update robot poses from observations
        for agent_id, obs in observations.items():
            if obs['active']:
                # Store pose as (x, y, theta)
                # Normalize positions to [0, 1] assuming max map size of 100
                x_norm = obs['position'][0] / 100.0
                y_norm = obs['position'][1] / 100.0
                theta_norm = obs['facing_direction'] / 3.0  # Normalize to [0, 1]
                self.robot_poses[agent_id] = (x_norm, y_norm, theta_norm)

        actions = {}

        for agent_id, obs in observations.items():
            if not obs['active']:
                actions[agent_id] = 3  # STAY
                continue

            # Epsilon-greedy
            if random.random() < self.epsilon:
                actions[agent_id] = random.randint(0, 3)
            else:
                # Prepare input
                state = self._prepare_input(agent_id, obs, info)
                map_tensor, pose_tensor = state

                with torch.no_grad():
                    q_values = self.q_networks[agent_id](
                        map_tensor.unsqueeze(0).to(device),
                        pose_tensor.unsqueeze(0).to(device)
                    )
                    actions[agent_id] = q_values.argmax().item()

        self.steps += 1
        return actions

    def _prepare_input(
        self,
        agent_id: int,
        obs: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the fixed-size input for the network.

        Returns:
            map_tensor: 44x44 padded map
            pose_tensor: 9-element tensor with robot poses
        """
        # Get the global map from info (unified view)
        global_map = info.get('global_map', obs['local_map'])

        # Convert to float and normalize
        map_array = global_map.astype(np.float32)
        map_array = (map_array + 1) / 7.0  # Normalize to [0, 1]

        # Pad to 44x44 if necessary
        h, w = map_array.shape
        if h < self.input_size or w < self.input_size:
            # Calculate padding
            pad_h = max(0, self.input_size - h)
            pad_w = max(0, self.input_size - w)

            # Pad symmetrically
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            # Pad with 0 (which represents unknown after normalization)
            map_array = np.pad(
                map_array,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
        elif h > self.input_size or w > self.input_size:
            # If larger, take center crop
            start_h = (h - self.input_size) // 2
            start_w = (w - self.input_size) // 2
            map_array = map_array[
                start_h:start_h + self.input_size,
                start_w:start_w + self.input_size
            ]

        # Ensure exactly 44x44
        assert map_array.shape == (self.input_size, self.input_size), f"Map shape {map_array.shape} != 44x44"

        # Create map tensor with channel dimension
        map_tensor = torch.tensor(map_array, dtype=torch.float32).unsqueeze(0)  # (1, 44, 44)

        # Prepare pose features
        pose_features = []

        # Get 2 other robots' poses
        other_robot_ids = [i for i in self.robot_poses.keys() if i != agent_id]

        # Add first 2 other robots' poses
        for i in range(2):
            if i < len(other_robot_ids):
                other_id = other_robot_ids[i]
                x, y, theta = self.robot_poses[other_id]
                pose_features.extend([x, y, theta])
            else:
                # If less than 2 other robots, use default values
                pose_features.extend([0.5, 0.5, 0.0])

        # Add current robot's pose
        if agent_id in self.robot_poses:
            x, y, theta = self.robot_poses[agent_id]
            pose_features.extend([x, y, theta])
        else:
            # Default if pose not available
            pose_features.extend([0.5, 0.5, 0.0])

        pose_tensor = torch.tensor(pose_features, dtype=torch.float32)

        return map_tensor, pose_tensor

    def update(
        self,
        observations: Dict[int, Any],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_observations: Dict[int, Any],
        dones: Dict[int, bool],
        info: Dict[str, Any],
        next_info: Dict[str, Any]
    ):
        """Update the Q-networks using experience replay."""
        # Store transitions
        for agent_id in observations:
            if observations[agent_id]['active']:
                state = self._prepare_input(agent_id, observations[agent_id], info)
                next_state = self._prepare_input(agent_id, next_observations[agent_id], next_info)

                self.replay_buffer.append((
                    agent_id,
                    state,
                    actions[agent_id],
                    rewards[agent_id],
                    next_state,
                    dones[agent_id]
                ))

        # Train if ready
        if len(self.replay_buffer) >= self.batch_size and self.steps % self.update_frequency == 0:
            self._train_step()

        # Update target networks
        if self.steps % self.target_update_frequency == 0:
            for agent_id in range(self.num_agents):
                self.target_networks[agent_id].load_state_dict(self.q_networks[agent_id].state_dict())

    def _train_step(self):
        """Train all agents using the full batch of experiences."""
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batch tensors
        map_batch = []
        pose_batch = []
        next_map_batch = []
        next_pose_batch = []
        actions_batch = []
        rewards_batch = []
        dones_batch = []

        for _, state, action, reward, next_state, done in batch:
            map_tensor, pose_tensor = state
            next_map_tensor, next_pose_tensor = next_state

            map_batch.append(map_tensor)
            pose_batch.append(pose_tensor)
            next_map_batch.append(next_map_tensor)
            next_pose_batch.append(next_pose_tensor)
            actions_batch.append(action)
            rewards_batch.append(reward)
            dones_batch.append(float(done))

        # Convert to tensors
        map_batch = torch.stack(map_batch).to(device)
        pose_batch = torch.stack(pose_batch).to(device)
        next_map_batch = torch.stack(next_map_batch).to(device)
        next_pose_batch = torch.stack(next_pose_batch).to(device)
        actions_batch = torch.tensor(actions_batch, dtype=torch.long).to(device)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        dones_batch = torch.tensor(dones_batch, dtype=torch.float32).to(device)

        # Train each agent on the same batch
        for agent_id in self.q_networks.keys():
            # Current Q values
            current_q_values = self.q_networks[agent_id](map_batch, pose_batch)
            current_q_values = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

            # Target Q values
            with torch.no_grad():
                next_q_values = self.target_networks[agent_id](next_map_batch, next_pose_batch)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards_batch + (1 - dones_batch) * self.gamma * max_next_q_values

            # Loss and optimization
            loss = F.mse_loss(current_q_values, target_q_values)
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_networks[agent_id].parameters(), 1.0)
            self.optimizers[agent_id].step()

    def save(self, path: str):
        """Save model."""
        save_dict = {
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        for agent_id in range(self.num_agents):
            save_dict[f'q_network_{agent_id}'] = self.q_networks[agent_id].state_dict()
            save_dict[f'optimizer_{agent_id}'] = self.optimizers[agent_id].state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=device)
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

        for agent_id in range(self.num_agents):
            self.q_networks[agent_id].load_state_dict(checkpoint[f'q_network_{agent_id}'])
            self.target_networks[agent_id].load_state_dict(checkpoint[f'q_network_{agent_id}'])
            self.optimizers[agent_id].load_state_dict(checkpoint[f'optimizer_{agent_id}'])