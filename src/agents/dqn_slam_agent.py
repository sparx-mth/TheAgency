"""
Custom DQN Agent with Fixed 10x19 Input - Single Network for All Drones

This agent uses:
- 10x19 global map input (what all drones have discovered)
- Current drone's pose (x, y, orientation)
- Single shared network for all drones
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
    DQN Network with fixed 10x19 input size.

    Input:
    - 10x19 global map (padded if necessary)
    - Current robot's pose (x, y, orientation)

    Output:
    - Q-values for 4 actions [FORWARD, TURN_LEFT, TURN_RIGHT, STAY]
    """

    def __init__(self, num_actions: int = 4):
        super(FixedSizeDQNetwork, self).__init__()

        # Modified CNN for 10x19 map processing
        # Note: Using smaller kernels and strides due to smaller input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 10x10 -> 10x10
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 10x10 -> 10x10
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0) # 10x10 -> 8x8

        # Calculate flattened size: 8 * 17 * 128 = 17,408
        conv_output_size = 8 * 8 * 128

        # Pose features: current robot (x,y,θ) = 3 features
        pose_features_size = 3

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
            map_input: Tensor of shape (batch, 1, 10, 19)
            pose_input: Tensor of shape (batch, 3) containing:
                       [current_robot_x, current_robot_y, current_robot_theta]

        Returns:
            Q-values of shape (batch, num_actions)
        """
        # Process map through CNN
        x = F.relu(self.conv1(map_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

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
    Custom DQN Agent with fixed 10x19 input using a single shared network.

    This agent:
    - Uses global map (what all drones have discovered)
    - Uses current drone's pose only
    - Shares a single network across all drones
    """

    def __init__(
        self,
        num_agents: int = 3,
        learning_rate: float = 1e-4,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        update_frequency: int = 4,
        target_update_frequency: int = 2000
    ):
        super().__init__(num_agents)

        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency

        # Fixed input size - changed to 10x19
        self.input_height = 10
        self.input_width = 10

        # Single shared network for all drones
        self.q_network = FixedSizeDQNetwork().to(device)
        self.target_network = FixedSizeDQNetwork().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer - now stores all experiences together
        self.replay_buffer = deque(maxlen=buffer_size)

        # Tracking
        self.steps = 0

    def reset(self):
        """Reset agent state for new episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_actions(self, observations: Dict[int, Any], info: Dict[str, Any]) -> Dict[int, int]:
        """Get actions for all agents using the shared network."""
        actions = {}

        # Get global map from info
        global_map = info.get('global_map', None)

        for agent_id, obs in observations.items():
            if not obs['active']:
                actions[agent_id] = 3  # STAY
                continue

            # Epsilon-greedy
            if random.random() < self.epsilon:
                actions[agent_id] = random.randint(0, 3)
            else:
                # Prepare input using global map and current drone's pose
                state = self._prepare_input(agent_id, obs, global_map)
                map_tensor, pose_tensor = state

                with torch.no_grad():
                    q_values = self.q_network(
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
        global_map: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the fixed-size input for the network.

        Returns:
            map_tensor: 10x19 padded global map
            pose_tensor: 3-element tensor with current drone's pose
        """
        # Use global map if available, otherwise use local map
        if global_map is not None:
            map_array = global_map.astype(np.float32)
        else:
            map_array = obs['local_map'].astype(np.float32)

        map_array = np.where(map_array == 2, 0, map_array)

        # Normalize to [0, 1]
        map_array = (map_array + 1) / 2.0  # Normalize from [-1, 1] to [0, 1]

        # Pad to 10x19 if necessary
        h, w = map_array.shape
        if h < self.input_height or w < self.input_width:
            # Calculate padding
            pad_h = max(0, self.input_height - h)
            pad_w = max(0, self.input_width - w)

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
        elif h > self.input_height or w > self.input_width:
            # If larger, take center crop
            start_h = max(0, (h - self.input_height) // 2)
            start_w = max(0, (w - self.input_width) // 2)
            map_array = map_array[
                start_h:start_h + self.input_height,
                start_w:start_w + self.input_width
            ]

        # Ensure exactly 10x19
        assert map_array.shape == (self.input_height, self.input_width), \
            f"Map shape {map_array.shape} != {self.input_height}x{self.input_width}"

        # Create map tensor with channel dimension
        map_tensor = torch.tensor(map_array, dtype=torch.float32).unsqueeze(0)  # (1, 10, 19)

        # Prepare pose features for current drone only
        # Normalize positions to [0, 1] assuming max map size of 100
        x_norm = obs['position'][0] / 100.0
        y_norm = obs['position'][1] / 100.0
        theta_norm = obs['facing_direction'] / 3.0  # Normalize to [0, 1]

        pose_tensor = torch.tensor([x_norm, y_norm, theta_norm], dtype=torch.float32)

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
        """Update the Q-network using experience replay."""
        # Get global maps
        global_map = info.get('global_map', None)
        next_global_map = next_info.get('global_map', None)

        # Store transitions for all active drones
        for agent_id in observations:
            if observations[agent_id]['active']:
                state = self._prepare_input(agent_id, observations[agent_id], global_map)
                next_state = self._prepare_input(agent_id, next_observations[agent_id], next_global_map)

                self.replay_buffer.append((
                    state,
                    actions[agent_id],
                    rewards[agent_id],
                    next_state,
                    dones[agent_id]
                ))

        # Train if ready
        if len(self.replay_buffer) >= self.batch_size and self.steps % self.update_frequency == 0:
            self._train_step()

        # Update target network
        if self.steps % self.target_update_frequency == 0:
            print('target network update')
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _train_step(self):
        """Train the shared network using experiences from all agents."""
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

        for state, action, reward, next_state, done in batch:
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

        # Current Q values
        current_q_values = self.q_network(map_batch, pose_batch)
        current_q_values = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_map_batch, next_pose_batch)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_batch + (1 - dones_batch) * self.gamma * max_next_q_values

        # Loss and optimization
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

    def save(self, path: str):
        """Save model."""
        save_dict = {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=device)
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])