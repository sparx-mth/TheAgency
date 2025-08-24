"""
cnn_feature_extractor.py - Custom CNN feature extractor for SLAM environment
Processes 2D map with CNN and other features with MLP.

Copy for evaluation to ensure model loading works.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class SLAMCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for SLAM environment.

    Processes the 2D global_map with CNN layers and combines with
    other features (positions, facings, active) using MLP.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Calculate total feature dimension
        cnn_output_dim = 64  # CNN output features
        other_features_dim = observation_space['positions'].shape[0] * 2 + \
                             observation_space['facings'].shape[0] + \
                             observation_space['active'].shape[0]

        super().__init__(observation_space, features_dim)

        # CNN for processing the 2D map
        map_shape = observation_space['global_map'].shape  # (height, width)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output size regardless of input map size
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, cnn_output_dim),
            nn.ReLU()
        )

        # MLP for combining CNN features with other features
        combined_dim = cnn_output_dim + other_features_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def preprocess_map_to_semantic(self, map_data):
        """
        Convert map to semantic channels for better learning.

        Map values:
        - UNKNOWN = -1
        - FREE_SPACE = 0
        - WALL = 1
        - ENTRY_POINT = 2
        - DOOR_CLOSED = 3
        - DOOR_OPEN = 4
        - WINDOW = 5
        - OUT_OF_BOUNDS = 6

        Creates 2 semantic channels:
        1. Exploration status (known vs unknown)
        2. Traversability (can move here)
        """
        batch_size = map_data.shape[0]
        height, width = map_data.shape[1], map_data.shape[2]
        device = map_data.device

        # Create 2 semantic channels
        channels = torch.zeros(batch_size, 2, height, width, device=device)

        # Channel 0: Exploration status (0 = unknown, 1 = explored)
        # Everything that's not -1 (UNKNOWN) is explored
        channels[:, 0, :, :] = (map_data != -1).float()

        # Channel 1: Traversability (0 = blocked, 1 = traversable)
        # FREE_SPACE (0), ENTRY_POINT (2), DOOR_OPEN (4)
        traversable = (map_data == 0) | (map_data == 2) | (map_data == 4)
        channels[:, 1, :, :] = traversable.float()

        return channels

    def forward(self, observations) -> torch.Tensor:
        # Process map with CNN
        map_data = observations['global_map'].float()  # (batch, height, width)
        # # Convert map to semantic channels
        map_data = self.preprocess_map_to_semantic(map_data)  # (batch, 2, height, width)
        # map_data = map_data.unsqueeze(1)  # Add channel dimension: (batch, 1, height, width)
        cnn_features = self.cnn(map_data)  # (batch, cnn_output_dim)

        # Flatten other features
        positions = observations['positions'].float().flatten(start_dim=1)  # (batch, num_agents*2)
        facings = observations['facings'].float()  # (batch, num_agents)
        active = observations['active'].float()  # (batch, num_agents)

        other_features = torch.cat([positions, facings, active], dim=1)

        # Combine all features
        combined = torch.cat([cnn_features, other_features], dim=1)

        # Final MLP processing
        return self.mlp(combined)