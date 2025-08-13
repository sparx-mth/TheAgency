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
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
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

    def forward(self, observations) -> torch.Tensor:
        # Process map with CNN
        map_data = observations['global_map'].float()  # (batch, height, width)
        map_data = map_data.unsqueeze(1)  # Add channel dimension: (batch, 1, height, width)
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