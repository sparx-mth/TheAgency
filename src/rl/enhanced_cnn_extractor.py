"""
enhanced_cnn_extractor.py - Enhanced CNN feature extractor for larger maps
Designed for complex 32x32 house map with multiple rooms and corridors.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class EnhancedSLAMCNNExtractor(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor for larger, more complex SLAM environments.

    Designed for 32x32 maps with multiple rooms, corridors, and complex spatial structure.
    Uses deeper CNN with more channels and spatial attention.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # Calculate feature dimensions
        cnn_output_dim = 256  # Larger CNN output for complex maps
        other_features_dim = observation_space['positions'].shape[0] * 2 + \
                             observation_space['facings'].shape[0] + \
                             observation_space['active'].shape[0]

        super().__init__(observation_space, features_dim)

        # Enhanced CNN for processing larger 2D maps
        map_shape = observation_space['global_map'].shape  # (height, width)

        self.cnn = nn.Sequential(
            # First block: capture local features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # Second block: capture room-level features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            # Third block: capture global structure
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4

            # Fourth block: high-level features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Ensure 2x2 output regardless of input size
            nn.Flatten(),

            # Dense layers for CNN features
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, cnn_output_dim),
            nn.ReLU()
        )

        # Enhanced MLP for combining features
        combined_dim = cnn_output_dim + other_features_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process map with enhanced CNN
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


class UltraEnhancedSLAMCNNExtractor(BaseFeaturesExtractor):
    """
    Ultra-enhanced CNN with attention mechanism for very complex maps.
    Use this if the enhanced version isn't enough.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 768):
        # Even larger feature dimensions
        cnn_output_dim = 512
        other_features_dim = observation_space['positions'].shape[0] * 2 + \
                             observation_space['facings'].shape[0] + \
                             observation_space['active'].shape[0]

        super().__init__(observation_space, features_dim)

        # Ultra-enhanced CNN with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512 * 2 * 2),
            nn.Sigmoid()
        )

        # Final CNN processing
        self.cnn_final = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, cnn_output_dim),
            nn.ReLU()
        )

        # Ultra-enhanced MLP
        combined_dim = cnn_output_dim + other_features_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process map with ultra-enhanced CNN
        map_data = observations['global_map'].float()
        map_data = map_data.unsqueeze(1)

        # Forward through conv layers
        x = self.conv1(map_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (batch, 512*2*2)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Final CNN processing
        cnn_features = self.cnn_final(x)

        # Process other features
        positions = observations['positions'].float().flatten(start_dim=1)
        facings = observations['facings'].float()
        active = observations['active'].float()
        other_features = torch.cat([positions, facings, active], dim=1)

        # Combine and process
        combined = torch.cat([cnn_features, other_features], dim=1)
        return self.mlp(combined)