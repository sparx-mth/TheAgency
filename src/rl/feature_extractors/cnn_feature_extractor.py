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

        # CNN for processing the 2D map - now expects 1 channel
        map_shape = observation_space['global_map'].shape  # (height, width)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Changed to 1 channel
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
        Convert map to single semantic channel with 3 values.

        Map values:
        - UNKNOWN = -1        → 0.0 (unknown)
        - FREE_SPACE = 0      → 0.5 (traversable)
        - WALL = 1            → 1.0 (blocked)
        - ENTRY_POINT = 2     → 0.5 (traversable)
        - DOOR_CLOSED = 3     → 1.0 (blocked)
        - DOOR_OPEN = 4       → 0.5 (traversable)
        - WINDOW = 5          → 1.0 (blocked)
        - OUT_OF_BOUNDS = 6   → 1.0 (blocked)

        Creates 1 semantic channel with values:
        - 0.0 = Unknown (unexplored)
        - 0.5 = Traversable (can move here)
        - 1.0 = Blocked (cannot move here)
        """
        batch_size = map_data.shape[0]
        height, width = map_data.shape[1], map_data.shape[2]
        device = map_data.device

        # Create 1 semantic channel
        channel = torch.zeros(batch_size, 1, height, width, device=device)

        # Set values based on map type
        semantic = torch.zeros_like(map_data, dtype=torch.float32)

        # Unknown areas (-1) → 0.0
        semantic[map_data == -1] = 0.0

        # Traversable areas → 0.5
        semantic[map_data == 0] = 0.5  # FREE_SPACE
        semantic[map_data == 2] = 0.5  # ENTRY_POINT
        semantic[map_data == 4] = 0.5  # DOOR_OPEN

        # Blocked areas → 1.0
        semantic[map_data == 1] = 1.0  # WALL
        semantic[map_data == 3] = 1.0  # DOOR_CLOSED
        semantic[map_data == 5] = 1.0  # WINDOW
        semantic[map_data == 6] = 1.0  # OUT_OF_BOUNDS

        channel[:, 0, :, :] = semantic

        return channel

    def forward(self, observations) -> torch.Tensor:
        # Process map with CNN
        map_data = observations['global_map'].float()  # (batch, height, width)
        # Convert map to semantic channel
        map_data = self.preprocess_map_to_semantic(map_data)  # (batch, 1, height, width)
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

class NavigationCNNExtractor(BaseFeaturesExtractor):
    """
    Simplified navigation feature extractor with only goal position.
    Uses 1-channel semantic preprocessing from SLAMCNNExtractor.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN for 1-channel semantic map
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Changed to 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # Calculate total features
        cnn_dim = 32 * 4 * 4
        num_agents = observation_space['positions'].shape[0]

        # Drone features: positions(2*n) + facings(n) + active(n)
        drone_dim = num_agents * 2 + num_agents + num_agents

        # Goal features: ONLY position (x, y)
        goal_dim = 2  # Just goal_position(2)

        other_dim = drone_dim + goal_dim

        # MLP for combining all features
        self.mlp = nn.Sequential(
            nn.Linear(cnn_dim + other_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )

    def preprocess_map_to_semantic(self, map_data):
        """
        Convert map to single semantic channel with 3 values.

        Creates 1 semantic channel with values:
        - 0.0 = Unknown (unexplored)
        - 0.5 = Traversable (can move here)
        - 1.0 = Blocked (cannot move here)
        """
        batch_size = map_data.shape[0]
        height, width = map_data.shape[1], map_data.shape[2]
        device = map_data.device

        # Create 1 semantic channel
        channel = torch.zeros(batch_size, 1, height, width, device=device)

        # Set values based on map type
        semantic = torch.zeros_like(map_data, dtype=torch.float32)

        # Unknown areas (-1) → 0.0
        semantic[map_data == -1] = 0.0

        # Traversable areas → 0.5
        semantic[map_data == 0] = 0.5  # FREE_SPACE
        semantic[map_data == 2] = 0.5  # ENTRY_POINT
        semantic[map_data == 4] = 0.5  # DOOR_OPEN

        # Blocked areas → 1.0
        semantic[map_data == 1] = 1.0  # WALL
        semantic[map_data == 3] = 1.0  # DOOR_CLOSED
        semantic[map_data == 5] = 1.0  # WINDOW
        semantic[map_data == 6] = 1.0  # OUT_OF_BOUNDS

        channel[:, 0, :, :] = semantic

        return channel

    def forward(self, observations) -> torch.Tensor:
        # Process map with semantic preprocessing
        map_data = observations['global_map'].float()
        semantic_map = self.preprocess_map_to_semantic(map_data)
        cnn_out = self.cnn(semantic_map)

        # Drone features
        positions = observations['positions'].float().flatten(start_dim=1)
        facings = observations['facings'].float()
        active = observations['active'].float()

        # Goal features - ONLY position
        goal_position = observations['goal_position'].float()

        # Concatenate all features
        other = torch.cat([
            positions, facings, active,  # Drone state
            goal_position  # Only goal position
        ], dim=1)

        # Combine CNN and other features
        combined = torch.cat([cnn_out, other], dim=1)

        # Final MLP processing
        return self.mlp(combined)