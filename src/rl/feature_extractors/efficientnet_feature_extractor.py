"""
efficientnet_feature_extractor.py - Fixed version with proper memory management
Allows full backbone training without memory leaks
Now includes semantic grouping for better neural network processing
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SLAMEfficientNetExtractor(BaseFeaturesExtractor):
    """
    EfficientNet-based feature extractor for SLAM environment.
    Fixed version that properly handles gradients to prevent memory leaks.
    Now includes semantic grouping for better map understanding.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256,
                 efficientnet_variant: str = 'b0', pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Args:
            observation_space: The observation space of the environment
            features_dim: Output dimension of the feature extractor
            efficientnet_variant: Which EfficientNet variant to use ('b0' to 'b7')
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the EfficientNet backbone during training
        """
        # Calculate other features dimension
        other_features_dim = (observation_space['positions'].shape[0] * 2 +
                              observation_space['facings'].shape[0] +
                              observation_space['active'].shape[0])

        super().__init__(observation_space, features_dim)

        # Select and initialize EfficientNet variant
        self.efficientnet_variant = efficientnet_variant
        self.freeze_backbone = freeze_backbone
        self.efficientnet = self._get_efficientnet(efficientnet_variant, pretrained)

        # Get the output dimension of EfficientNet
        efficientnet_output_dims = {
            'b0': 1280, 'b1': 1280, 'b2': 1408, 'b3': 1536,
            'b4': 1792, 'b5': 2048, 'b6': 2304, 'b7': 2560
        }
        efficientnet_out_dim = efficientnet_output_dims.get(efficientnet_variant, 1280)

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        # NO LONGER NEEDED - we create 3 semantic channels directly
        # self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, padding=0)

        # Projection layer to reduce EfficientNet output dimension
        self.projection = nn.Sequential(
            nn.Linear(efficientnet_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # MLP for combining CNN features with other features
        combined_dim = 256 + other_features_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

        # Store whether we need to resize
        self.needs_resize = True  # Will resize to 224x224 for EfficientNet

    def _get_efficientnet(self, variant: str, pretrained: bool):
        """Get the appropriate EfficientNet model."""
        if variant == 'b0':
            if pretrained:
                model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = efficientnet_b0(weights=None)
        elif variant == 'b1':
            model = models.efficientnet_b1(pretrained=pretrained)
        elif variant == 'b2':
            model = models.efficientnet_b2(pretrained=pretrained)
        elif variant == 'b3':
            model = models.efficientnet_b3(pretrained=pretrained)
        elif variant == 'b4':
            model = models.efficientnet_b4(pretrained=pretrained)
        elif variant == 'b5':
            model = models.efficientnet_b5(pretrained=pretrained)
        elif variant == 'b6':
            model = models.efficientnet_b6(pretrained=pretrained)
        elif variant == 'b7':
            model = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        # Remove the final classification layer
        model.classifier = nn.Identity()

        return model

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

        Creates 3 semantic channels:
        1. Exploration status (known vs unknown)
        2. Traversability (can move here)
        3. Obstacles (walls, barriers)
        """
        batch_size = map_data.shape[0]
        height, width = map_data.shape[1], map_data.shape[2]
        device = map_data.device

        # Create 3 semantic channels
        channels = torch.zeros(batch_size, 3, height, width, device=device)

        # Channel 0: Exploration status (0 = unknown, 1 = explored)
        # Everything that's not -1 (UNKNOWN) is explored
        channels[:, 0, :, :] = (map_data != -1).float()

        # Channel 1: Traversability (0 = blocked, 1 = traversable)
        # FREE_SPACE (0), ENTRY_POINT (2), DOOR_OPEN (4)
        traversable = (map_data == 0) | (map_data == 2) | (map_data == 4)
        channels[:, 1, :, :] = traversable.float()

        # Channel 2: Obstacles and features (0 = empty, 1 = obstacle/feature)
        # WALL (1), DOOR_CLOSED (3), WINDOW (5), OUT_OF_BOUNDS (6)
        obstacles = (map_data == 1) | (map_data == 3) | (map_data == 5) | (map_data == 6)
        channels[:, 2, :, :] = obstacles.float()

        return channels

    def forward(self, observations) -> torch.Tensor:
        # Process map with semantic grouping
        map_data = observations['global_map'].float()

        # Convert to 3 semantic channels (exploration, traversability, obstacles)
        map_data_semantic = self.preprocess_map_to_semantic(map_data)

        # Resize if needed - CRITICAL FIX: Don't create new ops in forward pass
        current_size = map_data_semantic.shape[-1]
        if self.needs_resize and current_size < 224:
            # Use simpler interpolation without creating persistent graph
            map_data_semantic = nn.functional.interpolate(
                map_data_semantic,
                size=(224, 224),
                mode='nearest'  # Changed from bilinear to nearest - faster and less memory
            )

        # Pass through EfficientNet (now accepts 3 channels naturally)
        efficientnet_features = self.efficientnet(map_data_semantic)

        # Project to lower dimension
        cnn_features = self.projection(efficientnet_features)

        # Flatten other features - ensure contiguous memory
        positions = observations['positions'].float().flatten(start_dim=1)
        facings = observations['facings'].float()
        active = observations['active'].float()

        other_features = torch.cat([positions, facings, active], dim=1)

        # Combine all features
        combined = torch.cat([cnn_features, other_features], dim=1)

        # Final MLP processing
        output = self.mlp(combined)

        # CRITICAL: Ensure output is contiguous to prevent memory fragmentation
        return output.contiguous()


class SLAMLightweightEfficientNetExtractor(BaseFeaturesExtractor):
    """
    A more lightweight version using a smaller custom EfficientNet-like architecture.
    Better for smaller input maps and faster training.
    Now includes semantic grouping for better map understanding.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Calculate other features dimension
        other_features_dim = (observation_space['positions'].shape[0] * 2 +
                              observation_space['facings'].shape[0] +
                              observation_space['active'].shape[0])

        super().__init__(observation_space, features_dim)

        # Custom lightweight EfficientNet-inspired architecture
        # Now expecting 3 input channels from semantic grouping
        self.cnn = nn.Sequential(
            # Initial convolution - now accepts 3 channels
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),  # Use inplace operations to save memory

            # MBConv-like blocks
            self._make_mbconv_block(32, 16, expansion=1),
            self._make_mbconv_block(16, 24, expansion=6),
            self._make_mbconv_block(24, 40, expansion=6),
            nn.MaxPool2d(2, 2),

            self._make_mbconv_block(40, 80, expansion=6),
            self._make_mbconv_block(80, 112, expansion=6),

            # Final layers
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(112, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2)
        )

        # MLP for combining CNN features with other features
        combined_dim = 256 + other_features_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.SiLU(inplace=True)
        )

    def _make_mbconv_block(self, in_channels, out_channels, expansion=6):
        """Create a Mobile Inverted Bottleneck Convolution block."""
        hidden_dim = in_channels * expansion

        if expansion == 1:
            return nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(in_channels, in_channels, kernel_size=3,
                          stride=1, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                # Pointwise convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            return nn.Sequential(
                # Expansion
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                          stride=1, padding=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                # Projection
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
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
        # Process map with semantic grouping
        map_data = observations['global_map'].float()

        # Convert to 3 semantic channels (exploration, traversability, obstacles)
        map_data_semantic = self.preprocess_map_to_semantic(map_data)

        # Pass through CNN (now processes 3 semantic channels)
        cnn_features = self.cnn(map_data_semantic)

        # Flatten other features
        positions = observations['positions'].float().flatten(start_dim=1)
        facings = observations['facings'].float()
        active = observations['active'].float()

        other_features = torch.cat([positions, facings, active], dim=1)

        # Combine all features
        combined = torch.cat([cnn_features, other_features], dim=1)

        # Final MLP processing
        output = self.mlp(combined)

        # Ensure contiguous output
        return output.contiguous()


# Alias for backward compatibility
SLAMCNNExtractor = SLAMLightweightEfficientNetExtractor