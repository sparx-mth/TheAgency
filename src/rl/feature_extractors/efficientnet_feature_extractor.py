"""
efficientnet_feature_extractor.py - EfficientNet-based feature extractor for SLAM environment
Uses pretrained EfficientNet for better feature extraction from 2D maps.
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

    Uses a pretrained EfficientNet-B0 (or custom variant) to process the 2D global_map
    and combines with other features (positions, facings, active) using MLP.
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
        self.efficientnet = self._get_efficientnet(efficientnet_variant, pretrained)

        # Get the output dimension of EfficientNet
        # EfficientNet-B0 outputs 1280 features, B1: 1280, B2: 1408, B3: 1536, etc.
        efficientnet_output_dims = {
            'b0': 1280, 'b1': 1280, 'b2': 1408, 'b3': 1536,
            'b4': 1792, 'b5': 2048, 'b6': 2304, 'b7': 2560
        }
        efficientnet_out_dim = efficientnet_output_dims.get(efficientnet_variant, 1280)

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        # Adapter layer to convert grayscale to RGB (EfficientNet expects 3 channels)
        # We'll expand the single channel to 3 channels
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, padding=0)

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
        # EfficientNet structure: features -> avgpool -> classifier
        # We only need the features part
        model.classifier = nn.Identity()

        return model

    def forward(self, observations) -> torch.Tensor:
        # Process map with EfficientNet
        map_data = observations['global_map'].float()  # (batch, height, width)
        map_data = map_data.unsqueeze(1)  # Add channel dimension: (batch, 1, height, width)

        # Convert grayscale to RGB-like format for EfficientNet
        map_data_rgb = self.channel_adapter(map_data)  # (batch, 3, height, width)

        # Resize if needed - EfficientNet works best with certain input sizes
        # B0: 224x224, B1: 240x240, B2: 260x260, B3: 300x300, B4: 380x380, etc.
        # For small maps (32x32), we might want to upsample
        current_size = map_data_rgb.shape[-1]
        if current_size < 224:
            # Upsample to at least 224x224 for better feature extraction
            map_data_rgb = nn.functional.interpolate(
                map_data_rgb,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

        # Pass through EfficientNet
        efficientnet_features = self.efficientnet(map_data_rgb)  # (batch, efficientnet_out_dim)

        # Project to lower dimension
        cnn_features = self.projection(efficientnet_features)  # (batch, 256)

        # Flatten other features
        positions = observations['positions'].float().flatten(start_dim=1)  # (batch, num_agents*2)
        facings = observations['facings'].float()  # (batch, num_agents)
        active = observations['active'].float()  # (batch, num_agents)

        other_features = torch.cat([positions, facings, active], dim=1)

        # Combine all features
        combined = torch.cat([cnn_features, other_features], dim=1)

        # Final MLP processing
        return self.mlp(combined)


class SLAMLightweightEfficientNetExtractor(BaseFeaturesExtractor):
    """
    A more lightweight version using a smaller custom EfficientNet-like architecture.
    Better for smaller input maps and faster training.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # Calculate other features dimension
        other_features_dim = (observation_space['positions'].shape[0] * 2 +
                              observation_space['facings'].shape[0] +
                              observation_space['active'].shape[0])

        super().__init__(observation_space, features_dim)

        # Custom lightweight EfficientNet-inspired architecture
        # Using MBConv blocks (Mobile Inverted Bottleneck Convolution)
        self.cnn = nn.Sequential(
            # Initial convolution
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),  # Swish activation used in EfficientNet

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
            nn.SiLU(),
            nn.Dropout(0.2)
        )

        # MLP for combining CNN features with other features
        combined_dim = 256 + other_features_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.SiLU()
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
                nn.SiLU(),
                # Pointwise convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            return nn.Sequential(
                # Expansion
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                          stride=1, padding=1, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # Projection
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, observations) -> torch.Tensor:
        # Process map with CNN
        map_data = observations['global_map'].float()  # (batch, height, width)
        map_data = map_data.unsqueeze(1)  # Add channel dimension: (batch, 1, height, width)

        # Pass through CNN
        cnn_features = self.cnn(map_data)  # (batch, 256)

        # Flatten other features
        positions = observations['positions'].float().flatten(start_dim=1)
        facings = observations['facings'].float()
        active = observations['active'].float()

        other_features = torch.cat([positions, facings, active], dim=1)

        # Combine all features
        combined = torch.cat([cnn_features, other_features], dim=1)

        # Final MLP processing
        return self.mlp(combined)


# Alias for backward compatibility if needed
SLAMCNNExtractor = SLAMLightweightEfficientNetExtractor