"""3D CNN model for voxel-based shape classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
import warnings


class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block with batch norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.2, inplace=True),
            "elu": nn.ELU(inplace=True),
            "gelu": nn.GELU(),
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.use_batch_norm:
            x = self.bn(x)
        
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block for deeper networks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.conv1 = ConvBlock3D(
            in_channels, out_channels, use_batch_norm=use_batch_norm,
            activation=activation
        )
        self.conv2 = ConvBlock3D(
            out_channels, out_channels, use_batch_norm=use_batch_norm,
            activation=activation
        )
        
        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.relu(out + identity)


class VoxelCNN(nn.Module):
    """
    3D CNN for voxel-based shape classification.
    
    Architecture optimized for Metal Performance Shaders on macOS.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        model_config = config['model']
        
        # Model parameters
        self.num_classes = model_config['num_classes']
        self.input_shape = model_config['input_shape']
        self.channels = model_config['channels']
        self.dropout_rate = model_config['dropout_rate']
        self.use_batch_norm = model_config['use_batch_norm']
        self.activation = model_config['activation']
        self.use_residual = model_config.get('use_residual', False)
        self.feature_extraction_mode = model_config.get('feature_extraction_mode', False)
        
        # Check device for MPS compatibility
        self.device = config['training']['device']
        self.use_mps_fallback = self.device == 'mps'
        
        if self.use_mps_fallback:
            warnings.warn(
                "3D pooling layers are not implemented for MPS device. "
                "Using strided convolutions instead. This may slightly affect model performance.",
                UserWarning
            )
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Calculate feature dimension
        self.feature_dim = self.channels[-1]
        
        # Classifier
        self.classifier = self._build_classifier()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Build the convolutional encoder."""
        layers = []
        in_channels = 1  # Binary voxel input
        
        for i, out_channels in enumerate(self.channels):
            if self.use_residual and i > 0:
                layers.append(
                    ResidualBlock3D(
                        in_channels, out_channels,
                        self.use_batch_norm, self.activation
                    )
                )
            else:
                layers.append(
                    ConvBlock3D(
                        in_channels, out_channels,
                        use_batch_norm=self.use_batch_norm,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate if i > 0 else 0
                    )
                )
            
            # Add pooling layer (except for last layer)
            if i < len(self.channels) - 1:
                # Note: Neither MaxPool3d nor AvgPool3d work on MPS currently
                # Use strided convolution as workaround
                if self.use_mps_fallback:
                    layers.append(
                        nn.Conv3d(out_channels, out_channels, 2, stride=2, padding=0)
                    )
                else:
                    layers.append(nn.MaxPool3d(2, 2))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self) -> nn.Sequential:
        """Build the classification head."""
        return nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input voxels."""
        # Add channel dimension if needed
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        # Encode
        features = self.encoder(x)
        
        # Global pooling
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input voxel tensor of shape (B, D, H, W) or (B, 1, D, H, W)
            
        Returns:
            Class logits or features depending on mode
        """
        features = self.extract_features(x)
        
        if self.feature_extraction_mode:
            return features
        
        logits = self.classifier(features)
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate activation maps for visualization."""
        if x.dim() == 4:
            x = x.unsqueeze(1)
        
        attention_maps = []
        
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, (ConvBlock3D, ResidualBlock3D)):
                attention_maps.append(x.clone())
        
        return attention_maps


def create_model(config: Dict[str, Any]) -> VoxelCNN:
    """
    Create a VoxelCNN model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VoxelCNN model instance
    """
    model = VoxelCNN(config)
    
    # Move to appropriate device
    device = config['training']['device']
    if device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
    elif device == 'cuda' and torch.cuda.is_available():
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    
    return model 