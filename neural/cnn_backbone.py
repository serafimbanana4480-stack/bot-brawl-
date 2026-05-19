"""
Spatial CNN Backbone for Grid Feature Extraction

Lightweight CNN architecture for processing the 21x21 spatial grid.
Extracts spatial features from the environment around the player.

Architecture:
- Input: (21, 21, num_classes) spatial grid
- Conv layers: 3 convolutional blocks with ReLU
- Output: 256-dim spatial embedding

Usage:
    from neural.cnn_backbone import SpatialCNN
    
    cnn = SpatialCNN(input_channels=15)
    embedding = cnn(grid)  # grid.shape = (B, 21, 21, 15)
    # embedding.shape = (B, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCNN(nn.Module):
    """
    Lightweight CNN for spatial feature extraction from game grid.
    
    Designed to be fast and efficient for real-time inference:
    - Small kernel sizes (3x3)
    - Reasonable channel counts (32, 64, 128)
    - Global average pooling for fixed-size output
    """
    
    def __init__(self, input_channels: int = 15, output_dim: int = 256):
        """
        Initialize SpatialCNN.
        
        Args:
            input_channels: Number of input channels (classes in schema)
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Output projection
        self.output_proj = nn.Linear(128, output_dim)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, H, W, C)
               If (B, H, W, C), will be permuted to (B, C, H, W)
               
        Returns:
            Output embedding of shape (B, output_dim)
        """
        # Handle channel-last to channel-first conversion
        if x.shape[-1] == x.shape[1] or x.shape[-1] < 50:
            # Assume (B, H, W, C) format
            x = x.permute(0, 3, 1, 2)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 32, 10, 10)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 64, 5, 5)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # (B, 128, 2, 2)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, 128, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 128)
        
        # Output projection
        x = self.dropout(x)
        x = self.output_proj(x)  # (B, output_dim)
        
        return x


class SpatialCNNSmall(nn.Module):
    """
    Even smaller CNN variant for resource-constrained environments.
    
    Uses fewer channels and layers for faster inference.
    """
    
    def __init__(self, input_channels: int = 15, output_dim: int = 128):
        """
        Initialize small SpatialCNN.
        
        Args:
            input_channels: Number of input channels
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.output_proj = nn.Linear(32, output_dim)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.shape[-1] == x.shape[1] or x.shape[-1] < 50:
            x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1).squeeze(-1)
        x = self.output_proj(x)
        
        return x
