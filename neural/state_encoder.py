"""
State Encoder MLP for Scalar State Features

Encodes scalar game state features (health, ammo, cooldowns, etc.)
into a compact embedding for fusion with spatial features.

Architecture:
- Input: 44 scalar features from GameState
- Hidden layers: 2 MLP layers (128 -> 64)
- Output: 64-dim state embedding

Usage:
    from neural.state_encoder import StateEncoder
    
    encoder = StateEncoder(input_dim=44, output_dim=64)
    embedding = encoder(state_vector)  # state_vector.shape = (B, 44)
    # embedding.shape = (B, 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """
    MLP encoder for scalar game state features.
    
    Processes non-spatial features like:
    - Player health, ammo, cooldowns
    - Enemy count, threat level
    - Match time remaining
    - Previous action history
    """
    
    def __init__(self, input_dim: int = 44, hidden_dim: int = 128, output_dim: int = 64):
        """
        Initialize StateEncoder.
        
        Args:
            input_dim: Number of input scalar features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through state encoder.
        
        Args:
            x: Input tensor of shape (B, input_dim)
            
        Returns:
            Output embedding of shape (B, output_dim)
        """
        # Handle batch size 1 for batch normalization
        if x.shape[0] == 1:
            # Use evaluation mode for single sample
            self.eval()
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        # No activation on final layer (linear output)
        
        return x


class StateEncoderSmall(nn.Module):
    """
    Smaller state encoder for resource-constrained environments.
    """
    
    def __init__(self, input_dim: int = 44, output_dim: int = 32):
        """
        Initialize small state encoder.
        
        Args:
            input_dim: Number of input features
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.shape[0] == 1:
            self.eval()
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
