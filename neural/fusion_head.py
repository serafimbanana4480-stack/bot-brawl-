"""
Fusion Network for Combining Multi-Modal Embeddings

Fuses spatial, state, and temporal embeddings into a unified representation.
Produces final policy and value heads for PPO training.

Architecture:
- Spatial embedding (256-dim) from CNN
- State embedding (64-dim) from MLP
- Temporal embedding (128-dim) from LSTM
- Fusion: Concatenation + MLP
- Output: Unified embedding (256-dim) + policy/value heads

Usage:
    from neural.fusion_head import FusionNetwork
    
    fusion = FusionNetwork(spatial_dim=256, state_dim=64, temporal_dim=128)
    policy_logits, value = fusion(spatial_emb, state_emb, temporal_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNetwork(nn.Module):
    """
    Fusion network combining multi-modal embeddings.
    
    Combines:
    - Spatial features from CNN (environment layout)
    - Scalar state features from MLP (health, ammo, etc.)
    - Temporal features from LSTM (history, patterns)
    """
    
    def __init__(
        self,
        spatial_dim: int = 256,
        state_dim: int = 64,
        temporal_dim: int = 128,
        fusion_dim: int = 256,
        num_actions: int = 12
    ):
        """
        Initialize FusionNetwork.
        
        Args:
            spatial_dim: Spatial embedding dimension from CNN
            state_dim: State embedding dimension from MLP
            temporal_dim: Temporal embedding dimension from LSTM
            fusion_dim: Fused embedding dimension
            num_actions: Number of output actions (UnifiedAction enum size)
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.state_dim = state_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.num_actions = num_actions
        
        # Fusion MLP
        input_dim = spatial_dim + state_dim + temporal_dim
        self.fusion_fc1 = nn.Linear(input_dim, fusion_dim)
        self.fusion_fc2 = nn.Linear(fusion_dim, fusion_dim)
        
        self.bn1 = nn.BatchNorm1d(fusion_dim)
        self.bn2 = nn.BatchNorm1d(fusion_dim)
        
        self.dropout = nn.Dropout(0.2)
        
        # Policy head (action logits)
        self.policy_head = nn.Linear(fusion_dim, num_actions)
        
        # Value head (state value estimate)
        self.value_head = nn.Linear(fusion_dim, 1)
        
    def forward(
        self,
        spatial_emb: torch.Tensor,
        state_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion network.
        
        Args:
            spatial_emb: Spatial embedding of shape (B, spatial_dim)
            state_emb: State embedding of shape (B, state_dim)
            temporal_emb: Temporal embedding of shape (B, temporal_dim)
            
        Returns:
            (policy_logits, value) tuple
            - policy_logits: Action logits of shape (B, num_actions)
            - value: State value estimate of shape (B, 1)
        """
        # Concatenate all embeddings
        x = torch.cat([spatial_emb, state_emb, temporal_emb], dim=-1)
        
        # Handle batch size 1 for batch normalization
        if x.shape[0] == 1:
            self.eval()
        
        # Fusion MLP
        x = self.fusion_fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fusion_fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Policy head
        policy_logits = self.policy_head(x)
        
        # Value head
        value = self.value_head(x)
        
        return policy_logits, value


class FusionNetworkSmall(nn.Module):
    """
    Smaller fusion network for resource-constrained environments.
    """
    
    def __init__(
        self,
        spatial_dim: int = 128,
        state_dim: int = 32,
        temporal_dim: int = 64,
        fusion_dim: int = 128,
        num_actions: int = 12
    ):
        """
        Initialize small fusion network.
        
        Args:
            spatial_dim: Spatial embedding dimension
            state_dim: State embedding dimension
            temporal_dim: Temporal embedding dimension
            fusion_dim: Fused embedding dimension
            num_actions: Number of output actions
        """
        super().__init__()
        
        input_dim = spatial_dim + state_dim + temporal_dim
        self.fusion_fc = nn.Linear(input_dim, fusion_dim)
        self.bn = nn.BatchNorm1d(fusion_dim)
        
        self.policy_head = nn.Linear(fusion_dim, num_actions)
        self.value_head = nn.Linear(fusion_dim, 1)
        
    def forward(
        self,
        spatial_emb: torch.Tensor,
        state_emb: torch.Tensor,
        temporal_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = torch.cat([spatial_emb, state_emb, temporal_emb], dim=-1)
        
        if x.shape[0] == 1:
            self.eval()
        
        x = self.fusion_fc(x)
        x = self.bn(x)
        x = F.relu(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
