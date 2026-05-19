"""
Temporal Memory Module (LSTM)

LSTM-based temporal memory for maintaining history of game states.
Enables the policy network to consider temporal patterns and sequences.

Architecture:
- Input: Sequential state embeddings (64-dim)
- LSTM: Single LSTM layer with 128 hidden units
- Output: 128-dim temporal embedding

Usage:
    from neural.temporal_memory import TemporalLSTM
    
    lstm = TemporalLSTM(input_dim=64, hidden_dim=128)
    temporal_emb, hidden = lstm(state_embeddings, hidden_state)
    # temporal_emb.shape = (B, 128)
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalLSTM(nn.Module):
    """
    LSTM-based temporal memory for sequence modeling.
    
    Maintains hidden state across timesteps to capture:
    - Enemy movement patterns
    - Player action sequences
    - Temporal dependencies in game state
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1):
        """
        Initialize TemporalLSTM.
        
        Args:
            input_dim: Input embedding dimension (from StateEncoder)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output projection (optional, can be identity)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape (B, T, input_dim) or (B, input_dim)
               If (B, input_dim), will add T=1 dimension
            hidden: Optional hidden state tuple (h_n, c_n)
            
        Returns:
            (output, hidden) tuple
            - output: Last hidden state of shape (B, hidden_dim)
            - hidden: Full hidden state tuple for next step
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)
        
        # Run LSTM
        output, hidden = self.lstm(x, hidden)
        
        # Take last timestep output
        output = output[:, -1, :]  # (B, hidden_dim)
        
        # Apply output projection
        output = self.output_proj(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Initialize hidden state to zeros.
        
        Args:
            batch_size: Batch size
            device: Tensor device
            
        Returns:
            Tuple of (h_0, c_0) initialized to zeros
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0)


class TemporalLSTMSmall(nn.Module):
    """
    Smaller LSTM variant for resource-constrained environments.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 64):
        """
        Initialize small temporal LSTM.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: LSTM hidden state dimension
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple] = None
    ) -> tuple[torch.Tensor, tuple]:
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        output, hidden = self.lstm(x, hidden)
        output = output[:, -1, :]
        
        return output, hidden
