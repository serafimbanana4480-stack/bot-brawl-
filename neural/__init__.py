"""
Neural Network Components for Soberana Omega Bot

This package contains the neural network architecture for the advanced
policy network that replaces Q-Learning tabular learning.

Architecture:
- SpatialGridBuilder: Converts detections to 21x21 grid
- SpatialCNN: Lightweight CNN for spatial features
- StateEncoder: MLP for scalar state features
- TemporalLSTM: LSTM for temporal memory
- FusionNetwork: Combines all embeddings for policy/value heads
- NeuralPolicy: Complete integrated policy network

Usage:
    from neural import NeuralPolicy
    
    policy = NeuralPolicy()
    action, confidence = policy.get_action(game_state, detections)
"""

from neural.grid_builder import SpatialGridBuilder
from neural.cnn_backbone import SpatialCNN
from neural.state_encoder import StateEncoder
from neural.temporal_memory import TemporalLSTM
from neural.fusion_head import FusionNetwork
from neural.neural_policy import NeuralPolicy

__all__ = [
    "SpatialGridBuilder",
    "SpatialCNN",
    "StateEncoder",
    "TemporalLSTM",
    "FusionNetwork",
    "NeuralPolicy",
]
