"""
Neural Policy Network - Complete Integration

End-to-end neural policy network that replaces Q-Learning tabular learning.
Integrates spatial CNN, state encoder, temporal LSTM, and fusion network.

Architecture:
- Input: Detections + GameState
- Grid Builder: 21x21 spatial grid
- SpatialCNN: 256-dim spatial embedding
- StateEncoder: 64-dim state embedding
- TemporalLSTM: 128-dim temporal embedding
- FusionNetwork: Policy + Value heads
- Output: Action (UnifiedAction) + confidence

Usage:
    from neural.neural_policy import NeuralPolicy
    from core.class_registry import get_schema
    
    policy = NeuralPolicy(schema="full")
    action, confidence = policy.get_action(game_state, detections)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from neural.grid_builder import SpatialGridBuilder
from neural.cnn_backbone import SpatialCNN
from neural.state_encoder import StateEncoder
from neural.temporal_memory import TemporalLSTM
from neural.fusion_head import FusionNetwork
from core.class_registry import UnifiedAction, get_schema, STATE_FEATURE_DIM


class NeuralPolicy(nn.Module):
    """
    Complete neural policy network for Brawl Stars bot.
    
    Replaces tabular Q-learning with a deep neural network that:
    - Processes spatial environment via CNN
    - Encodes scalar state features via MLP
    - Maintains temporal memory via LSTM
    - Fuses all modalities for policy/value prediction
    
    This enables:
    - Better generalization across maps
    - Handling of continuous state spaces
    - Learning complex spatial patterns
    - Temporal strategy planning
    """
    
    def __init__(
        self,
        schema: str = "full",
        spatial_dim: int = 256,
        state_dim: int = 64,
        temporal_dim: int = 128,
        fusion_dim: int = 256
    ):
        """
        Initialize NeuralPolicy.
        
        Args:
            schema: Class schema tier ("core", "extended", "full")
            spatial_dim: Spatial embedding dimension
            state_dim: State embedding dimension
            temporal_dim: Temporal embedding dimension
            fusion_dim: Fusion embedding dimension
        """
        super().__init__()
        
        self.schema = schema
        self.num_classes = len(get_schema(schema))
        self.num_actions = len(UnifiedAction)
        
        # Grid builder
        self.grid_builder = SpatialGridBuilder()
        
        # Spatial CNN
        self.spatial_cnn = SpatialCNN(
            input_channels=self.num_classes,
            output_dim=spatial_dim
        )
        
        # State encoder
        self.state_encoder = StateEncoder(
            input_dim=STATE_FEATURE_DIM,
            output_dim=state_dim
        )
        
        # Temporal LSTM
        self.temporal_lstm = TemporalLSTM(
            input_dim=state_dim,
            hidden_dim=temporal_dim
        )

        # Enemy attention for variable-length enemy sets
        self.enemy_feature_dim = 8
        self.enemy_embedding_dim = 32
        self.enemy_encoder = nn.Linear(self.enemy_feature_dim, self.enemy_embedding_dim)
        self.enemy_attention = nn.MultiheadAttention(
            embed_dim=self.enemy_embedding_dim,
            num_heads=4,
            batch_first=True,
        )
        self.enemy_context_proj = nn.Linear(self.enemy_embedding_dim, state_dim)
        
        # Fusion network
        self.fusion_network = FusionNetwork(
            spatial_dim=spatial_dim,
            state_dim=state_dim,
            temporal_dim=temporal_dim,
            fusion_dim=fusion_dim,
            num_actions=self.num_actions
        )
        
        # Hidden state for temporal memory
        self.hidden_state = None
        
    def forward(
        self,
        grid: torch.Tensor,
        state_features: torch.Tensor,
        enemy_context: Optional[torch.Tensor] = None,
        reset_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through neural policy.
        
        Args:
            grid: Spatial grid of shape (B, H, W, C)
            state_features: Scalar state features of shape (B, 44)
            reset_hidden: Whether to reset LSTM hidden state
            
        Returns:
            (policy_logits, value) tuple
        """
        # Reset hidden state if requested
        if reset_hidden or self.hidden_state is None:
            device = grid.device
            self.hidden_state = self.temporal_lstm.init_hidden(grid.shape[0], device)
        
        # Spatial encoding
        spatial_emb = self.spatial_cnn(grid)  # (B, spatial_dim)
        
        # State encoding
        state_emb = self.state_encoder(state_features)  # (B, state_dim)

        if enemy_context is not None:
            state_emb = state_emb + enemy_context
        
        # Temporal encoding
        temporal_emb, self.hidden_state = self.temporal_lstm(
            state_emb, self.hidden_state
        )  # (B, temporal_dim)
        
        # Fusion and policy/value prediction
        policy_logits, value = self.fusion_network(
            spatial_emb, state_emb, temporal_emb
        )
        
        return policy_logits, value
    
    def get_action(
        self,
        player_pos: Tuple[float, float],
        detections: Dict[str, List],
        game_state_vector: np.ndarray,
        deterministic: bool = False,
        game_state_object=None,
    ) -> Tuple[UnifiedAction, float]:
        """
        Get action from policy (inference mode).
        
        Args:
            player_pos: Player (x, y) position
            detections: Detection dict {class_name: [bbox, ...]}
            game_state_vector: Scalar state features as numpy array (44,)
            deterministic: Whether to use deterministic policy (argmax)
            
        Returns:
            (action, confidence) tuple
        """
        self.eval()
        
        with torch.no_grad():
            # Build spatial grid
            grid = self.grid_builder.build(player_pos, detections, self.schema)
            grid = torch.from_numpy(grid).float().unsqueeze(0)  # (1, 21, 21, C)
            
            # Convert state features
            state_features = torch.from_numpy(game_state_vector).float().unsqueeze(0)  # (1, 30)

            enemy_context = None
            if game_state_object is not None:
                enemy_context = self._encode_enemy_context(game_state_object, grid.device)
            
            # Forward pass
            policy_logits, value = self.forward(
                grid,
                state_features,
                enemy_context=enemy_context,
                reset_hidden=True,
            )
            
            # Get action probabilities
            action_probs = torch.softmax(policy_logits, dim=-1)
            
            if deterministic:
                # Argmax action
                action_idx = torch.argmax(action_probs, dim=-1).item()
                confidence = action_probs[0, action_idx].item()
            else:
                # Sample action
                action_idx = torch.multinomial(action_probs, 1).item()
                confidence = action_probs[0, action_idx].item()
            
            # Convert to UnifiedAction
            action = UnifiedAction(action_idx)
            
        return action, confidence

    def _encode_enemy_context(self, game_state, device: torch.device) -> torch.Tensor:
        """
        Encode per-enemy features using self-attention.

        The resulting context vector is projected to the state embedding
        dimension so it can be fused with scalar state features.
        """
        enemies = getattr(game_state, "enemy_history", None) or getattr(game_state, "enemies", []) or []
        if not enemies:
            return torch.zeros(1, self.state_encoder.fc3.out_features, device=device)

        now = time.time()
        enemy_rows = []
        for enemy in enemies[:6]:
            last_seen = float(getattr(enemy, "last_seen", now) or now)
            recency = max(0.0, min(1.0, (now - last_seen) / 10.0))
            velocity = getattr(enemy, "velocity", (0.0, 0.0)) or (0.0, 0.0)
            velocity_norm = min(1.0, float((velocity[0] ** 2 + velocity[1] ** 2) ** 0.5) / 500.0)
            enemy_rows.append([
                float(getattr(enemy, "health_estimate", 0.0)),
                1.0 if getattr(enemy, "has_super", False) else 0.0,
                float(getattr(enemy, "angle", 0.0)) / np.pi,
                1.0 if getattr(enemy, "is_attacking", False) else 0.0,
                float(getattr(enemy, "distance", 0.0)) / 500.0,
                recency,
                velocity_norm,
                float(getattr(enemy, "hp_estimate_confidence", 0.5)),
            ])

        enemy_tensor = torch.tensor(enemy_rows, dtype=torch.float32, device=device).unsqueeze(0)
        encoded = self.enemy_encoder(enemy_tensor)
        attended, _ = self.enemy_attention(encoded, encoded, encoded)
        pooled = attended.mean(dim=1)
        return self.enemy_context_proj(pooled)
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state."""
        self.hidden_state = None
    
    def extract_state_vector(self, game_state) -> np.ndarray:
        """
        Extract scalar state features from GameState object.
        
        Args:
            game_state: GameState object with 44 features
            
        Returns:
            Numpy array of shape (44,) with normalized features
        """
        # Extract scalar features (normalize to 0-1 range)
        features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)

        def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
            if value is None:
                return lower
            return max(lower, min(upper, float(value)))

        def _bool(value) -> float:
            return 1.0 if bool(value) else 0.0

        def _safe_getattr(name: str, default=0.0):
            return getattr(game_state, name, default)

        def _previous_action_code(action: Optional[str]) -> float:
            action_order = {
                "attack": 0,
                "move_to_enemy": 1,
                "retreat": 2,
                "use_super": 3,
                "collect_cube": 4,
                "idle": 5,
                "take_cover": 6,
                "hold_position": 7,
                "chase": 8,
                "heal_up": 9,
                "ambush": 10,
            }
            if not action:
                return 0.0
            return action_order.get(action.lower(), 0) / 10.0
        
        # Player state (11 features)
        features[0] = _clamp(_safe_getattr("player_health", 1.0))
        features[1] = _clamp(_safe_getattr("player_ammo", 3) / 10.0)
        features[2] = _bool(_safe_getattr("player_super_charged", False))
        features[3] = _bool(_safe_getattr("gadget_ready", False))
        features[4] = _bool(_safe_getattr("hypercharge_ready", False))
        features[5] = _clamp(_safe_getattr("cooldown_attack", 0.0))
        features[6] = _clamp(_safe_getattr("cooldown_super", 0.0))
        features[7] = _bool(_safe_getattr("is_moving", False))
        features[8] = _bool(_safe_getattr("is_in_bush", False))
        velocity = _safe_getattr("velocity", (0.0, 0.0)) or (0.0, 0.0)
        features[9] = _clamp((velocity[0]**2 + velocity[1]**2) ** 0.5 / 500.0)
        features[10] = _clamp(_safe_getattr("current_tactic", 0) / 10.0)
        
        # Spatial features (16 features)
        features[11] = _clamp(_safe_getattr("danger_score", 0.0))
        features[12] = _clamp(_safe_getattr("enemies_in_range", 0) / 5.0)
        features[13] = _clamp(_safe_getattr("escape_routes", 0) / 5.0)
        features[14] = _clamp((_safe_getattr("dist_nearest_enemy", 0.0) or 0.0) / 500.0)
        features[15] = _clamp((_safe_getattr("dist_nearest_cube", 0.0) or 0.0) / 500.0)
        features[16] = _clamp((_safe_getattr("dist_nearest_cover", 0.0) or 0.0) / 500.0)
        features[17] = _clamp((_safe_getattr("dist_nearest_safezone", 0.0) or 0.0) / 500.0)
        features[18] = _bool(_safe_getattr("line_of_sight_free", True))
        safe_direction = _safe_getattr("safe_direction", (0.0, 0.0)) or (0.0, 0.0)
        features[19] = max(-1.0, min(1.0, float(safe_direction[0])))
        features[20] = max(-1.0, min(1.0, float(safe_direction[1])))
        wall_proximity = _safe_getattr("wall_proximity", {}) or {}
        features[21] = _bool(wall_proximity.get("left"))
        features[22] = _bool(wall_proximity.get("right"))
        features[23] = _bool(wall_proximity.get("up"))
        features[24] = _bool(wall_proximity.get("down"))
        features[25] = _bool(_safe_getattr("bush_nearby", False))
        features[26] = _clamp(_safe_getattr("projectile_threat", 0.0))
        features[27] = _clamp(_safe_getattr("objective_pressure", 0.0))

        # Temporal features (10 features)
        features[28] = _clamp(_safe_getattr("time_since_enemy_seen", 0.0) / 10.0)
        features[29] = _previous_action_code(_safe_getattr("previous_action", None))
        features[30] = max(-1.0, min(1.0, float(velocity[0]) / 500.0))
        features[31] = max(-1.0, min(1.0, float(velocity[1]) / 500.0))
        features[32] = max(-1.0, min(1.0, float(_safe_getattr("enemy_last_seen_x", 0.0) or 0.0) / 500.0))
        features[33] = max(-1.0, min(1.0, float(_safe_getattr("enemy_last_seen_y", 0.0) or 0.0) / 500.0))
        features[34] = _clamp(_safe_getattr("enemy_last_hp", 0.0))
        features[35] = _bool(_safe_getattr("enemy_last_super", False))
        features[36] = max(-1.0, min(1.0, float(_safe_getattr("enemy_last_angle", 0.0)) / np.pi))
        features[37] = _bool(_safe_getattr("enemy_last_attack", False))

        # Per-enemy aggregates (6 features)
        nearest_enemy = _safe_getattr("nearest_enemy", None)
        if nearest_enemy is not None:
            features[38] = _clamp(getattr(nearest_enemy, "health_estimate", 0.0))
            features[39] = _bool(getattr(nearest_enemy, "has_super", False))
            features[40] = max(-1.0, min(1.0, float(getattr(nearest_enemy, "angle", 0.0)) / np.pi))
            features[41] = _bool(getattr(nearest_enemy, "is_attacking", False))
            features[42] = max(-1.0, min(1.0, float(_safe_getattr("enemy_last_seen_x", 0.0) or 0.0) / 500.0))
            features[43] = max(-1.0, min(1.0, float(_safe_getattr("enemy_last_seen_y", 0.0) or 0.0) / 500.0))
        else:
            features[42] = 0.0
            features[43] = 0.0

        return features


class NeuralPolicySmall(NeuralPolicy):
    """
    Smaller variant of NeuralPolicy for resource-constrained environments.
    """
    
    def __init__(self, schema: str = "core"):
        """Initialize small neural policy."""
        super().__init__(
            schema=schema,
            spatial_dim=128,
            state_dim=32,
            temporal_dim=64,
            fusion_dim=128
        )
