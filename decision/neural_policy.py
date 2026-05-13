"""
neural_policy.py

Neural policy wrapper for integrating BC and CQL policies with the decision system.

Provides a unified interface for using neural networks (BC, CQL) alongside
the existing rule-based system.

Features:
- Ensemble of multiple policies
- Confidence calibration
- Fallback to rule-based system
- Policy selection based on context
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import numpy as np

import torch

logger = logging.getLogger(__name__)


@dataclass
class PolicyOutput:
    """Output from a policy."""
    move_angle: float  # degrees 0-360
    attack: bool
    use_super: bool
    target_x: float  # normalized 0-1
    target_y: float  # normalized 0-1
    confidence: float  # 0.0 to 1.0
    source: str  # "bc", "cql", "rule"


class NeuralPolicy:
    """
    Wrapper for neural policies (BC, CQL).
    
    Provides unified interface and ensemble capabilities.
    """
    
    def __init__(
        self,
        bc_model_path: Optional[Path] = None,
        cql_model_path: Optional[Path] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.6,
    ):
        self.bc_model = None
        self.cql_model = None
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        
        # Load models if paths provided
        if bc_model_path and bc_model_path.exists():
            self._load_bc_policy(bc_model_path)
        
        if cql_model_path and cql_model_path.exists():
            self._load_cql_policy(cql_model_path)
    
    def _load_bc_policy(self, path: Path):
        """Load Behavior Cloning policy."""
        try:
            from rl_stubs.behavior_cloning import BehaviorCloningTrainer, BCConfig
            config = BCConfig(device=str(self.device))
            self.bc_trainer = BehaviorCloningTrainer(config)
            self.bc_trainer.load_policy(path)
            self.bc_model = self.bc_trainer.model
            logger.info(f"Loaded BC policy from {path}")
        except Exception as e:
            logger.error(f"Failed to load BC policy: {e}")
    
    def _load_cql_policy(self, path: Path):
        """Load CQL policy."""
        try:
            from rl_stubs.cql_trainer import CQLTrainer, CQLConfig
            config = CQLConfig(device=str(self.device))
            self.cql_trainer = CQLTrainer(config)
            self.cql_trainer.load_policy(path)
            self.cql_model = self.cql_trainer.actor
            logger.info(f"Loaded CQL policy from {path}")
        except Exception as e:
            logger.error(f"Failed to load CQL policy: {e}")
    
    def predict(
        self,
        image: np.ndarray,
        aux_state: np.ndarray,
        state_vector: Optional[np.ndarray] = None,
        use_ensemble: bool = True,
    ) -> PolicyOutput:
        """
        Predict action using available policies.
        
        Args:
            image: RGB image (H, W, 3)
            aux_state: Auxiliary state [health, ammo]
            state_vector: Optional state vector for CQL
            use_ensemble: Whether to use ensemble of policies
            
        Returns:
            PolicyOutput with action and confidence
        """
        outputs = []
        
        # BC prediction
        if self.bc_model is not None:
            try:
                bc_action = self.bc_trainer.predict(image, aux_state)
                bc_confidence = 0.8  # Default confidence for BC
                outputs.append({
                    'policy': 'bc',
                    'action': bc_action,
                    'confidence': bc_confidence
                })
            except Exception as e:
                logger.debug(f"BC prediction failed: {e}")
        
        # CQL prediction
        if self.cql_model is not None and state_vector is not None:
            try:
                cql_action = self.cql_trainer.predict(state_vector)
                cql_confidence = 0.85  # Default confidence for CQL
                outputs.append({
                    'policy': 'cql',
                    'action': cql_action,
                    'confidence': cql_confidence
                })
            except Exception as e:
                logger.debug(f"CQL prediction failed: {e}")
        
        # Select best output
        if not outputs:
            # Fallback to default
            return PolicyOutput(
                move_angle=0.0,
                attack=False,
                use_super=False,
                target_x=0.5,
                target_y=0.5,
                confidence=0.0,
                source="fallback"
            )
        
        if use_ensemble and len(outputs) > 1:
            # Ensemble: weighted average based on confidence
            total_confidence = sum(o['confidence'] for o in outputs)
            
            # Weighted average of actions
            weighted_action = {}
            for key in ['move_angle', 'target_x', 'target_y']:
                weighted_action[key] = sum(
                    o['action'][key] * o['confidence'] for o in outputs
                ) / total_confidence
            
            # Majority vote for binary actions
            attack_votes = sum(1 for o in outputs if o['action']['attack'])
            super_votes = sum(1 for o in outputs if o['action']['use_super'])
            
            return PolicyOutput(
                move_angle=weighted_action['move_angle'] * 360.0,
                attack=attack_votes > len(outputs) / 2,
                use_super=super_votes > len(outputs) / 2,
                target_x=weighted_action['target_x'],
                target_y=weighted_action['target_y'],
                confidence=max(o['confidence'] for o in outputs),
                source="ensemble"
            )
        else:
            # Use single best policy
            best = max(outputs, key=lambda x: x['confidence'])
            action = best['action']
            return PolicyOutput(
                move_angle=action['move_angle'] * 360.0,
                attack=action['attack'],
                use_super=action['use_super'],
                target_x=action['target_x'],
                target_y=action['target_y'],
                confidence=best['confidence'],
                source=best['policy']
            )
    
    def is_available(self) -> bool:
        """Check if any neural policy is available."""
        return self.bc_model is not None or self.cql_model is not None
    
    def get_policy_info(self) -> Dict:
        """Get information about loaded policies."""
        return {
            'bc_loaded': self.bc_model is not None,
            'cql_loaded': self.cql_model is not None,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold,
        }


class HybridDecisionSystem:
    """
    Hybrid decision system combining rule-based and neural policies.
    
    Uses neural policies when confidence is high, falls back to rules
    when uncertain or in critical situations.
    """
    
    def __init__(
        self,
        neural_policy: NeuralPolicy,
        rule_engine,
        confidence_threshold: float = 0.7,
    ):
        self.neural_policy = neural_policy
        self.rule_engine = rule_engine
        self.confidence_threshold = confidence_threshold
        
        # Statistics
        self.neural_decisions = 0
        self.rule_decisions = 0
    
    def decide(
        self,
        game_state,
        image: np.ndarray,
        aux_state: np.ndarray,
        state_vector: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Make a decision using hybrid approach.
        
        Args:
            game_state: Current game state
            image: Current frame
            aux_state: Auxiliary state [health, ammo]
            state_vector: Optional state vector for CQL
            
        Returns:
            Decision dictionary with action and metadata
        """
        # Try neural policy first
        if self.neural_policy.is_available():
            try:
                neural_output = self.neural_policy.predict(
                    image, aux_state, state_vector
                )
                
                # Use neural if confidence is high
                if neural_output.confidence >= self.confidence_threshold:
                    self.neural_decisions += 1
                    return {
                        'action': neural_output,
                        'source': 'neural',
                        'confidence': neural_output.confidence,
                        'policy_used': neural_output.source,
                    }
            except Exception as e:
                logger.debug(f"Neural policy failed: {e}")
        
        # Fallback to rule-based
        self.rule_decisions += 1
        rule_decision = self.rule_engine.evaluate_engagement(game_state)
        
        # Convert rule decision to action format
        if rule_decision:
            best = rule_decision[0]
            return {
                'action': PolicyOutput(
                    move_angle=0.0,  # Would calculate from target
                    attack='attack' in best.tactic.value.lower(),
                    use_super=False,
                    target_x=best.target_position[0] if best.target_position else 0.5,
                    target_y=best.target_position[1] if best.target_position else 0.5,
                    confidence=best.priority,
                    source='rule',
                ),
                'source': 'rule',
                'confidence': best.priority,
                'policy_used': best.tactic.value,
            }
        
        # Ultimate fallback
        return {
            'action': PolicyOutput(
                move_angle=0.0,
                attack=False,
                use_super=False,
                target_x=0.5,
                target_y=0.5,
                confidence=0.0,
                source='fallback',
            ),
            'source': 'fallback',
            'confidence': 0.0,
            'policy_used': 'fallback',
        }
    
    def get_statistics(self) -> Dict:
        """Get decision statistics."""
        total = self.neural_decisions + self.rule_decisions
        return {
            'neural_decisions': self.neural_decisions,
            'rule_decisions': self.rule_decisions,
            'neural_ratio': self.neural_decisions / total if total > 0 else 0.0,
            'total_decisions': total,
        }


def main():
    """Test neural policy wrapper."""
    logging.basicConfig(level=logging.INFO)
    
    # Create neural policy (no models loaded for test)
    policy = NeuralPolicy()
    
    print(f"Policy available: {policy.is_available()}")
    print(f"Policy info: {policy.get_policy_info()}")


if __name__ == "__main__":
    main()
