"""
PPO Trainer for Neural Policy Network

Proximal Policy Optimization (PPO) trainer for the neural policy network.
Implements the PPO-Clip algorithm for stable policy gradient learning.

Usage:
    from training.ppo_trainer import PPOTrainer
    from neural.neural_policy import NeuralPolicy
    
    policy = NeuralPolicy()
    trainer = PPOTrainer(policy)
    trainer.train(num_epochs=100)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# MLflow tracking (optional)
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None  # type: ignore[assignment, misc]


class PPOTrainer:
    """
    PPO trainer for neural policy network.
    
    Implements PPO-Clip algorithm with:
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus for exploration
    - GAE (Generalized Advantage Estimation)
    """
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        mlflow_enabled: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "soberana_omega_ppo",
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy: Neural policy network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            mlflow_enabled: Whether to log metrics to MLflow
            mlflow_tracking_uri: MLflow tracking URI (directory or server)
            mlflow_experiment_name: MLflow experiment name
        """
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Training statistics
        self.total_steps = 0
        self.episodes = 0
        
        # MLflow configuration
        self.mlflow_enabled = mlflow_enabled and HAS_MLFLOW
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self._mlflow_run_active = False
        
        if self.mlflow_enabled and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            logger.info(f"[PPO] MLflow tracking enabled: {mlflow_tracking_uri}")
        elif self.mlflow_enabled:
            mlflow.set_experiment(mlflow_experiment_name)
            logger.info("[PPO] MLflow tracking enabled (default URI)")
        
    def _start_mlflow_run(self, run_name: Optional[str] = None):
        """Start an MLflow run if tracking is enabled."""
        if not self.mlflow_enabled or self._mlflow_run_active:
            return
        try:
            mlflow.start_run(run_name=run_name)
            self._mlflow_run_active = True
            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": self.optimizer.defaults["lr"],
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
            })
            logger.info("[PPO] MLflow run started")
        except Exception as e:
            logger.warning(f"[PPO] Failed to start MLflow run: {e}")
            self._mlflow_run_active = False
    
    def _end_mlflow_run(self, checkpoint_path: Optional[str] = None):
        """End the MLflow run and optionally log an artifact."""
        if not self.mlflow_enabled or not self._mlflow_run_active:
            return
        try:
            if checkpoint_path and Path(checkpoint_path).exists():
                mlflow.log_artifact(checkpoint_path)
                logger.info(f"[PPO] Checkpoint logged to MLflow: {checkpoint_path}")
            mlflow.end_run()
            self._mlflow_run_active = False
            logger.info("[PPO] MLflow run ended")
        except Exception as e:
            logger.warning(f"[PPO] Failed to end MLflow run: {e}")
            self._mlflow_run_active = False
    
    def _log_mlflow_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to MLflow if tracking is enabled."""
        if not self.mlflow_enabled or not self._mlflow_run_active:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"[PPO] Failed to log MLflow metrics: {e}")
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards per timestep
            values: List of value predictions per timestep
            dones: List of done flags per timestep
            
        Returns:
            (advantages, returns) tuple
        """
        advantages = []
        gae = 0.0
        
        # Compute GAE in reverse
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        
        return advantages, returns
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        grids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            states: Batch of states (B, state_dim)
            actions: Batch of actions
            old_log_probs: Batch of old action log probabilities
            advantages: Batch of advantages
            returns: Batch of returns
            grids: Optional batch of spatial grids (B, H, W, C)
            
        Returns:
            Dictionary of training losses and statistics
        """
        # Forward pass
        if grids is not None:
            policy_logits, values = self.policy(grid=grids, state_features=states)
        else:
            policy_logits, values = self.policy(states)
        
        # Compute new log probabilities
        action_dist = Categorical(logits=policy_logits)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        # Policy loss (negative because we maximize)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Statistics
        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }
        
        self.total_steps += states.shape[0]
        
        return stats
    
    def train_epoch(
        self,
        batch_size: int = 64,
        num_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch (placeholder for actual training loop).
        
        Args:
            batch_size: Batch size for training
            num_epochs: Number of PPO epochs per update
            
        Returns:
            Dictionary of average training statistics
        """
        # This is a placeholder - actual training requires experience buffer
        # See train() method for full training loop
        return {"placeholder": 0.0}
    
    def train(
        self,
        experience_buffer,
        num_updates: int = 100,
        batch_size: int = 64,
        ppo_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train policy using PPO.
        
        Args:
            experience_buffer: Experience buffer with (state, action, reward, done, value)
            num_updates: Number of PPO updates
            batch_size: Batch size for training
            ppo_epochs: Number of PPO epochs per update
            checkpoint_path: Path to save final checkpoint (logged as MLflow artifact)
            run_name: Optional name for the MLflow run
            
        Returns:
            Dictionary of training statistics per update
        """
        # Start MLflow run
        self._start_mlflow_run(run_name=run_name)
        
        all_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
        }
        
        for update in range(num_updates):
            # Sample batch from experience buffer
            batch = experience_buffer.sample(batch_size)
            
            # Compute advantages and returns using GAE
            advantages, returns = self.compute_gae(
                batch["rewards"],
                batch["values"],
                batch["dones"]
            )
            
            # Convert to tensors
            states = torch.FloatTensor(batch["states"])
            actions = torch.LongTensor(batch["actions"])
            old_log_probs = torch.FloatTensor(batch["old_log_probs"])
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # Optional spatial grids
            grids = None
            if "grids" in batch:
                grids = torch.FloatTensor(batch["grids"])
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO epochs
            for _ in range(ppo_epochs):
                stats = self.train_step(
                    states, actions, old_log_probs, advantages, returns, grids=grids
                )
                
                for key in all_stats:
                    all_stats[key].append(stats[key])
            
            self.episodes += 1
            
            # Log metrics to MLflow every update
            self._log_mlflow_metrics({
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
                "total_loss": stats["total_loss"],
            }, step=update)
            
            if update % 10 == 0:
                print(f"Update {update}/{num_updates}, "
                      f"Policy Loss: {stats['policy_loss']:.4f}, "
                      f"Value Loss: {stats['value_loss']:.4f}, "
                      f"Entropy: {stats['entropy']:.4f}")
        
        # Save checkpoint and end MLflow run
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)
        self._end_mlflow_run(checkpoint_path=checkpoint_path)
        
        return all_stats
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.episodes = checkpoint["episodes"]
