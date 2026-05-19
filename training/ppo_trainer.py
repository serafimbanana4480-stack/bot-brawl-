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
        max_grad_norm: float = 0.5
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
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Batch of old action log probabilities
            advantages: Batch of advantages
            returns: Batch of returns
            
        Returns:
            Dictionary of training losses and statistics
        """
        # Forward pass
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
        ppo_epochs: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train policy using PPO.
        
        Args:
            experience_buffer: Experience buffer with (state, action, reward, done, value)
            num_updates: Number of PPO updates
            batch_size: Batch size for training
            ppo_epochs: Number of PPO epochs per update
            
        Returns:
            Dictionary of training statistics per update
        """
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
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO epochs
            for _ in range(ppo_epochs):
                stats = self.train_step(
                    states, actions, old_log_probs, advantages, returns
                )
                
                for key in all_stats:
                    all_stats[key].append(stats[key])
            
            self.episodes += 1
            
            if update % 10 == 0:
                print(f"Update {update}/{num_updates}, "
                      f"Policy Loss: {stats['policy_loss']:.4f}, "
                      f"Value Loss: {stats['value_loss']:.4f}, "
                      f"Entropy: {stats['entropy']:.4f}")
        
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
