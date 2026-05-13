"""
cql_trainer.py

Conservative Q-Learning (CQL) Offline RL trainer for Brawl Stars bot.

Implements offline reinforcement learning from replay data without interacting
with the environment during training.

Architecture:
  - Actor network (policy) - warm-started from BC
  - Critic network (Q-function) - learns from offline data
  - Conservative Q-update to prevent overestimation
  - Target networks for stability

Training data format:
  Replay buffer with (state, action, reward, next_state, done) tuples

Reward shaping:
  kill = +1.0, death = -1.0, damage_dealt = +0.1, damage_taken = -0.05
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CQLConfig:
    replay_buffer_dir: Path = Path("dataset/replay_buffer")
    output_model_path: Path = Path("models/cql_policy.pt")
    bc_model_path: Optional[Path] = None  # Path to BC model for warm-start
    
    # Training hyperparameters
    batch_size: int = 256
    epochs: int = 100
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update coefficient
    
    # CQL-specific
    cql_alpha: float = 1.0  # Conservative Q-learning weight
    cql_temp: float = 1.0  # Temperature for CQL
    
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2
    
    # Validation
    val_split: float = 0.1
    device: str = "cpu"


@dataclass
class CQLTrainResult:
    trained: bool = False
    epochs_completed: int = 0
    final_q_loss: Optional[float] = None
    final_actor_loss: Optional[float] = None
    final_cql_loss: Optional[float] = None
    model_path: Optional[Path] = None
    error: Optional[str] = None


class ReplayBuffer:
    """
    Replay buffer for offline RL training.
    
    Stores transitions in format (state, action, reward, next_state, done).
    Supports prioritized experience replay.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0
    ):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Prioritized sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices
        )
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path: Path):
        """Save buffer to disk."""
        data = {
            'buffer': list(self.buffer),
            'priorities': list(self.priorities),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    def load(self, path: Path):
        """Load buffer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.buffer = deque(data['buffer'], maxlen=self.capacity)
        self.priorities = deque(data['priorities'], maxlen=self.capacity)


class ActorNetwork(nn.Module):
    """
    Actor network (policy) for CQL.
    
    Takes state and outputs action distribution.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, state):
        return self.net(state)


class CriticNetwork(nn.Module):
    """
    Critic network (Q-function) for CQL.
    
    Takes state-action pair and outputs Q-value.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class CQLTrainer:
    """
    Conservative Q-Learning trainer for offline RL.
    
    Implements CQL algorithm to learn from offline replay data
    without overestimating Q-values.
    """
    
    def __init__(self, config: Optional[CQLConfig] = None):
        self.config = config or CQLConfig()
        self.device = torch.device(self.config.device)
        
        # Networks
        self.actor = None
        self.critic = None
        self.critic_target = None
        
        # Optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training state
        self.total_steps = 0
    
    def _create_networks(self, state_dim: int, action_dim: int):
        """Create actor and critic networks."""
        self.actor = ActorNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        
        # Copy critic to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate_critic)
        
        # Warm-start from BC if available
        if self.config.bc_model_path and self.config.bc_model_path.exists():
            logger.info(f"Warm-starting actor from BC model: {self.config.bc_model_path}")
            try:
                from rl_stubs.behavior_cloning import BCPolicyNet
                bc_model = BCPolicyNet().to(self.device)
                bc_model.load_state_dict(torch.load(self.config.bc_model_path, weights_only=True))
                
                # Copy backbone weights (simplified - would need proper architecture matching)
                logger.info("BC model loaded (note: architecture matching may need adjustment)")
            except Exception as e:
                logger.warning(f"Failed to load BC model: {e}")
    
    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def _compute_cql_loss(self, states, actions):
        """
        Compute Conservative Q-Learning loss.
        
        This penalizes overestimation of Q-values for actions not taken
        in the dataset.
        """
        # Sample random actions for CQL penalty
        batch_size = states.shape[0]
        random_actions = torch.randn(batch_size, actions.shape[-1]).to(self.device)
        random_actions = torch.clamp(random_actions, -1, 1)
        
        # Current actions (from dataset)
        current_actions = actions
        
        # Actor actions
        actor_actions = self.actor(states)
        
        # Compute Q-values for all action sets
        q_current = self.critic(states, current_actions)
        q_random = self.critic(states, random_actions)
        q_actor = self.critic(states, actor_actions)
        
        # CQL loss: logsumexp of Q-values minus Q-value of dataset action
        q_all = torch.cat([q_random, q_actor], dim=1)
        logsumexp_q = torch.logsumexp(q_all / self.config.cql_temp, dim=1, keepdim=True) * self.config.cql_temp
        
        cql_loss = logsumexp_q - q_current
        
        return cql_loss.mean()
    
    def train_step(self, batch):
        """Perform one training step."""
        states, actions, rewards, next_states, dones, indices = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            # Target Q-values
            next_actions = self.actor(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.config.gamma * (1 - dones) * target_q
        
        # Current Q-values
        current_q = self.critic(states, actions)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # CQL loss
        cql_loss = self._compute_cql_loss(states, actions)
        
        # Total critic loss
        total_critic_loss = critic_loss + self.config.cql_alpha * cql_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_q = self.critic(states, actor_actions)
        actor_loss = -actor_q.mean()  # Maximize Q-value
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target
        self._soft_update_target()
        
        # Update priorities based on TD error
        td_error = torch.abs(current_q - target_q).detach().cpu().numpy()
        priorities = 1.0 / (td_error.flatten() + 1e-6)
        self.replay_buffer.update_priorities(indices, priorities.tolist())
        
        return {
            'critic_loss': critic_loss.item(),
            'cql_loss': cql_loss.item(),
            'actor_loss': actor_loss.item(),
            'total_critic_loss': total_critic_loss.item(),
        }
    
    def load_replay_buffer(self, path: Path):
        """Load replay buffer from disk."""
        if path.exists():
            self.replay_buffer.load(path)
            logger.info(f"Loaded replay buffer with {len(self.replay_buffer)} transitions")
        else:
            logger.warning(f"Replay buffer not found at {path}")
    
    def train(self) -> CQLTrainResult:
        """
        Train CQL model.
        
        Returns:
            CQLTrainResult with training statistics
        """
        # Load replay buffer
        buffer_path = self.config.replay_buffer_dir / "replay_buffer.json"
        self.load_replay_buffer(buffer_path)
        
        if len(self.replay_buffer) < 1000:
            return CQLTrainResult(
                trained=False,
                error=f"Insufficient replay data: {len(self.replay_buffer)} < 1000"
            )
        
        # Determine state and action dimensions from first sample
        sample_state, sample_action, _, _, _ = self.replay_buffer.buffer[0]
        state_dim = sample_state.shape[0] if isinstance(sample_state, np.ndarray) else len(sample_state)
        action_dim = sample_action.shape[0] if isinstance(sample_action, np.ndarray) else len(sample_action)
        
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        
        # Create networks
        self._create_networks(state_dim, action_dim)
        
        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_losses = []
            
            # Sample batches
            num_batches = len(self.replay_buffer) // self.config.batch_size
            for batch_idx in range(num_batches):
                batch = self.replay_buffer.sample(self.config.batch_size)
                losses = self.train_step(batch)
                epoch_losses.append(losses)
            
            # Average losses for epoch
            avg_losses = {
                key: np.mean([loss[key] for loss in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            logger.info(
                f"Epoch {epoch}: "
                f"Critic Loss = {avg_losses['critic_loss']:.4f}, "
                f"CQL Loss = {avg_losses['cql_loss']:.4f}, "
                f"Actor Loss = {avg_losses['actor_loss']:.4f}"
            )
            
            # Early stopping
            total_loss = avg_losses['total_critic_loss']
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
                # Save best model
                self.config.output_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'config': self.config.__dict__,
                }, self.config.output_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return CQLTrainResult(
            trained=True,
            epochs_completed=epoch + 1,
            final_q_loss=avg_losses['critic_loss'],
            final_actor_loss=avg_losses['actor_loss'],
            final_cql_loss=avg_losses['cql_loss'],
            model_path=self.config.output_model_path,
        )
    
    def load_policy(self, path: Optional[Path] = None):
        """Load trained CQL policy for inference."""
        model_path = path or self.config.output_model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        checkpoint = torch.load(model_path, weights_only=True)
        
        # Determine dimensions from config
        config_dict = checkpoint.get('config', {})
        # Would need to extract state_dim and action_dim from config or data
        
        # For now, assume dimensions (would need proper handling)
        self._create_networks(state_dim=10, action_dim=5)  # Placeholder
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor.eval()
        self.critic.eval()
        
        logger.info(f"Loaded CQL policy from {model_path}")
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict action for given state.
        
        Args:
            state: State vector
            
        Returns:
            Action vector
        """
        if self.actor is None:
            raise RuntimeError("Model not loaded. Call load_policy() first.")
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
        
        return action.cpu().numpy()[0]


def main():
    """Test CQL trainer."""
    logging.basicConfig(level=logging.INFO)
    
    config = CQLConfig()
    trainer = CQLTrainer(config)
    
    # Create dummy replay buffer for testing
    for i in range(1000):
        state = np.random.randn(10)
        action = np.random.randn(5)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        trainer.replay_buffer.add(state, action, reward, next_state, done)
    
    # Train
    result = trainer.train()
    print(f"Training result: {result}")


if __name__ == "__main__":
    main()
