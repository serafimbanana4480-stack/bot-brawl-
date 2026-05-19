"""
Experience Buffer for RL Training

Circular buffer for storing and sampling experience tuples
for PPO training. Supports prioritized sampling and episode segmentation.

Usage:
    from core.experience_buffer import ExperienceBuffer
    
    buffer = ExperienceBuffer(capacity=10000)
    buffer.add(state, action, reward, done, value, log_prob)
    batch = buffer.sample(batch_size=64)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import random


class ExperienceBuffer:
    """
    Circular buffer for storing experience tuples.
    
    Stores:
    - States (grid + scalar features)
    - Actions
    - Rewards
    - Dones (episode termination flags)
    - Values (value function predictions)
    - Log probabilities (action log probs from policy)
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.size = 0
        self.pointer = 0
        
        # Storage arrays
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.old_log_probs = []
        
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """
        Add an experience to the buffer.
        
        Args:
            state: State representation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            value: Value function prediction
            log_prob: Action log probability
        """
        if self.size < self.capacity:
            # Add new experience
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.old_log_probs.append(log_prob)
            self.size += 1
        else:
            # Overwrite old experience (circular buffer)
            self.states[self.pointer] = state
            self.actions[self.pointer] = action
            self.rewards[self.pointer] = reward
            self.dones[self.pointer] = done
            self.values[self.pointer] = value
            self.old_log_probs[self.pointer] = log_prob
        
        self.pointer = (self.pointer + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary with batched data
        """
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            "states": np.array([self.states[i] for i in indices]),
            "actions": np.array([self.actions[i] for i in indices]),
            "rewards": np.array([self.rewards[i] for i in indices]),
            "dones": np.array([self.dones[i] for i in indices]),
            "values": np.array([self.values[i] for i in indices]),
            "old_log_probs": np.array([self.old_log_probs[i] for i in indices]),
        }
    
    def sample_episode(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample a complete episode (sequence of experiences).
        
        Returns:
            Dictionary with episode data or None if no complete episodes
        """
        # Find episode boundaries (where done=True)
        episode_starts = [0]
        episode_ends = []
        
        for i in range(self.size):
            if self.dones[i]:
                episode_ends.append(i)
                if i + 1 < self.size:
                    episode_starts.append(i + 1)
        
        if not episode_ends:
            return None  # No complete episodes
        
        # Select random episode
        episode_idx = random.randint(0, len(episode_ends) - 1)
        start = episode_starts[episode_idx]
        end = episode_ends[episode_idx]
        
        return {
            "states": np.array(self.states[start:end+1]),
            "actions": np.array(self.actions[start:end+1]),
            "rewards": np.array(self.rewards[start:end+1]),
            "dones": np.array(self.dones[start:end+1]),
            "values": np.array(self.values[start:end+1]),
            "old_log_probs": np.array(self.old_log_probs[start:end+1]),
        }
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.old_log_probs = []
        self.size = 0
        self.pointer = 0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    Experience buffer with prioritized sampling.
    
    Samples experiences with probability proportional to their TD-error,
    focusing learning on high-error experiences.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized experience buffer.
        
        Args:
            capacity: Maximum number of experiences
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
        priority: Optional[float] = None
    ):
        """
        Add experience with priority.
        
        Args:
            state: State representation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            value: Value function prediction
            log_prob: Action log probability
            priority: Optional priority (defaults to max_priority)
        """
        if priority is None:
            priority = self.max_priority
        
        if self.size < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.old_log_probs.append(log_prob)
            self.size += 1
        else:
            self.states[self.pointer] = state
            self.actions[self.pointer] = action
            self.rewards[self.pointer] = reward
            self.dones[self.pointer] = done
            self.values[self.pointer] = value
            self.old_log_probs[self.pointer] = log_prob
        
        self.priorities[self.pointer] = priority
        self.pointer = (self.pointer + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample batch with priority-based sampling.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            
        Returns:
            (batch_data, indices, weights) tuple
        """
        if self.size == 0:
            return {}, np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        return {
            "states": np.array([self.states[i] for i in indices]),
            "actions": np.array([self.actions[i] for i in indices]),
            "rewards": np.array([self.rewards[i] for i in indices]),
            "dones": np.array([self.dones[i] for i in indices]),
            "values": np.array([self.values[i] for i in indices]),
            "old_log_probs": np.array([self.old_log_probs[i] for i in indices]),
        }, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices to update
            priorities: New priority values
        """
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
