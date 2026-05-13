"""
CQL (Conservative Q-Learning) Training Script
Simplified implementation for demonstration purposes.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from typing import List, Dict


class CQLAgent(nn.Module):
    """Simplified CQL agent for offline RL."""

    def __init__(self, state_dim: int = 32, action_dim: int = 8, hidden_dim: int = 256):
        super(CQLAgent, self).__init__()

        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q_value = self.q_net(sa)
        return q_value


def load_replay_buffer(data_path: str):
    """Load replay buffer from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    states = torch.tensor([t['state'] for t in data], dtype=torch.float32)
    actions = torch.tensor([t['action'] for t in data], dtype=torch.float32)
    next_states = torch.tensor([t['next_state'] for t in data], dtype=torch.float32)
    rewards = torch.tensor([t['reward'] for t in data], dtype=torch.float32)
    dones = torch.tensor([t['done'] for t in data], dtype=torch.float32)

    return states, actions, next_states, rewards, dones


def train_cql(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    conservative_weight: float = 5.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models/cql"
):
    """
    Train CQL agent on offline replay buffer.

    Args:
        data_path: Path to replay buffer
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        conservative_weight: Conservative penalty weight
        device: Device to train on
        save_dir: Directory to save model
    """
    print("="*60)
    print("CQL Training")
    print("="*60)
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Conservative weight: {conservative_weight}")
    print(f"Device: {device}")
    print("="*60)

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load replay buffer
    print("\nLoading replay buffer...")
    states, actions, next_states, rewards, dones = load_replay_buffer(data_path)
    print(f"Loaded {len(states)} transitions")

    # Split into train/val
    train_size = int(0.8 * len(states))
    val_size = len(states) - train_size

    train_states = states[:train_size]
    train_actions = actions[:train_size]
    train_next_states = next_states[:train_size]
    train_rewards = rewards[:train_size]
    train_dones = dones[:train_size]

    val_states = states[train_size:]
    val_actions = actions[train_size:]
    val_next_states = next_states[train_size:]
    val_rewards = rewards[train_size:]
    val_dones = dones[train_size:]

    # Initialize agent
    print("\nInitializing CQL agent...")
    agent = CQLAgent().to(device)
    target_agent = CQLAgent().to(device)
    target_agent.load_state_dict(agent.state_dict())

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        agent.train()

        # Mini-batch training
        indices = torch.randperm(len(train_states))
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_states), batch_size):
            batch_indices = indices[i:i+batch_size]

            batch_states = train_states[batch_indices].to(device)
            batch_actions = train_actions[batch_indices].to(device)
            batch_next_states = train_next_states[batch_indices].to(device)
            batch_rewards = train_rewards[batch_indices].to(device)
            batch_dones = train_dones[batch_indices].to(device)

            # Current Q-values
            current_q = agent(batch_states, batch_actions)

            # Target Q-values (simplified: use reward + gamma * max Q)
            with torch.no_grad():
                next_actions = batch_actions  # Use same actions for simplicity
                next_q = target_agent(batch_next_states, next_actions)
                target_q = batch_rewards + 0.99 * (1 - batch_dones) * next_q.squeeze()

            # Q-loss
            q_loss = loss_fn(current_q.squeeze(), target_q)

            # Conservative penalty (CQL): penaliza Q-values de ações OOD vs ações do dataset
            # Normaliza ações aleatórias para o mesmo intervalo das ações reais
            random_actions = torch.randn_like(batch_actions)
            random_actions = random_actions / (random_actions.norm(dim=-1, keepdim=True) + 1e-8) \
                * batch_actions.norm(dim=-1, keepdim=True)
            random_q = agent(batch_states, random_actions)
            # Penalidade: E[Q(s, a_ood)] - E[Q(s, a_dataset)] deve ser ≤ 0 (ações dataset devem ter Q maior)
            # Sem clamp para que o gradiente sempre seja computado
            conservative_loss = random_q.mean() - current_q.mean()

            # Total loss
            loss = q_loss + conservative_weight * conservative_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Update target network
        if epoch % 5 == 0:
            target_agent.load_state_dict(agent.state_dict())

        # Validation
        agent.eval()
        with torch.no_grad():
            val_q = agent(val_states.to(device), val_actions.to(device))
            val_loss = loss_fn(val_q.squeeze(), val_rewards.to(device))

        avg_train_loss = total_loss / num_batches

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(agent.state_dict(), f"{save_dir}/best_cql_agent.pt")
            print(f"  Saved best model (val_loss: {val_loss.item():.4f})")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}/best_cql_agent.pt")

    return agent


if __name__ == "__main__":
    # Configuration
    data_path = "dataset/cql/replay_buffer.json"
    save_dir = "models/cql"

    # Train agent
    agent = train_cql(
        data_path=data_path,
        epochs=50,
        batch_size=64,
        learning_rate=1e-4,
        conservative_weight=5.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=save_dir
    )

    print("\nCQL training pipeline complete!")
