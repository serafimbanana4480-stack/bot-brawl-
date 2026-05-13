"""
Behavior Cloning Training Script
Trains a policy network to imitate human gameplay actions.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List
import torch.nn.functional as F
from torchvision import models, transforms


class BrawlStarsBCDataset(Dataset):
    """Dataset for behavior cloning from gameplay data."""

    def __init__(self, data_path: str):
        """
        Load BC dataset from JSON file.

        Args:
            data_path: Path to bc_dataset.json
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Flatten episodes into individual frame-action pairs
        self.samples = []
        for episode in self.data:
            for frame in episode['frames']:
                self.samples.append({
                    'state': frame['state'],
                    'action': frame['action']
                })

        print(f"Loaded {len(self.samples)} frame-action pairs from {len(self.data)} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert state to tensor (8 features)
        state = torch.tensor([
            sample['state']['player_x'] / 1000.0,
            sample['state']['player_y'] / 1000.0,
            sample['state']['health'] / 100.0,
            sample['state']['ammo'] / 10.0,
            sample['state']['enemy_distance'] / 500.0,
            sample['state']['enemy_angle'] / 360.0,
            float(sample['state']['obstacle_nearby']),
            float(sample['state']['powerup_available'])
        ], dtype=torch.float32)

        # Convert action to tensor (5 discrete actions)
        move_dir_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'none': 4}
        action = torch.tensor([
            move_dir_map[sample['action']['move_direction']],
            float(sample['action']['attack']),
            float(sample['action']['use_ability']),
            sample['action']['target_x'] / 1000.0,
            sample['action']['target_y'] / 1000.0
        ], dtype=torch.float32)

        return state, action


class BCPolicyNet(nn.Module):
    """Policy network for behavior cloning."""

    def __init__(self, state_dim: int = 8, action_dim: int = 5, hidden_dim: int = 256):
        super(BCPolicyNet, self).__init__()

        # Feature extractor (simple MLP)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Action heads
        self.move_head = nn.Linear(hidden_dim // 2, 5)  # 5 movement directions
        self.attack_head = nn.Linear(hidden_dim // 2, 2)  # binary attack
        self.ability_head = nn.Linear(hidden_dim // 2, 2)  # binary ability
        self.target_head = nn.Linear(hidden_dim // 2, 2)  # target x, y

    def forward(self, state):
        features = self.feature_net(state)

        move_logits = self.move_head(features)
        attack_logits = self.attack_head(features)
        ability_logits = self.ability_head(features)
        target_coords = torch.sigmoid(self.target_head(features))

        return move_logits, attack_logits, ability_logits, target_coords


def train_bc_model(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models/bc"
):
    """
    Train behavior cloning model.

    Args:
        data_path: Path to BC dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save model
    """
    print("="*60)
    print("Behavior Cloning Training")
    print("="*60)
    print(f"Data: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("="*60)

    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    dataset = BrawlStarsBCDataset(data_path)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Initialize model
    print("\nInitializing model...")
    model = BCPolicyNet().to(device)

    # Loss functions
    move_criterion = nn.CrossEntropyLoss()
    attack_criterion = nn.CrossEntropyLoss()
    ability_criterion = nn.CrossEntropyLoss()
    target_criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_move_acc = 0.0
        train_attack_acc = 0.0
        train_ability_acc = 0.0

        for batch_idx, (states, actions) in enumerate(train_loader):
            states, actions = states.to(device), actions.to(device)

            # Forward pass
            move_logits, attack_logits, ability_logits, target_coords = model(states)

            # Compute losses
            move_loss = move_criterion(move_logits, actions[:, 0].long())
            attack_loss = attack_criterion(attack_logits, actions[:, 1].long())
            ability_loss = ability_criterion(ability_logits, actions[:, 2].long())
            target_loss = target_criterion(target_coords, actions[:, 3:5])

            # Combined loss
            loss = move_loss + attack_loss + ability_loss + target_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            train_move_acc += (move_logits.argmax(dim=1) == actions[:, 0].long()).float().mean().item()
            train_attack_acc += (attack_logits.argmax(dim=1) == actions[:, 1].long()).float().mean().item()
            train_ability_acc += (ability_logits.argmax(dim=1) == actions[:, 2].long()).float().mean().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_move_acc = 0.0
        val_attack_acc = 0.0
        val_ability_acc = 0.0

        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)

                move_logits, attack_logits, ability_logits, target_coords = model(states)

                move_loss = move_criterion(move_logits, actions[:, 0].long())
                attack_loss = attack_criterion(attack_logits, actions[:, 1].long())
                ability_loss = ability_criterion(ability_logits, actions[:, 2].long())
                target_loss = target_criterion(target_coords, actions[:, 3:5])

                loss = move_loss + attack_loss + ability_loss + target_loss

                val_loss += loss.item()
                val_move_acc += (move_logits.argmax(dim=1) == actions[:, 0].long()).float().mean().item()
                val_attack_acc += (attack_logits.argmax(dim=1) == actions[:, 1].long()).float().mean().item()
                val_ability_acc += (ability_logits.argmax(dim=1) == actions[:, 2].long()).float().mean().item()

        # Calculate averages
        train_loss /= len(train_loader)
        train_move_acc /= len(train_loader)
        train_attack_acc /= len(train_loader)
        train_ability_acc /= len(train_loader)

        val_loss /= len(val_loader)
        val_move_acc /= len(val_loader)
        val_attack_acc /= len(val_loader)
        val_ability_acc /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Move Acc: {train_move_acc:.4f} | Attack Acc: {train_attack_acc:.4f} | Ability Acc: {train_ability_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Move Acc: {val_move_acc:.4f} | Attack Acc: {val_attack_acc:.4f} | Ability Acc: {val_ability_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_bc_policy.pt")
            print(f"  Saved best model (val_loss: {val_loss:.4f})")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}/best_bc_policy.pt")

    return model


if __name__ == "__main__":
    # Configuration
    data_path = "dataset/bc/bc_dataset.json"
    save_dir = "models/bc"

    # Train model
    model = train_bc_model(
        data_path=data_path,
        epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=save_dir
    )

    print("\nBehavior cloning training pipeline complete!")
