"""
behavior_cloning.py

Behavior Cloning (BC) trainer for the Brawl Stars bot.

Implements supervised imitation learning from human gameplay data.

Architecture:
  Input:  game frame (H, W, 3) + auxiliary state vector (health, ammo, position)
  Output: discrete action (move_angle, attack, super, target_x, target_y)

Training data format:
  Each sample: {
    "frame_path": str,        ← PNG from screenshot_recorder
    "action": {
        "move_angle": float,  ← degrees 0-360
        "attack": bool,
        "use_super": bool,
        "target_x": float,    ← relative screen coords 0-1
        "target_y": float,
    },
    "game_state": str,        ← "combat" | "roaming" | "retreat"
    "annotated_by": str,      ← "human" | "auto"
  }
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    dataset_dir: Path = Path("dataset/labeled")
    output_model_path: Path = Path("models/bc_policy.pt")
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 3e-4
    val_split: float = 0.1
    device: str = "cpu"
    image_size: Tuple[int, int] = (224, 224)


@dataclass
class BCTrainResult:
    trained: bool = False
    epochs_completed: int = 0
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    model_path: Optional[Path] = None
    error: Optional[str] = None


class BrawlStarsDataset(Dataset):
    """Dataset for Brawl Stars behavior cloning."""
    
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load all JSON files
        self.samples = []
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    sample = json.load(f)
                    sample['json_path'] = str(json_file)
                    self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        frame_path = Path(sample['frame_path'])
        if not frame_path.is_absolute():
            frame_path = self.data_dir / frame_path
        
        image = cv2.imread(str(frame_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Extract action
        action = sample['action']
        
        # Convert action to tensor
        move_angle = torch.tensor(action.get('move_angle', 0.0) / 360.0, dtype=torch.float32)
        attack = torch.tensor(1.0 if action.get('attack', False) else 0.0, dtype=torch.float32)
        use_super = torch.tensor(1.0 if action.get('use_super', False) else 0.0, dtype=torch.float32)
        target_x = torch.tensor(action.get('target_x', 0.5), dtype=torch.float32)
        target_y = torch.tensor(action.get('target_y', 0.5), dtype=torch.float32)
        
        # Combine into action vector
        action_vector = torch.stack([move_angle, attack, use_super, target_x, target_y])
        
        # Auxiliary state (simplified)
        aux_state = torch.tensor([1.0, 3.0], dtype=torch.float32)  # [health, ammo]
        
        return image, aux_state, action_vector


class BCPolicyNet(nn.Module):
    """
    Neural network for behavior cloning.
    
    Architecture:
    - ResNet18 backbone for image features
    - MLP for auxiliary state
    - Fusion layer
    - Output heads for discrete and continuous actions
    """
    
    def __init__(self, aux_state_dim: int = 2):
        super().__init__()
        
        # Image backbone (ResNet18)
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Auxiliary state MLP
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        
        # Fusion layer
        image_feat_dim = 512  # ResNet18 output
        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Action heads
        self.move_angle_head = nn.Linear(128, 1)  # Continuous
        self.attack_head = nn.Linear(128, 1)  # Binary
        self.super_head = nn.Linear(128, 1)  # Binary
        self.target_head = nn.Linear(128, 2)  # Continuous (x, y)
    
    def forward(self, image, aux_state):
        # Extract image features
        image_features = self.backbone(image)
        image_features = image_features.flatten(1)
        
        # Process auxiliary state
        aux_features = self.aux_mlp(aux_state)
        
        # Fuse features
        fused = torch.cat([image_features, aux_features], dim=1)
        fused = self.fusion(fused)
        
        # Predict actions
        move_angle = torch.sigmoid(self.move_angle_head(fused)).squeeze(-1)
        attack = torch.sigmoid(self.attack_head(fused)).squeeze(-1)
        use_super = torch.sigmoid(self.super_head(fused)).squeeze(-1)
        target = torch.sigmoid(self.target_head(fused))
        
        return {
            'move_angle': move_angle,
            'attack': attack,
            'use_super': use_super,
            'target': target,
        }


class BehaviorCloningTrainer:
    """
    Trains a policy network via supervised imitation of human gameplay.
    """
    
    def __init__(self, config: Optional[BCConfig] = None):
        self.config = config or BCConfig()
        self.device = torch.device(self.config.device)
        self.model = None
        self.optimizer = None
        
    def check_dataset(self) -> dict:
        """Returns info about available labeled data."""
        d = self.config.dataset_dir
        if not d.exists():
            return {"exists": False, "samples": 0, "path": str(d)}
        samples = list(d.glob("*.json"))
        return {"exists": True, "samples": len(samples), "path": str(d)}
    
    def _create_model(self):
        """Create the policy network."""
        self.model = BCPolicyNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def _train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCELoss()
        
        for batch_idx, (images, aux_states, actions) in enumerate(dataloader):
            images = images.to(self.device)
            aux_states = aux_states.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images, aux_states)
            
            # Calculate losses
            loss_move = criterion_mse(predictions['move_angle'], actions[:, 0])
            loss_attack = criterion_bce(predictions['attack'], actions[:, 1])
            loss_super = criterion_bce(predictions['use_super'], actions[:, 2])
            loss_target = criterion_mse(predictions['target'], actions[:, 3:5])
            
            # Combined loss
            loss = loss_move + loss_attack + loss_super + loss_target
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def _validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCELoss()
        
        with torch.no_grad():
            for images, aux_states, actions in dataloader:
                images = images.to(self.device)
                aux_states = aux_states.to(self.device)
                actions = actions.to(self.device)
                
                predictions = self.model(images, aux_states)
                
                loss_move = criterion_mse(predictions['move_angle'], actions[:, 0])
                loss_attack = criterion_bce(predictions['attack'], actions[:, 1])
                loss_super = criterion_bce(predictions['use_super'], actions[:, 2])
                loss_target = criterion_mse(predictions['target'], actions[:, 3:5])
                
                loss = loss_move + loss_attack + loss_super + loss_target
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self) -> BCTrainResult:
        """
        Train the behavior cloning model.
        """
        # Check dataset
        dataset_info = self.check_dataset()
        if not dataset_info['exists'] or dataset_info['samples'] == 0:
            return BCTrainResult(
                trained=False,
                error=f"No dataset found at {dataset_info['path']}. "
                      "Run gameplay_recorder.py to collect data."
            )
        
        logger.info(f"Training on {dataset_info['samples']} samples")
        
        # Create dataset and dataloader
        try:
            full_dataset = BrawlStarsDataset(self.config.dataset_dir)
            
            # Split train/val
            train_size = int((1 - self.config.val_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
        except Exception as e:
            return BCTrainResult(
                trained=False,
                error=f"Failed to create dataset: {e}"
            )
        
        # Create model
        self._create_model()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader)
            
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.config.output_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.config.output_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(self.config.output_model_path, weights_only=True))
        
        return BCTrainResult(
            trained=True,
            epochs_completed=epoch + 1,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            model_path=self.config.output_model_path,
        )
    
    def load_policy(self, path: Optional[Path] = None):
        """Load a trained BC policy for inference."""
        model_path = path or self.config.output_model_path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self._create_model()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        logger.info(f"Loaded BC policy from {model_path}")
    
    def predict(self, image: np.ndarray, aux_state: np.ndarray) -> Dict:
        """
        Predict action for given image and state.
        
        Args:
            image: RGB image (H, W, 3)
            aux_state: Auxiliary state [health, ammo]
            
        Returns:
            Dictionary with predicted actions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_policy() first.")
        
        # Preprocess image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        aux_tensor = torch.tensor(aux_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor, aux_tensor)
        
        # Convert to numpy
        return {
            'move_angle': predictions['move_angle'].cpu().item() * 360.0,
            'attack': predictions['attack'].cpu().item() > 0.5,
            'use_super': predictions['use_super'].cpu().item() > 0.5,
            'target_x': predictions['target'][0].cpu().item(),
            'target_y': predictions['target'][1].cpu().item(),
        }


def main():
    """Test behavior cloning trainer."""
    logging.basicConfig(level=logging.INFO)
    
    config = BCConfig()
    trainer = BehaviorCloningTrainer(config)
    
    # Check dataset
    dataset_info = trainer.check_dataset()
    print(f"Dataset info: {dataset_info}")
    
    if dataset_info['samples'] > 0:
        result = trainer.train()
        print(f"Training result: {result}")
    else:
        print("No dataset available. Run gameplay_recorder.py first.")


if __name__ == "__main__":
    main()
