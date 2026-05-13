"""Imitation Learning - Learn from expert demonstrations"""

import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class Demonstration:
    observation: np.ndarray
    action: int
    reward: float
    done: bool
    timestamp: float


class ImitationLearning:
    def __init__(self, state_dim: int, action_dim: int,
                 model: Optional[Any] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model
        self.demonstrations: List[Demonstration] = []
        self.pretrain_threshold = 1000
        self.learning_rate = 0.001
        
    def add_demonstration(self, observation: np.ndarray, action: int,
                        reward: float = 0.0, done: bool = False):
        demo = Demonstration(
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            timestamp=time.time(),
        )
        self.demonstrations.append(demo)
        
        if len(self.demonstrations) >= self.pretrain_threshold:
            self.pretrain()
    
    def pretrain(self):
        if len(self.demonstrations) < self.pretrain_threshold:
            return
        
        logger.info(f"Pretraining on {len(self.demonstrations)} demonstrations...")

        X = np.array([d.observation for d in self.demonstrations])
        y = np.array([d.action for d in self.demonstrations])

        if self.model is None:
            self.model = self._create_model()

        self._train_model(X, y)

        logger.info("Pretraining complete!")
    
    def _create_model(self):
        try:
            import torch
            import torch.nn as nn
            
            class ExpertPolicy(nn.Module):
                def __init__(self, state_dim, action_dim):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(state_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, action_dim),
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            self.model = ExpertPolicy(self.state_dim, self.action_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.CrossEntropyLoss()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            return self.model
        except ImportError:
            return None
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        if self.model is None or not hasattr(self.model, 'train'):
            return
        
        try:
            import torch
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            for epoch in range(epochs):
                self.model.train()
                
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    def predict(self, state: np.ndarray) -> int:
        if self.model is None:
            return np.random.randint(0, self.action_dim)
        
        try:
            import torch
            
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                outputs = self.model(state_tensor)
                action = torch.argmax(outputs, dim=1).item()
            
            return action
        except Exception:
            return np.random.randint(0, self.action_dim)
    
    def save(self, path: str):
        if self.model is None:
            return
        
        try:
            import torch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'demonstrations_count': len(self.demonstrations),
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def load(self, path: str):
        try:
            import torch
            checkpoint = torch.load(path, map_location=self.device if hasattr(self, 'device') else 'cpu', weights_only=True)
            if self.model is None:
                self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Load error: {e}")
