"""
movement_predictor.py

Advanced movement prediction system for Brawl Stars bot.

Implements neural network-based prediction using LSTM to forecast
enemy movement patterns, improving aim assist and dodging capabilities.

Features:
- LSTM-based trajectory prediction
- Kalman filter smoothing
- Multi-method prediction ensemble
- Confidence-based prediction selection
- Integration with existing tracker
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MovementPrediction:
    """Predicted movement for an object."""
    predicted_position: Tuple[float, float]  # (x, y)
    velocity: Tuple[float, float]  # (vx, vy)
    confidence: float  # 0.0 to 1.0
    prediction_horizon: float  # seconds into future
    method: str  # "kalman", "lstm", "linear", "ensemble"


class KalmanFilterPredictor:
    """
    Kalman filter for smooth movement prediction.
    
    Uses constant velocity model for short-term prediction.
    """
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 0.5):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.covariance = np.eye(4) * 1.0
        
        # Transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
    
    def predict(self, dt: float = 0.1) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Predict position and velocity.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            (predicted_position, velocity)
        """
        # Update transition matrix for dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        # Predict
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + np.eye(4) * self.process_noise
        
        position = (self.state[0], self.state[1])
        velocity = (self.state[2], self.state[3])
        
        return position, velocity
    
    def update(self, measurement: Tuple[float, float]):
        """Update with new measurement."""
        z = np.array(measurement)
        
        # Innovation
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + np.eye(2) * self.measurement_noise
        
        # Kalman gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Update
        self.state = self.state + K @ y
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance


class LSTMPredictor(nn.Module):
    """
    LSTM-based movement predictor.
    
    Learns movement patterns from historical position sequences.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, 2)  # Predict (dx, dy)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            
        Returns:
            Predicted delta position for each timestep (batch, seq_len, 2)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        delta = self.fc(lstm_out)   # (batch, seq_len, 2)
        return delta


class MovementPredictor:
    """
    Advanced movement prediction system.
    
    Combines multiple prediction methods for robust movement forecasting.
    """
    
    def __init__(
        self,
        history_length: int = 10,
        use_lstm: bool = True,
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.history_length = history_length
        self.use_lstm = use_lstm
        self.device = torch.device(device)
        
        # Position history for each tracked object
        self.position_history: Dict[int, deque] = {}
        
        # Predictors
        self.kalman_predictors: Dict[int, KalmanFilterPredictor] = {}
        self.lstm_predictor = None
        
        # Initialize LSTM if enabled
        if self.use_lstm:
            self._init_lstm(model_path)
    
    def _init_lstm(self, model_path: Optional[Path]):
        """Initialize LSTM predictor."""
        try:
            self.lstm_predictor = LSTMPredictor().to(self.device)
            
            if model_path and model_path.exists():
                self.lstm_predictor.load_state_dict(torch.load(model_path, weights_only=True))
                logger.info(f"Loaded LSTM model from {model_path}")
            else:
                logger.info("LSTM predictor initialized (untrained)")
        except Exception as e:
            logger.error(f"Failed to initialize LSTM: {e}")
            self.lstm_predictor = None
            self.use_lstm = False
    
    def add_object(self, track_id: int, position: Tuple[float, float]):
        """
        Add or update object position.
        
        Args:
            track_id: Unique identifier for the object
            position: (x, y) position
        """
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=self.history_length)
            self.kalman_predictors[track_id] = KalmanFilterPredictor()
        
        self.position_history[track_id].append(position)
        
        # Update Kalman filter
        self.kalman_predictors[track_id].update(position)
    
    def get_velocity(self, track_id: int) -> Tuple[float, float]:
        """
        Get current velocity estimate for an object.
        
        Args:
            track_id: Object identifier
            
        Returns:
            (vx, vy) velocity
        """
        if track_id not in self.position_history:
            return (0.0, 0.0)
        
        history = self.position_history[track_id]
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Calculate velocity from last two positions
        pos1 = history[-2]
        pos2 = history[-1]
        dt = 1.0  # Assume 1 second between frames
        
        vx = (pos2[0] - pos1[0]) / dt
        vy = (pos2[1] - pos1[1]) / dt
        
        return (vx, vy)
    
    def predict_kalman(
        self,
        track_id: int,
        dt: float = 0.1,
        horizon: float = 0.5,
    ) -> MovementPrediction:
        """
        Predict using Kalman filter.
        
        Args:
            track_id: Object identifier
            dt: Time step
            horizon: Prediction horizon in seconds
            
        Returns:
            MovementPrediction
        """
        if track_id not in self.kalman_predictors:
            return self._create_fallback_prediction()
        
        kf = self.kalman_predictors[track_id]
        
        # Predict for horizon
        current_pos = self.position_history[track_id][-1]
        
        # Step through horizon
        for _ in range(int(horizon / dt)):
            pos, vel = kf.predict(dt)
        
        return MovementPrediction(
            predicted_position=pos,
            velocity=vel,
            confidence=0.75,  # Kalman confidence
            prediction_horizon=horizon,
            method="kalman",
        )
    
    def predict_lstm(
        self,
        track_id: int,
        horizon: float = 0.5,
    ) -> MovementPrediction:
        """
        Predict using LSTM network.
        
        Args:
            track_id: Object identifier
            horizon: Prediction horizon in seconds
            
        Returns:
            MovementPrediction
        """
        if not self.use_lstm or self.lstm_predictor is None:
            return self._create_fallback_prediction()
        
        if track_id not in self.position_history:
            return self._create_fallback_prediction()
        
        history = self.position_history[track_id]
        if len(history) < self.history_length:
            # Not enough history, use linear extrapolation
            return self._predict_linear(track_id, horizon)
        
        # Prepare input sequence
        positions = list(history)
        # Calculate deltas
        deltas = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            deltas.append([dx, dy])
        
        # Pad if needed
        while len(deltas) < self.history_length:
            deltas.insert(0, [0.0, 0.0])
        
        # Convert to tensor
        x = torch.FloatTensor(deltas).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            delta = self.lstm_predictor(x)
        
        # Extrapolate to horizon
        current_pos = positions[-1]
        steps = int(horizon / 0.1)  # Assume 0.1s per step
        pred_pos = current_pos
        
        for _ in range(steps):
            pred_pos = (pred_pos[0] + delta[0].item(), pred_pos[1] + delta[1].item())
        
        # Estimate velocity
        velocity = (delta[0].item() / 0.1, delta[1].item() / 0.1)
        
        return MovementPrediction(
            predicted_position=pred_pos,
            velocity=velocity,
            confidence=0.85,  # LSTM confidence
            prediction_horizon=horizon,
            method="lstm",
        )
    
    def _predict_linear(
        self,
        track_id: int,
        horizon: float = 0.5,
    ) -> MovementPrediction:
        """Predict using linear extrapolation."""
        if track_id not in self.position_history:
            return self._create_fallback_prediction()
        
        history = self.position_history[track_id]
        if len(history) < 2:
            return self._create_fallback_prediction()
        
        # Calculate velocity
        vel = self.get_velocity(track_id)
        current_pos = history[-1]
        
        # Extrapolate
        pred_x = current_pos[0] + vel[0] * horizon
        pred_y = current_pos[1] + vel[1] * horizon
        
        return MovementPrediction(
            predicted_position=(pred_x, pred_y),
            velocity=vel,
            confidence=0.5,  # Linear confidence
            prediction_horizon=horizon,
            method="linear",
        )
    
    def _create_fallback_prediction(self) -> MovementPrediction:
        """Create fallback prediction when data insufficient."""
        return MovementPrediction(
            predicted_position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            confidence=0.0,
            prediction_horizon=0.5,
            method="fallback",
        )
    
    def predict(
        self,
        track_id: int,
        method: str = "ensemble",
        horizon: float = 0.5,
    ) -> MovementPrediction:
        """
        Predict movement using specified method.
        
        Args:
            track_id: Object identifier
            method: Prediction method ("kalman", "lstm", "linear", "ensemble")
            horizon: Prediction horizon in seconds
            
        Returns:
            MovementPrediction
        """
        if method == "ensemble":
            return self._predict_ensemble(track_id, horizon)
        elif method == "kalman":
            return self.predict_kalman(track_id, horizon=horizon)
        elif method == "lstm":
            return self.predict_lstm(track_id, horizon)
        elif method == "linear":
            return self._predict_linear(track_id, horizon)
        else:
            return self._create_fallback_prediction()
    
    def _predict_ensemble(
        self,
        track_id: int,
        horizon: float = 0.5,
    ) -> MovementPrediction:
        """
        Ensemble prediction combining multiple methods.
        
        Weighted average based on confidence.
        """
        predictions = []
        
        # Get predictions from all available methods
        predictions.append(self.predict_kalman(track_id, horizon=horizon))
        
        if self.use_lstm:
            predictions.append(self.predict_lstm(track_id, horizon=horizon))
        
        predictions.append(self._predict_linear(track_id, horizon))
        
        # Weighted average
        total_confidence = sum(p.confidence for p in predictions)
        
        if total_confidence == 0:
            return self._create_fallback_prediction()
        
        # Weighted position
        weighted_x = sum(p.predicted_position[0] * p.confidence for p in predictions) / total_confidence
        weighted_y = sum(p.predicted_position[1] * p.confidence for p in predictions) / total_confidence
        
        # Weighted velocity
        weighted_vx = sum(p.velocity[0] * p.confidence for p in predictions) / total_confidence
        weighted_vy = sum(p.velocity[1] * p.confidence for p in predictions) / total_confidence
        
        # Max confidence
        max_confidence = max(p.confidence for p in predictions)
        
        return MovementPrediction(
            predicted_position=(weighted_x, weighted_y),
            velocity=(weighted_vx, weighted_vy),
            confidence=max_confidence,
            prediction_horizon=horizon,
            method="ensemble",
        )
    
    def reset_object(self, track_id: int):
        """Reset tracking for an object (e.g., after death or match end)."""
        if track_id in self.position_history:
            self.position_history[track_id].clear()
        
        if track_id in self.kalman_predictors:
            self.kalman_predictors[track_id] = KalmanFilterPredictor()
    
    def reset_all(self):
        """Reset all tracking."""
        self.position_history.clear()
        self.kalman_predictors.clear()
    
    def train_lstm(self, sequences: List[np.ndarray], epochs: int = 50):
        """
        Train LSTM predictor on movement sequences.
        
        Args:
            sequences: List of position sequences
            epochs: Number of training epochs
        """
        if not self.use_lstm or self.lstm_predictor is None:
            logger.warning("LSTM not available for training")
            return
        
        # Prepare training data
        # sequences should be list of (seq_len, 2) arrays
        
        # Convert to deltas
        train_data = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            deltas = []
            for i in range(1, len(seq)):
                dx = seq[i][0] - seq[i-1][0]
                dy = seq[i][1] - seq[i-1][1]
                deltas.append([dx, dy])
            
            if len(deltas) >= self.history_length:
                # Create sequences of length history_length
                for i in range(len(deltas) - self.history_length):
                    train_data.append(deltas[i:i+self.history_length])
        
        if not train_data:
            logger.warning("No training data available")
            return
        
        # Convert to tensor
        train_data = np.array(train_data, dtype=np.float32)
        train_tensor = torch.FloatTensor(train_data).to(self.device)
        
        # Train
        optimizer = torch.optim.Adam(self.lstm_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            self.lstm_predictor.train()
            
            # Forward
            outputs = self.lstm_predictor(train_tensor)
            
            # Predict next delta (shifted by 1)
            targets = train_tensor[:, 1:, :]
            outputs = outputs[:, :-1, :]
            
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info("LSTM training completed")
        
        # Save model
        model_path = Path(__file__).parent.parent / "models" / "movement_predictor_lstm.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.lstm_predictor.state_dict(), model_path)
        logger.info(f"LSTM model saved to {model_path}")


def main():
    """Test movement predictor."""
    logging.basicConfig(level=logging.INFO)
    
    predictor = MovementPredictor(use_lstm=False)
    
    # Simulate tracking
    for i in range(10):
        x = 100 + i * 10
        y = 200 + i * 5
        predictor.add_object(1, (x, y))
    
    # Predict
    prediction = predictor.predict(1, method="ensemble", horizon=0.5)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
