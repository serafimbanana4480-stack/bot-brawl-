"""
model_registry.py

Model registry for versioning, A/B testing, and rollback.

Manages multiple model versions, tracks performance, and provides
rollback capabilities for the continuous learning system.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    model_path: str
    version: str
    created_at: str
    model_type: str  # "yolo", "bc", "cql", "neural_policy"
    training_data: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    is_deployed: bool = False
    is_active: bool = False
    parent_model_id: Optional[str] = None  # For tracking lineage


@dataclass
class ModelPerformance:
    """Performance tracking for a model."""
    model_id: str
    timestamp: str
    matches_played: int
    win_rate: float
    average_score: float
    kda_ratio: float
    custom_metrics: Dict[str, float]


class ModelRegistry:
    """
    Registry for managing model versions and performance.
    
    Provides versioning, A/B testing, and rollback capabilities.
    """
    
    def __init__(self, registry_dir: Path):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_file = self.registry_dir / "models.json"
        self.performance_file = self.registry_dir / "performance.json"
        
        self.models: Dict[str, ModelMetadata] = {}
        self.performance_history: List[ModelPerformance] = []
        
        self._load_models()
        self._load_performance()
    
    def _load_models(self):
        """Load model metadata from file."""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self.models[model_id] = ModelMetadata(**model_data)
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save model metadata to file."""
        try:
            data = {model_id: asdict(model) for model_id, model in self.models.items()}
            with open(self.models_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_performance(self):
        """Load performance history from file."""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = [ModelPerformance(**item) for item in data]
                logger.info(f"Loaded {len(self.performance_history)} performance records")
            except Exception as e:
                logger.error(f"Error loading performance: {e}")
    
    def _save_performance(self):
        """Save performance history to file."""
        try:
            data = [asdict(perf) for perf in self.performance_history]
            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance: {e}")
    
    def _generate_model_id(self, model_type: str) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"{model_type}_{timestamp}_{random_suffix}"
    
    def register_model(
        self,
        model_path: Path,
        model_type: str,
        version: str,
        training_data: str,
        training_metrics: Dict[str, float],
        validation_metrics: Dict[str, float],
        parent_model_id: Optional[str] = None
    ) -> str:
        """
        Register a new model.
        
        Args:
            model_path: Path to model file
            model_type: Type of model ("yolo", "bc", "cql", "neural_policy")
            version: Version string
            training_data: Description of training data
            training_metrics: Metrics from training
            validation_metrics: Metrics from validation
            parent_model_id: Optional parent model ID for lineage tracking
            
        Returns:
            Model ID
        """
        model_id = self._generate_model_id(model_type)
        
        # Copy model to registry storage
        storage_dir = self.registry_dir / "models" / model_type
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_id}.pt"
        storage_path = storage_dir / model_filename
        
        try:
            shutil.copy(str(model_path), str(storage_path))
        except Exception as e:
            logger.error(f"Error copying model: {e}")
            return ""
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_path=str(storage_path),
            version=version,
            created_at=datetime.now().isoformat(),
            model_type=model_type,
            training_data=training_data,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            parent_model_id=parent_model_id
        )
        
        self.models[model_id] = metadata
        self._save_models()
        
        logger.info(f"Registered model {model_id} of type {model_type}")
        
        return model_id
    
    def deploy_model(self, model_id: str) -> bool:
        """
        Deploy a model (mark as active).
        
        Args:
            model_id: Model ID to deploy
            
        Returns:
            True if deployment successful
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        # Undeploy current active model of same type
        model_type = self.models[model_id].model_type
        for mid, model in self.models.items():
            if model.model_type == model_type and model.is_active:
                model.is_active = False
                model.is_deployed = False
        
        # Deploy new model
        self.models[model_id].is_active = True
        self.models[model_id].is_deployed = True
        
        self._save_models()
        
        logger.info(f"Deployed model {model_id}")
        
        return True
    
    def get_active_model(self, model_type: str) -> Optional[ModelMetadata]:
        """
        Get currently active model of a specific type.
        
        Args:
            model_type: Type of model
            
        Returns:
            ModelMetadata or None if no active model
        """
        for model in self.models.values():
            if model.model_type == model_type and model.is_active:
                return model
        return None
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        """
        List all models, optionally filtered by type.
        
        Args:
            model_type: Optional model type filter
            
        Returns:
            List of ModelMetadata
        """
        if model_type:
            return [m for m in self.models.values() if m.model_type == model_type]
        return list(self.models.values())
    
    def rollback_model(self, model_type: str, steps: int = 1) -> Optional[str]:
        """
        Rollback to previous model version.
        
        Args:
            model_type: Type of model to rollback
            steps: Number of versions to rollback
            
        Returns:
            Model ID of rolled back model or None
        """
        # Get all models of type, sorted by creation date
        models = [m for m in self.models.values() if m.model_type == model_type]
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        # Find current active model
        current_idx = None
        for i, model in enumerate(models):
            if model.is_active:
                current_idx = i
                break
        
        if current_idx is None:
            logger.error(f"No active model found for type {model_type}")
            return None
        
        # Rollback
        target_idx = current_idx + steps
        if target_idx >= len(models):
            logger.error(f"Cannot rollback {steps} steps - not enough versions")
            return None
        
        target_model = models[target_idx]
        self.deploy_model(target_model.model_id)
        
        logger.info(f"Rolled back to model {target_model.model_id}")
        
        return target_model.model_id
    
    def record_performance(
        self,
        model_id: str,
        matches_played: int,
        win_rate: float,
        average_score: float,
        kda_ratio: float,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record performance data for a model.
        
        Args:
            model_id: Model ID
            matches_played: Number of matches played
            win_rate: Win rate (0.0 to 1.0)
            average_score: Average score
            kda_ratio: Kill/Death/Assist ratio
            custom_metrics: Optional additional metrics
        """
        performance = ModelPerformance(
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            matches_played=matches_played,
            win_rate=win_rate,
            average_score=average_score,
            kda_ratio=kda_ratio,
            custom_metrics=custom_metrics or {}
        )
        
        self.performance_history.append(performance)
        self._save_performance()
        
        logger.info(f"Recorded performance for model {model_id}")
    
    def get_model_performance(self, model_id: str, n_recent: int = 10) -> List[ModelPerformance]:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model ID
            n_recent: Number of recent records to return
            
        Returns:
            List of ModelPerformance
        """
        model_perf = [p for p in self.performance_history if p.model_id == model_id]
        return model_perf[-n_recent:]
    
    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, float]:
        """
        Compare performance of two models.
        
        Args:
            model_id1: First model ID
            model_id2: Second model ID
            
        Returns:
            Dictionary with comparison metrics
        """
        perf1 = self.get_model_performance(model_id1)
        perf2 = self.get_model_performance(model_id2)
        
        if not perf1 or not perf2:
            return {"error": "Insufficient performance data"}
        
        # Calculate averages
        avg1 = {
            "win_rate": sum(p.win_rate for p in perf1) / len(perf1),
            "average_score": sum(p.average_score for p in perf1) / len(perf1),
            "kda_ratio": sum(p.kda_ratio for p in perf1) / len(perf1)
        }
        
        avg2 = {
            "win_rate": sum(p.win_rate for p in perf2) / len(perf2),
            "average_score": sum(p.average_score for p in perf2) / len(perf2),
            "kda_ratio": sum(p.kda_ratio for p in perf2) / len(perf2)
        }
        
        comparison = {
            "model1": model_id1,
            "model2": model_id2,
            "model1_metrics": avg1,
            "model2_metrics": avg2,
            "difference": {
                "win_rate": avg1["win_rate"] - avg2["win_rate"],
                "average_score": avg1["average_score"] - avg2["average_score"],
                "kda_ratio": avg1["kda_ratio"] - avg2["kda_ratio"]
            }
        }
        
        return comparison
    
    def cleanup_old_models(self, model_type: str, keep_n: int = 5):
        """
        Clean up old models, keeping only the most recent N.
        
        Args:
            model_type: Type of model to cleanup
            keep_n: Number of models to keep
        """
        models = [m for m in self.models.values() if m.model_type == model_type]
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        # Keep active models
        to_delete = [m for m in models[keep_n:] if not m.is_active]
        
        for model in to_delete:
            try:
                # Delete model file
                model_path = Path(model.model_path)
                if model_path.exists():
                    model_path.unlink()
                
                # Remove from registry
                del self.models[model.model_id]
                
                logger.info(f"Deleted old model {model.model_id}")
            except Exception as e:
                logger.error(f"Error deleting model {model.model_id}: {e}")
        
        self._save_models()
    
    def get_registry_summary(self) -> Dict:
        """Get summary of registry state."""
        return {
            "total_models": len(self.models),
            "models_by_type": {
                model_type: len([m for m in self.models.values() if m.model_type == model_type])
                for model_type in ["yolo", "bc", "cql", "neural_policy"]
            },
            "active_models": len([m for m in self.models.values() if m.is_active]),
            "performance_records": len(self.performance_history)
        }
