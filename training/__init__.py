"""
Training and auto-learning module for Brawl Stars bot.
"""

from .retrain import (
    PerformanceMonitor,
    RetrainOrchestrator,
    ContinuousLearner,
    PerformanceMetrics,
    RetrainTrigger,
    create_continuous_learner
)

try:
    from .model_registry import ModelRegistry, ModelMetadata
except ImportError:
    pass

try:
    from .real_reward_system import RealRewardCalculator, GameMetrics
except ImportError:
    pass

try:
    from .continuous_training_pipeline import ContinuousTrainingPipeline, PipelineMetrics
except ImportError:
    pass

try:
    from .unified_training_system import UnifiedTrainingSystem
except ImportError:
    pass

__all__ = [
    "PerformanceMonitor",
    "RetrainOrchestrator",
    "ContinuousLearner",
    "PerformanceMetrics",
    "RetrainTrigger",
    "create_continuous_learner",
]
