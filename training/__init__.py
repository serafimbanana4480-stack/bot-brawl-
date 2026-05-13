"""
Training and auto-learning module for Brawl Stars bot.
"""

from .auto_labeler import (
    AutoLabeler,
    HeuristicLabeler,
    TemplateMatcher,
    AutoLabel,
    auto_label_dataset
)

from .retrain import (
    PerformanceMonitor,
    RetrainOrchestrator,
    ContinuousLearner,
    PerformanceMetrics,
    RetrainTrigger,
    create_continuous_learner
)

__all__ = [
    # Auto-labeling
    "AutoLabeler",
    "HeuristicLabeler",
    "TemplateMatcher",
    "AutoLabel",
    "auto_label_dataset",
    
    # Retraining
    "PerformanceMonitor",
    "RetrainOrchestrator",
    "ContinuousLearner",
    "PerformanceMetrics",
    "RetrainTrigger",
    "create_continuous_learner",
]
