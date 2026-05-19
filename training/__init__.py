"""
Training and auto-learning module for Brawl Stars bot.

Exports all training modules: YOLO training, RL/PPO, hyperparameter tuning,
semi-supervised learning, dataset validation, augmentation, continuous training,
retraining orchestration, model registry, and reward calculation.

Usage:
    from training import HyperparameterTuner, SemiSupervisedTrainer, get_schema
    from training import AdvancedAugmenter, validate_dataset
    from training import ContinuousTrainingPipeline, PPOTrainer
"""

# ---------------------------------------------------------------------------
# Class schema (core dependency, always available)
# ---------------------------------------------------------------------------
from .class_schema import (
    CORE_CLASSES, EXTENDED_CLASSES, FULL_CLASSES,
    LEGACY_CORE_LABEL_MAP, CLASS_MAP, ROBOFLOW_CLASS_NAMES, KEEP_CLASSES,
    get_schema, get_full_schema, schema_name, remap_label_id,
    get_canonical_name, get_class_id,
    validate_schema_completeness, get_schema_coverage,
    migrate_old_detection,
)

# ---------------------------------------------------------------------------
# Schema dataset builder (always available)
# ---------------------------------------------------------------------------
from .schema_dataset_builder import build_schema_dataset

# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------
try:
    from .validate_dataset import (
        validate_structure, count_classes, check_image_quality,
        check_bbox_sizes, generate_report,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
try:
    from .advanced_augmentation import AdvancedAugmenter, AugmentationPolicy
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
try:
    from .hyperparameter_tuner import (
        HyperparameterTuner, TuningCandidate, TuningTrialResult,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Semi-supervised / pseudo-labeling
# ---------------------------------------------------------------------------
try:
    from .semi_supervised_trainer import SemiSupervisedTrainer, PseudoLabelConfig
except ImportError:
    pass

# ---------------------------------------------------------------------------
# PPO Trainer (RL)
# ---------------------------------------------------------------------------
try:
    from .ppo_trainer import PPOTrainer
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Enhanced training pipeline (YOLO)
# ---------------------------------------------------------------------------
try:
    from .enhanced_training_pipeline import (
        DataCurator, EnhancedAutoLabeler,
        create_data_yaml, train_yolo, validate_model,
        STANDARD_CLASSES, CORE_STANDARD_CLASSES, BRAWLSTARS_HSV,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Retraining & performance monitoring
# ---------------------------------------------------------------------------
try:
    from .retrain import (
        PerformanceMonitor, RetrainOrchestrator, ContinuousLearner,
        PerformanceMetrics, RetrainTrigger, create_continuous_learner,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Model registry (training version)
# ---------------------------------------------------------------------------
try:
    from .model_registry import ModelRegistry, ModelMetadata, ModelPerformance
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Real reward system
# ---------------------------------------------------------------------------
try:
    from .real_reward_system import RealRewardCalculator, GameMetrics
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Continuous training pipeline
# ---------------------------------------------------------------------------
try:
    from .continuous_training_pipeline import ContinuousTrainingPipeline, PipelineMetrics
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Unified training system
# ---------------------------------------------------------------------------
try:
    from .unified_training_system import (
        UnifiedTrainingSystem, TrainingMonitor, TrainingMetrics, TrainingReward,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# SOTA training pipeline (orchestration)
# ---------------------------------------------------------------------------
try:
    from .sota_training_pipeline import run_training
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Complete training workflow
# ---------------------------------------------------------------------------
try:
    from .complete_training_workflow import (
        step_capture, step_download_roboflow, step_merge_datasets,
        step_validate, step_train, step_final_validation, check_emulator,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
try:
    from .synthetic_data_generator import (
        PhysicsEngine, SyntheticGameState, SyntheticDataGenerator,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Training validator
# ---------------------------------------------------------------------------
try:
    from .training_validator import ModelValidator, ValidationMetrics
except ImportError:
    pass

# ============================================================================
# __all__
# ============================================================================
__all__ = [
    # Class schema
    "CORE_CLASSES", "EXTENDED_CLASSES", "FULL_CLASSES",
    "LEGACY_CORE_LABEL_MAP", "CLASS_MAP", "ROBOFLOW_CLASS_NAMES", "KEEP_CLASSES",
    "get_schema", "get_full_schema", "schema_name", "remap_label_id",
    "get_canonical_name", "get_class_id",
    "validate_schema_completeness", "get_schema_coverage",
    "migrate_old_detection",

    # Dataset
    "build_schema_dataset",
    "validate_structure", "count_classes", "check_image_quality",
    "check_bbox_sizes", "generate_report",

    # Augmentation
    "AdvancedAugmenter", "AugmentationPolicy",

    # Hyperparameter tuning
    "HyperparameterTuner", "TuningCandidate", "TuningTrialResult",

    # Semi-supervised
    "SemiSupervisedTrainer", "PseudoLabelConfig",

    # PPO / RL
    "PPOTrainer",

    # YOLO pipeline
    "DataCurator", "EnhancedAutoLabeler",
    "create_data_yaml", "train_yolo", "validate_model",
    "STANDARD_CLASSES", "CORE_STANDARD_CLASSES", "BRAWLSTARS_HSV",

    # Retraining
    "PerformanceMonitor", "RetrainOrchestrator", "ContinuousLearner",
    "PerformanceMetrics", "RetrainTrigger", "create_continuous_learner",

    # Model registry
    "ModelRegistry", "ModelMetadata", "ModelPerformance",

    # Rewards
    "RealRewardCalculator", "GameMetrics",

    # Continuous training
    "ContinuousTrainingPipeline", "PipelineMetrics",

    # Unified system
    "UnifiedTrainingSystem", "TrainingMonitor", "TrainingMetrics", "TrainingReward",

    # SOTA
    "run_training",

    # Complete workflow
    "step_capture", "step_download_roboflow", "step_merge_datasets",
    "step_validate", "step_train", "step_final_validation", "check_emulator",

    # Synthetic data
    "PhysicsEngine", "SyntheticGameState", "SyntheticDataGenerator",

    # Validation
    "ModelValidator", "ValidationMetrics",
]