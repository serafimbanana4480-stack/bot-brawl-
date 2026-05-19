# Dependencies Manifest

This document tracks all optional dependencies used throughout the Brawl Stars Bot project with their purposes and fallback behaviors.

## Critical Dependencies (Required)

These dependencies are required for core functionality. The bot will not function without them.

- **opencv-python (cv2)**: Image processing, screenshot capture, auto-labeling
- **numpy**: Numerical operations, array handling
- **ultralytics**: YOLO model inference
- **torch**: PyTorch for ML models (TensorRT optimization)

## Optional Dependencies

These dependencies provide enhanced functionality but have fallbacks if not installed.

### wrapper.py

| Import | Purpose | Fallback Behavior |
|--------|---------|-------------------|
| `from realtime_logs import get_log_manager` | Structured logging | Uses standard logging, logs warning |
| `from core.world_model import WorldModel` | Spatial/temporal memory | Sets to None, world_model features disabled |
| `from core.pressure_map import PressureMap` | Tactical pressure analysis | Sets to None, pressure analysis disabled |
| `from core.cover_system import CoverSystem` | Cover detection | Sets to None, cover system disabled |
| `from core.occupancy_grid import OccupancyGrid` | Spatial occupancy tracking | Sets to None, occupancy grid disabled |
| `from core.behavioral_profile import BehavioralProfile` | Session-level personality | Sets to None, behavioral profiling disabled |
| `from core.central_coordinator import CentralCoordinator` | Central decision coordination | Sets to None, uses individual decision systems |
| `from decision.utility_ai import UtilityAI` | Utility-based action selection | Sets to None, falls back to rule-based |
| `from decision.sticky_target import StickyTarget` | Target commitment system | Sets to None, may target-switch frequently |
| `from decision.intent_system import IntentSystem` | Persistent strategic goals | Sets to None, no persistent goals |
| `from decision.enemy_intention import EnemyIntentionPredictor` | Enemy behavior prediction | Sets to None, no intention prediction |
| `from decision.meta_awareness import MetaAwareness` | Meta-game strategy adaptation | Sets to None, no meta adaptation |
| `from pylaai_real.rl_engine import CombatQLearning` | Q-Learning reinforcement learning | Sets to None, RL disabled |
| `from pylaai_real.elo_tracker import ELOTracker` | Per-brawler ELO tracking | Sets to None, ELO tracking disabled |
| `from meta_learning import MetaLearningSystem` | Hyperparameter adaptation | Sets to None, auto-tuning disabled |
| `from world_model_integration import WorldModelIntegrator` | Bridge vision to world model | Sets to None, integration disabled |
| `from auto_tuner import AutoTuner` | Automatic parameter tuning | Sets to None, auto-tuning disabled |
| `from behavioral_profile_system import BehavioralProfileSystem` | Extended behavioral profiling | Sets to None, extended profiling disabled |
| `from auto_calibrator import AutoCalibrator` | Automatic calibration | Sets to None, manual calibration required |
| `from pylaai_real.ocr_state_detector import OCRStateDetector` | OCR-based state detection | Sets to None, pixel heuristics only |
| `from pylaai_real.debug_visualizer import DebugVisualizer` | Debug visualization | Sets to None, no debug overlays |
| `from core.async_pipeline import AsyncPipeline` | Async inference pipeline | Sets to None, synchronous inference |
| `from core.class_registry import UnifiedAction` | Unified action space | Sets to None, uses legacy actions |

### vision_engine.py

| Import | Purpose | Fallback Behavior |
|--------|---------|-------------------|
| `from vision.tracker import ByteTracker` | Object tracking | Uses SORT tracker or no tracking |
| `from vision.state import StateExtractor` | State extraction | Uses basic detection only |

### vision/tracker.py

| Import | Purpose | Fallback Behavior |
|--------|---------|-------------------|
| `from yolox.tracker.byte_tracker import BYTETracker` | ByteTrack implementation | Uses fallback tracker |

### vision/state.py

| Import | Purpose | Fallback Behavior |
|--------|---------|-------------------|
| `from yolox.tracker.byte_tracker import BYTETracker` | ByteTrack for state tracking | Uses fallback |

### vision/map_analyzer.py

| Import | Purpose | Fallback Behavior |
|--------|---------|-------------------|
| `from vision.game_feature_extractor import GameFeatureExtractor` | Game feature extraction | Uses basic detection only |

### vision/game_feature_extractor.py

| Import | Purpose | Fallback Behavior |
|--------|---------|-------------------|
| `from vision.tracker import ByteTracker` | Tracking for features | Uses fallback tracker |

## Recommended Installation

For full functionality, install all optional dependencies:

```bash
pip install opencv-python numpy ultralytics torch
```

For minimal functionality (core bot without advanced features):

```bash
pip install opencv-python numpy ultralytics torch
```

## Dependency Health Check

A startup script should verify which optional dependencies are available and log the status:

```python
def check_dependencies():
    """Check and log availability of optional dependencies."""
    optional_deps = {
        'realtime_logs': 'Structured logging',
        'core.world_model': 'World model',
        'core.pressure_map': 'Pressure map',
        'core.cover_system': 'Cover system',
        'core.occupancy_grid': 'Occupancy grid',
        'core.behavioral_profile': 'Behavioral profile',
        'core.central_coordinator': 'Central coordinator',
        'decision.utility_ai': 'Utility AI',
        'decision.sticky_target': 'Sticky target',
        'decision.intent_system': 'Intent system',
        'decision.enemy_intention': 'Enemy intention',
        'decision.meta_awareness': 'Meta awareness',
        'pylaai_real.rl_engine': 'RL engine',
        'pylaai_real.elo_tracker': 'ELO tracker',
        'meta_learning': 'Meta learning',
        'world_model_integration': 'World model integration',
        'auto_tuner': 'Auto tuner',
        'behavioral_profile_system': 'Behavioral profile system',
        'auto_calibrator': 'Auto calibrator',
        'pylaai_real.ocr_state_detector': 'OCR state detector',
        'pylaai_real.debug_visualizer': 'Debug visualizer',
        'core.async_pipeline': 'Async pipeline',
        'core.class_registry': 'Class registry',
    }
    
    available = []
    missing = []
    
    for module, feature in optional_deps.items():
        try:
            __import__(module)
            available.append(f"✓ {feature} ({module})")
        except ImportError:
            missing.append(f"✗ {feature} ({module})")
    
    logger.info("=== Dependency Status ===")
    logger.info("Available:")
    for item in available:
        logger.info(f"  {item}")
    logger.info("Missing (optional features disabled):")
    for item in missing:
        logger.info(f"  {item}")
    
    return len(available), len(missing)
```

## Future Improvements

1. **Fail-fast on critical dependencies**: Instead of silent fallbacks, fail with clear error messages for truly critical features
2. **Dependency groups in pyproject.toml**: Define dependency groups (core, full, dev) for easier installation
3. **Version pinning**: Pin specific versions for reproducibility
4. **Dependency health dashboard**: Add to the web dashboard for real-time dependency status
