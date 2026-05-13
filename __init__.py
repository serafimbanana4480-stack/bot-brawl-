"""
PylaAI Integration Module for Brawl Stars Bot

Public API:
- PylaAIEnhanced: Main bot wrapper
- SafetySystem / SafetyConfig: Anti-ban safety controls
- HumanizationEngine / HumanizationConfig: Human-like behavior
- get_emulator_detector: Emulator auto-detection
- get_model_downloader: YOLO model management

Internal components (Detect, Movement, BrawlerQueue) are NOT exported.
Use them via PylaAIEnhanced. (Fix Error #24)
"""

from importlib import import_module


def _load(module_name: str):
    """Load local modules when available without making import brittle."""
    try:
        return import_module(f".{module_name}", __name__)
    except Exception:
        try:
            return import_module(module_name)
        except Exception:
            return None


# Show legal disclaimer on import.
_load("DISCLAIMER")

# Public interface only. Keep imports lazy and optional so test discovery
# does not fail if one integration module is temporarily broken.
_safety_system = _load("safety_system")
_humanization = _load("humanization")
_emulator_detector = _load("emulator_detector")
_realtime_logs = _load("realtime_logs")
_model_downloader = _load("model_downloader")
_adb_resilient = _load("adb_resilient")
_dataset_pipeline = _load("dataset_pipeline")

SafetySystem = getattr(_safety_system, "SafetySystem", None)
SafetyConfig = getattr(_safety_system, "SafetyConfig", None)
HumanizationEngine = getattr(_humanization, "HumanizationEngine", None)
HumanizationConfig = getattr(_humanization, "HumanizationConfig", None)
get_emulator_detector = getattr(_emulator_detector, "get_emulator_detector", None)
EmulatorInfo = getattr(_emulator_detector, "EmulatorInfo", None)
get_log_manager = getattr(_realtime_logs, "get_log_manager", None)
get_model_downloader = getattr(_model_downloader, "get_model_downloader", None)
ResilientADB = getattr(_adb_resilient, "ResilientADB", None)
DatasetPipeline = getattr(_dataset_pipeline, "DatasetPipeline", None)

__all__ = [
    'PylaAIEnhanced',
    'SafetySystem', 'SafetyConfig',
    'HumanizationEngine', 'HumanizationConfig',
    'get_emulator_detector', 'EmulatorInfo',
    'get_log_manager', 'get_model_downloader',
    'ResilientADB',
    'DatasetPipeline',
]
