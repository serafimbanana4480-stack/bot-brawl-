"""Compatibility package for legacy `brawl_bot.*` imports.

The repository historically exposed its modules at the flat project root,
but several tests and integration points still import them through the
`brawl_bot` package name. This shim keeps both import styles working.
"""

from importlib import import_module
import sys


_ALIASES = {
    "auto_tuner": "auto_tuner",
    "match_controller": "match_controller",
    "safety_system": "safety_system",
    "humanization": "humanization",
    "emulator_detector": "emulator_detector",
    "emulator_controller": "emulator_controller",
    "realtime_logs": "realtime_logs",
    "model_downloader": "model_downloader",
    "adb_resilient": "adb_resilient",
    "dataset_pipeline": "dataset_pipeline",
    "tracker": "tracker",
    "analysis": "analysis",
    "training": "training",
    "vision": "vision",
    "decision": "decision",
    "control": "control",
    "core": "core",
    "automation": "automation",
    "pylaai_real": "pylaai_real",
    # Typo alias used in some test files (missing second 'a')
    "pylai_real": "pylaai_real",
    "rl_stubs": "rl_stubs",
    # Modules imported by wrapper.py via relative imports
    # NOTE: "wrapper" is NOT aliased here because it uses relative imports
    # and must be imported as brawl_bot.wrapper, not as a top-level module.
    "diagnostic_overlay": "diagnostic_overlay",
    "vision_engine": "vision_engine",
    "verify_installation": "verify_installation",
}


def _alias_modules() -> None:
    for alias, target in _ALIASES.items():
        try:
            module = import_module(target)
        except Exception:
            continue
        sys.modules[f"{__name__}.{alias}"] = module


_alias_modules()

_wrapper = None

def __getattr__(name):
    """Lazy-load wrapper.PylaAIEnhanced on first access."""
    global _wrapper
    if name == "PylaAIEnhanced":
        if _wrapper is None:
            try:
                _wrapper = import_module(f"{__name__}.wrapper")
            except Exception:
                return None
        return getattr(_wrapper, "PylaAIEnhanced", None)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from safety_system import SafetySystem, SafetyConfig  # noqa: E402
from humanization import HumanizationEngine, HumanizationConfig  # noqa: E402
from emulator_detector import get_emulator_detector, EmulatorInfo  # noqa: E402
from realtime_logs import get_log_manager  # noqa: E402
from model_downloader import get_model_downloader  # noqa: E402
from adb_resilient import ResilientADB  # noqa: E402
from dataset_pipeline import DatasetPipeline  # noqa: E402

__all__ = [
    "PylaAIEnhanced",
    "SafetySystem",
    "SafetyConfig",
    "HumanizationEngine",
    "HumanizationConfig",
    "get_emulator_detector",
    "EmulatorInfo",
    "get_log_manager",
    "get_model_downloader",
    "ResilientADB",
    "DatasetPipeline",
]
