"""brawl_bot/wrapper.py — re-export shim for the root-level wrapper module.

The real ``wrapper.py`` lives at the project root.  This shim loads it
via importlib.util so that ``brawl_bot.wrapper.PylaAIEnhanced`` resolves
correctly without the fragile fake-package approach.
"""

import importlib.util as _util
import os as _os
import sys as _sys

# Load the real wrapper.py from the project root (parent directory)
_wrapper_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "wrapper.py")

# Ensure all top-level modules that wrapper.py needs are importable
# (they live in the project root, which is on sys.path via main.py)
# Pre-register them as brawl_bot submodules so relative imports resolve
from importlib import import_module as _import_module

# Map of submodule names that wrapper.py imports via relative imports
# to their top-level equivalents (already on sys.path)
_SUBMODULE_MAP = {
    "pylaai_real": "pylaai_real",
    "diagnostic_overlay": "diagnostic_overlay",
    "match_controller": "match_controller",
    "model_downloader": "model_downloader",
    "emulator_detector": "emulator_detector",
    "emulator_controller": "emulator_controller",
    "safety_system": "safety_system",
    "humanization": "humanization",
    "auto_tuner": "auto_tuner",
    "decision": "decision",
    "core": "core",
    "dataset": "dataset",
}

# Register top-level modules as brawl_bot submodules so relative imports work
for _alias, _target in _SUBMODULE_MAP.items():
    _fq_name = f"brawl_bot.{_alias}"
    if _fq_name not in _sys.modules:
        try:
            _mod = _import_module(_target)
            _sys.modules[_fq_name] = _mod
        except Exception:
            pass

# Now load wrapper.py with __package__ = "brawl_bot" so relative imports resolve
_spec = _util.spec_from_file_location(
    "brawl_bot.wrapper",
    _wrapper_path,
    submodule_search_locations=None,
)
_real_wrapper = _util.module_from_spec(_spec)
_real_wrapper.__package__ = "brawl_bot"
_sys.modules["brawl_bot.wrapper"] = _real_wrapper
_spec.loader.exec_module(_real_wrapper)

# Re-export the public API
PylaAIEnhanced = _real_wrapper.PylaAIEnhanced
