"""
conftest.py — project-wide pytest configuration.

Registers compatibility shims so that tests which import from legacy or
alternate package paths work without a separate package installation.

Problems solved:
1. Tests compute ``_repo_root`` as 3 levels above ``tests/``, putting the
   wrong directory on sys.path.  We insert the real project root early.

2. ``brawl_bot/__init__.py`` is heavy and triggers slow imports.  We stub
   the ``brawl_bot`` package in sys.modules before anything touches it, and
   pre-register every sub-module alias directly so no __init__.py chains run.

3. ``wrapper.py`` at root uses relative imports; we skip loading it at import
   time and instead lazily provide ``PylaAIEnhanced`` through a thin shim.

4. Tests import ``soberana_omega.backend.api.brawl_stars_routes``.
   We shim that name to point at ``api/brawl_stars_routes``.

5. pytest's monkeypatch.setattr("backend.brawl_bot.pylaai_real.lobby_automator.time.sleep", ...)
   traverses with getattr(), so every intermediate module must be set as an
   *attribute* on its parent module, not just in sys.modules.
"""

import sys
import types
import os
import importlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1: Real project root on sys.path FIRST.
# ---------------------------------------------------------------------------
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pkg(dotted_name: str) -> types.ModuleType:
    """Return the stub namespace package for *dotted_name*, creating it and
    all parent stubs as needed.  Each stub is also set as an attribute on its
    parent so that getattr-based traversal (monkeypatch) works."""
    parts = dotted_name.split(".")
    parent = None
    accumulated = ""
    for part in parts:
        accumulated = f"{accumulated}.{part}" if accumulated else part
        if accumulated not in sys.modules:
            pkg = types.ModuleType(accumulated)
            pkg.__path__ = []
            pkg.__package__ = accumulated
            sys.modules[accumulated] = pkg
        else:
            pkg = sys.modules[accumulated]
        if parent is not None:
            setattr(parent, part, pkg)
        parent = pkg
    return pkg


def _bind(full_name: str, real_target: str) -> None:
    """Import *real_target* and register it under *full_name* in sys.modules.
    Also sets the module as an attribute on its parent stub so getattr works."""
    if full_name in sys.modules:
        return
    try:
        mod = importlib.import_module(real_target)
    except Exception:
        return
    sys.modules[full_name] = mod
    # Set as attribute on the parent stub (if one exists)
    if "." in full_name:
        parent_name, attr = full_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, mod)


# ---------------------------------------------------------------------------
# Step 2: Stub brawl_bot so its heavy __init__.py never runs, and
#         pre-register every sub-module alias.
# ---------------------------------------------------------------------------
_ensure_pkg("brawl_bot")

_BRAWL_BOT_FLAT = [
    "realtime_logs", "model_downloader", "adb_resilient", "dataset_pipeline",
    "safety_system", "humanization", "emulator_detector", "emulator_controller",
    "match_controller", "auto_tuner", "tracker", "diagnostic_overlay",
]
for _n in _BRAWL_BOT_FLAT:
    _bind(f"brawl_bot.{_n}", _n)

# pylaai_real sub-modules
_ensure_pkg("brawl_bot.pylaai_real")
_bind("brawl_bot.pylaai_real", "pylaai_real")
for _sub in ("state_finder", "state_manager", "lobby_automator", "play",
             "screenshot_taker", "progress_observer", "detect", "movement",
             "screen_automation"):
    _bind(f"brawl_bot.pylaai_real.{_sub}", f"pylaai_real.{_sub}")

# pylai_real (typo alias — several tests import this instead of pylaai_real)
_ensure_pkg("brawl_bot.pylai_real")
_bind("brawl_bot.pylai_real", "pylaai_real")
for _sub in ("state_finder", "state_manager", "lobby_automator", "play",
             "screenshot_taker", "progress_observer", "detect", "movement",
             "screen_automation"):
    _bind(f"brawl_bot.pylai_real.{_sub}", f"pylaai_real.{_sub}")


# ---------------------------------------------------------------------------
# Step 3: backend.brawl_bot.* aliases — with attribute chain for monkeypatch.
# ---------------------------------------------------------------------------
_ensure_pkg("backend")
_ensure_pkg("backend.brawl_bot")

for _alias, _target in [
    ("backend.brawl_bot.safety_system",        "safety_system"),
    ("backend.brawl_bot.humanization",         "humanization"),
    ("backend.brawl_bot.emulator_detector",    "emulator_detector"),
    ("backend.brawl_bot.emulator_controller",  "emulator_controller"),
    ("backend.brawl_bot.match_controller",     "match_controller"),
    ("backend.brawl_bot.auto_tuner",           "auto_tuner"),
    ("backend.brawl_bot.diagnostic_overlay",   "diagnostic_overlay"),
    ("backend.brawl_bot.api",                  "api.brawl_stars_routes"),
    ("backend.brawl_bot.api.brawl_stars_routes", "api.brawl_stars_routes"),
]:
    _bind(_alias, _target)

# Wrapper — provide a thin shim that exposes PylaAIEnhanced without loading
# the real wrapper.py at import time (it uses relative imports and pulls in
# heavy ML deps).
def _make_wrapper_shim() -> types.ModuleType:
    shim = types.ModuleType("backend.brawl_bot.wrapper")
    shim.__package__ = "backend.brawl_bot"

    class _PylaAIEnhancedPlaceholder:
        """Thin stand-in loaded lazily from the real wrapper.py."""
        _real_class = None

        @classmethod
        def _load_real(cls):
            if cls._real_class is not None:
                return cls._real_class
            try:
                import importlib.util as _ilu
                import types as _t
                _PKG = "_brawl_bot_compat_pkg"
                if _PKG not in sys.modules:
                    _p = _t.ModuleType(_PKG)
                    _p.__path__ = []
                    sys.modules[_PKG] = _p
                    for _rel, _abs in [
                        ("pylaai_real",                  "pylaai_real"),
                        ("pylaai_real.state_finder",     "pylaai_real.state_finder"),
                        ("pylaai_real.state_manager",    "pylaai_real.state_manager"),
                        ("pylaai_real.screenshot_taker", "pylaai_real.screenshot_taker"),
                        ("pylaai_real.lobby_automator",  "pylaai_real.lobby_automator"),
                        ("pylaai_real.progress_observer","pylaai_real.progress_observer"),
                        ("pylaai_real.play",             "pylaai_real.play"),
                        ("pylaai_real.detect",           "pylaai_real.detect"),
                        ("pylaai_real.movement",         "pylaai_real.movement"),
                        ("pylaai_real.screen_automation","pylaai_real.screen_automation"),
                        ("diagnostic_overlay",           "diagnostic_overlay"),
                        ("match_controller",             "match_controller"),
                        ("model_downloader",             "model_downloader"),
                        ("emulator_detector",            "emulator_detector"),
                        ("safety_system",                "safety_system"),
                        ("humanization",                 "humanization"),
                        ("auto_tuner",                   "auto_tuner"),
                        ("decision",                     "decision"),
                        ("decision.brawler_selector",    "decision.brawler_selector"),
                    ]:
                        try:
                            sys.modules[f"{_PKG}.{_rel}"] = importlib.import_module(_abs)
                        except Exception:
                            pass
                spec = _ilu.spec_from_file_location(
                    f"{_PKG}.wrapper",
                    os.path.join(_repo_root, "wrapper.py"),
                )
                mod = _ilu.module_from_spec(spec)
                mod.__package__ = _PKG
                sys.modules[f"{_PKG}.wrapper"] = mod
                spec.loader.exec_module(mod)
                cls._real_class = mod.PylaAIEnhanced
            except Exception as exc:
                raise ImportError(f"Could not load real PylaAIEnhanced: {exc}") from exc
            return cls._real_class

        def __new__(cls, *args, **kwargs):
            real = cls._load_real()
            return real.__new__(real)

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    # Also provide get_emulator_detector at the module level (used by tests)
    try:
        from emulator_detector import get_emulator_detector
        shim.get_emulator_detector = get_emulator_detector
    except Exception:
        pass

    shim.PylaAIEnhanced = _PylaAIEnhancedPlaceholder
    return shim


_wrapper_shim = _make_wrapper_shim()
sys.modules["backend.brawl_bot.wrapper"] = _wrapper_shim
_bb = sys.modules.get("backend.brawl_bot")
if _bb is not None:
    setattr(_bb, "wrapper", _wrapper_shim)

# pylaai_real under backend.brawl_bot — with full attribute chain
_ensure_pkg("backend.brawl_bot.pylaai_real")
_bind("backend.brawl_bot.pylaai_real", "pylaai_real")
for _sub in ("state_finder", "state_manager", "lobby_automator", "play",
             "screenshot_taker", "progress_observer", "detect", "movement",
             "screen_automation"):
    _bind(f"backend.brawl_bot.pylaai_real.{_sub}", f"pylaai_real.{_sub}")

# pylai_real (typo alias) under backend.brawl_bot
_ensure_pkg("backend.brawl_bot.pylai_real")
_bind("backend.brawl_bot.pylai_real", "pylaai_real")
for _sub in ("state_finder", "state_manager", "lobby_automator", "play",
             "screenshot_taker", "progress_observer", "detect", "movement",
             "screen_automation"):
    _bind(f"backend.brawl_bot.pylai_real.{_sub}", f"pylaai_real.{_sub}")

# Ensure the attribute chain for monkeypatch traversal:
# backend.brawl_bot.pylaai_real.<sub>.time  (for patching time.sleep)
for _sub in ("state_manager", "lobby_automator", "play", "progress_observer"):
    for _prefix in ("backend.brawl_bot.pylaai_real", "backend.brawl_bot.pylai_real",
                    "brawl_bot.pylaai_real", "brawl_bot.pylai_real"):
        _mod = sys.modules.get(f"{_prefix}.{_sub}")
        if _mod is not None and not hasattr(_mod, "time"):
            try:
                import time as _time_mod
                setattr(_mod, "time", _time_mod)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Step 4: soberana_omega shim
# ---------------------------------------------------------------------------
def _ensure_soberana_shim() -> None:
    try:
        real_routes = importlib.import_module("api.brawl_stars_routes")
    except Exception:
        return
    _ensure_pkg("soberana_omega")
    _ensure_pkg("soberana_omega.backend")
    _ensure_pkg("soberana_omega.backend.api")
    sys.modules["soberana_omega.backend.api.brawl_stars_routes"] = real_routes
    _api = sys.modules.get("soberana_omega.backend.api")
    if _api is not None:
        setattr(_api, "brawl_stars_routes", real_routes)


_ensure_soberana_shim()
