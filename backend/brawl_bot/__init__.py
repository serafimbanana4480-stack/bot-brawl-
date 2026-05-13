"""Compatibility bridge for `backend.brawl_bot.*` imports."""

from importlib import import_module
import sys

_ROOT_PACKAGE = import_module("brawl_bot")


def _mirror(alias: str, target: str) -> None:
    """Register *target* module under the ``backend.brawl_bot.<alias>`` name."""
    try:
        module = import_module(target)
    except Exception:
        return
    sys.modules[f"{__name__}.{alias}"] = module


for _alias in (
    "safety_system",
    "humanization",
    "emulator_detector",
    "realtime_logs",
    "model_downloader",
    "adb_resilient",
    "dataset_pipeline",
    "auto_tuner",
    "match_controller",
    # Flat-root modules also needed via backend.brawl_bot path
    "wrapper",
    "diagnostic_overlay",
):
    _mirror(_alias, _alias)

# Mirror pylaai_real package and its public sub-modules
_mirror("pylaai_real", "pylaai_real")
for _sub in (
    "state_finder",
    "state_manager",
    "lobby_automator",
    "play",
    "screenshot_taker",
    "progress_observer",
    "detect",
    "movement",
    "screen_automation",
):
    _mirror(f"pylaai_real.{_sub}", f"pylaai_real.{_sub}")

# Mirror the api package (api/brawl_stars_routes.py) under two names
_mirror("api", "api")
_mirror("api.brawl_stars_routes", "api.brawl_stars_routes")

from brawl_bot import *  # noqa: F401,F403,E402
