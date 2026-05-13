import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

# Ensure repository root is importable.
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Make the package path importable for the module under test.
_pkg_root = Path(__file__).resolve().parents[4] / "soberana-omega"
_pkg = types.ModuleType("soberana_omega")
_pkg.__path__ = [str(_pkg_root)]
sys.modules["soberana_omega"] = _pkg

_backend_pkg = types.ModuleType("soberana_omega.backend")
_backend_pkg.__path__ = [str(_pkg_root / "backend")]
sys.modules["soberana_omega.backend"] = _backend_pkg

_api_pkg = types.ModuleType("soberana_omega.backend.api")
_api_pkg.__path__ = [str(_pkg_root / "backend" / "api")]
sys.modules["soberana_omega.backend.api"] = _api_pkg

routes = importlib.import_module("backend.brawl_bot.api")


class DummyBot:
    def __init__(self):
        self.running = True

    def get_status(self):
        return {
            "running": True,
            "current_state": "lobby",
            "current_brawler": "Colt",
            "queue": [{"name": "Colt"}],
            "safety": {"safe": True},
            "diagnostics": {
                "diagnostic_mode": True,
                "lobby": {"step": "press_play_done"},
                "screen_state": "play",
                "progress": {"wins": 1},
            },
        }


def test_get_diagnostics_returns_consolidated_snapshot(monkeypatch):
    monkeypatch.setattr(routes, "_bot_instance", DummyBot())

    resp = asyncio.run(routes.get_diagnostics())

    assert resp["success"] is True
    diag = resp["diagnostics"]
    assert diag["bot_running"] is True
    assert diag["current_state"] == "lobby"
    assert diag["current_brawler"] == "Colt"
    assert diag["diagnostic_mode"] is True
    assert diag["lobby"]["step"] == "press_play_done"
    assert diag["screen_state"] == "play"
