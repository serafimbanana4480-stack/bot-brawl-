import asyncio
from types import SimpleNamespace
from pathlib import Path
import importlib
import types
import sys

# Make package name importable even though the top-level directory has a hyphen
# Map the package name 'soberana_omega' to the existing 'soberana-omega' folder on disk
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

routes = importlib.import_module("soberana_omega.backend.api.brawl_stars_routes")


class DummyEmulator:
    def __init__(self, name, adb_id=None):
        self.name = name
        self.adb_id = adb_id
        self.type = "unknown"
        self.window_title = name
        self.connected = False


class DummyDetector:
    def __init__(self):
        self.detect_called = False
        self.available_emulators = []

    def detect_all(self):
        # Mark that detection was attempted and leave available_emulators empty
        self.detect_called = True
        self.available_emulators = []
        return self.available_emulators

    def get_emulator_by_name(self, name: str):
        # Ensure detect_all is invoked lazily if state is empty
        if not self.available_emulators:
            self.detect_all()
        for e in self.available_emulators:
            if e.name == name or e.window_title == name:
                return e
        return None


def test_detect_all_called_and_connect_tolerates_empty_detector(monkeypatch):
    dummy = DummyDetector()

    # Monkeypatch the module-level get_emulator_detector used by the routes
    monkeypatch.setattr(routes, "get_emulator_detector", lambda: dummy)

    # Call the connect handler for a non-existent emulator
    resp = asyncio.run(routes.connect_emulator("no-such-emulator"))

    # detect_all should have been invoked (via get_emulator_by_name lazy call)
    assert dummy.detect_called is True, "detect_all was not called by the connect flow"

    # Response for missing emulator is a structured not-found message (local reality)
    assert isinstance(resp, dict)
    assert resp.get("success") is False
    assert "não encontrado" in resp.get("message", "").lower()


def test_list_emulators_returns_diagnostics_when_none(monkeypatch):
    dummy = DummyDetector()
    monkeypatch.setattr(routes, "get_emulator_detector", lambda: dummy)

    # list_emulators is async; run it
    resp = asyncio.run(routes.list_emulators())

    # When no emulators detected, route should return diagnostics structure
    assert isinstance(resp, dict)
    assert resp.get("emulators") == []
    assert "diagnostics" in resp or "error" in resp
