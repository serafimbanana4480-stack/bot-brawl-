import asyncio
from pathlib import Path
import importlib
import types
import sys

# Make package name importable even though the top-level directory has a hyphen
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


class DummyBot:
    def __init__(self):
        self.setup_called = 0
        self.running = False

    def setup(self):
        # Return True the first time, False afterwards to simulate behavior toggles
        self.setup_called += 1
        return self.setup_called == 1

    def start(self):
        # Simulate a no-op start that marks running True
        if self.running:
            return False
        self.running = True
        return True

    def stop(self):
        if not self.running:
            return False
        self.running = False
        return True

    def get_status(self):
        return {"running": self.running}


def test_setup_then_start_and_status(monkeypatch):
    dummy = DummyBot()

    # Monkeypatch the module-level singleton so get_bot() returns our dummy
    monkeypatch.setattr(routes, "_bot_instance", dummy)

    # Call setup (first call should return success True)
    setup_resp = asyncio.run(routes.setup_bot())
    assert isinstance(setup_resp, dict)
    assert setup_resp.get("success") is True

    # Call start - should return success True and set running
    start_resp = asyncio.run(routes.start_bot())
    assert isinstance(start_resp, dict)
    assert start_resp.get("success") is True

    # get_status should reflect running
    status_resp = asyncio.run(routes.get_status())
    assert status_resp.get("running") is True

    # Simulate subsequent setup failure: reset _bot_instance to a new DummyBot whose setup returns False
    dummy2 = DummyBot()
    # make first call return False by incrementing internal counter
    dummy2.setup_called = 1
    monkeypatch.setattr(routes, "_bot_instance", dummy2)

    setup_resp2 = asyncio.run(routes.setup_bot())
    assert isinstance(setup_resp2, dict)
    assert setup_resp2.get("success") is False
