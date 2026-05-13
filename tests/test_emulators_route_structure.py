import pytest
import types
import asyncio
import importlib.util
from pathlib import Path


# Helper to load the route module by file path
def load_routes_module():
    # project_root = "bot brawl/" directory (parent of "tests/")
    project_root = Path(__file__).parent.parent
    module_path = project_root / 'api' / 'brawl_stars_routes.py'
    import sys
    cwd = str(project_root)
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    spec = importlib.util.spec_from_file_location('brawl_stars_routes', str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DummyDetector:
    def __init__(self, emulators=None):
        self._emulators = emulators or []

    def detect_all(self):
        return self._emulators

    def get_emulator_by_name(self, name):
        for e in self._emulators:
            if e.name == name:
                return e
        return None


@pytest.mark.asyncio
async def test_list_emulators_returns_structure_when_none(monkeypatch, tmp_path):
    routes = load_routes_module()

    # Patch get_emulator_detector to return a detector that finds nothing
    monkeypatch.setattr(routes, 'get_emulator_detector', lambda: DummyDetector([]))

    # Ensure no verify_installation_report.json exists at the known location
    repo_root = routes.Path(__file__).parent.parent.parent
    report_path = repo_root / "backend" / "brawl_bot" / "verify_installation_report.json"
    if report_path.exists():
        report_path.unlink()

    resp = await routes.list_emulators()

    assert isinstance(resp, dict), "response must be a dict"
    assert 'emulators' in resp and isinstance(resp['emulators'], list)
    assert 'count' in resp and resp['count'] == 0
    assert 'diagnostics' in resp and isinstance(resp['diagnostics'], dict)

    diag = resp['diagnostics']
    # diagnostics should include keys we expect (adb_path, psutil_available, pywin32_available)
    assert 'adb_path' in diag
    assert 'psutil_available' in diag
    assert 'pywin32_available' in diag
    assert 'last_verify_installation_report' in diag


@pytest.mark.asyncio
async def test_list_emulators_returns_emulator_list(monkeypatch):
    routes = load_routes_module()

    # Create a simple emulator-like object
    class E:
        def __init__(self):
            self.name = 'emu1'
            self.type = 'nox'
            self.adb_id = '127.0.0.1:5555'
            self.window_title = 'Emu Window'
            self.connected = False

    monkeypatch.setattr(routes, 'get_emulator_detector', lambda: DummyDetector([E()]))

    resp = await routes.list_emulators()
    assert isinstance(resp, dict)
    assert resp['count'] == 1
    assert isinstance(resp['emulators'], list)
    item = resp['emulators'][0]
    assert item['name'] == 'emu1'
    assert 'adb_id' in item
