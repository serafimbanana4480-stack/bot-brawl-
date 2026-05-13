import pytest
import importlib.util
from pathlib import Path


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
async def test_connect_uses_get_adb_path_and_returns_outputs(monkeypatch):
    routes = load_routes_module()

    # Create a simple emulator-like object
    class E:
        def __init__(self):
            self.name = 'emu1'
            self.type = 'nox'
            self.adb_id = '127.0.0.1:5555'
            self.window_title = 'Emu Window'
            self.connected = False

    detector = DummyDetector([E()])
    monkeypatch.setattr(routes, 'get_emulator_detector', lambda: detector)

    # Monkeypatch get_adb_path directly on the routes module
    monkeypatch.setattr(routes, 'get_adb_path', lambda: 'fake-adb-binary')

    # Capture what subprocess.run is called with and return a failing result
    class FakeResult:
        def __init__(self, returncode=1, stdout='failed to connect', stderr='error details'):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    called = {}

    def fake_run(cmd, capture_output=False, text=False, timeout=None, **kwargs):
        called['cmd'] = cmd
        called['kwargs'] = kwargs
        return FakeResult(returncode=1, stdout='some stdout output', stderr='some stderr output')

    import subprocess
    monkeypatch.setattr(subprocess, 'run', fake_run)

    # Call the connect endpoint
    resp = await routes.connect_emulator('emu1')

    assert isinstance(resp, dict)
    assert resp['success'] is False
    assert 'adb' in resp and isinstance(resp['adb'], dict)
    adb_info = resp['adb']
    assert adb_info['chosen_adb_path'] == 'fake-adb-binary'
    assert adb_info['stdout'] == 'some stdout output'
    assert adb_info['stderr'] == 'some stderr output'

    # Ensure the subprocess was invoked with the chosen binary as first arg
    assert isinstance(called.get('cmd'), list)
    assert called['cmd'][0] == 'fake-adb-binary'
    assert 'connect' in called['cmd']
    assert '127.0.0.1:5555' in called['cmd']
