import sys
import types

from brawl_bot.emulator_detector import EmulatorDetector


def test_detect_flow(monkeypatch):
    # Create fake win32gui module
    fake_win32gui = types.SimpleNamespace()

    def IsWindowVisible(hwnd):
        return True

    def GetWindowText(hwnd):
        return "BlueStacks App Player"

    def GetWindowThreadProcessId(hwnd):
        return (0, 12345)

    def EnumWindows(callback, lparam):
        # Simulate a single window handle
        callback(1, lparam)

    fake_win32gui.IsWindowVisible = IsWindowVisible
    fake_win32gui.GetWindowText = GetWindowText
    fake_win32gui.GetWindowThreadProcessId = GetWindowThreadProcessId
    fake_win32gui.EnumWindows = EnumWindows

    # Create fake psutil module for Process
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def name(self):
            return "HD-Player.exe"

    fake_psutil = types.SimpleNamespace()
    fake_psutil.Process = FakeProcess

    # Inject fake modules into sys.modules
    monkeypatch.setitem(sys.modules, 'win32gui', fake_win32gui)
    monkeypatch.setitem(sys.modules, 'win32process', types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil)

    detector = EmulatorDetector()
    window_results = detector.detect_window_emulators()

    assert isinstance(window_results, list)
    assert any(r.type == 'bluestacks' for r in window_results), "Window detection should find a BlueStacks emulator"

    # Now test process-based detection directly
    class FakeProc:
        def __init__(self, name):
            self.info = {'name': name, 'pid': 100, 'exe': name}
            self.pid = 100

    def fake_process_iter(attrs):
        return [FakeProc('HD-Player.exe'), FakeProc('not_an_emulator.exe')]

    fake_psutil2 = types.SimpleNamespace()
    fake_psutil2.process_iter = fake_process_iter
    fake_psutil2.NoSuchProcess = Exception
    fake_psutil2.AccessDenied = Exception
    fake_psutil2.ZombieProcess = Exception

    monkeypatch.setitem(sys.modules, 'psutil', fake_psutil2)

    proc_results = detector._detect_by_process()
    assert isinstance(proc_results, list)
    assert any(r.type == 'bluestacks' for r in proc_results), "Process detection should find a BlueStacks emulator"
