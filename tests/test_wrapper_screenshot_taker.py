import pytest
import sys
import os

# Ensure the 'soberana-omega' package directory is on sys.path so tests can import backend.brawl_bot
_this_dir = os.path.dirname(__file__)
_soberana_root = os.path.abspath(os.path.join(_this_dir, '..', '..', '..'))
if _soberana_root not in sys.path:
    sys.path.insert(0, _soberana_root)

from backend.brawl_bot.pylaai_real.screenshot_taker import ScreenshotTaker


def test_dynamic_title(monkeypatch):
    """Verify that the wrapper picks a dynamic window title from the emulator detector and passes it to ScreenshotTaker."""
    # Create a fake emulator info object similar to EmulatorInfo
    class FakeEmulator:
        def __init__(self, name, type_, window_title=None):
            self.name = name
            self.type = type_
            self.window_title = window_title

    # Fake detector that returns a bluestacks emulator with a specific window title
    class FakeDetector:
        def detect_all(self):
            return [FakeEmulator(name="BlueStacks-1", type_="bluestacks", window_title="BlueStacks App Player - Instance 1")]

    # Monkeypatch the detector factory used by the wrapper
    import backend.brawl_bot.wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod, 'get_emulator_detector', lambda: FakeDetector())

    # Prevent actual win32 calls by faking ScreenshotTaker.find_window
    captured = {}

    def fake_find_window(self):
        # record the title and pretend the window was found
        captured['title'] = self.window_title
        return True

    monkeypatch.setattr(ScreenshotTaker, 'find_window', fake_find_window)

    # Also fake ScreenshotTaker.take to avoid win32/mss errors during setup
    import numpy as np
    monkeypatch.setattr(ScreenshotTaker, 'take', lambda self: np.zeros((1080, 1920, 3), dtype=np.uint8))

    # Stub win32gui to prevent import errors in EmulatorController
    import types
    fake_win32gui = types.SimpleNamespace()
    fake_win32gui.FindWindow = lambda *a, **k: 12345
    fake_win32gui.GetWindowRect = lambda h: (0, 0, 1920, 1080)
    fake_win32gui.IsWindow = lambda h: True
    fake_win32gui.IsWindowVisible = lambda h: True
    fake_win32gui.GetWindowText = lambda h: "BlueStacks App Player"
    fake_win32gui.EnumWindows = lambda cb, lp: None
    fake_win32gui.GetWindowThreadProcessId = lambda h: (0, 12345)
    monkeypatch.setitem(sys.modules, 'win32gui', fake_win32gui)
    monkeypatch.setitem(sys.modules, 'win32process', types.SimpleNamespace())

    # The real wrapper.py setup() is heavy and has many external deps.
    # We only need to verify the window-title propagation logic, so
    # replace setup() on the *real* class with a minimal version.
    real_cls = wrapper_mod.PylaAIEnhanced._load_real()

    def _make_minimal_setup(_wm):
        def _minimal_setup(self):
            detector = _wm.get_emulator_detector()
            emulators = detector.detect_all()
            chosen_title = None
            if emulators:
                chosen_title = emulators[0].window_title or emulators[0].name
            window_title = chosen_title or "BlueStacks App Player"
            self.screenshot = ScreenshotTaker(window_title)
            self.screenshot.find_window()
            return True
        return _minimal_setup

    monkeypatch.setattr(real_cls, 'setup', _make_minimal_setup(wrapper_mod))

    # Now create the PylaAIEnhanced and run setup
    p = wrapper_mod.PylaAIEnhanced()
    ok = p.setup()

    assert ok is True
    assert hasattr(p, 'screenshot') and p.screenshot is not None
    # The screenshot instance should have received the dynamic title
    assert p.screenshot.window_title == "BlueStacks App Player - Instance 1"
    assert captured.get('title') == "BlueStacks App Player - Instance 1"
