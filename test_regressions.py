from types import SimpleNamespace

import numpy as np

from pylaai_real.detect import Detect
from pylaai_real.state_manager import StateManager
import screenshot_recorder


class _FakeArray:
    def __init__(self, values):
        self._values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBox:
    def __init__(self, cls_id, conf, bbox):
        self.cls = np.array([cls_id])
        self.conf = np.array([conf])
        self.xyxy = [_FakeArray(bbox)]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, img, conf=0.5):
        self.calls.append(conf)
        if conf >= 0.4:
            return [_FakeResult([])]
        return [_FakeResult([_FakeBox(2, 0.11, [10, 20, 30, 40])])]


def test_detect_falls_back_to_lower_confidence_when_high_conf_finds_nothing():
    model = _FakeModel()
    detector = Detect(model=model, classes={2: "enemy"}, conf=0.40)

    detections = detector.detect_objects(np.zeros((32, 32, 3), dtype=np.uint8))

    assert model.calls == [0.4, 0.1]
    assert detections == {"enemy": [[10, 20, 30, 40]]}


def test_screenshot_recorder_get_adb_path_works_when_run_directly():
    adb_path = screenshot_recorder._get_adb_path()

    assert isinstance(adb_path, str)
    assert adb_path


class _FakeQueue:
    def get_current(self):
        return SimpleNamespace(name="colt", game_mode=None)


class _FakeLobby:
    def __init__(self):
        self.queue = _FakeQueue()
        self.press_play_calls = 0

    def press_play(self):
        self.press_play_calls += 1
        return True


class _FakeMatchController:
    def __init__(self):
        self.calls = []
        self.current_match = None

    def start_match(self, game_mode, brawler):
        self.calls.append((game_mode, brawler))
        return True


def test_handle_lobby_registers_match_before_switching_to_loading():
    manager = StateManager(
        screenshot_taker=None,
        state_finder=None,
        lobby_automator=_FakeLobby(),
        progress_observer=None,
        play_logic=None,
        match_controller=_FakeMatchController(),
        emulator_controller=None,
        screen_automation=None,
        movement=None,
        diagnostic_mode=False,
        reward_bridge=None,
        data_collector=None,
        brawler_selector=None,
        observability=None,
        unified_state_detector=None,
        ocr_detector=None,
        lobby=None,
        rl_engine=None,
        learning_mode_controller=None,
        auto_fix_engine=None,
    )

    manager._handle_lobby()

    assert manager.match_controller.calls == [("showdown", "colt")]
    assert manager.current_state == "loading"
