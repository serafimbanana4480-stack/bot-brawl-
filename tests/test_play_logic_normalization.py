import os
import sys
from dataclasses import dataclass

# Ensure repository root is importable.
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pylaai_real.play import PlayLogic


class DummyMovement:
    def get_tactical_movement(self, player, enemies, walls, power_cubes):
        return "W"


class DummyEmulatorController:
    def __init__(self, active=True):
        self.active = active
        self.taps = []
        self.swipes = []

    def get_status_snapshot(self):
        return {"window_active": self.active, "window_title": "BlueStacks App Player"}

    def ensure_window_active(self):
        self.active = True
        return True

    def tap_scaled(self, x, y):
        self.taps.append((x, y))
        return True

    def swipe_scaled(self, x1, y1, x2, y2, duration=300):
        self.swipes.append((x1, y1, x2, y2, duration))
        return True


@dataclass
class DummyDetection:
    class_name: str
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int


def test_normalize_detection_map_handles_dict_and_detection_objects():
    logic = PlayLogic(detect_main=None, detect_enemies=None, movement=DummyMovement())

    dict_input = {
        "player": [[10, 20, 30, 40]],
        "enemy": [(50, 60, 70, 80)],
    }
    normalized_dict = logic._normalize_detection_map(dict_input)
    assert normalized_dict["player"] == [[10, 20, 30, 40]]
    assert normalized_dict["enemy"] == [[50, 60, 70, 80]]

    objects_input = [
        DummyDetection("enemy", 10, 20, 30, 40, 25, 40),
        DummyDetection("player", 100, 120, 30, 40, 115, 140),
    ]
    normalized_objects = logic._normalize_detection_map(objects_input)
    assert normalized_objects["enemy"] == [[10, 20, 40, 60]]
    assert normalized_objects["player"] == [[100, 120, 130, 160]]


def test_get_enemy_id_accepts_list_bbox_without_crashing():
    logic = PlayLogic(detect_main=None, detect_enemies=None, movement=DummyMovement())
    enemy_id = logic._get_enemy_id([10, 20, 40, 60])
    assert isinstance(enemy_id, int)


def test_play_logic_records_window_and_action_snapshot():
    class DummyDetector:
        def detect_objects(self, _frame):
            return {
                "Player": [[100, 100, 160, 160]],
                "Enemy": [[250, 100, 300, 150]],
            }

    logic = PlayLogic(
        detect_main=DummyDetector(),
        detect_enemies=None,
        movement=DummyMovement(),
        emulator_controller=DummyEmulatorController(active=True),
    )

    import numpy as np
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert logic.play_round(frame) is True
    snapshot = logic.get_last_combat_snapshot()
    assert snapshot["state"] == "combat_ok"
    assert snapshot["window_active"] is True
    assert snapshot["attack_taken"] is True
    assert snapshot["target_position"] is not None
