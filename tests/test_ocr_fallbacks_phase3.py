import os
import sys
from pathlib import Path

import numpy as np

# Ensure the repository root is importable.
_this_dir = os.path.dirname(__file__)

from backend.brawl_bot.pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig
from backend.brawl_bot.pylaai_real.progress_observer import ProgressObserver


class DummyController:
    def __init__(self):
        self.taps = []
        self.swipes = []

    def tap_scaled(self, x, y):
        self.taps.append((x, y))

    def swipe_scaled(self, x1, y1, x2, y2, **kwargs):
        self.swipes.append((x1, y1, x2, y2))


class DummyTesseract:
    def __init__(self, text):
        self.text = text

    def image_to_string(self, _image):
        return self.text


class DummyEasyOCR:
    def __init__(self, results):
        self.results = results

    def readtext(self, _image):
        return self.results


def test_progress_observer_accepts_pytesseract_fallback_for_result_detection():
    observer = ProgressObserver()
    observer._ocr_available = True
    observer._ocr_backend = "pytesseract"
    observer.reader = DummyTesseract("VICTORY")

    screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert observer._find_result_ocr(screenshot) is True
    stats = observer.get_stats()
    assert stats["wins"] == 1
    assert stats["ocr_available"] is True


def test_lobby_automator_selects_brawler_with_pytesseract_fallback(monkeypatch):
    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="Colt"))

    lobby = LobbyAutomator(queue=queue, emulator_controller=DummyController(), diagnostic_mode=True)
    lobby._ocr_reader = DummyTesseract("Colt\n")
    lobby._ocr_backend = "pytesseract"

    monkeypatch.setattr("backend.brawl_bot.pylaai_real.lobby_automator.time.sleep", lambda *_args, **_kwargs: None)

    screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert lobby.select_brawler("Colt", lambda: screenshot) is True
    assert lobby.emulator_controller.taps
    report = lobby.get_diagnostic_report()
    assert report["step"] == "brawler_selected"
    assert report["details"]["backend"] == "pytesseract"


def test_lobby_automator_reports_card_position_and_confirms_selection(monkeypatch):
    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="Colt"))

    controller = DummyController()
    lobby = LobbyAutomator(queue=queue, emulator_controller=controller, diagnostic_mode=True)
    lobby._ocr_reader = DummyEasyOCR([
        ([(100, 100), (220, 100), (220, 180), (100, 180)], "Colt", 0.93),
        ([(1200, 120), (1340, 120), (1340, 200), (1200, 200)], "Shelly", 0.81),
    ])
    lobby._ocr_backend = "easyocr"

    monkeypatch.setattr("backend.brawl_bot.pylaai_real.lobby_automator.time.sleep", lambda *_args, **_kwargs: None)

    screenshots = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(10)]

    assert lobby.select_brawler("Colt", lambda: screenshots.pop(0)) is True
    assert controller.taps, "Expected a tap on the matched card"
    report = lobby.get_diagnostic_report()
    assert report["step"] == "brawler_selected"
    assert report["details"]["confirmed"] is True
    assert report["details"]["card_position"] == "left"


def test_lobby_automator_uses_grid_navigation_when_candidates_span_multiple_columns(monkeypatch):
    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="Colt"))

    controller = DummyController()
    lobby = LobbyAutomator(queue=queue, emulator_controller=controller, diagnostic_mode=True)
    lobby._ocr_reader = DummyEasyOCR([
        ([(100, 100), (220, 100), (220, 180), (100, 180)], "Bull", 0.90),
        ([(1200, 120), (1340, 120), (1340, 200), (1200, 200)], "Shelly", 0.85),
    ])
    lobby._ocr_backend = "easyocr"

    monkeypatch.setattr("backend.brawl_bot.pylaai_real.lobby_automator.time.sleep", lambda *_args, **_kwargs: None)

    screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)

    assert lobby.select_brawler("Colt", lambda: screenshot) is False
    assert controller.swipes, "Expected a navigation swipe for grid exploration"
    report = lobby.get_diagnostic_report()
    assert report["step"] == "brawler_not_found"
