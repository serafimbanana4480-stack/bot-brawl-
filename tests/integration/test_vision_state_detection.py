"""Integration tests for vision-based state detection using real screenshots."""

from pathlib import Path

import cv2
import pytest

from pylaai_real.unified_state_detector import UnifiedStateDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _collect_screenshots():
    cases = []
    # Lobby screenshots
    lobby_patterns = [
        "auto_test_093747_c1.png",
        "auto_test_093814_c10.png",
        "auto_test_093844_c20.png",
        "auto_play_094657_in_game.png",
        "auto_play_094900_in_game.png",
        "auto_play_095139_in_game.png",
        "screenshot_current.png",
        "screenshot_latest.png",
    ]
    for p in lobby_patterns:
        path = PROJECT_ROOT / p
        if path.exists():
            cases.append((path, "lobby"))
    # Matchmaking screenshots
    mm_patterns = [
        "auto_play_100328_matchmaking.png",
        "auto_play_100502_matchmaking.png",
        "auto_play_100818_matchmaking.png",
        "screenshot_now.png",
    ]
    for p in mm_patterns:
        path = PROJECT_ROOT / p
        if path.exists():
            cases.append((path, "matchmaking"))
    return cases


SCREENSHOT_CASES = _collect_screenshots()


@pytest.mark.parametrize(
    "screenshot_path,expected_state",
    SCREENSHOT_CASES,
    ids=[f"{p.name}:{s}" for p, s in SCREENSHOT_CASES],
)
def test_detect_state_from_screenshot(screenshot_path, expected_state):
    detector = UnifiedStateDetector(images_path=PROJECT_ROOT / "images")
    image = cv2.imread(str(screenshot_path))
    assert image is not None, f"Failed to load {screenshot_path}"
    result = detector.detect(image)
    assert result.state == expected_state, (
        f"Expected {expected_state} for {screenshot_path.name}, got {result.state} "
        f"(conf={result.confidence:.2f}, method={result.method})"
    )
