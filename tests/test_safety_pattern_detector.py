"""
Testes adicionais para safety_system.py — PatternDetector e MovementAnalyzer
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from safety_system import PatternDetector, MovementAnalyzer


class TestPatternDetector:
    def test_empty_detector(self):
        pd = PatternDetector()
        assert not pd.detect_perfect_timing()
        assert not pd.detect_perfect_aim()
        assert not pd.detect_repeated_patterns()
        assert not pd.detect_burst()
        assert pd.get_suspicion_score() == 0

    def test_burst_detection(self):
        pd = PatternDetector()
        now = __import__('time').time()
        for i in range(15):
            pd.action_window.append(now)
        assert pd.detect_burst()

    def test_click_history_bounded(self):
        pd = PatternDetector()
        for i in range(150):
            pd.record_click(float(i), float(i))
        assert len(pd.click_times) <= pd.max_history
        assert len(pd.click_positions) <= pd.max_history


class TestMovementAnalyzer:
    def test_record_swipe(self):
        ma = MovementAnalyzer()
        ma.record_swipe(0, 0, 100, 100, duration=1.0)
        assert len(ma.movements) == 1
        assert len(ma.swipes) == 1
        assert ma.get_average_velocity() > 0

    def test_record_tap(self):
        ma = MovementAnalyzer()
        ma.record_tap(50, 50)
        assert len(ma.movements) == 1
        assert len(ma.taps) == 1
        assert ma.get_average_velocity() == 0

    def test_velocity_variance(self):
        ma = MovementAnalyzer()
        assert ma.get_velocity_variance() == 0
        ma.record_swipe(0, 0, 100, 0, duration=1.0)
        ma.record_swipe(0, 0, 200, 0, duration=1.0)
        assert ma.get_velocity_variance() >= 0

    def test_max_acceleration(self):
        ma = MovementAnalyzer()
        assert ma.get_max_acceleration() == 0
        ma.record_swipe(0, 0, 100, 0, duration=1.0)
        ma.record_swipe(0, 0, 300, 0, duration=0.5)
        assert ma.get_max_acceleration() >= 0
