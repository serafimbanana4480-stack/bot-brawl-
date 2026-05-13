"""
Testes para humanization.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from humanization import BezierCurve, WindMouse, HumanizationEngine, HumanizationConfig


class TestBezierCurve:
    def test_get_point_bounds(self):
        curve = BezierCurve((0, 0), (1, 1), (2, 2), (3, 3))
        p = curve.get_point(0.5)
        assert 0 <= p[0] <= 3
        assert 0 <= p[1] <= 3

    def test_get_point_clamps(self):
        curve = BezierCurve((0, 0), (1, 1), (2, 2), (3, 3))
        p_low = curve.get_point(-0.5)
        p_high = curve.get_point(1.5)
        assert p_low == pytest.approx((0.0, 0.0))
        assert p_high == pytest.approx((3.0, 3.0))

    def test_generate_path_length(self):
        curve = BezierCurve((0, 0), (1, 2), (3, 2), (4, 4))
        path = curve.generate_path(num_points=50)
        assert len(path) == 50


class TestWindMouse:
    def test_generate_path_direction_right(self):
        wm = WindMouse()
        path = wm.generate_path((0, 0), (100, 0), max_step=20)
        assert len(path) > 0
        # O último ponto deve estar perto do destino
        assert abs(path[-1][0] - 100) < 25
        assert abs(path[-1][1] - 0) < 25

    def test_generate_path_direction_left(self):
        wm = WindMouse()
        path = wm.generate_path((100, 0), (0, 0), max_step=20)
        assert len(path) > 0
        assert abs(path[-1][0] - 0) < 25
        assert abs(path[-1][1] - 0) < 25

    def test_generate_path_direction_up(self):
        wm = WindMouse()
        path = wm.generate_path((0, 100), (0, 0), max_step=20)
        assert len(path) > 0
        assert abs(path[-1][0] - 0) < 25
        assert abs(path[-1][1] - 0) < 25

    def test_generate_path_with_timing(self):
        wm = WindMouse()
        points = wm.generate_path_with_timing((0, 0), (100, 100), base_speed=500.0)
        assert len(points) > 0
        assert len(points[0]) == 3  # x, y, timestamp
        # Timestamps devem ser crescentes
        for i in range(1, len(points)):
            assert points[i][2] >= points[i-1][2]


class TestHumanizationEngine:
    def test_initialization(self):
        engine = HumanizationEngine()
        assert engine is not None

    def test_get_delay_range(self):
        cfg = HumanizationConfig(min_delay=0.1, max_delay=0.5)
        engine = HumanizationEngine(cfg)
        delay = engine.get_delay("attack")
        assert cfg.min_delay <= delay <= cfg.max_delay

    def test_disabled_engine(self):
        cfg = HumanizationConfig(enabled=False)
        engine = HumanizationEngine(cfg)
        delay = engine.get_delay("attack")
        assert delay == 0.0
