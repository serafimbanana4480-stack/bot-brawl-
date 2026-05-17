"""
tests/test_adversarial_humanization.py

Tests for adversarial humanization (Phase 4).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.adversarial_humanization import (
    AdversarialHumanizationConfig,
    AdversarialHumanizer,
    BiometricRhythm,
    FingerprintRotator,
    ActionClusterer,
)
from core.adapters.adversarial_input_adapter import AdversarialInputAdapter
from core.ports.input_port import InputAction, InputPort


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cfg():
    return AdversarialHumanizationConfig(
        enabled=True,
        tap_jitter_sigma=2.0,
        miss_tap_probability=1.0,  # force miss for testing
        delayed_reaction_probability=1.0,  # force delay
        overshoot_probability=1.0,  # force overshoot
        reaction_time_base_ms=200.0,
        log_normal_sigma=0.1,
        fatigue_decay_per_hour=0.0,  # disable fatigue for timing tests
        rotation_interval_minutes=9999.0,  # disable rotation for most tests
        rotation_mutation_sigma=0.0,  # no mutation so tests are deterministic
        cluster_burst_prob=0.0,  # disable clustering for most tests
    )


@pytest.fixture
def disabled_cfg():
    return AdversarialHumanizationConfig(enabled=False)


# ------------------------------------------------------------------
# BiometricRhythm
# ------------------------------------------------------------------

class TestBiometricRhythm:
    def test_multiplier_near_one_at_start(self):
        rhythm = BiometricRhythm(AdversarialHumanizationConfig())
        m = rhythm.get_timing_multiplier()
        assert 0.7 < m < 1.3

    def test_multiplier_changes_over_time(self):
        cfg = AdversarialHumanizationConfig(circadian_amplitude=0.2, fatigue_decay_per_hour=0.1)
        rhythm = BiometricRhythm(cfg)
        m1 = rhythm.get_timing_multiplier()
        # Pretend 30 minutes elapsed -> circadian peak
        rhythm.session_start -= 1800
        m2 = rhythm.get_timing_multiplier()
        # Should be different due to circadian + fatigue
        assert m1 != pytest.approx(m2, abs=0.01)


# ------------------------------------------------------------------
# FingerprintRotator
# ------------------------------------------------------------------

class TestFingerprintRotator:
    def test_rotation_not_triggered_early(self):
        rot = FingerprintRotator(AdversarialHumanizationConfig(rotation_interval_minutes=60.0))
        assert not rot._should_rotate()

    def test_rotation_triggered_after_interval(self):
        rot = FingerprintRotator(AdversarialHumanizationConfig(rotation_interval_minutes=0.0))
        rot._last_rotation = time.time() - 120  # 2 minutes ago
        assert rot._should_rotate()

    def test_mutation_changes_values(self):
        base = AdversarialHumanizationConfig(tap_jitter_sigma=1.0)
        rot = FingerprintRotator(base)
        mutated = rot._mutate(base)
        assert mutated.tap_jitter_sigma != pytest.approx(base.tap_jitter_sigma, abs=0.001)
        assert mutated.enabled == base.enabled

    def test_active_config_returns_mutated(self):
        base = AdversarialHumanizationConfig(rotation_interval_minutes=0.0)
        rot = FingerprintRotator(base)
        cfg1 = rot.active_config
        cfg2 = rot.active_config
        # After rotation, subsequent calls return same mutated config
        assert cfg1.tap_jitter_sigma == pytest.approx(cfg2.tap_jitter_sigma)


# ------------------------------------------------------------------
# ActionClusterer
# ------------------------------------------------------------------

class TestActionClusterer:
    def test_cluster_lifecycle(self):
        cfg = AdversarialHumanizationConfig(
            cluster_burst_prob=1.0,
            cluster_actions=(2, 2),
            cluster_pause_ms=(500.0, 500.0),
        )
        clusterer = ActionClusterer(cfg)

        # First call starts cluster (2 actions), no delay
        assert clusterer.should_delay() is None
        # Second call decrements to 1, no delay
        assert clusterer.should_delay() is None
        # Third call ends cluster, returns pause
        pause = clusterer.should_delay()
        assert pause is not None and pause >= 500.0
        # Fourth call during pause returns remaining delay
        assert clusterer.should_delay() is not None

    def test_no_cluster_when_prob_zero(self):
        cfg = AdversarialHumanizationConfig(cluster_burst_prob=0.0)
        clusterer = ActionClusterer(cfg)
        for _ in range(10):
            assert clusterer.should_delay() is None


# ------------------------------------------------------------------
# AdversarialHumanizer
# ------------------------------------------------------------------

class TestAdversarialHumanizer:
    def test_disabled_returns_action_unchanged(self, disabled_cfg):
        humanizer = AdversarialHumanizer(disabled_cfg)
        action = InputAction(action_type="tap", x=0.5, y=0.5)
        result = humanizer.perturb(action, 1920, 1080)
        assert result.x == pytest.approx(0.5)
        assert result.y == pytest.approx(0.5)

    def test_jitter_changes_coordinates(self, cfg):
        humanizer = AdversarialHumanizer(cfg)
        action = InputAction(action_type="tap", x=0.5, y=0.5)
        result = humanizer.perturb(action, 1920, 1080)
        # Jitter is small but non-zero on average
        assert result.x != pytest.approx(0.5, abs=0.0001) or result.y != pytest.approx(0.5, abs=0.0001)
        assert 0.0 <= result.x <= 1.0
        assert 0.0 <= result.y <= 1.0

    def test_intentional_miss(self, cfg):
        humanizer = AdversarialHumanizer(cfg)
        action = InputAction(action_type="tap", x=0.5, y=0.5)
        result = humanizer.perturb(action, 1920, 1080)
        # miss_tap_probability=1.0 guarantees a miss offset
        assert result.x != pytest.approx(0.5, abs=0.001) or result.y != pytest.approx(0.5, abs=0.001)

    def test_delayed_reaction(self, cfg):
        humanizer = AdversarialHumanizer(cfg)
        action = InputAction(action_type="tap", x=0.5, y=0.5, duration_ms=100)
        result = humanizer.perturb(action, 1920, 1080)
        assert result.duration_ms > 100

    def test_swipe_overshoot(self, cfg):
        humanizer = AdversarialHumanizer(cfg)
        action = InputAction(action_type="swipe", x=0.2, y=0.2, x2=0.8, y2=0.8, duration_ms=300)
        result = humanizer.perturb(action, 1920, 1080)
        assert result.x2 != pytest.approx(0.8, abs=0.001)
        assert result.y2 != pytest.approx(0.8, abs=0.001)

    def test_sleep_reaction_time(self, cfg):
        humanizer = AdversarialHumanizer(cfg)
        t0 = time.time()
        humanizer.sleep_reaction_time()
        elapsed = (time.time() - t0) * 1000
        assert elapsed >= 50.0

    def test_stats_tracking(self, cfg):
        humanizer = AdversarialHumanizer(cfg)
        for _ in range(5):
            action = InputAction(action_type="tap", x=0.5, y=0.5)
            humanizer.perturb(action, 1920, 1080)
        stats = humanizer.get_stats()
        assert stats["actions_processed"] == 5
        assert stats["misses_injected"] >= 1
        assert stats["delays_injected"] >= 1
        assert "timing_multiplier" in stats


# ------------------------------------------------------------------
# AdversarialInputAdapter
# ------------------------------------------------------------------

class MockInputPort(InputPort):
    def __init__(self):
        self.executed: list[InputAction] = []

    def initialize(self) -> bool:
        return True

    def execute(self, action: InputAction) -> bool:
        self.executed.append(action)
        return True

    def tap(self, x: float, y: float, duration_ms: int = 100) -> bool:
        return self.execute(InputAction(action_type="tap", x=x, y=y, duration_ms=duration_ms))

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300) -> bool:
        return self.execute(InputAction(
            action_type="swipe", x=x1, y=y1, x2=x2, y2=y2, duration_ms=duration_ms
        ))

    def health_check(self):
        return {"ok": True}

    def shutdown(self):
        pass


class TestAdversarialInputAdapter:
    def test_execute_perturbs_and_delegates(self, cfg):
        primary = MockInputPort()
        humanizer = AdversarialHumanizer(cfg)
        adapter = AdversarialInputAdapter(primary, humanizer)

        action = InputAction(action_type="tap", x=0.5, y=0.5)
        adapter.execute(action)

        assert len(primary.executed) == 1
        executed = primary.executed[0]
        assert executed.x != pytest.approx(0.5, abs=0.0001) or executed.y != pytest.approx(0.5, abs=0.0001)

    def test_health_check_merges_stats(self, cfg):
        primary = MockInputPort()
        humanizer = AdversarialHumanizer(cfg)
        adapter = AdversarialInputAdapter(primary, humanizer)

        health = adapter.health_check()
        assert health["ok"] is True
        assert "adversarial_humanization" in health

    def test_tap_convenience(self, cfg):
        primary = MockInputPort()
        humanizer = AdversarialHumanizer(cfg)
        adapter = AdversarialInputAdapter(primary, humanizer)

        adapter.tap(0.5, 0.5)
        assert len(primary.executed) == 1

    def test_swipe_convenience(self, cfg):
        primary = MockInputPort()
        humanizer = AdversarialHumanizer(cfg)
        adapter = AdversarialInputAdapter(primary, humanizer)

        adapter.swipe(0.2, 0.2, 0.8, 0.8)
        assert len(primary.executed) == 1
        assert primary.executed[0].action_type == "swipe"
