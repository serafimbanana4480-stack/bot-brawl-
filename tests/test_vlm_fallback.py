"""
tests/test_vlm_fallback.py

Tests for VLM fallback system (Phase 3: anti-UI-change resilience).
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.vlm_fallback import VLMFallback, VLMConfig, VLMResult
from core.adapters.vlm_vision_adapter import VLMVisionAdapter
from core.ports.vision_port import GameStateSnapshot, HUDState, DetectedObject


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def dummy_image():
    """A small dummy screenshot."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def vlm_config():
    return VLMConfig(
        enabled=True,
        provider="openai",
        model="gpt-4o-mini",
        api_key="fake-key",
        timeout_seconds=1.0,
        max_calls_per_minute=10,
        max_calls_per_hour=100,
        cache_ttl_seconds=5.0,
        fallback_threshold=0.35,
    )


@pytest.fixture
def vlm_disabled_config():
    return VLMConfig(enabled=False)


# ------------------------------------------------------------------
# VLMFallback unit tests
# ------------------------------------------------------------------

class TestVLMConfig:
    def test_default_config(self):
        cfg = VLMConfig()
        assert cfg.enabled is False
        assert cfg.provider == "openai"
        assert cfg.fallback_threshold == 0.35

    def test_custom_config(self, vlm_config):
        assert vlm_config.enabled is True
        assert vlm_config.provider == "openai"
        assert vlm_config.fallback_threshold == 0.35


class TestShouldFallback:
    def test_disabled_never_fallbacks(self, dummy_image):
        vlm = VLMFallback(config=VLMConfig(enabled=False))
        assert vlm.should_fallback(primary_confidence=0.0) is False
        assert vlm.should_fallback(primary_confidence=0.1, frames_without_detection=10) is False

    def test_low_confidence_triggers(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        assert vlm.should_fallback(primary_confidence=0.2) is True
        assert vlm.should_fallback(primary_confidence=0.34) is True
        assert vlm.should_fallback(primary_confidence=0.35) is False
        assert vlm.should_fallback(primary_confidence=0.5) is False

    def test_frames_without_detection_triggers(self, vlm_config):
        vlm = VLMFallback(config=vlm_config)
        assert vlm.should_fallback(primary_confidence=0.5, frames_without_detection=5) is True
        assert vlm.should_fallback(primary_confidence=0.5, frames_without_detection=4) is False


class TestCacheAndRateLimit:
    def test_cache_hit(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        fake_result = VLMResult(game_phase="lobby", confidence=0.9)

        with patch.object(vlm, "_call_openai", return_value=fake_result) as mock_caller:
            r1 = vlm.analyze(dummy_image)
            r2 = vlm.analyze(dummy_image)

        assert r1.cached is False
        assert r2.cached is True
        assert mock_caller.call_count == 1

    def test_cache_expires(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        fake_result = VLMResult(game_phase="lobby", confidence=0.9)

        with patch.object(vlm, "_call_openai", return_value=fake_result) as mock_caller:
            r1 = vlm.analyze(dummy_image)
            time.sleep(6)  # wait for cache TTL (5s) to expire
            r2 = vlm.analyze(dummy_image)

        assert r1.cached is False
        assert r2.cached is False
        assert mock_caller.call_count == 2

    def test_rate_limit_blocks(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        vlm._call_history = [time.time()] * 10  # 10 calls "just now"

        r = vlm.analyze(dummy_image)
        assert r.error == "rate_limited"

    def test_stats(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        fake_result = VLMResult(game_phase="lobby", confidence=0.9)

        with patch.object(vlm, "_call_openai", return_value=fake_result) as mock_caller:
            vlm.analyze(dummy_image)
            assert mock_caller.call_count == 1

        stats = vlm.get_stats()
        assert stats["enabled"] is True
        assert stats["total_calls"] == 1
        assert stats["calls_last_hour"] == 1
        assert stats["cache_size"] == 1


class TestParseResponse:
    def test_valid_json(self, vlm_config):
        vlm = VLMFallback(config=vlm_config)
        text = json.dumps({
            "game_phase": "combat",
            "confidence": 0.85,
            "elements": [
                {"type": "enemy", "x_norm": 0.5, "y_norm": 0.6, "description": "near"}
            ],
        })
        result = vlm._parse_json_response(text)
        assert result.game_phase == "combat"
        assert result.confidence == 0.85
        assert len(result.detected_elements) == 1
        assert result.error is None

    def test_markdown_fences(self, vlm_config):
        vlm = VLMFallback(config=vlm_config)
        text = "```json\n" + json.dumps({"game_phase": "lobby", "confidence": 0.9}) + "\n```"
        result = vlm._parse_json_response(text)
        assert result.game_phase == "lobby"
        assert result.error is None

    def test_invalid_json(self, vlm_config):
        vlm = VLMFallback(config=vlm_config)
        result = vlm._parse_json_response("not json at all")
        assert result.error is not None
        assert result.confidence == 0.0


class TestProviderMocks:
    def test_openai_mock(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        fake_resp = {
            "choices": [{"message": {"content": json.dumps({"game_phase": "victory", "confidence": 0.95})}}]
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = fake_resp
            mock_post.return_value.raise_for_status = lambda: None
            result = vlm._call_openai(dummy_image)

        assert result.game_phase == "victory"
        assert result.model_used.startswith("openai")

    def test_anthropic_mock(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        vlm.cfg.provider = "anthropic"
        fake_resp = {
            "content": [{"text": json.dumps({"game_phase": "defeat", "confidence": 0.8})}]
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = fake_resp
            mock_post.return_value.raise_for_status = lambda: None
            result = vlm._call_anthropic(dummy_image)

        assert result.game_phase == "defeat"
        assert result.model_used.startswith("anthropic")

    def test_local_mock(self, vlm_config, dummy_image):
        vlm = VLMFallback(config=vlm_config)
        vlm.cfg.provider = "local"
        fake_resp = {"message": {"content": json.dumps({"game_phase": "rewards", "confidence": 0.7})}}

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = fake_resp
            mock_post.return_value.raise_for_status = lambda: None
            result = vlm._call_local(dummy_image)

        assert result.game_phase == "rewards"
        assert result.model_used.startswith("local")


# ------------------------------------------------------------------
# VLMVisionAdapter unit tests
# ------------------------------------------------------------------

class MockVisionPort:
    """Fake VisionPort for testing the composite adapter."""

    def __init__(self, confidence: float = 1.0, phase: str = "in_game"):
        self.confidence = confidence
        self.phase = phase
        self.call_count = 0

    def initialize(self) -> bool:
        return True

    def capture_and_perceive(self):
        self.call_count += 1
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return GameStateSnapshot(
            screenshot=img,
            game_phase=self.phase,
            metadata={"state_confidence": self.confidence},
        )

    def get_detected_objects(self, class_filter=None):
        return []

    def health_check(self):
        return {"ok": True}

    def shutdown(self):
        pass


class TestVLMVisionAdapter:
    def test_no_fallback_when_confidence_high(self, vlm_config):
        primary = MockVisionPort(confidence=0.9, phase="in_game")
        vlm = VLMFallback(config=vlm_config)
        adapter = VLMVisionAdapter(primary=primary, vlm=vlm)

        with patch.object(vlm, "analyze") as mock_vlm:
            snapshot = adapter.capture_and_perceive()

        mock_vlm.assert_not_called()
        assert snapshot.game_phase == "in_game"

    def test_fallback_when_confidence_low(self, vlm_config):
        primary = MockVisionPort(confidence=0.2, phase="unknown")
        vlm = VLMFallback(config=vlm_config)
        adapter = VLMVisionAdapter(primary=primary, vlm=vlm)

        fake_vlm_result = VLMResult(
            game_phase="combat",
            confidence=0.8,
            detected_elements=[{"type": "enemy", "x_norm": 0.5, "y_norm": 0.5}],
        )

        with patch.object(vlm, "analyze", return_value=fake_vlm_result):
            snapshot = adapter.capture_and_perceive()

        assert snapshot.game_phase == "combat"
        assert "vlm_phase_override" in snapshot.metadata
        assert len(snapshot.detected_objects) == 1
        assert snapshot.detected_objects[0].class_name == "enemy"

    def test_fallback_vlm_fails_gracefully(self, vlm_config):
        primary = MockVisionPort(confidence=0.2, phase="unknown")
        vlm = VLMFallback(config=vlm_config)
        adapter = VLMVisionAdapter(primary=primary, vlm=vlm)

        with patch.object(vlm, "analyze", return_value=VLMResult(error="timeout")):
            snapshot = adapter.capture_and_perceive()

        # Should return primary snapshot unchanged
        assert snapshot.game_phase == "unknown"

    def test_health_check_includes_vlm_stats(self, vlm_config):
        primary = MockVisionPort(confidence=0.9)
        vlm = VLMFallback(config=vlm_config)
        adapter = VLMVisionAdapter(primary=primary, vlm=vlm)

        health = adapter.health_check()
        assert health["vlm_enabled"] is True
        assert "vlm_stats" in health

    def test_disabled_vlm_skips_analysis(self, dummy_image):
        primary = MockVisionPort(confidence=0.1)
        vlm = VLMFallback(config=VLMConfig(enabled=False))
        adapter = VLMVisionAdapter(primary=primary, vlm=vlm)

        with patch.object(vlm, "analyze") as mock_vlm:
            snapshot = adapter.capture_and_perceive()

        mock_vlm.assert_not_called()
        assert snapshot.game_phase == "in_game"

    def test_merge_elements_format_error(self, vlm_config):
        """Ensure bad element data doesn't crash the adapter."""
        primary = MockVisionPort(confidence=0.1)
        vlm = VLMFallback(config=vlm_config)
        adapter = VLMVisionAdapter(primary=primary, vlm=vlm)

        fake_vlm_result = VLMResult(
            game_phase="combat",
            confidence=0.8,
            detected_elements=[
                {"type": "bad_element", "x_norm": "not_a_number"},
            ],
        )

        with patch.object(vlm, "analyze", return_value=fake_vlm_result):
            snapshot = adapter.capture_and_perceive()

        # Should not crash; element with bad data is skipped
        assert snapshot.game_phase == "combat"
