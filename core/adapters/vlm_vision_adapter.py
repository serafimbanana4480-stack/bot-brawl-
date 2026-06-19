"""
core/adapters/vlm_vision_adapter.py

Composite vision adapter that wraps a primary VisionPort and adds
VLM fallback when the primary pipeline has low confidence.

This is the anti-UI-change resilience layer:
    - If YOLO + template matching produce low-confidence results,
      the screenshot is sent to a VLM for semantic understanding.
    - VLM results are merged into the GameStateSnapshot.
    - Caching ensures we don't burn API budget on static frames.
"""

from __future__ import annotations

import logging
from typing import Any

from core.ports.vision_port import (
    DetectedObject,
    GameStateSnapshot,
    VisionPort,
)
from core.vlm_fallback import VLMFallback

logger = logging.getLogger(__name__)


class VLMVisionAdapter(VisionPort):
    """
    Decorator around a primary VisionPort that adds VLM fallback.

    Usage:
        primary = VisionAdapter(screenshot_taker=..., detector=...)
        vlm = VLMFallback(VLMConfig(enabled=True, provider="openai", ...))
        composite = VLMVisionAdapter(primary=primary, vlm=vlm)
    """

    def __init__(
        self,
        primary: VisionPort,
        vlm: VLMFallback,
        merge_elements: bool = True,
        override_phase_on_low_confidence: bool = True,
    ):
        self._primary = primary
        self._vlm = vlm
        self._merge = merge_elements
        self._override_phase = override_phase_on_low_confidence
        self._frames_without_detection = 0
        self._vlm_calls = 0
        self._vlm_hits = 0

    # ------------------------------------------------------------------
    # VisionPort implementation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        ok = self._primary.initialize()
        logger.info("[VLM_VISION] Initialized (primary_ok=%s, vlm_enabled=%s)", ok, self._vlm.cfg.enabled)
        return ok

    def capture_and_perceive(self) -> GameStateSnapshot | None:
        snapshot = self._primary.capture_and_perceive()
        if snapshot is None:
            self._frames_without_detection += 1
            # Even primary failed — try VLM as last resort
            if self._vlm.cfg.enabled:
                return self._try_vlm_fallback(snapshot)
            return None

        self._frames_without_detection = 0
        primary_conf = snapshot.metadata.get("state_confidence", 1.0)

        if self._vlm.should_fallback(primary_confidence=primary_conf):
            return self._try_vlm_fallback(snapshot)

        return snapshot

    def get_detected_objects(self, class_filter: list[str] | None = None) -> list[DetectedObject]:
        return self._primary.get_detected_objects(class_filter)

    def health_check(self) -> dict[str, Any]:
        primary = self._primary.health_check()
        return {
            **primary,
            "vlm_enabled": self._vlm.cfg.enabled,
            "vlm_calls": self._vlm_calls,
            "vlm_hits": self._vlm_hits,
            "vlm_stats": self._vlm.get_stats(),
        }

    def shutdown(self) -> None:
        self._primary.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_vlm_fallback(self, snapshot: GameStateSnapshot | None) -> GameStateSnapshot | None:
        """Run VLM on the snapshot screenshot and merge results."""
        if snapshot is None or snapshot.screenshot is None:
            logger.debug("[VLM_VISION] No screenshot available for VLM fallback")
            return snapshot

        self._vlm_calls += 1
        vlm_result = self._vlm.analyze(snapshot.screenshot)

        if vlm_result.error:
            logger.warning("[VLM_VISION] VLM failed: %s", vlm_result.error)
            return snapshot

        self._vlm_hits += 1
        logger.info(
            "[VLM_VISION] Fallback used: phase=%s conf=%.2f model=%s cached=%s",
            vlm_result.game_phase,
            vlm_result.confidence,
            vlm_result.model_used,
            vlm_result.cached,
        )

        # Build a new merged snapshot
        merged = GameStateSnapshot(
            screenshot=snapshot.screenshot,
            detected_objects=list(snapshot.detected_objects),
            hud=snapshot.hud,
            game_phase=snapshot.game_phase,
            player_pos=snapshot.player_pos,
            timestamp=snapshot.timestamp,
            latency_ms=snapshot.latency_ms + vlm_result.latency_ms,
            resolution=snapshot.resolution,
            metadata=dict(snapshot.metadata),
        )

        # Override game phase if VLM is more confident
        if self._override_phase and vlm_result.confidence > 0.5:
            merged.game_phase = vlm_result.game_phase
            merged.metadata["vlm_phase_override"] = True

        # Merge detected elements from VLM into detected_objects
        if self._merge and vlm_result.detected_elements:
            for elem in vlm_result.detected_elements:
                try:
                    x_norm = float(elem.get("x_norm", 0.5))
                    y_norm = float(elem.get("y_norm", 0.5))
                    w_norm = float(elem.get("w_norm", 0.1))
                    h_norm = float(elem.get("h_norm", 0.1))
                    merged.detected_objects.append(
                        DetectedObject(
                            class_name=elem.get("type", "unknown"),
                            confidence=vlm_result.confidence * 0.9,  # slightly discount VLM
                            bbox=(
                                max(0.0, x_norm - w_norm / 2),
                                max(0.0, y_norm - h_norm / 2),
                                min(1.0, x_norm + w_norm / 2),
                                min(1.0, y_norm + h_norm / 2),
                            ),
                            center=(x_norm, y_norm),
                        )
                    )
                except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                    logger.debug("[VLM_VISION] Element merge error: %s", e)

        merged.metadata["vlm"] = {
            "confidence": vlm_result.confidence,
            "model": vlm_result.model_used,
            "cached": vlm_result.cached,
            "latency_ms": vlm_result.latency_ms,
            "raw_elements": vlm_result.detected_elements,
        }

        return merged
