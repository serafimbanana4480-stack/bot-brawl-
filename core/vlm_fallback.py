"""
core/vlm_fallback.py

Vision Language Model (VLM) fallback for anti-UI-change resilience.

When template matching, YOLO, and pixel heuristics all fail or have low
confidence (e.g., after a game update changed the UI), this module sends
the screenshot to a VLM and asks it to identify the current game phase
and key UI elements.

Supported backends (priority order):
    1. OpenAI GPT-4o / GPT-4V (via openai SDK)
    2. Anthropic Claude 3 (via anthropic SDK)
    3. Google Gemini (via google.generativeai)
    4. Local Qwen2-VL / LLaVA (via HTTP API or transformers)

Design goals:
    - Never block the main loop (> 500ms timeout)
    - Aggressive caching (same screenshot hash → reuse answer)
    - Cost guard (max calls per minute, max daily budget)
    - Graceful degradation (if VLM fails, return low-confidence fallback)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VLMResult:
    """Structured result from a VLM inference."""
    game_phase: str = "unknown"
    confidence: float = 0.0
    detected_elements: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    model_used: str = "none"
    latency_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None


@dataclass
class VLMConfig:
    """Configuration for VLM fallback behavior."""
    enabled: bool = False
    provider: str = "openai"  # openai | anthropic | google | local
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout_seconds: float = 3.0
    max_calls_per_minute: int = 10
    max_calls_per_hour: int = 100
    cache_ttl_seconds: float = 30.0
    fallback_threshold: float = 0.35  # trigger VLM when primary confidence < this
    max_image_size: Tuple[int, int] = (512, 512)  # resize before encoding


class VLMFallback:
    """
    High-level VLM fallback client with caching, rate limiting, and
    multi-provider support.
    """

    _SYSTEM_PROMPT = (
        "You are analyzing a screenshot from the mobile game Brawl Stars. "
        "Respond ONLY with valid JSON in this exact schema:\n"
        '{"game_phase": "lobby|matchmaking|loading|countdown|combat|victory|defeat|rewards|unknown", '
        '"confidence": 0.0-1.0, '
        '"elements": [{"type": "play_button|brawler_icon|enemy|super_button|joystick|health_bar", '
        '"x_norm": 0.0-1.0, "y_norm": 0.0-1.0, "description": "..."}]}'
    )

    def __init__(self, config: Optional[VLMConfig] = None):
        self.cfg = config or VLMConfig()
        self._call_history: List[float] = []  # timestamps of API calls
        self._cache: Dict[str, Tuple[VLMResult, float]] = {}  # hash -> (result, timestamp)
        # Providers resolved dynamically so monkey-patching works in tests

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_fallback(
        self,
        primary_confidence: float,
        frames_without_detection: int = 0,
    ) -> bool:
        """Return True if the VLM fallback should be triggered."""
        if not self.cfg.enabled:
            return False
        if primary_confidence < self.cfg.fallback_threshold:
            return True
        if frames_without_detection >= 5:
            return True
        return False

    def analyze(self, screenshot: np.ndarray) -> VLMResult:
        """
        Analyze a screenshot with the VLM.
        Returns cached result if available and not expired.
        """
        t0 = time.time()
        img_hash = self._hash_image(screenshot)

        # Cache hit?
        if img_hash in self._cache:
            result, ts = self._cache[img_hash]
            if time.time() - ts < self.cfg.cache_ttl_seconds:
                from dataclasses import replace
                cached_result = replace(result, cached=True, latency_ms=(time.time() - t0) * 1000)
                logger.debug("[VLM] Cache hit for hash %s...", img_hash[:8])
                return cached_result

        # Rate limit check
        if not self._check_rate_limits():
            logger.warning("[VLM] Rate limit exceeded — skipping VLM call")
            return VLMResult(
                error="rate_limited",
                confidence=0.0,
                latency_ms=(time.time() - t0) * 1000,
            )

        # Dispatch to provider (resolved dynamically for testability)
        provider_method = getattr(self, f"_call_{self.cfg.provider}", self._call_openai)
        try:
            result = provider_method(screenshot)
        except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[VLM] Provider %s failed: %s", self.cfg.provider, e)
            result = VLMResult(error=str(e), confidence=0.0)

        result.latency_ms = (time.time() - t0) * 1000
        if not result.error:
            self._cache[img_hash] = (result, time.time())
            self._call_history.append(time.time())

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        now = time.time()
        recent = [t for t in self._call_history if now - t < 3600]
        return {
            "enabled": self.cfg.enabled,
            "provider": self.cfg.provider,
            "total_calls": len(self._call_history),
            "calls_last_hour": len(recent),
            "cache_size": len(self._cache),
            "rate_limited_now": not self._check_rate_limits(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_image(img: np.ndarray) -> str:
        """Fast perceptual hash for caching."""
        small = img[::8, ::8] if img.shape[0] > 64 else img
        return hashlib.sha1(small.tobytes()).hexdigest()[:16]

    def _check_rate_limits(self) -> bool:
        now = time.time()
        self._call_history = [t for t in self._call_history if now - t < 3600]
        per_minute = len([t for t in self._call_history if now - t < 60])
        per_hour = len(self._call_history)
        return (
            per_minute < self.cfg.max_calls_per_minute
            and per_hour < self.cfg.max_calls_per_hour
        )

    def _encode_image(self, img: np.ndarray) -> str:
        """Resize and base64-encode an image for API transport."""
        import cv2
        h, w = img.shape[:2]
        target_w, target_h = self.cfg.max_image_size
        if w > target_w or h > target_h:
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode("ascii")

    def _parse_json_response(self, text: str) -> VLMResult:
        """Extract JSON from LLM response (handles markdown fences)."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove fence markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("[VLM] JSON parse failed: %s", e)
            return VLMResult(
                raw_response=text,
                error=f"json_parse: {e}",
                confidence=0.0,
            )

        elements = data.get("elements", [])
        if not isinstance(elements, list):
            elements = []

        return VLMResult(
            game_phase=data.get("game_phase", "unknown"),
            confidence=float(data.get("confidence", 0.0)),
            detected_elements=elements,
            raw_response=text,
        )

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_openai(self, img: np.ndarray) -> VLMResult:
        import requests
        api_key = self.cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        b64 = self._encode_image(img)
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "What is the current game phase and where are "
                                "the key UI elements?"
                            ),
                        },
                    ],
                },
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        url = self.cfg.api_base or "https://api.openai.com/v1/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.cfg.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        result = self._parse_json_response(text)
        result.model_used = f"openai:{self.cfg.model}"
        return result

    def _call_anthropic(self, img: np.ndarray) -> VLMResult:
        import requests
        api_key = self.cfg.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        b64 = self._encode_image(img)
        payload = {
            "model": self.cfg.model or "claude-3-haiku-20240307",
            "max_tokens": 256,
            "system": self._SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "What is the current game phase and where are "
                                "the key UI elements?"
                            ),
                        },
                    ],
                }
            ],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=self.cfg.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        result = self._parse_json_response(text)
        result.model_used = f"anthropic:{self.cfg.model}"
        return result

    def _call_google(self, img: np.ndarray) -> VLMResult:
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("google.generativeai not installed")

        api_key = self.cfg.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.cfg.model or "gemini-1.5-flash")
        import cv2
        _, buf = cv2.imencode(".png", img)
        response = model.generate_content(
            [self._SYSTEM_PROMPT, buf.tobytes()],
            generation_config={"max_output_tokens": 256, "temperature": 0.0},
            request_options={"timeout": int(self.cfg.timeout_seconds * 1000)},
        )
        text = response.text
        result = self._parse_json_response(text)
        result.model_used = f"google:{self.cfg.model}"
        return result

    def _call_local(self, img: np.ndarray) -> VLMResult:
        """Call a local VLM via HTTP (e.g., Ollama, vLLM, llama-cpp)."""
        import requests
        b64 = self._encode_image(img)
        base_url = self.cfg.api_base or "http://localhost:11434"
        payload = {
            "model": self.cfg.model or "llava",
            "messages": [
                {
                    "role": "user",
                    "content": self._SYSTEM_PROMPT,
                    "images": [b64],
                }
            ],
            "stream": False,
        }
        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=self.cfg.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"]
        result = self._parse_json_response(text)
        result.model_used = f"local:{self.cfg.model}"
        return result
