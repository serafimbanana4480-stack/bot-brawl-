"""
core/adaptive_screenshot.py

Adaptive Screenshot Cache for Brawl Stars bot.

Solves the "screenshot cache not adaptive" problem. The current cache
likely uses a fixed 150ms TTL, which means:
- In combat: stale data (enemies move fast, 150ms = outdated positions)
- In lobby: wasted captures (screen barely changes, 150ms = too short)

This module provides adaptive caching:
- Combat mode: 30-50ms TTL (fresh data critical for aiming)
- Transition mode: 80-100ms TTL (loading screens, state changes)
- Lobby mode: 200-500ms TTL (screen barely changes)
- Menu mode: 300-600ms TTL (static screens)

Also provides:
- Frame deduplication (skip identical frames)
- Priority-based capture (important events get fresh screenshots)
- Integration with game state for automatic mode switching

Usage:
    cache = AdaptiveScreenshotCache()

    # Each frame:
    screenshot = cache.get_screenshot(capture_fn, game_state="in_game")

    # Or with priority:
    screenshot = cache.get_screenshot(capture_fn, game_state="in_game", priority="high")
"""

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CacheMode(Enum):
    """Screenshot cache modes based on game state."""
    COMBAT = "combat"          # In-game, enemies visible
    IN_GAME = "in_game"        # In-game, no enemies (safe)
    TRANSITION = "transition"  # Loading, state changes
    LOBBY = "lobby"           # Lobby, brawler select
    MENU = "menu"             # Static menus, popups


# TTL (time-to-live) in milliseconds for each mode
MODE_TTL = {
    CacheMode.COMBAT: 40,       # Very fresh data needed
    CacheMode.IN_GAME: 80,     # Moderate freshness
    CacheMode.TRANSITION: 100,  # State changes
    CacheMode.LOBBY: 250,      # Slow changes
    CacheMode.MENU: 400,       # Nearly static
}

# Priority overrides: high priority always gets fresh screenshot
PRIORITY_TTL = {
    "critical": 0,     # Always fresh
    "high": 20,        # Very fresh
    "normal": None,    # Use mode default
    "low": None,       # Use mode default
}


@dataclass
class CacheEntry:
    """A cached screenshot entry."""
    screenshot: np.ndarray = None
    timestamp: float = 0.0
    mode: CacheMode = CacheMode.LOBBY
    hash_value: str = ""
    capture_time_ms: float = 0.0


class AdaptiveScreenshotCache:
    """
    Adaptive screenshot cache that adjusts TTL based on game state.

    Features:
    - Automatic mode detection from game state
    - Priority-based capture (important events get fresh data)
    - Frame deduplication (skip identical frames)
    - Statistics tracking
    - Thread-safe
    """

    # Game state → cache mode mapping
    STATE_MODE_MAP = {
        "in_game": CacheMode.IN_GAME,
        "in_game_combat": CacheMode.COMBAT,
        "loading": CacheMode.TRANSITION,
        "lobby": CacheMode.LOBBY,
        "menu": CacheMode.MENU,
        "brawler_select": CacheMode.LOBBY,
        "victory": CacheMode.TRANSITION,
        "defeat": CacheMode.TRANSITION,
        "unknown": CacheMode.LOBBY,
    }

    def __init__(self, default_ttl_ms: float = 150.0):
        self._default_ttl = default_ttl_ms / 1000.0  # Convert to seconds
        self._cache: CacheEntry | None = None
        self._current_mode: CacheMode = CacheMode.LOBBY
        self._lock = threading.RLock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._dedup_skips = 0
        self._total_captures = 0
        self._avg_capture_time_ms = 0.0
        self._capture_times: list = []

        # Custom TTL overrides
        self._custom_ttls: dict[CacheMode, float] = {}

        logger.info("[ADAPTIVE_SCREENSHOT] Initialized (default_ttl=%.0fms)",
                     default_ttl_ms)

    def set_custom_ttl(self, mode: CacheMode, ttl_ms: float):
        """Override TTL for a specific cache mode."""
        self._custom_ttls[mode] = ttl_ms / 1000.0

    def get_screenshot(self, capture_fn: Callable[[], np.ndarray],
                       game_state: str = "lobby",
                       priority: str = "normal",
                       enemies_visible: bool = False) -> np.ndarray:
        """
        Get a screenshot, using cache when appropriate.

        Args:
            capture_fn: Function that captures a new screenshot
            game_state: Current game state string
            priority: "critical", "high", "normal", "low"
            enemies_visible: Whether enemies are currently visible

        Returns:
            Screenshot as numpy array
        """
        # Determine cache mode
        mode = self.STATE_MODE_MAP.get(game_state, CacheMode.LOBBY)

        # Upgrade to combat mode if enemies visible
        if mode == CacheMode.IN_GAME and enemies_visible:
            mode = CacheMode.COMBAT

        self._current_mode = mode

        # Check priority override
        priority_ttl = PRIORITY_TTL.get(priority)
        if priority_ttl is not None and priority_ttl == 0:
            # Critical priority — always fresh capture
            return self._capture_new(capture_fn, mode)

        # Get TTL for current mode
        ttl = self._get_ttl(mode, priority)

        with self._lock:
            # Check cache validity
            if self._cache is not None:
                age = time.time() - self._cache.timestamp

                if age < ttl and self._cache.screenshot is not None:
                    # Cache hit
                    self._hits += 1
                    return self._cache.screenshot

            # Cache miss — need new capture
            self._misses += 1

        return self._capture_new(capture_fn, mode)

    def invalidate(self):
        """Force invalidate the cache (next call will capture fresh)."""
        with self._lock:
            self._cache = None

    def get_current_mode(self) -> CacheMode:
        """Get current cache mode."""
        return self._current_mode

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / max(1, total) * 100

            return {
                "current_mode": self._current_mode.value,
                "current_ttl_ms": round(self._get_ttl(self._current_mode) * 1000),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 1),
                "dedup_skips": self._dedup_skips,
                "avg_capture_ms": round(self._avg_capture_time_ms, 1),
                "total_captures": self._total_captures,
            }

    # --- Internal ---

    def _get_ttl(self, mode: CacheMode, priority: str = "normal") -> float:
        """Get TTL in seconds for a given mode and priority."""
        # Priority override
        priority_ttl = PRIORITY_TTL.get(priority)
        if priority_ttl is not None:
            return priority_ttl / 1000.0

        # Custom TTL
        if mode in self._custom_ttls:
            return self._custom_ttls[mode]

        # Mode default
        ttl_ms = MODE_TTL.get(mode, 150)
        return ttl_ms / 1000.0

    def _capture_new(self, capture_fn: Callable, mode: CacheMode) -> np.ndarray:
        """Capture a new screenshot and update cache."""
        start = time.time()

        try:
            screenshot = capture_fn()
        except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[ADAPTIVE_SCREENSHOT] Capture failed: %s", e)
            # Return cached screenshot if available
            if self._cache and self._cache.screenshot is not None:
                return self._cache.screenshot
            raise

        capture_time = (time.time() - start) * 1000

        # Compute hash for deduplication
        hash_value = ""
        if screenshot is not None:
            try:
                # Downsample for faster hashing
                small = screenshot[::10, ::10] if screenshot.size > 10000 else screenshot
                hash_value = hashlib.md5(small.tobytes()).hexdigest()[:16]
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

        with self._lock:
            # Check for deduplication
            if (self._cache is not None and
                self._cache.hash_value == hash_value and
                hash_value != ""):
                # Same frame — extend cache lifetime
                self._dedup_skips += 1
                self._cache.timestamp = time.time()  # Refresh timestamp
                return self._cache.screenshot

            # Update cache
            self._cache = CacheEntry(
                screenshot=screenshot,
                timestamp=time.time(),
                mode=mode,
                hash_value=hash_value,
                capture_time_ms=capture_time,
            )

            # Update stats
            self._total_captures += 1
            self._capture_times.append(capture_time)
            if len(self._capture_times) > 50:
                self._capture_times = self._capture_times[-50:]
            self._avg_capture_time_ms = (
                sum(self._capture_times) / len(self._capture_times)
            )

        return screenshot
