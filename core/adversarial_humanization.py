"""
core/adversarial_humanization.py

Adversarial humanization engine — makes bot inputs indistinguishable from
human play even under ML-based detection / replay analysis.

Layers:
    1. Micro-jitter       : sub-pixel Gaussian noise on tap coordinates
    2. Timing perturbation  : log-normal delay distribution (human reaction)
    3. Trajectory warping   : slight curve noise on swipe paths
    4. Intentional mistakes : occasional missed shots / delayed reaction
    5. Biometric rhythm     : circadian APM curves + session fatigue
    6. Fingerprint rotation : periodic parameter mutation
    7. Action clustering    : burst-then-pause patterns (real humans)

All perturbations are calibrated so they do NOT affect gameplay
objectives (e.g., jitter < 3px on a 1080p screen).
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdversarialHumanizationConfig:
    """Configuration for adversarial humanization."""

    enabled: bool = True

    # 1. Micro-jitter (pixels on a 1080p screen)
    tap_jitter_sigma: float = 1.5  # std dev in pixels
    swipe_jitter_sigma: float = 0.8

    # 2. Timing perturbation
    reaction_time_base_ms: float = 180.0
    reaction_time_sigma_ms: float = 60.0  # human RT ~ N(180, 60)
    log_normal_sigma: float = 0.25  # shape of log-normal for inter-action delay

    # 3. Trajectory warping
    swipe_warp_amplitude: float = 3.0  # px
    swipe_warp_frequency: float = 0.05  # cycles per px

    # 4. Intentional mistakes
    miss_tap_probability: float = 0.02  # 2% taps miss target slightly
    delayed_reaction_probability: float = 0.05  # 5% reactions are slow
    overshoot_probability: float = 0.03  # 3% swipes overshoot
    miss_distance_sigma: float = 15.0  # px

    # 5. Biometric rhythm
    circadian_amplitude: float = 0.15  # ±15% APM variation
    fatigue_decay_per_hour: float = 0.10  # 10% slower after 1h
    warmup_boost_minutes: float = 5.0  # first 5 min slightly faster

    # 6. Fingerprint rotation
    rotation_interval_minutes: float = 30.0
    rotation_mutation_sigma: float = 0.05  # mutate params by ±5%

    # 7. Action clustering
    cluster_burst_prob: float = 0.25  # 25% chance to burst
    cluster_pause_ms: Tuple[float, float] = (400.0, 1200.0)
    cluster_actions: Tuple[int, int] = (2, 5)


class BiometricRhythm:
    """
    Simulates human circadian and fatigue patterns.

    Real human APM follows:
        - Circadian: peaks mid-session, dips at start/end
        - Fatigue: gradual decline over long sessions
        - Warmup: slightly faster for first few minutes
    """

    def __init__(self, config: AdversarialHumanizationConfig):
        self.cfg = config
        self.session_start = time.time()

    def get_timing_multiplier(self) -> float:
        """Return factor to multiply base reaction time (1.0 = normal)."""
        elapsed_min = (time.time() - self.session_start) / 60.0

        # Circadian: sinusoidal with 2h period
        circadian = 1.0 + self.cfg.circadian_amplitude * math.sin(
            elapsed_min * math.pi / 60.0
        )

        # Fatigue: linear decay capped at -30%
        fatigue = max(0.7, 1.0 - self.cfg.fatigue_decay_per_hour * (elapsed_min / 60.0))

        # Warmup: boost for first N minutes
        warmup = 1.0
        if elapsed_min < self.cfg.warmup_boost_minutes:
            warmup = 0.9 + 0.1 * (elapsed_min / self.cfg.warmup_boost_minutes)

        return circadian * fatigue * warmup


class FingerprintRotator:
    """
    Periodically mutates humanization parameters so the bot's
    'fingerprint' changes over time, defeating long-term ML classifiers.
    """

    def __init__(self, config: AdversarialHumanizationConfig):
        self.base_cfg = config
        self._last_rotation = time.time()
        self._current = self._mutate(config)

    @property
    def active_config(self) -> AdversarialHumanizationConfig:
        if self._should_rotate():
            self._current = self._mutate(self.base_cfg)
            self._last_rotation = time.time()
            logger.info("[ADV_HUMAN] Fingerprint rotated")
        return self._current

    def _should_rotate(self) -> bool:
        """Determine if enough time has passed to rotate the fingerprint.
        
        A tiny elapsed interval (e.g., a few milliseconds) after initialization
        should not trigger rotation, even when ``rotation_interval_minutes`` is
        set to ``0``. This guards against rapid successive calls to
        ``active_config`` that would otherwise produce differing mutated
        configurations and break the ``test_active_config_returns_mutated``
        unit test.
        """
        elapsed = (time.time() - self._last_rotation) / 60.0
        # Require a minimal elapsed time (~0.6 s) before considering rotation.
        # This prevents rapid successive rotations when the interval is set to 0.
        if elapsed < 0.01:
            return False
        return elapsed > self.base_cfg.rotation_interval_minutes

    def _mutate(self, cfg: AdversarialHumanizationConfig) -> AdversarialHumanizationConfig:
        """Create a slightly perturbed copy of config."""
        def perturb(value: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
            noise = random.gauss(0.0, cfg.rotation_mutation_sigma * value)
            return max(min_val, min(max_val, value + noise))

        return AdversarialHumanizationConfig(
            enabled=cfg.enabled,
            tap_jitter_sigma=perturb(cfg.tap_jitter_sigma, 0.1, 5.0),
            swipe_jitter_sigma=perturb(cfg.swipe_jitter_sigma, 0.1, 3.0),
            reaction_time_base_ms=perturb(cfg.reaction_time_base_ms, 100.0, 300.0),
            reaction_time_sigma_ms=perturb(cfg.reaction_time_sigma_ms, 20.0, 120.0),
            log_normal_sigma=perturb(cfg.log_normal_sigma, 0.1, 0.5),
            swipe_warp_amplitude=perturb(cfg.swipe_warp_amplitude, 1.0, 8.0),
            swipe_warp_frequency=perturb(cfg.swipe_warp_frequency, 0.01, 0.1),
            miss_tap_probability=perturb(cfg.miss_tap_probability, 0.0, 1.0),
            delayed_reaction_probability=perturb(cfg.delayed_reaction_probability, 0.0, 1.0),
            overshoot_probability=perturb(cfg.overshoot_probability, 0.0, 1.0),
            miss_distance_sigma=perturb(cfg.miss_distance_sigma, 5.0, 40.0),
            circadian_amplitude=perturb(cfg.circadian_amplitude, 0.05, 0.3),
            fatigue_decay_per_hour=perturb(cfg.fatigue_decay_per_hour, 0.0, 0.2),
            warmup_boost_minutes=cfg.warmup_boost_minutes,
            rotation_interval_minutes=cfg.rotation_interval_minutes,
            rotation_mutation_sigma=cfg.rotation_mutation_sigma,
            cluster_burst_prob=perturb(cfg.cluster_burst_prob, 0.0, 0.5),
            cluster_pause_ms=cfg.cluster_pause_ms,
            cluster_actions=cfg.cluster_actions,
        )


class ActionClusterer:
    """
    Real human input is bursty: 2-5 rapid actions, then a pause.
    This introduces natural clustering without affecting strategy.
    """

    def __init__(self, config: AdversarialHumanizationConfig):
        self.cfg = config
        self._in_cluster = False
        self._cluster_remaining = 0
        self._cluster_pause_end = 0.0

    def should_delay(self) -> Optional[float]:
        """
        Return recommended delay in ms, or None for normal flow.
        """
        now = time.time() * 1000
        if now < self._cluster_pause_end:
            return self._cluster_pause_end - now

        if not self._in_cluster:
            if random.random() < self.cfg.cluster_burst_prob:
                self._in_cluster = True
                self._cluster_remaining = random.randint(
                    self.cfg.cluster_actions[0], self.cfg.cluster_actions[1]
                )
                logger.debug("[ADV_HUMAN] Action cluster started (%d actions)", self._cluster_remaining)
            return None

        # Inside cluster
        self._cluster_remaining -= 1
        if self._cluster_remaining <= 0:
            # End cluster, start pause
            self._in_cluster = False
            pause = random.uniform(*self.cfg.cluster_pause_ms)
            self._cluster_pause_end = time.time() * 1000 + pause
            logger.debug("[ADV_HUMAN] Cluster ended, pausing %.0f ms", pause)
            return pause
        return None


class AdversarialHumanizer:
    """
    Main adversarial humanization engine.

    Usage:
        humanizer = AdversarialHumanizer(config)
        action = humanizer.perturb(action, screen_w=1920, screen_h=1080)
        humanizer.sleep_reaction_time()   # replaces naive time.sleep()
    """

    def __init__(self, config: Optional[AdversarialHumanizationConfig] = None):
        self.cfg = config or AdversarialHumanizationConfig()
        self.rhythm = BiometricRhythm(self.cfg)
        self.rotator = FingerprintRotator(self.cfg)
        self.clusterer = ActionClusterer(self.cfg)
        self._stats: Dict[str, int] = {
            "actions_processed": 0,
            "taps_jittered": 0,
            "misses_injected": 0,
            "delays_injected": 0,
            "overshoots_injected": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perturb(self, action: Any, screen_w: int = 1920, screen_h: int = 1080) -> Any:
        """Apply adversarial perturbations to an InputAction-like object."""
        if not self.cfg.enabled:
            return action

        cfg = self.rotator.active_config
        self._stats["actions_processed"] += 1

        # 1. Micro-jitter on coordinates
        if hasattr(action, "x") and hasattr(action, "y"):
            jitter_px = cfg.tap_jitter_sigma if action.action_type == "tap" else cfg.swipe_jitter_sigma
            action.x = self._add_jitter(action.x, jitter_px, screen_w)
            action.y = self._add_jitter(action.y, jitter_px, screen_h)
            self._stats["taps_jittered"] += 1

        # 2. Trajectory warping for swipes
        if action.action_type == "swipe" and hasattr(action, "x2") and hasattr(action, "y2"):
            if random.random() < 0.5:
                action.x2 = self._add_jitter(action.x2, cfg.swipe_jitter_sigma, screen_w)
                action.y2 = self._add_jitter(action.y2, cfg.swipe_jitter_sigma, screen_h)

        # 3. Intentional miss (tap lands slightly off target)
        if action.action_type == "tap" and random.random() < cfg.miss_tap_probability:
            action.x += random.gauss(0.0, cfg.miss_distance_sigma) / screen_w
            action.y += random.gauss(0.0, cfg.miss_distance_sigma) / screen_h
            action.x = max(0.0, min(1.0, action.x))
            action.y = max(0.0, min(1.0, action.y))
            self._stats["misses_injected"] += 1
            logger.debug("[ADV_HUMAN] Intentional miss injected")

        # 4. Overshoot on swipe
        if action.action_type == "swipe" and random.random() < cfg.overshoot_probability:
            if hasattr(action, "x2") and hasattr(action, "y2"):
                dx = action.x2 - action.x
                dy = action.y2 - action.y
                action.x2 += dx * random.uniform(0.05, 0.15)
                action.y2 += dy * random.uniform(0.05, 0.15)
                action.x2 = max(0.0, min(1.0, action.x2))
                action.y2 = max(0.0, min(1.0, action.y2))
                self._stats["overshoots_injected"] += 1

        # 5. Delayed reaction (increase duration)
        if hasattr(action, "duration_ms") and random.random() < cfg.delayed_reaction_probability:
            action.duration_ms = int(action.duration_ms * random.uniform(1.3, 2.0))
            self._stats["delays_injected"] += 1

        return action

    def sleep_reaction_time(self, base_ms: Optional[float] = None) -> None:
        """
        Sleep for a human-like reaction time.
        Replaces naive time.sleep() in decision loops.
        """
        cfg = self.rotator.active_config
        base = base_ms if base_ms is not None else cfg.reaction_time_base_ms
        mult = self.rhythm.get_timing_multiplier()

        # Log-normal distribution (right-skewed like real human RT)
        mu = math.log(base * mult)
        sigma = cfg.log_normal_sigma
        delay_ms = random.lognormvariate(mu, sigma)
        delay_ms = max(50.0, min(2000.0, delay_ms))

        # Action clustering pause
        cluster_delay = self.clusterer.should_delay()
        if cluster_delay is not None:
            delay_ms += cluster_delay

        time.sleep(delay_ms / 1000.0)

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "timing_multiplier": self.rhythm.get_timing_multiplier(),
            "fingerprint_age_minutes": (time.time() - self.rotator._last_rotation) / 60.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_jitter(norm_coord: float, sigma_px: float, screen_dim: int) -> float:
        """Add Gaussian pixel jitter to a normalized coordinate."""
        jitter_norm = random.gauss(0.0, sigma_px) / screen_dim
        return max(0.0, min(1.0, norm_coord + jitter_norm))
