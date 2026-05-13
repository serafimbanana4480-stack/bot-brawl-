"""
safety_gate.py

Runtime safety gate that sits between the RL policy and the ADB executor.
Blocks or throttles actions that would be flagged as bot-like by anti-cheat.

This is FUNCTIONAL (not a stub) — it enforces hard limits regardless of
what the policy outputs.

Checks performed before every action:
  1. Action rate limit (APM cap)
  2. Perfect-aim detection guard (small forced variance on attack coords)
  3. Minimum inter-action delay (reaction time floor)
  4. Session length limit (auto-pause after X minutes)
  5. Cooldown after suspicious patterns (rapid attack bursts)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    max_apm: int = 60               # max actions per minute
    min_reaction_ms: int = 180      # minimum ms between any two actions
    max_session_minutes: float = 90 # session auto-pause threshold
    break_duration_range: Tuple[float, float] = (30.0, 90.0)  # seconds
    aim_variance_px: float = 5.0    # forced pixel variance on attack targets
    burst_threshold: int = 5        # actions in burst window triggers cooldown
    burst_window_sec: float = 1.0   # burst detection window
    burst_cooldown_sec: float = 2.0 # forced pause after burst detected


@dataclass
class SafetyState:
    session_start: float = field(default_factory=time.time)
    last_action_time: float = 0.0
    action_times: list = field(default_factory=list)  # rolling 60s window
    forced_break_until: float = 0.0
    total_actions: int = 0
    total_blocked: int = 0
    total_modified: int = 0


class SafetyGate:
    """
    Runtime safety filter for bot actions.

    Usage:
        gate = SafetyGate()
        allowed, modified_action = gate.check(action)
        if allowed:
            executor.run(modified_action)
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        self.state = SafetyState()

    def check(self, action: Dict) -> Tuple[bool, Dict]:
        """
        Evaluate an action before execution.

        Args:
            action: dict with keys like 'type', 'target_x', 'target_y', etc.

        Returns:
            (allowed: bool, modified_action: dict)
            If allowed=False, the caller MUST NOT execute the action.
        """
        now = time.time()
        modified = dict(action)

        # 1. Forced break check
        if now < self.state.forced_break_until:
            wait = self.state.forced_break_until - now
            logger.debug(f"SafetyGate: action blocked — forced break for {wait:.1f}s more")
            self.state.total_blocked += 1
            return False, modified

        # 2. Session length check
        session_elapsed_min = (now - self.state.session_start) / 60.0
        if session_elapsed_min >= self.config.max_session_minutes:
            break_dur = random.uniform(*self.config.break_duration_range)
            self.state.forced_break_until = now + break_dur
            logger.warning(
                f"SafetyGate: session limit reached ({session_elapsed_min:.1f}min). "
                f"Pausing {break_dur:.0f}s."
            )
            self.state.total_blocked += 1
            return False, modified

        # 3. Minimum reaction time
        if self.state.last_action_time > 0:
            elapsed_ms = (now - self.state.last_action_time) * 1000
            if elapsed_ms < self.config.min_reaction_ms:
                logger.debug(
                    f"SafetyGate: action blocked — too fast ({elapsed_ms:.0f}ms < "
                    f"{self.config.min_reaction_ms}ms)"
                )
                self.state.total_blocked += 1
                return False, modified

        # 4. APM check (rolling 60s window)
        self.state.action_times = [t for t in self.state.action_times if now - t < 60.0]
        if len(self.state.action_times) >= self.config.max_apm:
            logger.debug(f"SafetyGate: action blocked — APM limit ({self.config.max_apm})")
            self.state.total_blocked += 1
            return False, modified

        # 5. Burst detection
        recent = [t for t in self.state.action_times if now - t < self.config.burst_window_sec]
        if len(recent) >= self.config.burst_threshold:
            self.state.forced_break_until = now + self.config.burst_cooldown_sec
            logger.warning(
                f"SafetyGate: burst pattern detected ({len(recent)} actions in "
                f"{self.config.burst_window_sec}s). Cooldown {self.config.burst_cooldown_sec}s."
            )
            self.state.total_blocked += 1
            return False, modified

        # 6. Aim variance injection (anti perfect-aim detection)
        if "target_x" in modified and "target_y" in modified:
            var = self.config.aim_variance_px
            modified["target_x"] += random.gauss(0, var)
            modified["target_y"] += random.gauss(0, var)
            self.state.total_modified += 1

        # Action is allowed — record it
        self.state.action_times.append(now)
        self.state.last_action_time = now
        self.state.total_actions += 1
        return True, modified

    def force_break(self, duration_sec: Optional[float] = None) -> None:
        """Manually trigger a forced break."""
        if duration_sec is None:
            duration_sec = random.uniform(*self.config.break_duration_range)
        self.state.forced_break_until = time.time() + duration_sec
        logger.info(f"SafetyGate: manual forced break of {duration_sec:.0f}s")

    def reset_session(self) -> None:
        """Reset session timer (call after a real break)."""
        self.state.session_start = time.time()
        self.state.action_times.clear()
        self.state.last_action_time = 0.0
        self.state.forced_break_until = 0.0
        logger.info("SafetyGate: session reset")

    def get_stats(self) -> Dict:
        return {
            "total_actions": self.state.total_actions,
            "total_blocked": self.state.total_blocked,
            "total_modified": self.state.total_modified,
            "session_elapsed_min": (time.time() - self.state.session_start) / 60.0,
            "current_apm": len([t for t in self.state.action_times if time.time() - t < 60.0]),
            "in_forced_break": time.time() < self.state.forced_break_until,
        }
