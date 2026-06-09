"""
core/repetition_guard.py

Repetition Guard — extracted from core/lobby_fsm.py (now deprecated).

Guards against repetitive actions and stuck loops:
- Action cooldown: prevents repeating the same action too quickly
- Loop detection: detects when the same sequence of actions repeats
- Stuck recovery: forces state reset when loop detected
- Action history tracking
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ActionRecord:
    """Record of a performed action."""
    action: str
    target: str = ""         # What was clicked/interacted with
    timestamp: float = 0.0
    result: str = ""         # "success", "fail", "timeout"


class RepetitionGuard:
    """
    Guards against repetitive actions and stuck loops.

    Features:
    - Action cooldown: prevents repeating the same action too quickly
    - Loop detection: detects when the same sequence of actions repeats
    - Stuck recovery: forces state reset when loop detected
    - Action history tracking
    """

    # Minimum time between identical actions (seconds)
    DEFAULT_COOLDOWN = 2.0

    # Action-specific cooldowns
    ACTION_COOLDOWNS = {
        "click_play": 3.0,
        "select_brawler": 2.0,
        "click_confirm": 2.0,
        "close_popup": 1.5,
        "tap_screen": 0.5,
        "swipe": 0.3,
    }

    # Loop detection: if this many identical sequences occur, it's a loop
    LOOP_THRESHOLD = 3

    # Maximum sequence length to check for loops
    MAX_SEQUENCE_CHECK = 6

    def __init__(self):
        self._action_history: List[ActionRecord] = []
        self._cooldown_timers: Dict[str, float] = {}
        self._loop_detected = False
        self._loop_count = 0
        self._lock = threading.RLock()

        logger.info("[REPETITION_GUARD] Initialized")

    def can_execute(self, action: str, target: str = "") -> Tuple[bool, str]:
        """
        Check if an action can be executed (not on cooldown, not in a loop).

        Returns (can_execute, reason).
        """
        with self._lock:
            # Check cooldown
            cooldown = self.ACTION_COOLDOWNS.get(action, self.DEFAULT_COOLDOWN)
            key = f"{action}:{target}" if target else action
            last_time = self._cooldown_timers.get(key, 0)

            now = time.time()
            if now - last_time < cooldown:
                remaining = cooldown - (now - last_time)
                return (False, f"cooldown:{remaining:.1f}s")

            # Check for active loop
            if self._loop_detected:
                return (False, "loop_detected")

            return (True, "ok")

    def record_action(self, action: str, target: str = "",
                      result: str = "success"):
        """
        Record that an action was performed.

        Args:
            action: Action name
            target: Target identifier
            result: "success", "fail", "timeout"
        """
        now = time.time()

        with self._lock:
            # Update cooldown timer
            key = f"{action}:{target}" if target else action
            self._cooldown_timers[key] = now

            # Record action
            self._action_history.append(ActionRecord(
                action=action,
                target=target,
                timestamp=now,
                result=result,
            ))
            if len(self._action_history) > 200:
                self._action_history = self._action_history[-200:]

            # Check for loops
            if result == "fail" or result == "timeout":
                self._check_for_loops()

    def force_cooldown(self, action: str, cooldown_seconds: float):
        """Force a specific cooldown for an action."""
        with self._lock:
            self._cooldown_timers[action] = time.time()
            self.ACTION_COOLDOWNS[action] = cooldown_seconds

    def reset_loop(self):
        """Reset loop detection state."""
        with self._lock:
            self._loop_detected = False
            self._loop_count = 0
            logger.info("[REPETITION_GUARD] Loop state reset")

    def is_loop_detected(self) -> bool:
        """Check if a loop is currently detected."""
        with self._lock:
            return self._loop_detected

    def get_loop_info(self) -> Dict:
        """Get information about detected loops."""
        with self._lock:
            return {
                "loop_detected": self._loop_detected,
                "loop_count": self._loop_count,
                "recent_actions": [
                    {"action": a.action, "target": a.target, "result": a.result}
                    for a in self._action_history[-10:]
                ],
            }

    def get_stats(self) -> Dict:
        """Get guard statistics."""
        with self._lock:
            total = len(self._action_history)
            failures = sum(1 for a in self._action_history if a.result != "success")

            return {
                "total_actions": total,
                "failures": failures,
                "failure_rate": round(failures / max(1, total) * 100, 1),
                "loops_detected": self._loop_count,
                "active_cooldowns": len(self._cooldown_timers),
            }

    # --- Internal ---

    def _check_for_loops(self):
        """Check recent action history for repeating sequences."""
        if len(self._action_history) < 4:
            return

        # Get recent failed actions
        recent = self._action_history[-self.MAX_SEQUENCE_CHECK * 2:]

        # Try different sequence lengths
        for seq_len in range(2, self.MAX_SEQUENCE_CHECK + 1):
            if len(recent) < seq_len * self.LOOP_THRESHOLD:
                continue

            # Get the last sequence
            last_seq = [(a.action, a.target) for a in recent[-seq_len:]]

            # Check if this sequence repeats
            matches = 0
            for i in range(len(recent) - seq_len, -1, -seq_len):
                check_seq = [(a.action, a.target) for a in recent[i:i + seq_len]]
                if check_seq == last_seq:
                    matches += 1
                else:
                    break

            if matches >= self.LOOP_THRESHOLD:
                self._loop_detected = True
                self._loop_count += 1
                logger.warning(
                    "[REPETITION_GUARD] Loop detected! Sequence of %d actions "
                    "repeated %d times: %s",
                    seq_len, matches, last_seq
                )
                return
