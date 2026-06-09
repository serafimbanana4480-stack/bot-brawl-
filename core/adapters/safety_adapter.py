"""
core/adapters/safety_adapter.py

Adapter: SafetySystem + AntiBanSystem -> SafetyPort
"""

from __future__ import annotations

import logging
from typing import Any

from core.ports.safety_port import SafetyPort, SafetyStatus

logger = logging.getLogger(__name__)


class SafetyAdapter(SafetyPort):
    """Wraps SafetySystem / AntiBanSystem to satisfy SafetyPort."""

    def __init__(self, safety_system=None, anti_ban=None):
        self._safety = safety_system
        self._anti_ban = anti_ban

    def check_before_action(self, action_type: str) -> SafetyStatus:
        if self._safety is None:
            return SafetyStatus(can_continue=True)
        try:
            # Simplified check — SafetySystem may have specific methods
            status = SafetyStatus(can_continue=True)
            if hasattr(self._safety, "check_apm"):
                if not self._safety.check_apm():
                    status.should_pause = True
                    status.warning_message = "APM limit reached"
            return status
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug(f"[SAFETY_ADAPTER] Check error: {e}")
            return SafetyStatus(can_continue=True)

    def check_before_match(self) -> SafetyStatus:
        if self._safety is None:
            return SafetyStatus(can_continue=True)
        try:
            status = SafetyStatus(can_continue=True)
            if hasattr(self._safety, "check_session_duration"):
                if not self._safety.check_session_duration():
                    status.should_stop = True
                    status.warning_message = "Max session duration reached"
            return status
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug(f"[SAFETY_ADAPTER] Match check error: {e}")
            return SafetyStatus(can_continue=True)

    def record_action(self, action_type: str, duration_ms: float = 0.0) -> None:
        if self._safety is not None and hasattr(self._safety, "record_action"):
            try:
                self._safety.record_action(action_type)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

    def record_match_end(self, result: str, duration_sec: float = 0.0) -> None:
        if self._safety is not None and hasattr(self._safety, "record_match"):
            try:
                self._safety.record_match(result, duration_sec)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass

    def health_check(self):
        return {
            "safety_available": self._safety is not None,
            "anti_ban_available": self._anti_ban is not None,
        }
