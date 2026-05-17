"""
core/ports/safety_port.py

Safety Port — abstract interface for anti-ban and operational safety.

Adapters:
    - SafetySystemAdapter (safety_system.py)
    - AntiBanAdapter (pylaai_real/anti_ban_advanced.py)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SafetyStatus:
    can_continue: bool = True
    should_pause: bool = False
    should_stop: bool = False
    warning_message: str = ""
    metrics: Dict[str, Any] = None  # type: ignore[assignment]


class SafetyPort(abc.ABC):
    """Abstract safety/anti-ban interface."""

    @abc.abstractmethod
    def check_before_action(self, action_type: str) -> SafetyStatus:
        """Pre-action safety check (rate limits, APM, session limits)."""
        ...

    @abc.abstractmethod
    def check_before_match(self) -> SafetyStatus:
        """Pre-match safety check (trophies, session duration)."""
        ...

    @abc.abstractmethod
    def record_action(self, action_type: str, duration_ms: float = 0.0) -> None:
        """Record an action for APM tracking."""
        ...

    @abc.abstractmethod
    def record_match_end(self, result: str, duration_sec: float = 0.0) -> None:
        """Record match completion for session tracking."""
        ...

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        ...
