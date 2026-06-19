"""
core/ports/input_port.py

Input Port — abstract interface for game control / input dispatch.

The orchestrator depends only on this interface, not on:
- ADB commands
- Win32 SendMessage
- DirectInput
- Emulator specifics

Adapters:
    - ADBInputAdapter (emulator_controller.py)
    - DirectInputAdapter (future)
    - ReplayInputAdapter (for testing / replays)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputAction:
    """Normalized input action (resolution-independent)."""
    action_type: str  # "tap", "swipe", "long_press", "key"
    x: float = 0.0   # 0.0–1.0 (normalized screen width)
    y: float = 0.0   # 0.0–1.0 (normalized screen height)
    x2: float = 0.0  # for swipe
    y2: float = 0.0  # for swipe
    duration_ms: int = 100
    keycode: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InputPort(abc.ABC):
    """Abstract input/control interface."""

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Connect to input mechanism."""
        ...

    @abc.abstractmethod
    def execute(self, action: InputAction) -> bool:
        """Execute one input action. Return success/failure."""
        ...

    @abc.abstractmethod
    def tap(self, x: float, y: float, duration_ms: int = 100) -> bool:
        """Convenience: tap at normalized coordinates."""
        ...

    @abc.abstractmethod
    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300) -> bool:
        """Convenience: swipe between normalized coordinates."""
        ...

    @abc.abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return input health: connected, lag, errors, etc."""
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        ...
