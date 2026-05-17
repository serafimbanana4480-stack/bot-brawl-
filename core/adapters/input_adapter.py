"""
core/adapters/input_adapter.py

Adapter: EmulatorController -> InputPort
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from core.ports.input_port import InputAction, InputPort

logger = logging.getLogger(__name__)


class InputAdapter(InputPort):
    """Wraps EmulatorController to satisfy InputPort."""

    def __init__(self, emulator_controller: Any = None):
        self._controller = emulator_controller

    def initialize(self) -> bool:
        if self._controller is None:
            return False
        return True

    def execute(self, action: InputAction) -> bool:
        if self._controller is None:
            return False
        try:
            if action.action_type == "tap":
                return self.tap(action.x, action.y, action.duration_ms)
            elif action.action_type == "swipe":
                return self.swipe(action.x, action.y, action.x2, action.y2, action.duration_ms)
            elif action.action_type == "key" and action.keycode is not None:
                return self._controller.keyevent(action.keycode)
            else:
                logger.warning(f"[INPUT_ADAPTER] Unknown action type: {action.action_type}")
                return False
        except Exception as e:
            logger.error(f"[INPUT_ADAPTER] Execute failed: {e}")
            return False

    def tap(self, x: float, y: float, duration_ms: int = 100) -> bool:
        if self._controller is None:
            return False
        try:
            # Use tap_scaled if available (handles resolution scaling)
            if hasattr(self._controller, "tap_scaled"):
                return self._controller.tap_scaled(int(x * 1920), int(y * 1080))
            return self._controller.tap(int(x * 1920), int(y * 1080))
        except Exception as e:
            logger.error(f"[INPUT_ADAPTER] Tap failed: {e}")
            return False

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300) -> bool:
        if self._controller is None:
            return False
        try:
            if hasattr(self._controller, "swipe_scaled"):
                return self._controller.swipe_scaled(
                    int(x1 * 1920), int(y1 * 1080),
                    int(x2 * 1920), int(y2 * 1080),
                    duration_ms
                )
            return self._controller.swipe(
                int(x1 * 1920), int(y1 * 1080),
                int(x2 * 1920), int(y2 * 1080),
                duration_ms
            )
        except Exception as e:
            logger.error(f"[INPUT_ADAPTER] Swipe failed: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        return {
            "connected": self._controller is not None,
            "controller_type": type(self._controller).__name__ if self._controller else "None",
        }

    def shutdown(self) -> None:
        pass
