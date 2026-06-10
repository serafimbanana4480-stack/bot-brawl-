"""
core/adapters/input_adapter.py

Adapter: EmulatorController -> InputPort
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, Optional

from core.ports.input_port import InputAction, InputPort

logger = logging.getLogger(__name__)


class InputAdapter(InputPort):
    """Wraps EmulatorController to satisfy InputPort with an async input queue."""

    def __init__(self, emulator_controller: Any = None):
        self._controller = emulator_controller
        self._input_queue: queue.Queue = queue.Queue(maxsize=20)
        self._input_stop = threading.Event()
        self._input_thread: Optional[threading.Thread] = None

    def initialize(self) -> bool:
        if self._controller is None:
            return False
        self._input_stop.clear()
        self._input_thread = threading.Thread(
            target=self._input_loop, daemon=True, name="input-worker"
        )
        self._input_thread.start()
        return True

    def execute(self, action: InputAction) -> bool:
        try:
            self._input_queue.put(action, block=False)
            return True
        except queue.Full:
            logger.warning("[INPUT_ADAPTER] Input queue full, dropping action")
            return False

    def tap(self, x: float, y: float, duration_ms: int = 100) -> bool:
        return self.execute(InputAction(action_type="tap", x=x, y=y, duration_ms=duration_ms))

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300) -> bool:
        return self.execute(InputAction(action_type="swipe", x=x1, y=y1, x2=x2, y2=y2, duration_ms=duration_ms))

    def _input_loop(self) -> None:
        while not self._input_stop.is_set():
            try:
                action = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._dispatch_action(action)

    def _dispatch_action(self, action: InputAction) -> None:
        if self._controller is None:
            return
        try:
            if action.action_type == "tap":
                if hasattr(self._controller, "tap_scaled"):
                    self._controller.tap_scaled(int(action.x * 1920), int(action.y * 1080))
                else:
                    self._controller.tap(int(action.x * 1920), int(action.y * 1080))
            elif action.action_type == "swipe":
                if hasattr(self._controller, "swipe_scaled"):
                    self._controller.swipe_scaled(
                        int(action.x * 1920), int(action.y * 1080),
                        int(action.x2 * 1920), int(action.y2 * 1080),
                        action.duration_ms
                    )
                else:
                    self._controller.swipe(
                        int(action.x * 1920), int(action.y * 1080),
                        int(action.x2 * 1920), int(action.y2 * 1080),
                        action.duration_ms
                    )
            elif action.action_type == "key" and action.keycode is not None:
                self._controller.keyevent(action.keycode)
            else:
                logger.warning(f"[INPUT_ADAPTER] Unknown action type: {action.action_type}")
        except Exception as e:
            logger.error(f"[INPUT_ADAPTER] Dispatch failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        return {
            "connected": self._controller is not None,
            "controller_type": type(self._controller).__name__ if self._controller else "None",
            "queue_size": self._input_queue.qsize(),
        }

    def shutdown(self) -> None:
        self._input_stop.set()
        if self._input_thread and self._input_thread.is_alive():
            self._input_thread.join(timeout=2.0)
