"""
core/adapters/adversarial_input_adapter.py

Decorator InputPort that wraps another InputPort and applies
adversarial humanization perturbations before executing actions.

This allows zero-intrusion integration: the orchestrator still
sees a normal InputPort, but every action is humanized before
dispatch to ADB/Win32.
"""

from __future__ import annotations

import logging
from typing import Any

from core.adversarial_humanization import (
    AdversarialHumanizer,
)
from core.ports.input_port import InputAction, InputPort

logger = logging.getLogger(__name__)


class AdversarialInputAdapter(InputPort):
    """
    Wraps a primary InputPort and applies adversarial humanization.

    Usage in factory:
        primary = InputAdapter(emulator_controller)
        humanizer = AdversarialHumanizer(config)
        wrapped = AdversarialInputAdapter(primary, humanizer)
    """

    def __init__(
        self,
        primary: InputPort,
        humanizer: AdversarialHumanizer,
        screen_resolution: tuple[int, int] = (1920, 1080),
    ):
        self._primary = primary
        self._humanizer = humanizer
        self._screen_w, self._screen_h = screen_resolution

    # ------------------------------------------------------------------
    # InputPort implementation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        return self._primary.initialize()

    def execute(self, action: InputAction) -> bool:
        perturbed = self._humanizer.perturb(action, self._screen_w, self._screen_h)
        # Optional: humanized reaction delay before executing
        self._humanizer.sleep_reaction_time()
        return self._primary.execute(perturbed)

    def tap(self, x: float, y: float, duration_ms: int = 100) -> bool:
        action = InputAction(action_type="tap", x=x, y=y, duration_ms=duration_ms)
        return self.execute(action)

    def swipe(self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300) -> bool:
        action = InputAction(
            action_type="swipe",
            x=x1,
            y=y1,
            x2=x2,
            y2=y2,
            duration_ms=duration_ms,
        )
        return self.execute(action)

    def health_check(self) -> dict[str, Any]:
        primary = self._primary.health_check()
        return {
            **primary,
            "adversarial_humanization": self._humanizer.get_stats(),
        }

    def shutdown(self) -> None:
        self._primary.shutdown()
