"""
core/ports/

Ports & Adapters architecture for Soberana Omega.

This module defines the abstract interfaces (ports) that the bot orchestrator
depends on. Concrete implementations (adapters) live in their respective
modules and are injected at startup.

Benefits:
- Testability: mock any port for unit testing
- Swapability: replace vision engine, input method, or decision system
- No circular dependencies: core/ports only defines contracts
- Clean architecture: orchestrator doesn't know about Win32, ADB, YOLO, etc.

Example:
    from core.ports import VisionPort, InputPort, DecisionPort
    from core.orchestrator import BotOrchestrator

    orchestrator = BotOrchestrator(
        vision=MyVisionAdapter(),
        input=MyInputAdapter(),
        decision=MyDecisionAdapter(),
        ...
    )
"""

from __future__ import annotations

from .vision_port import VisionPort, GameStateSnapshot, DetectedObject, HUDState
from .input_port import InputPort, InputAction
from .decision_port import DecisionPort, Decision, DecisionContext
from .safety_port import SafetyPort, SafetyStatus
from .telemetry_port import TelemetryPort, MetricEvent
from .persistence_port import PersistencePort

__all__ = [
    # Vision
    "VisionPort",
    "GameStateSnapshot",
    "DetectedObject",
    "HUDState",
    # Input
    "InputPort",
    "InputAction",
    # Decision
    "DecisionPort",
    "Decision",
    "DecisionContext",
    # Safety
    "SafetyPort",
    "SafetyStatus",
    # Telemetry
    "TelemetryPort",
    "MetricEvent",
    # Persistence
    "PersistencePort",
]
