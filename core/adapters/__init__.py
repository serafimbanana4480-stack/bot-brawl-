"""
core/adapters/

Concrete adapters that implement the core/ports interfaces.
Each adapter wraps an existing subsystem and bridges it to the Port contract.

This allows the orchestrator to remain independent of:
- YOLO/Ultralytics
- ADB/Win32
- Q-Learning/NeuralPolicy specifics
- Any particular safety or telemetry backend

Adapters:
    - VisionAdapter: wraps Detect + UnifiedStateDetector + ScreenshotTaker
    - InputAdapter: wraps EmulatorController
    - DecisionAdapter: wraps RLBridge / UtilityAI
    - SafetyAdapter: wraps SafetySystem / AntiBanSystem
    - TelemetryAdapter: wraps ObservabilityCollector
    - PersistenceAdapter: wraps StatePersistence
"""

from core.adapters.vision_adapter import VisionAdapter
from core.adapters.input_adapter import InputAdapter
from core.adapters.decision_adapter import DecisionAdapter
from core.adapters.safety_adapter import SafetyAdapter
from core.adapters.telemetry_adapter import TelemetryAdapter
from core.adapters.persistence_adapter import PersistenceAdapter

__all__ = [
    "VisionAdapter",
    "InputAdapter",
    "DecisionAdapter",
    "SafetyAdapter",
    "TelemetryAdapter",
    "PersistenceAdapter",
]
