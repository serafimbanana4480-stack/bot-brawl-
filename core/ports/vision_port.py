"""
core/ports/vision_port.py

Vision Port — abstract interface for game perception.

The orchestrator depends only on this interface, not on:
- YOLO/Ultralytics
- OpenCV
- Screenshot capture mechanism (Win32, ADB, etc.)
- OCR engines

Adapters:
    - YOLOVisionAdapter  (pylaai_real/detect.py)
    - MultimodalVisionAdapter (vision/multimodal_pipeline.py)
    - VLMFallbackAdapter (future: CLIP/Florence-2)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectedObject:
    """Single detected object in game world."""
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized 0-1)
    center: tuple[float, float]
    track_id: int | None = None
    velocity: tuple[float, float] | None = None
    hp_ratio: float | None = None


@dataclass
class HUDState:
    """Extracted HUD information."""
    hp_ratio: float = 1.0
    ammo_ratio: float = 1.0
    super_ready: bool = False
    match_time: float = 120.0
    team_score: int = 0
    enemy_score: int = 0
    power_cubes: int = 0
    gems: int = 0


@dataclass
class GameStateSnapshot:
    """Complete snapshot of the game state at one timestep."""
    screenshot: np.ndarray | None = None  # Raw image (optional, for debug)
    detected_objects: list[DetectedObject] = field(default_factory=list)
    hud: HUDState = field(default_factory=HUDState)
    game_phase: str = "unknown"  # lobby, countdown, combat, defeated, victory, etc.
    player_pos: tuple[float, float] = (0.5, 0.5)
    timestamp: float = 0.0
    latency_ms: float = 0.0
    resolution: tuple[int, int] = (1920, 1080)
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionPort(abc.ABC):
    """Abstract vision/perception interface."""

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize vision engine (load models, connect to capture, etc.)."""
        ...

    @abc.abstractmethod
    def capture_and_perceive(self) -> GameStateSnapshot | None:
        """
        Capture a frame and return structured game state.
        Returns None on failure (allows orchestrator to handle gracefully).
        """
        ...

    @abc.abstractmethod
    def get_detected_objects(self, class_filter: list[str] | None = None) -> list[DetectedObject]:
        """Return last detected objects, optionally filtered by class."""
        ...

    @abc.abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return health status: fps, errors, model loaded, etc."""
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        ...
