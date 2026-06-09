"""
core/subsystems/__init__.py

Cohesive subsystems extracted from the PylaAIEnhanced God Class.
Each subsystem owns its lifecycle (init, setup, start, stop).
"""

from .emulator_subsystem import EmulatorSubsystem
from .vision_subsystem import VisionSubsystem
from .safety_subsystem import SafetySubsystem
from .decision_subsystem import DecisionSubsystem
from .learning_subsystem import LearningSubsystem
from .ui_subsystem import UISubsystem

__all__ = [
    "EmulatorSubsystem",
    "VisionSubsystem",
    "SafetySubsystem",
    "DecisionSubsystem",
    "LearningSubsystem",
    "UISubsystem",
]
