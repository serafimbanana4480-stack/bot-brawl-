"""
core/subsystems/__init__.py

Cohesive subsystems extracted from the PylaAIEnhanced God Class.
Each subsystem owns its lifecycle (init, setup, start, stop).
"""

from .decision_subsystem import DecisionSubsystem
from .emulator_subsystem import EmulatorSubsystem
from .learning_subsystem import LearningSubsystem
from .safety_subsystem import SafetySubsystem
from .ui_subsystem import UISubsystem
from .vision_subsystem import VisionSubsystem

__all__ = [
    "EmulatorSubsystem",
    "VisionSubsystem",
    "SafetySubsystem",
    "DecisionSubsystem",
    "LearningSubsystem",
    "UISubsystem",
]
