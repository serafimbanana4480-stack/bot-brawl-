"""
core/ports/decision_port.py

Decision Port — abstract interface for combat/strategic decision making.

The orchestrator depends only on this interface, not on:
- Q-Learning tables
- NeuralPolicy / PPO
- UtilityAI heuristics
- Meta-learning systems

Adapters:
    - NeuralDecisionAdapter (neural/rl_bridge.py)
    - UtilityAIAdapter (decision/utility_ai.py)
    - HybridDecisionAdapter (combines multiple)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DecisionContext:
    """Context passed to the decision system for one frame."""
    player_hp: float = 1.0
    player_pos: Tuple[float, float] = (0.5, 0.5)
    enemies: List[Dict[str, Any]] = field(default_factory=list)
    allies: List[Dict[str, Any]] = field(default_factory=list)
    detected_objects: List[Any] = field(default_factory=list)
    hud_state: Optional[Any] = None
    game_phase: str = "unknown"
    match_time_remaining: float = 120.0
    can_attack: bool = True
    can_super: bool = False
    brawler_name: str = "default"
    map_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    """Output from the decision system."""
    action_type: str = "idle"  # attack, move, retreat, super, collect, idle, ...
    target_pos: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    reasoning: str = ""  # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionPort(abc.ABC):
    """Abstract decision-making interface."""

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Load models, Q-tables, or heuristics."""
        ...

    @abc.abstractmethod
    def decide(self, context: DecisionContext) -> Decision:
        """
        Given game context, return a decision.
        Must be fast (< 20ms) for real-time play.
        """
        ...

    @abc.abstractmethod
    def learn(self, context: DecisionContext, decision: Decision, reward: float) -> None:
        """
        Online learning signal. May be a no-op for static systems.
        Called every frame by the orchestrator.
        """
        ...

    @abc.abstractmethod
    def start_episode(self, brawler: str, map_name: Optional[str] = None) -> None:
        """Called at match start."""
        ...

    @abc.abstractmethod
    def end_episode(self, result: str, rank: int = 0) -> None:
        """Called at match end. Trigger training if needed."""
        ...

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return decision system health."""
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Clean up."""
        ...
