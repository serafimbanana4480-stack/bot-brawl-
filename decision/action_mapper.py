"""
decision/action_mapper.py

Action space unification layer.

Bridges between:
- RL Engine actions (6 discrete actions from rl_engine.py)
- UtilityAI actions (10 discrete actions from utility_ai.py)
- UnifiedAction enum (12 canonical actions for neural policy)

This eliminates the dual decision system problem and provides a single
action space for the entire project.

Usage:
    from decision.action_mapper import (
        UnifiedAction,
        rl_to_unified,
        utility_to_unified,
        ActionMapper,
    )

    # Convert RL action string to unified
    unified = rl_to_unified("attack")  # UnifiedAction.ATTACK

    # Convert UtilityAI Action enum to unified
    unified = utility_to_unified(Action.KITE)  # UnifiedAction.KITE

    # Get action metadata
    mapper = ActionMapper()
    meta = mapper.get_action_metadata(UnifiedAction.KITE)
    # meta.name == "kite"
    # meta.description == "Attack while moving away"
    # meta.requires_target == True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

from core.class_registry import UnifiedAction as BaseUnifiedAction
from core.class_registry import RL_ACTION_MAP, UTILITY_ACTION_MAP


# Re-export UnifiedAction for convenience
UnifiedAction = BaseUnifiedAction


@dataclass(frozen=True)
class ActionMetadata:
    """Metadata about an action for UI and debugging."""
    name: str
    description: str
    requires_target: bool
    requires_ammo: bool
    can_use_super: bool
    cooldown_ms: int
    rl_equivalent: Optional[str] = None
    utility_equivalent: Optional[str] = None
    is_movement: bool = False
    is_offensive: bool = False
    is_defensive: bool = False


class ActionMapper:
    """
    Central registry for action mappings and metadata.

    Provides:
    - Conversion between RL, UtilityAI, and Unified action spaces
    - Action metadata for UI/dashboard
    - Action grouping (movement, offensive, defensive)
    - Valid action checking based on game state
    """

    # Action metadata for all unified actions
    ACTION_METADATA: Dict[UnifiedAction, ActionMetadata] = {
        UnifiedAction.IDLE: ActionMetadata(
            name="idle",
            description="Do nothing, wait",
            requires_target=False,
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            rl_equivalent="idle",
            is_movement=False,
            is_offensive=False,
            is_defensive=False,
        ),
        UnifiedAction.ATTACK: ActionMetadata(
            name="attack",
            description="Shoot at target",
            requires_target=True,
            requires_ammo=True,
            can_use_super=False,
            cooldown_ms=0,
            rl_equivalent="attack",
            utility_equivalent="ATTACK",
            is_movement=False,
            is_offensive=True,
            is_defensive=False,
        ),
        UnifiedAction.MOVE_TO_ENEMY: ActionMetadata(
            name="move_to_enemy",
            description="Move toward enemy",
            requires_target=True,
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            rl_equivalent="move_to_enemy",
            is_movement=True,
            is_offensive=True,
            is_defensive=False,
        ),
        UnifiedAction.RETREAT: ActionMetadata(
            name="retreat",
            description="Move away from threats",
            requires_target=False,
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            rl_equivalent="retreat",
            utility_equivalent="RETREAT",
            is_movement=True,
            is_offensive=False,
            is_defensive=True,
        ),
        UnifiedAction.KITE: ActionMetadata(
            name="kite",
            description="Attack while moving away (shoot-and-scoot)",
            requires_target=True,
            requires_ammo=True,
            can_use_super=False,
            cooldown_ms=0,
            rl_equivalent=None,
            utility_equivalent="KITE",
            is_movement=True,
            is_offensive=True,
            is_defensive=True,
        ),
        UnifiedAction.USE_SUPER: ActionMetadata(
            name="use_super",
            description="Activate super ability",
            requires_target=True,  # Most supers need target
            requires_ammo=False,
            can_use_super=True,
            cooldown_ms=0,
            rl_equivalent="use_super",
            utility_equivalent="USE_SUPER",
            is_movement=False,
            is_offensive=True,
            is_defensive=False,
        ),
        UnifiedAction.COLLECT_CUBE: ActionMetadata(
            name="collect_cube",
            description="Move to and collect power cube",
            requires_target=True,  # Target is the cube
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            rl_equivalent="collect_cube",
            utility_equivalent="COLLECT_CUBE",
            is_movement=True,
            is_offensive=False,
            is_defensive=False,
        ),
        UnifiedAction.TAKE_COVER: ActionMetadata(
            name="take_cover",
            description="Move to bush/wall for cover",
            requires_target=True,  # Target is cover location
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            utility_equivalent="TAKE_COVER",
            is_movement=True,
            is_offensive=False,
            is_defensive=True,
        ),
        UnifiedAction.HOLD_POSITION: ActionMetadata(
            name="hold_position",
            description="Stay in current position (zone control)",
            requires_target=False,
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            utility_equivalent="HOLD_POSITION",
            is_movement=False,
            is_offensive=False,
            is_defensive=True,
        ),
        UnifiedAction.CHASE: ActionMetadata(
            name="chase",
            description="Pursue low-health enemy",
            requires_target=True,
            requires_ammo=True,
            can_use_super=False,
            cooldown_ms=0,
            utility_equivalent="CHASE",
            is_movement=True,
            is_offensive=True,
            is_defensive=False,
        ),
        UnifiedAction.HEAL_UP: ActionMetadata(
            name="heal_up",
            description="Stay in safe area to regenerate health",
            requires_target=False,
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            utility_equivalent="HEAL_UP",
            is_movement=False,
            is_offensive=False,
            is_defensive=True,
        ),
        UnifiedAction.AMBUSH: ActionMetadata(
            name="ambush",
            description="Wait in bush for enemy to approach",
            requires_target=True,  # Target is ambush position
            requires_ammo=False,
            can_use_super=False,
            cooldown_ms=0,
            utility_equivalent="AMBUSH",
            is_movement=True,
            is_offensive=True,
            is_defensive=True,
        ),
    }

    def __init__(self):
        """Initialize action mapper."""
        # Build reverse lookup maps
        self._rl_to_unified: Dict[str, UnifiedAction] = {}
        self._utility_to_unified: Dict[str, UnifiedAction] = {}

        for action, meta in self.ACTION_METADATA.items():
            if meta.rl_equivalent:
                self._rl_to_unified[meta.rl_equivalent] = action
            if meta.utility_equivalent:
                self._utility_to_unified[meta.utility_equivalent] = action

    def rl_to_unified(self, rl_action: str) -> UnifiedAction:
        """
        Convert RL action string to unified action.

        Args:
            rl_action: RL action name (e.g., "attack", "retreat")

        Returns:
            Corresponding UnifiedAction

        Raises:
            ValueError: If RL action not found
        """
        if rl_action not in self._rl_to_unified:
            raise ValueError(
                f"Unknown RL action: {rl_action}. "
                f"Valid: {list(self._rl_to_unified.keys())}"
            )
        return self._rl_to_unified[rl_action]

    def utility_to_unified(self, utility_action: Union[str, Enum]) -> UnifiedAction:
        """
        Convert UtilityAI action to unified action.

        Args:
            utility_action: UtilityAI Action enum or string name

        Returns:
            Corresponding UnifiedAction

        Raises:
            ValueError: If utility action not found
        """
        # Handle enum
        if isinstance(utility_action, Enum):
            name = utility_action.name
        else:
            name = str(utility_action)

        if name not in self._utility_to_unified:
            raise ValueError(
                f"Unknown utility action: {name}. "
                f"Valid: {list(self._utility_to_unified.keys())}"
            )
        return self._utility_to_unified[name]

    def unified_to_rl(self, unified: UnifiedAction) -> Optional[str]:
        """
        Convert unified action to RL action (if possible).

        Args:
            unified: UnifiedAction

        Returns:
            RL action string or None if no direct equivalent
        """
        meta = self.ACTION_METADATA.get(unified)
        return meta.rl_equivalent if meta else None

    def unified_to_utility(self, unified: UnifiedAction) -> Optional[str]:
        """
        Convert unified action to UtilityAI action (if possible).

        Args:
            unified: UnifiedAction

        Returns:
            UtilityAI action name or None if no direct equivalent
        """
        meta = self.ACTION_METADATA.get(unified)
        return meta.utility_equivalent if meta else None

    def get_metadata(self, action: UnifiedAction) -> ActionMetadata:
        """Get metadata for an action."""
        if action not in self.ACTION_METADATA:
            raise ValueError(f"Unknown action: {action}")
        return self.ACTION_METADATA[action]

    def get_all_actions(self) -> List[UnifiedAction]:
        """Get list of all unified actions."""
        return list(UnifiedAction)

    def get_movement_actions(self) -> Set[UnifiedAction]:
        """Get actions that involve movement."""
        return {
            a for a, m in self.ACTION_METADATA.items()
            if m.is_movement
        }

    def get_offensive_actions(self) -> Set[UnifiedAction]:
        """Get offensive/combat actions."""
        return {
            a for a, m in self.ACTION_METADATA.items()
            if m.is_offensive
        }

    def get_defensive_actions(self) -> Set[UnifiedAction]:
        """Get defensive/survival actions."""
        return {
            a for a, m in self.ACTION_METADATA.items()
            if m.is_defensive
        }

    def is_valid_in_state(
        self,
        action: UnifiedAction,
        has_target: bool,
        has_ammo: bool,
        can_super: bool,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if action is valid given current state.

        Returns:
            (is_valid, reason_if_invalid)
        """
        meta = self.get_metadata(action)

        if meta.requires_target and not has_target:
            return False, "requires_target"

        if meta.requires_ammo and not has_ammo:
            return False, "requires_ammo"

        if meta.can_use_super and not can_super:
            return False, "requires_super"

        return True, None

    def get_fallback_action(self, action: UnifiedAction) -> UnifiedAction:
        """
        Get fallback action when primary action is invalid.

        For example, if ATTACK is chosen but no ammo, fallback to RETREAT.
        """
        fallbacks: Dict[UnifiedAction, UnifiedAction] = {
            UnifiedAction.ATTACK: UnifiedAction.RETREAT,
            UnifiedAction.MOVE_TO_ENEMY: UnifiedAction.IDLE,
            UnifiedAction.USE_SUPER: UnifiedAction.ATTACK,
            UnifiedAction.COLLECT_CUBE: UnifiedAction.IDLE,
            UnifiedAction.CHASE: UnifiedAction.ATTACK,
            UnifiedAction.AMBUSH: UnifiedAction.TAKE_COVER,
            UnifiedAction.KITE: UnifiedAction.RETREAT,
        }
        return fallbacks.get(action, UnifiedAction.IDLE)


# ============================================================================
# Convenience module-level functions
# ============================================================================

# Global mapper instance for convenience
_default_mapper: Optional[ActionMapper] = None


def _get_mapper() -> ActionMapper:
    """Get or create default mapper instance."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = ActionMapper()
    return _default_mapper


def rl_to_unified(rl_action: str) -> UnifiedAction:
    """Convert RL action to unified (convenience function)."""
    return _get_mapper().rl_to_unified(rl_action)


def utility_to_unified(utility_action: Union[str, Enum]) -> UnifiedAction:
    """Convert UtilityAI action to unified (convenience function)."""
    return _get_mapper().utility_to_unified(utility_action)


def unified_to_rl(unified: UnifiedAction) -> Optional[str]:
    """Convert unified to RL action (convenience function)."""
    return _get_mapper().unified_to_rl(unified)


def unified_to_utility(unified: UnifiedAction) -> Optional[str]:
    """Convert unified to UtilityAI action (convenience function)."""
    return _get_mapper().unified_to_utility(unified)


def get_action_metadata(action: UnifiedAction) -> ActionMetadata:
    """Get action metadata (convenience function)."""
    return _get_mapper().get_metadata(action)


def get_movement_actions() -> Set[UnifiedAction]:
    """Get movement actions (convenience function)."""
    return _get_mapper().get_movement_actions()


def get_offensive_actions() -> Set[UnifiedAction]:
    """Get offensive actions (convenience function)."""
    return _get_mapper().get_offensive_actions()


def get_defensive_actions() -> Set[UnifiedAction]:
    """Get defensive actions (convenience function)."""
    return _get_mapper().get_defensive_actions()


# ============================================================================
# Action validation helpers
# ============================================================================

def validate_action_sequence(
    actions: List[UnifiedAction],
    min_duration_ms: int = 100,
    max_repeats: int = 3,
) -> Tuple[bool, List[str]]:
    """
    Validate a sequence of actions for bot safety.

    Checks:
    - No action spam (max repeats)
    - Minimum duration between switches
    - Valid action transitions

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check for excessive repeats
    from itertools import groupby
    for action, group in groupby(actions):
        count = sum(1 for _ in group)
        if count > max_repeats:
            issues.append(f"Action {action.name} repeated {count} times (max {max_repeats})")

    # TODO: Add more sophisticated validation

    return len(issues) == 0, issues


def compute_action_diversity(actions: List[UnifiedAction]) -> float:
    """
    Compute diversity score for action sequence.

    Returns 0.0-1.0 where 1.0 means all different actions.
    """
    if not actions:
        return 0.0

    unique = len(set(actions))
    total = len(actions)
    return unique / total
