"""
Finite State Machine for Brawl Stars bot decision making.
States: IDLE -> SEARCH -> ENGAGE -> RETREAT -> RECOVER
"""

from enum import Enum, auto
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
import time
import random


class BotState(Enum):
    """Main bot states."""
    IDLE = auto()
    SEARCH = auto()
    ENGAGE = auto()
    RETREAT = auto()
    RECOVER = auto()


class StateTransition:
    """Represents a state transition condition."""
    
    def __init__(
        self,
        from_state: BotState,
        to_state: BotState,
        condition: Callable,
        priority: int = 0
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.priority = priority
        self.last_check = 0.0


@dataclass
class StateContext:
    """Context passed to state handlers."""
    game_state: object  # Will be GameState from vision.state
    bot_instance: object  # Reference to bot for actions
    last_state_change: float = 0.0
    state_data: Dict = None
    neural_policy: Optional[object] = None  # Neural policy for decision making
    current_image: Optional[object] = None  # Current frame image
    aux_state: Optional[object] = None  # Auxiliary state [health, ammo]
    
    def __post_init__(self):
        if self.state_data is None:
            self.state_data = {}


class StateMachine:
    """
    Finite State Machine for bot behavior.
    Manages transitions between IDLE, SEARCH, ENGAGE, RETREAT, RECOVER.
    """
    
    def __init__(self):
        self.current_state = BotState.IDLE
        self.previous_state = None
        self.state_entry_time = time.time()
        
        # Transition rules
        self.transitions: List[StateTransition] = []
        
        # State handlers
        self.handlers: Dict[BotState, Callable] = {}
        
        # Minimum time in state (to prevent rapid switching)
        self.min_state_duration: Dict[BotState, float] = {
            BotState.IDLE: 0.5,
            BotState.SEARCH: 1.0,
            BotState.ENGAGE: 0.5,
            BotState.RETREAT: 2.0,
            BotState.RECOVER: 1.0,
        }
        
        # State entry/exit callbacks
        self.on_enter: Dict[BotState, Callable] = {}
        self.on_exit: Dict[BotState, Callable] = {}
        
    def register_handler(self, state: BotState, handler: Callable):
        """Register a handler for a state."""
        self.handlers[state] = handler
        
    def register_transition(
        self,
        from_state: BotState,
        to_state: BotState,
        condition: Callable,
        priority: int = 0
    ):
        """Register a state transition."""
        self.transitions.append(StateTransition(
            from_state, to_state, condition, priority
        ))
        
    def can_transition(self) -> bool:
        """Check if enough time has passed in current state."""
        min_duration = self.min_state_duration.get(self.current_state, 0.0)
        return (time.time() - self.state_entry_time) >= min_duration
        
    def transition_to(self, new_state: BotState, context: StateContext):
        """Force transition to a new state."""
        if new_state == self.current_state:
            return
        
        # Exit current state
        if self.current_state in self.on_exit:
            self.on_exit[self.current_state](context)
        
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = time.time()
        context.last_state_change = self.state_entry_time
        
        # Enter new state
        if new_state in self.on_enter:
            self.on_enter[new_state](context)
    
    def update(self, context: StateContext) -> BotState:
        """
        Update state machine based on current context.
        
        Args:
            context: Current game context
            
        Returns:
            Current state after any transitions
        """
        if not self.can_transition():
            return self.current_state
        
        # Check all valid transitions
        valid_transitions = [
            t for t in self.transitions
            if t.from_state == self.current_state and t.condition(context)
        ]
        
        if valid_transitions:
            # Sort by priority (higher = more urgent)
            valid_transitions.sort(key=lambda t: -t.priority)
            best_transition = valid_transitions[0]
            
            self.transition_to(best_transition.to_state, context)
        
        return self.current_state
    
    def execute(self, context: StateContext):
        """Execute current state's handler."""
        if self.current_state in self.handlers:
            return self.handlers[self.current_state](context)
        
    def get_state_duration(self) -> float:
        """Get time spent in current state."""
        return time.time() - self.state_entry_time
    
    def is_stuck(self, max_duration: float = 30.0) -> bool:
        """Check if stuck in same state too long."""
        return self.get_state_duration() > max_duration
    
    def should_use_neural_policy(self, context: StateContext) -> bool:
        """
        Check if neural policy should be used for decision making.
        
        Returns True if neural policy is available and confidence is high.
        """
        if context.neural_policy is None:
            return False
        
        try:
            policy_info = context.neural_policy.get_policy_info()
            if not policy_info.get('bc_loaded') and not policy_info.get('cql_loaded'):
                return False
            
            # Check if we have required data for neural prediction
            if context.current_image is None or context.aux_state is None:
                return False
            
            return True
        except Exception as e:
            logger.debug(f"Neural policy check failed: {e}")
            return False
    
    def get_neural_decision(self, context: StateContext) -> Optional[Dict]:
        """
        Get decision from neural policy.
        
        Returns decision dict or None if neural policy unavailable.
        """
        if not self.should_use_neural_policy(context):
            return None
        
        try:
            import numpy as np
            # Convert image to numpy if needed
            image = context.current_image
            if hasattr(image, 'shape'):
                image_np = image
            else:
                # Assume it's already in correct format
                image_np = image
            
            aux_state = context.aux_state
            if hasattr(aux_state, 'shape'):
                aux_state_np = aux_state
            else:
                # Convert to numpy array
                aux_state_np = np.array([aux_state['health'], aux_state['ammo']])
            
            # Get neural prediction
            output = context.neural_policy.predict(
                image_np,
                aux_state_np,
                use_ensemble=True
            )
            
            return {
                'move_angle': output.move_angle,
                'attack': output.attack,
                'use_super': output.use_super,
                'target': (output.target_x, output.target_y),
                'confidence': output.confidence,
                'source': output.source,
            }
        except Exception as e:
            logger.debug(f"Neural decision failed: {e}")
            return None


class BrawlStarsStateMachine(StateMachine):
    """
    Pre-configured state machine for Brawl Stars.
    """
    
    def __init__(self):
        super().__init__()
        self._setup_default_transitions()
        
    def _setup_default_transitions(self):
        """Setup default transition rules for Brawl Stars."""
        
        # IDLE -> SEARCH: No enemies visible and player alive
        self.register_transition(
            BotState.IDLE,
            BotState.SEARCH,
            lambda ctx: (
                ctx.game_state.player_state.value == "alive" and
                len(ctx.game_state.enemies) == 0
            ),
            priority=1
        )
        
        # IDLE -> ENGAGE: Enemy visible and safe to engage
        self.register_transition(
            BotState.IDLE,
            BotState.ENGAGE,
            lambda ctx: (
                ctx.game_state.player_state.value == "alive" and
                len(ctx.game_state.enemies) > 0 and
                ctx.game_state.can_engage
            ),
            priority=5
        )
        
        # IDLE -> RETREAT: In danger even if idle
        self.register_transition(
            BotState.IDLE,
            BotState.RETREAT,
            lambda ctx: (
                ctx.game_state.should_retreat or
                ctx.game_state.danger_score > 0.8
            ),
            priority=10  # High priority
        )
        
        # SEARCH -> ENGAGE: Found enemy
        self.register_transition(
            BotState.SEARCH,
            BotState.ENGAGE,
            lambda ctx: (
                len(ctx.game_state.enemies) > 0 and
                ctx.game_state.can_engage
            ),
            priority=5
        )
        
        # SEARCH -> RETREAT: Danger detected while searching
        self.register_transition(
            BotState.SEARCH,
            BotState.RETREAT,
            lambda ctx: ctx.game_state.should_retreat,
            priority=10
        )
        
        # ENGAGE -> RETREAT: Health low or too many enemies
        self.register_transition(
            BotState.ENGAGE,
            BotState.RETREAT,
            lambda ctx: (
                ctx.game_state.should_retreat or
                (ctx.game_state.player_health < 0.3 and len(ctx.game_state.enemies) > 0)
            ),
            priority=10
        )
        
        # ENGAGE -> SEARCH: Enemy killed or lost
        self.register_transition(
            BotState.ENGAGE,
            BotState.SEARCH,
            lambda ctx: (
                len(ctx.game_state.enemies) == 0 or
                (ctx.game_state.nearest_enemy and ctx.game_state.nearest_enemy.distance > 500)
            ),
            priority=3
        )
        
        # ENGAGE -> IDLE: No threats and health good
        self.register_transition(
            BotState.ENGAGE,
            BotState.IDLE,
            lambda ctx: (
                len(ctx.game_state.enemies) == 0 and
                ctx.game_state.player_health > 0.8
            ),
            priority=1
        )
        
        # RETREAT -> RECOVER: Safe from danger
        self.register_transition(
            BotState.RETREAT,
            BotState.RECOVER,
            lambda ctx: (
                ctx.game_state.danger_score < 0.3 and
                ctx.game_state.player_health < 1.0
            ),
            priority=5
        )
        
        # RETREAT -> SEARCH: Safe and healthy
        self.register_transition(
            BotState.RETREAT,
            BotState.SEARCH,
            lambda ctx: (
                ctx.game_state.danger_score < 0.3 and
                ctx.game_state.player_health >= 0.8
            ),
            priority=4
        )
        
        # RETREAT -> IDLE: Completely safe
        self.register_transition(
            BotState.RETREAT,
            BotState.IDLE,
            lambda ctx: (
                ctx.game_state.danger_score < 0.1 and
                len(ctx.game_state.enemies) == 0
            ),
            priority=2
        )
        
        # RECOVER -> SEARCH: Healed and ready
        self.register_transition(
            BotState.RECOVER,
            BotState.SEARCH,
            lambda ctx: (
                ctx.game_state.player_health >= 0.9 and
                ctx.game_state.danger_score < 0.3
            ),
            priority=5
        )
        
        # RECOVER -> RETREAT: Danger during recovery
        self.register_transition(
            BotState.RECOVER,
            BotState.RETREAT,
            lambda ctx: ctx.game_state.danger_score > 0.5,
            priority=10
        )


def create_default_state_machine() -> BrawlStarsStateMachine:
    """Factory function to create a pre-configured state machine."""
    return BrawlStarsStateMachine()
