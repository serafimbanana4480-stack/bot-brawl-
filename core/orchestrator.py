"""
Main orchestrator for Brawl Stars bot.
Integrates all modules and manages the main execution loop.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Callable
from dataclasses import dataclass
import logging

from ..vision.vision_engine import YOLOv8VisionEngine
from ..vision.tracker import ByteTracker
from ..vision.state import StateExtractor, GameState

from ..decision.state_machine import (
    BrawlStarsStateMachine,
    BotState,
    StateContext,
    create_default_state_machine
)
from ..decision.rules import RuleEngine, Tactic
from ..decision.scorer import TargetScorer, create_default_scorers

from ..training.retrain import ContinuousLearner, create_continuous_learner


@dataclass
class BotConfig:
    """Configuration for bot orchestration."""
    # Vision
    models_dir: Path
    confidence_threshold: float = 0.5
    
    # Decision
    reaction_delay_min: float = 0.08  # 80ms
    reaction_delay_max: float = 0.22  # 220ms
    
    # Safety
    max_apm: int = 180
    trophy_limit: int = 500
    
    # Training
    enable_auto_learning: bool = True
    dataset_dir: Optional[Path] = None
    
    # Performance
    target_fps: int = 30
    frame_skip: int = 1


class BrawlStarsOrchestrator:
    """
    Main orchestrator that integrates all bot systems.
    Manages the game loop, state transitions, and action execution.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger("orchestrator")
        
        # Initialize components
        self.vision: Optional[YOLOv8VisionEngine] = None
        self.tracker: Optional[ByteTracker] = None
        self.state_extractor: Optional[StateExtractor] = None
        self.state_machine: Optional[BrawlStarsStateMachine] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.target_scorer: Optional[TargetScorer] = None
        self.learner: Optional[ContinuousLearner] = None
        
        # Runtime state
        self.is_running: bool = False
        self.current_state: Optional[GameState] = None
        self.frame_count: int = 0
        self.last_action_time: float = 0
        self.actions_this_minute: int = 0
        self.apm_reset_time: float = 0
        
        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_action: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Threading
        self._main_loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def initialize(self) -> bool:
        """
        Initialize all bot components.
        
        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing bot orchestrator...")
        
        try:
            # Vision system
            self.logger.info("Initializing vision system...")
            self.vision = YOLOv8VisionEngine(
                confidence_threshold=self.config.confidence_threshold
            )
            if not self.vision.load_models(self.config.models_dir):
                self.logger.error("Failed to load vision models")
                return False
            
            # Tracking
            self.tracker = ByteTracker(max_age=30, min_hits=3)
            
            # State extraction
            self.state_extractor = StateExtractor()
            
            # Decision system
            self.state_machine = create_default_state_machine()
            self._setup_state_handlers()
            
            # Rules and scoring
            self.rule_engine = RuleEngine()
            self.target_scorer, _, _ = create_default_scorers()
            
            # Auto-learning
            if self.config.enable_auto_learning and self.config.dataset_dir:
                self.learner = create_continuous_learner(
                    log_dir=self.config.dataset_dir / "logs",
                    dataset_dir=self.config.dataset_dir,
                    models_dir=self.config.models_dir
                )
            
            self.logger.info("Initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            if self.on_error:
                self.on_error("initialization", str(e))
            return False
    
    def _setup_state_handlers(self):
        """Setup handlers for each bot state."""
        
        def handle_idle(context: StateContext):
            """Handle IDLE state - wait and scan."""
            self.logger.debug("State: IDLE - scanning for enemies")
            # Slow scan movement
            # Check for enemies more frequently
            pass
        
        def handle_search(context: StateContext):
            """Handle SEARCH state - actively look for enemies."""
            self.logger.debug("State: SEARCH - patrolling")
            # Patrol towards center or common areas
            # Check bushes
            pass
        
        def handle_engage(context: StateContext):
            """Handle ENGAGE state - attack target."""
            self.logger.debug("State: ENGAGE - attacking")
            
            if not context.game_state.enemies:
                return
            
            # Rank targets and pick best
            ranked = self.target_scorer.rank_targets(
                context.game_state.enemies,
                context.game_state.player_position,
                context.game_state.player_health,
                context.game_state.walls
            )
            
            if ranked:
                best_target = ranked[0]
                self.logger.info(f"Targeting enemy {best_target.target_id}: {best_target.reasoning}")
                
                # Execute attack
                self._execute_attack(best_target, context)
        
        def handle_retreat(context: StateContext):
            """Handle RETREAT state - get to safety."""
            self.logger.debug("State: RETREAT - seeking safety")
            
            # Evaluate retreat options
            decisions = self.rule_engine.evaluate_retreat(context.game_state)
            
            if decisions:
                best = decisions[0]
                self.logger.info(f"Retreating: {best.reasoning}")
                self._execute_retreat(best, context)
        
        def handle_recover(context: StateContext):
            """Handle RECOVER state - heal and reposition."""
            self.logger.debug("State: RECOVER - healing")
            
            # Evaluate recovery options
            decisions = self.rule_engine.evaluate_recovery(context.game_state)
            
            if decisions:
                best = decisions[0]
                self.logger.info(f"Recovering: {best.reasoning}")
                self._execute_recovery(best, context)
        
        # Register handlers
        self.state_machine.register_handler(BotState.IDLE, handle_idle)
        self.state_machine.register_handler(BotState.SEARCH, handle_search)
        self.state_machine.register_handler(BotState.ENGAGE, handle_engage)
        self.state_machine.register_handler(BotState.RETREAT, handle_retreat)
        self.state_machine.register_handler(BotState.RECOVER, handle_recover)
    
    def _execute_attack(self, target_score, context: StateContext):
        """Execute attack on target."""
        # Find the actual enemy object
        enemy = None
        for e in context.game_state.enemies:
            if e.track_id == target_score.target_id:
                enemy = e
                break
        
        if not enemy:
            return
        
        # Aim at enemy
        # Fire if in range
        # Use abilities optimally
        
        if self.on_action:
            self.on_action("attack", enemy)
    
    def _execute_retreat(self, decision, context: StateContext):
        """Execute retreat maneuver."""
        if decision.target_position:
            # Move to safe position
            # Fire back if enemies pursuing
            pass
        
        if self.on_action:
            self.on_action("retreat", decision)
    
    def _execute_recovery(self, decision, context: StateContext):
        """Execute recovery action."""
        if decision.target_position:
            # Move to recovery spot
            pass
        
        if self.on_action:
            self.on_action("recover", decision)
    
    def _check_apm_limit(self) -> bool:
        """Check if APM limit reached."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.apm_reset_time >= 60:
            self.actions_this_minute = 0
            self.apm_reset_time = current_time
        
        return self.actions_this_minute < self.config.max_apm
    
    def _apply_reaction_delay(self):
        """Apply human-like reaction delay."""
        delay = self._humanized_delay()
        time.sleep(delay)
    
    def _humanized_delay(self) -> float:
        """Generate human-like reaction delay."""
        import random
        
        # Normal distribution around 150ms with variance
        base_delay = random.gauss(0.15, 0.04)
        
        # Clamp to configured range
        return max(
            self.config.reaction_delay_min,
            min(self.config.reaction_delay_max, base_delay)
        )
    
    def process_frame(self, frame) -> Optional[GameState]:
        """
        Process a single game frame.
        
        Args:
            frame: Screenshot/image from game
            
        Returns:
            Updated GameState or None
        """
        self.frame_count += 1
        
        # Skip frames if needed
        if self.frame_count % (self.config.frame_skip + 1) != 0:
            return self.current_state
        
        # Run vision inference
        detections = self.vision.detect(frame)
        
        # Update tracking
        tracks = self.tracker.update(detections)
        
        # Extract state
        self.current_state = self.state_extractor.extract_state(tracks)
        
        # Update state machine
        if self.current_state:
            context = StateContext(
                game_state=self.current_state,
                bot_instance=self
            )
            
            new_state = self.state_machine.update(context)
            
            if new_state != self.state_machine.current_state:
                if self.on_state_change:
                    self.on_state_change(self.state_machine.current_state, new_state)
            
            # Execute state handler
            self.state_machine.execute(context)
        
        return self.current_state
    
    def start(self):
        """Start the bot orchestrator."""
        if self.is_running:
            return
        
        self.logger.info("Starting bot orchestrator...")
        self.is_running = True
        self._stop_event.clear()
        
        # Start auto-learning if enabled
        if self.learner:
            self.learner.start()
        
        self.logger.info("Bot orchestrator running")
    
    def stop(self):
        """Stop the bot orchestrator."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping bot orchestrator...")
        self.is_running = False
        self._stop_event.set()
        
        # Stop auto-learning
        if self.learner:
            self.learner.stop()
        
        self.logger.info("Bot orchestrator stopped")
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            "running": self.is_running,
            "current_state": self.state_machine.current_state.name if self.state_machine else None,
            "frame_count": self.frame_count,
            "apm": self.actions_this_minute,
            "game_state": self.state_extractor.get_state_summary(self.current_state) if self.state_extractor and self.current_state else None
        }


def create_bot_orchestrator(
    models_dir: str,
    dataset_dir: Optional[str] = None,
    **kwargs
) -> BrawlStarsOrchestrator:
    """
    Factory function to create bot orchestrator.
    
    Args:
        models_dir: Directory with YOLO models
        dataset_dir: Directory for auto-learning data
        **kwargs: Additional config options
        
    Returns:
        Configured BrawlStarsOrchestrator
    """
    config = BotConfig(
        models_dir=Path(models_dir),
        dataset_dir=Path(dataset_dir) if dataset_dir else None,
        **kwargs
    )
    
    return BrawlStarsOrchestrator(config)
