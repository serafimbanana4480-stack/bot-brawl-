"""
core/tactical_bridge.py

Tactical Planner Bridge — connects enterprise tactical agents
to the real game pipeline.

The enterprise/ directory has sophisticated agents (TacticalPlannerAgent,
NavigationAgent, SupervisorAgent, MemorySystem) but they are dead code —
never called by play.py or wrapper.py.

This bridge:
1. Creates and manages enterprise agent instances
2. Translates real game state into enterprise agent format
3. Translates enterprise agent outputs back to real game actions
4. Handles graceful fallback if enterprise code fails
5. Provides a clean API for wrapper.py integration

Usage in wrapper.py:
    bridge = TacticalBridge()
    bridge.initialize(world_model, occupancy_grid, pressure_map)
    
    # Each frame:
    tactical_input = bridge.build_tactical_context(detections, player_state)
    plan = bridge.get_tactical_plan(tactical_input)
    
    # Use plan for high-level decisions
    if plan.should_retreat:
        ...
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TacticalPhase(Enum):
    """Match phases for tactical planning."""
    EARLY = "early"       # First 30s: farm, position
    MID = "mid"           # 30-90s: contest, control
    LATE = "late"         # 90s+: survive, final fights


@dataclass
class TacticalPlan:
    """Output from the tactical planner."""
    phase: TacticalPhase = TacticalPhase.EARLY
    should_retreat: bool = False
    should_push: bool = False
    should_farm: bool = False
    should_hold: bool = False
    priority_target_id: Optional[int] = None
    safe_zone: Optional[Tuple[float, float]] = None
    danger_zone: Optional[Tuple[float, float]] = None
    recommended_position: Optional[Tuple[float, float]] = None
    team_strategy: str = "default"  # "push_left", "push_right", "hold_center", "default"
    confidence: float = 0.0
    reasoning: str = ""


class TacticalBridge:
    """
    Bridge between enterprise tactical agents and the real game pipeline.
    
    Attempts to use enterprise agents if available, falls back to
    heuristic tactical planning if they fail or aren't importable.
    """

    def __init__(self):
        self._enterprise_available = False
        self._tactical_agent = None
        self._supervisor_agent = None
        self._navigation_agent = None
        self._memory_system = None
        
        # Fallback tactical state
        self._current_plan = TacticalPlan()
        self._match_start_time: float = 0.0
        self._last_plan_time: float = 0.0
        self._plan_interval: float = 1.0  # Re-plan every 1s
        
        # References to core systems
        self._world_model = None
        self._occupancy_grid = None
        self._pressure_map = None
        
        self._lock = threading.RLock()
        
        # Try to import enterprise agents
        self._try_import_enterprise()
        
        logger.info("[TACTICAL_BRIDGE] Initialized (enterprise=%s)",
                     self._enterprise_available)
    
    def _try_import_enterprise(self):
        """Try to import enterprise agents. Graceful failure."""
        try:
            from enterprise.agents.tactical import TacticalPlannerAgent
            self._tactical_agent = TacticalPlannerAgent()
            self._enterprise_available = True
            logger.info("[TACTICAL_BRIDGE] Enterprise tactical agent loaded")
        except (ImportError, Exception) as e:
            logger.info("[TACTICAL_BRIDGE] Enterprise not available: %s", e)
            self._enterprise_available = False
    
    def initialize(self, world_model=None, occupancy_grid=None, pressure_map=None):
        """Set references to core systems."""
        self._world_model = world_model
        self._occupancy_grid = occupancy_grid
        self._pressure_map = pressure_map
        logger.info("[TACTICAL_BRIDGE] Core systems linked")
    
    def start_match(self, game_mode: str = "showdown"):
        """Called when a new match starts."""
        self._match_start_time = time.time()
        self._current_plan = TacticalPlan(
            phase=TacticalPhase.EARLY,
            should_farm=True,
            reasoning="match_start",
        )
    
    def build_tactical_context(self, detections: List[Dict],
                                player_state: Dict) -> Dict:
        """
        Build tactical context from real game data.
        
        Translates YOLO detections and player state into a format
        suitable for tactical planning.
        """
        now = time.time()
        match_time = now - self._match_start_time if self._match_start_time > 0 else 0
        
        # Determine phase
        if match_time < 30:
            phase = TacticalPhase.EARLY
        elif match_time < 90:
            phase = TacticalPhase.MID
        else:
            phase = TacticalPhase.LATE
        
        context = {
            "match_time": match_time,
            "phase": phase.value,
            "game_mode": player_state.get("game_mode", "showdown"),
            "health": player_state.get("health", 1.0),
            "ammo": player_state.get("ammo", 3),
            "has_super": player_state.get("has_super", False),
            "position": player_state.get("position", (640, 600)),
            "enemies": [],
            "allies": [],
            "power_cubes": [],
            "pressure": 0.0,
            "danger": 0.0,
        }
        
        # Process detections
        for det in detections:
            cls = det.get("class_name", "").lower()
            if cls in ("enemy", "opponent"):
                context["enemies"].append(det)
            elif cls in ("ally", "teammate"):
                context["allies"].append(det)
            elif cls in ("powerup", "power_cube"):
                context["power_cubes"].append(det)
        
        # Add pressure/danger from PressureMap
        if self._pressure_map:
            pos = context["position"]
            context["pressure"] = self._pressure_map.get_pressure_at(pos[0], pos[1])
            context["danger"] = self._pressure_map.get_influence_at(pos[0], pos[1])
        
        return context
    
    def get_tactical_plan(self, context: Dict) -> TacticalPlan:
        """
        Get the current tactical plan.
        
        Uses enterprise agent if available, otherwise heuristic planning.
        Re-plans at intervals (not every frame).
        """
        now = time.time()
        
        with self._lock:
            # Check if we need to re-plan
            if now - self._last_plan_time < self._plan_interval:
                return self._current_plan
            
            self._last_plan_time = now
            
            # Try enterprise first
            if self._enterprise_available and self._tactical_agent:
                try:
                    plan = self._plan_enterprise(context)
                    if plan:
                        self._current_plan = plan
                        return plan
                except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                    logger.warning("[TACTICAL_BRIDGE] Enterprise planning failed: %s", e)
            
            # Fallback: heuristic planning
            plan = self._plan_heuristic(context)
            self._current_plan = plan
            return plan
    
    def get_current_plan(self) -> TacticalPlan:
        """Get current plan without re-evaluation."""
        with self._lock:
            return self._current_plan
    
    def _plan_enterprise(self, context: Dict) -> Optional[TacticalPlan]:
        """Use enterprise tactical agent for planning."""
        # This would call the enterprise agent's plan method
        # For now, return None to trigger fallback
        return None
    
    def _plan_heuristic(self, context: Dict) -> TacticalPlan:
        """
        Heuristic tactical planning when enterprise is unavailable.
        
        Simple rules-based planning that considers:
        - Match phase (early=farm, mid=contest, late=survive)
        - Health and pressure
        - Enemy count and positions
        - Game mode
        """
        phase_str = context.get("phase", "early")
        phase = TacticalPhase(phase_str) if phase_str in [p.value for p in TacticalPhase] else TacticalPhase.EARLY
        health = context.get("health", 1.0)
        pressure = context.get("pressure", 0.0)
        enemies = context.get("enemies", [])
        game_mode = context.get("game_mode", "showdown")
        has_super = context.get("has_super", False)
        cubes = len(context.get("power_cubes", []))
        
        plan = TacticalPlan(phase=phase)
        reasons = []
        
        # Phase-based defaults
        if phase == TacticalPhase.EARLY:
            plan.should_farm = True
            reasons.append("early_farm")
        elif phase == TacticalPhase.MID:
            plan.should_hold = True
            reasons.append("mid_hold")
        else:
            plan.should_retreat = health < 0.5
            reasons.append("late_survive" if plan.should_retreat else "late_fight")
        
        # Override based on health/pressure
        if health < 0.3 or pressure > 4.0:
            plan.should_retreat = True
            plan.should_push = False
            plan.should_farm = False
            reasons.append("critical_retreat")
        
        # Super available + healthy = push opportunity
        if has_super and health > 0.6 and len(enemies) > 0:
            plan.should_push = True
            plan.should_retreat = False
            reasons.append("super_push")
        
        # Game mode overrides
        if game_mode in ("gem_grab", "hot_zone"):
            plan.should_hold = True
            plan.team_strategy = "hold_center"
        elif game_mode == "brawl_ball":
            plan.should_push = True
            plan.team_strategy = "push_left"
        elif game_mode == "heist":
            plan.should_push = True
            plan.team_strategy = "push_right"
        
        # Safe zone from pressure map
        if self._pressure_map and pressure > 2.0:
            pos = context.get("position", (640, 600))
            safe_dir = self._pressure_map.get_safest_direction(pos[0], pos[1])
            if safe_dir != (0.0, 0.0):
                plan.safe_zone = (
                    pos[0] + safe_dir[0] * 200,
                    pos[1] + safe_dir[1] * 200,
                )
        
        # Priority target
        if enemies:
            # Target lowest-health enemy
            best = min(enemies, key=lambda e: e.get("health", 1.0))
            plan.priority_target_id = best.get("track_id")
        
        plan.reasoning = " | ".join(reasons)
        plan.confidence = 0.6  # Heuristic confidence
        
        return plan
    
    def end_match(self, result: str = "unknown"):
        """Called when match ends."""
        logger.info("[TACTICAL_BRIDGE] Match ended: %s", result)
        self._match_start_time = 0.0
