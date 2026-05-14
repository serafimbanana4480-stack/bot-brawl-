"""
map_strategy.py

Map-specific strategy generation for Brawl Stars.

Generates adaptive strategies based on map analysis:
- Initial positioning
- Rotation patterns
- Power cube routes
- Team positioning
- Late game strategies

Features:
- Strategy templates per map type
- Dynamic strategy adjustment
- Pathfinding integration
- Team coordination patterns
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np

logger = logging.getLogger(__name__)


class MapType(Enum):
    """Map categories."""
    OPEN = "open"  # Few walls, long sightlines
    CLOSED = "closed"  # Many walls, tight spaces
    SYMMETRICAL = "symmetrical"  # Balanced layout
    ASYMMETRICAL = "asymmetrical"  # Unbalanced layout
    MID_FOCUSED = "mid_focused"  # Center control important
    SPAWN_FOCUSED = "spawn_focused"  # Spawn points important


class GameMode(Enum):
    """Brawl Stars game modes."""
    SHOWDOWN = "showdown"
    GEM_GRAB = "gem_grab"
    BRAWL_BALL = "brawl_ball"
    HEIST = "heist"
    BOUNTY = "bounty"
    SIEGE = "siege"
    HOT_ZONE = "hot_zone"
    KNOCKOUT = "knockout"


# Mode-specific strategy modifiers
MODE_MODIFIERS = {
    GameMode.SHOWDOWN: {
        "aggression": 0.3,  # Low aggression - survival matters
        "cube_priority": 0.8,  # High cube priority
        "team_spread": 0.0,  # No team in solo showdown
        "retreat_threshold": 0.5,  # Retreat when HP < 50%
    },
    GameMode.GEM_GRAB: {
        "aggression": 0.5,
        "cube_priority": 0.0,
        "team_spread": 0.5,
        "retreat_threshold": 0.4,
        "gem_hold_priority": 0.9,  # Prioritize holding gems
    },
    GameMode.BRAWL_BALL: {
        "aggression": 0.7,  # High aggression
        "cube_priority": 0.0,
        "team_spread": 0.3,
        "retreat_threshold": 0.3,
        "ball_priority": 0.95,  # Prioritize ball possession
    },
    GameMode.HEIST: {
        "aggression": 0.8,  # Very aggressive - damage safe
        "cube_priority": 0.0,
        "team_spread": 0.4,
        "retreat_threshold": 0.6,  # Don't retreat much
        "safe_damage_priority": 0.9,  # Prioritize safe damage to safe
    },
    GameMode.BOUNTY: {
        "aggression": 0.2,  # Very careful - deaths are costly
        "cube_priority": 0.0,
        "team_spread": 0.6,
        "retreat_threshold": 0.7,  # Retreat early
    },
    GameMode.HOT_ZONE: {
        "aggression": 0.6,
        "cube_priority": 0.0,
        "team_spread": 0.4,
        "retreat_threshold": 0.4,
        "zone_control_priority": 0.85,
    },
}


@dataclass
class MapStrategy:
    """Strategy for a specific map."""
    map_name: str
    map_type: MapType
    
    # Initial positioning
    initial_position: Tuple[float, float]  # Normalized (0-1)
    initial_action: str  # "aggressive", "defensive", "mid_control"
    
    # Movement patterns
    patrol_route: List[Tuple[float, float]]  # Waypoints
    rotation_pattern: str  # "clockwise", "counter_clockwise", "adaptive"
    
    # Power cube strategy
    cube_route: List[Tuple[float, float]]
    cube_timing: str  # "early", "mid", "late", "continuous"
    
    # Team positioning (for 3v3)
    team_spread: float  # 0.0 (tight) to 1.0 (spread)
    support_positions: List[Tuple[float, float]]
    
    # Late game
    late_game_position: Tuple[float, float]
    late_game_action: str
    
    # Map-specific tactics
    special_tactics: List[str]


class MapStrategyGenerator:
    """
    Generates strategies based on map analysis.
    
    Creates adaptive strategies tailored to map layout and game mode.
    """
    
    def __init__(self, strategies_file: Optional[Path] = None):
        self.strategies_file = strategies_file or Path(__file__).parent.parent / "data" / "map_strategies.json"
        self.strategies: Dict[str, MapStrategy] = {}
        self._load_strategies()
    
    def _load_strategies(self):
        """Load predefined strategies from file."""
        if self.strategies_file.exists():
            try:
                with open(self.strategies_file, 'r') as f:
                    data = json.load(f)
                
                for name, strategy_data in data.items():
                    self.strategies[name] = MapStrategy(
                        map_name=name,
                        map_type=MapType(strategy_data.get('map_type', 'open')),
                        initial_position=tuple(strategy_data.get('initial_position', (0.5, 0.5))),
                        initial_action=strategy_data.get('initial_action', 'mid_control'),
                        patrol_route=[tuple(p) for p in strategy_data.get('patrol_route', [])],
                        rotation_pattern=strategy_data.get('rotation_pattern', 'adaptive'),
                        cube_route=[tuple(p) for p in strategy_data.get('cube_route', [])],
                        cube_timing=strategy_data.get('cube_timing', 'mid'),
                        team_spread=strategy_data.get('team_spread', 0.5),
                        support_positions=[tuple(p) for p in strategy_data.get('support_positions', [])],
                        late_game_position=tuple(strategy_data.get('late_game_position', (0.5, 0.5))),
                        late_game_action=strategy_data.get('late_game_action', 'defensive'),
                        special_tactics=strategy_data.get('special_tactics', []),
                    )
                
                logger.info(f"Loaded {len(self.strategies)} predefined strategies")
            except Exception as e:
                logger.error(f"Failed to load strategies: {e}")
    
    def classify_map(self, openness: float, symmetry: float, complexity: float) -> MapType:
        """
        Classify map type based on metrics.
        
        Args:
            openness: 0.0 (closed) to 1.0 (open)
            symmetry: 0.0 (asymmetric) to 1.0 (symmetric)
            complexity: 0.0 (simple) to 1.0 (complex)
            
        Returns:
            MapType classification
        """
        if openness > 0.7:
            return MapType.OPEN
        elif openness < 0.3:
            return MapType.CLOSED
        elif symmetry > 0.8:
            return MapType.SYMMETRICAL
        elif symmetry < 0.3:
            return MapType.ASYMMETRICAL
        elif complexity > 0.7:
            return MapType.MID_FOCUSED
        else:
            return MapType.SPAWN_FOCUSED
    
    def generate_strategy(
        self,
        map_name: str,
        map_type: MapType,
        game_mode: str = "showdown",
    ) -> MapStrategy:
        """
        Generate strategy for a map.
        
        Args:
            map_name: Name of the map
            map_type: Type of map
            game_mode: Game mode (showdown, gem_grab, etc.)
            
        Returns:
            MapStrategy object
        """
        # Check if we have a predefined strategy
        if map_name in self.strategies:
            logger.info(f"Using predefined strategy for {map_name}")
            return self.strategies[map_name]
        
        # Generate strategy based on map type
        strategy = self._generate_from_type(map_name, map_type, game_mode)
        
        # Save to database
        self.strategies[map_name] = strategy
        self._save_strategies()
        
        return strategy
    
    def _generate_from_type(self, map_name: str, map_type: MapType, game_mode: str) -> MapStrategy:
        """Generate strategy based on map type."""
        
        if map_type == MapType.OPEN:
            return self._generate_open_strategy(map_name, game_mode)
        elif map_type == MapType.CLOSED:
            return self._generate_closed_strategy(map_name, game_mode)
        elif map_type == MapType.SYMMETRICAL:
            return self._generate_symmetrical_strategy(map_name, game_mode)
        elif map_type == MapType.ASYMMETRICAL:
            return self._generate_asymmetrical_strategy(map_name, game_mode)
        elif map_type == MapType.MID_FOCUSED:
            return self._generate_mid_focused_strategy(map_name, game_mode)
        else:  # SPAWN_FOCUSED
            return self._generate_spawn_focused_strategy(map_name, game_mode)
    
    def _generate_open_strategy(self, map_name: str, game_mode: str) -> MapStrategy:
        """Generate strategy for open maps."""
        return MapStrategy(
            map_name=map_name,
            map_type=MapType.OPEN,
            initial_position=(0.5, 0.3),
            initial_action="mid_control",
            patrol_route=[
                (0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.5, 0.5)
            ],
            rotation_pattern="clockwise",
            cube_route=[(0.5, 0.5), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7)],
            cube_timing="continuous",
            team_spread=0.7,
            support_positions=[(0.3, 0.5), (0.7, 0.5)],
            late_game_position=(0.5, 0.5),
            late_game_action="aggressive",
            special_tactics=["long_range_engagement", "flanking_routes"],
        )
    
    def _generate_closed_strategy(self, map_name: str, game_mode: str) -> MapStrategy:
        """Generate strategy for closed maps."""
        return MapStrategy(
            map_name=map_name,
            map_type=MapType.CLOSED,
            initial_position=(0.5, 0.5),
            initial_action="defensive",
            patrol_route=[
                (0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6), (0.5, 0.5)
            ],
            rotation_pattern="adaptive",
            cube_route=[(0.5, 0.5), (0.4, 0.6), (0.6, 0.4)],
            cube_timing="mid",
            team_spread=0.3,
            support_positions=[(0.45, 0.5), (0.55, 0.5)],
            late_game_position=(0.5, 0.5),
            late_game_action="defensive",
            special_tactics=["corner_control", "ambush_points"],
        )
    
    def _generate_symmetrical_strategy(self, map_name: str, game_mode: str) -> MapStrategy:
        """Generate strategy for symmetrical maps."""
        return MapStrategy(
            map_name=map_name,
            map_type=MapType.SYMMETRICAL,
            initial_position=(0.5, 0.5),
            initial_action="mid_control",
            patrol_route=[
                (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7), (0.5, 0.5)
            ],
            rotation_pattern="clockwise",
            cube_route=[(0.5, 0.5)],
            cube_timing="mid",
            team_spread=0.5,
            support_positions=[(0.3, 0.3), (0.7, 0.7)],
            late_game_position=(0.5, 0.5),
            late_game_action="aggressive",
            special_tactics=["mirror_positions", "flank_symmetrically"],
        )
    
    def _generate_asymmetrical_strategy(self, map_name: str, game_mode: str) -> MapStrategy:
        """Generate strategy for asymmetrical maps."""
        return MapStrategy(
            map_name=map_name,
            map_type=MapType.ASYMMETRICAL,
            initial_position=(0.6, 0.5),  # Favor one side
            initial_action="aggressive",
            patrol_route=[
                (0.6, 0.5), (0.8, 0.5), (0.7, 0.3), (0.7, 0.7), (0.6, 0.5)
            ],
            rotation_pattern="adaptive",
            cube_route=[(0.7, 0.5), (0.8, 0.5), (0.6, 0.5)],
            cube_timing="early",
            team_spread=0.4,
            support_positions=[(0.5, 0.5), (0.8, 0.5)],
            late_game_position=(0.7, 0.5),
            late_game_action="aggressive",
            special_tactics=["exploit_asymmetry", "control_advantage_side"],
        )
    
    def _generate_mid_focused_strategy(self, map_name: str, game_mode: str) -> MapStrategy:
        """Generate strategy for mid-focused maps."""
        return MapStrategy(
            map_name=map_name,
            map_type=MapType.MID_FOCUSED,
            initial_position=(0.5, 0.5),
            initial_action="mid_control",
            patrol_route=[
                (0.5, 0.5), (0.4, 0.5), (0.6, 0.5), (0.5, 0.4), (0.5, 0.6)
            ],
            rotation_pattern="clockwise",
            cube_route=[(0.5, 0.5)],
            cube_timing="continuous",
            team_spread=0.4,
            support_positions=[(0.4, 0.5), (0.6, 0.5)],
            late_game_position=(0.5, 0.5),
            late_game_action="aggressive",
            special_tactics=["mid_domination", "deny_mid"],
        )
    
    def _generate_spawn_focused_strategy(self, map_name: str, game_mode: str) -> MapStrategy:
        """Generate strategy for spawn-focused maps."""
        return MapStrategy(
            map_name=map_name,
            map_type=MapType.SPAWN_FOCUSED,
            initial_position=(0.5, 0.2),  # Near spawn
            initial_action="defensive",
            patrol_route=[
                (0.5, 0.2), (0.3, 0.3), (0.7, 0.3), (0.5, 0.4), (0.5, 0.2)
            ],
            rotation_pattern="counter_clockwise",
            cube_route=[(0.5, 0.3), (0.3, 0.3), (0.7, 0.3)],
            cube_timing="early",
            team_spread=0.3,
            support_positions=[(0.3, 0.2), (0.7, 0.2)],
            late_game_position=(0.5, 0.5),
            late_game_action="aggressive",
            special_tactics=["spawn_control", "rotate_to_mid"],
        )
    
    def _save_strategies(self):
        """Save strategies to file."""
        self.strategies_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for name, strategy in self.strategies.items():
            data[name] = {
                'map_type': strategy.map_type.value,
                'initial_position': strategy.initial_position,
                'initial_action': strategy.initial_action,
                'patrol_route': strategy.patrol_route,
                'rotation_pattern': strategy.rotation_pattern,
                'cube_route': strategy.cube_route,
                'cube_timing': strategy.cube_timing,
                'team_spread': strategy.team_spread,
                'support_positions': strategy.support_positions,
                'late_game_position': strategy.late_game_position,
                'late_game_action': strategy.late_game_action,
                'special_tactics': strategy.special_tactics,
            }
        
        with open(self.strategies_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.strategies)} strategies")
    
    def get_strategy(self, map_name: str, map_type: MapType, game_mode: str = "showdown") -> MapStrategy:
        """
        Get strategy for a map, adjusted for game mode.
        
        Args:
            map_name: Name of the map
            map_type: Type of map
            game_mode: Game mode
            
        Returns:
            MapStrategy object (adjusted for game mode if recognized)
        """
        strategy = self.generate_strategy(map_name, map_type, game_mode)
        
        # Apply game mode modifiers
        try:
            mode = GameMode(game_mode)
            strategy = self.adjust_strategy_for_mode(strategy, mode)
        except ValueError:
            logger.debug(f"Unknown game mode '{game_mode}', using base strategy")
        
        return strategy

    def get_mode_modifier(self, game_mode: GameMode) -> Dict:
        """Get strategy modifier parameters for a game mode.
        
        Args:
            game_mode: GameMode enum value
            
        Returns:
            Dict of modifier values (aggression, cube_priority, etc.)
            Returns empty dict if mode has no specific modifiers.
        """
        return MODE_MODIFIERS.get(game_mode, {})

    def adjust_strategy_for_mode(self, strategy: MapStrategy, game_mode: GameMode) -> MapStrategy:
        """Adjust an existing strategy based on game mode modifiers.
        
        Applies mode-specific parameters like aggression level,
        retreat threshold, and special priorities (gems, ball, etc.)
        to an existing base strategy.
        
        Args:
            strategy: Base MapStrategy to adjust
            game_mode: GameMode enum value
            
        Returns:
            Adjusted MapStrategy (same object, modified in-place)
        """
        modifiers = self.get_mode_modifier(game_mode)
        if not modifiers:
            return strategy

        # Adjust initial action based on aggression level
        aggression = modifiers.get("aggression", 0.5)
        if aggression >= 0.7:
            strategy.initial_action = "aggressive"
        elif aggression <= 0.3:
            strategy.initial_action = "defensive"
        else:
            strategy.initial_action = "mid_control"

        # Adjust team spread
        team_spread = modifiers.get("team_spread", 0.5)
        strategy.team_spread = team_spread

        # Mode-specific adjustments
        if game_mode == GameMode.SHOWDOWN:
            # Showdown: prioritize power cubes
            cube_timing = modifiers.get("cube_priority", 0.0) > 0.5
            if cube_timing:
                strategy.cube_timing = "early"
            # Late game: play defensively (survival)
            strategy.late_game_action = "defensive"

        elif game_mode == GameMode.GEM_GRAB:
            # Gem Grab: center control is critical
            strategy.initial_position = (0.5, 0.5)  # Center
            strategy.initial_action = "mid_control"
            strategy.late_game_action = "defensive"  # Hold gems

        elif game_mode == GameMode.BRAWL_BALL:
            # Brawl Ball: aggressive, push forward
            strategy.initial_action = "aggressive"
            strategy.rotation_pattern = "adaptive"
            strategy.late_game_action = "aggressive"

        elif game_mode == GameMode.HEIST:
            # Heist: rush enemy safe
            strategy.initial_action = "aggressive"
            strategy.late_game_action = "aggressive"

        elif game_mode == GameMode.BOUNTY:
            # Bounty: stay alive, long range
            strategy.initial_action = "defensive"
            strategy.team_spread = 0.7  # Spread out to avoid team wipes
            strategy.late_game_action = "defensive"

        elif game_mode == GameMode.HOT_ZONE:
            # Hot Zone: control zones
            strategy.initial_position = (0.5, 0.5)
            strategy.initial_action = "mid_control"

        logger.info(
            f"[MAP_STRATEGY] Adjusted strategy for {game_mode.value}: "
            f"action={strategy.initial_action}, spread={strategy.team_spread:.1f}"
        )

        return strategy


def main():
    """Test map strategy generator."""
    logging.basicConfig(level=logging.INFO)
    
    generator = MapStrategyGenerator()
    
    # Test strategy generation
    strategy = generator.generate_strategy("test_map", MapType.OPEN, "showdown")
    print(f"Generated strategy: {strategy}")


if __name__ == "__main__":
    main()
