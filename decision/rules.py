"""
Rule-based strategy engine for Brawl Stars bot.
Implements tactical decision making based on game state.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import random
import math


class Tactic(Enum):
    """Available tactical maneuvers."""
    ENGAGE_CLOSE = "engage_close"
    ENGAGE_RANGED = "engage_ranged"
    HARASS = "harass"
    FLANK = "flank"
    AMBUSH = "ambush"
    RETREAT_DEFENSIVE = "retreat_defensive"
    RETREAT_AGGRESSIVE = "retreat_aggressive"
    CIRCLE_STRAFE = "circle_strafe"
    TAKE_COVER = "take_cover"
    HEAL_UP = "heal_up"
    PATROL = "patrol"
    HOLD_POSITION = "hold_position"


@dataclass
class TacticalDecision:
    """A tactical decision with execution parameters."""
    tactic: Tactic
    target_position: Optional[Tuple[float, float]]
    priority: float  # 0.0 to 1.0
    reasoning: str
    execution_params: Dict


class RuleEngine:
    """
    Rule-based strategy engine.
    Evaluates game state and recommends tactics.
    """
    
    def __init__(
        self,
        engagement_range: float = 200.0,
        optimal_range: float = 150.0,
        danger_health_threshold: float = 0.3,
        retreat_health_threshold: float = 0.2
    ):
        self.engagement_range = engagement_range
        self.optimal_range = optimal_range
        self.danger_health_threshold = danger_health_threshold
        self.retreat_health_threshold = retreat_health_threshold
        
    def _calculate_flank_position(
        self,
        target_pos: Tuple[float, float],
        player_pos: Tuple[float, float],
        walls: List
    ) -> Tuple[float, float]:
        """
        Calculate a flanking position around target.
        """
        # Vector from target to player
        dx = player_pos[0] - target_pos[0]
        dy = player_pos[1] - target_pos[1]
        
        # Perpendicular vector (90 degree rotation)
        flank_dx = -dy
        flank_dy = dx
        
        # Normalize and scale
        length = math.sqrt(flank_dx**2 + flank_dy**2)
        if length > 0:
            flank_dx = flank_dx / length * self.engagement_range
            flank_dy = flank_dy / length * self.engagement_range
        
        # Add some randomness (left or right flank)
        if random.random() < 0.5:
            flank_dx = -flank_dx
            flank_dy = -flank_dy
        
        return (target_pos[0] + flank_dx, target_pos[1] + flank_dy)
    
    def _find_safe_retreat_point(
        self,
        player_pos: Tuple[float, float],
        enemies: List,
        bushes: List,
        walls: List
    ) -> Optional[Tuple[float, float]]:
        """
        Find a safe point to retreat to.
        Priority: safe bushes > away from enemies > random direction
        """
        if not enemies:
            return None
        
        # Calculate centroid of enemies (danger center)
        enemy_centroid = (
            sum(e.position[0] for e in enemies) / len(enemies),
            sum(e.position[1] for e in enemies) / len(enemies)
        )
        
        # Vector away from enemies
        dx = player_pos[0] - enemy_centroid[0]
        dy = player_pos[1] - enemy_centroid[1]
        
        # Normalize
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx = dx / length * 300  # Retreat 300 units
            dy = dy / length * 300
        
        # Check for safe bushes in that general direction
        retreat_direction = (dx, dy)
        best_bush = None
        best_score = float('inf')
        
        for bush in bushes:
            if bush.enemies_nearby > 0:
                continue
            
            # Bush should be in retreat direction
            bush_dx = bush.center[0] - player_pos[0]
            bush_dy = bush.center[1] - player_pos[1]
            
            # Score based on alignment with retreat direction
            alignment = (bush_dx * retreat_direction[0] + bush_dy * retreat_direction[1])
            distance = math.sqrt(bush_dx**2 + bush_dy**2)
            
            if alignment > 0 and distance < 400:  # Bush is in retreat direction
                score = distance - alignment * 0.1
                if score < best_score:
                    best_score = score
                    best_bush = bush
        
        if best_bush:
            return best_bush.center
        
        # No safe bush, just retreat away
        return (player_pos[0] + dx, player_pos[1] + dy)
    
    def evaluate_engagement(
        self,
        game_state
    ) -> List[TacticalDecision]:
        """
        Evaluate engagement opportunities and return ranked decisions.
        """
        decisions = []
        
        if not game_state.enemies or not game_state.player_position:
            return decisions
        
        nearest = game_state.nearest_enemy
        if not nearest:
            return decisions
        
        player_pos = game_state.player_position
        
        # Decision 1: Close engagement (shotgun/rush brawlers)
        if nearest.distance < self.engagement_range * 0.8:
            if game_state.player_health > 0.5:
                decisions.append(TacticalDecision(
                    tactic=Tactic.ENGAGE_CLOSE,
                    target_position=nearest.position,
                    priority=0.7 + (1 - nearest.health_estimate) * 0.3,
                    reasoning="Close range advantage, enemy health low",
                    execution_params={
                        "strafe": True,
                        "auto_aim": True,
                        "shoot_while_moving": True
                    }
                ))
        
        # Decision 2: Ranged engagement (sharpshooters)
        if nearest.distance > self.engagement_range * 0.5:
            if game_state.can_engage:
                # Find optimal firing position
                optimal_pos = self._calculate_optimal_position(
                    player_pos, nearest.position, game_state.walls
                )
                decisions.append(TacticalDecision(
                    tactic=Tactic.ENGAGE_RANGED,
                    target_position=optimal_pos or nearest.position,
                    priority=0.6,
                    reasoning="Maintain optimal range",
                    execution_params={
                        "strafe": True,
                        "leading_shots": True,
                        "peek_shoot": True
                    }
                ))
        
        # Decision 3: Harassment (chip damage)
        if game_state.player_health > 0.7:
            decisions.append(TacticalDecision(
                tactic=Tactic.HARASS,
                target_position=nearest.position,
                priority=0.4,
                reasoning="Safe harassment opportunity",
                execution_params={
                    "shoot_and_retreat": True,
                    "max_shots": 1
                }
            ))
        
        # Decision 4: Flanking
        if len(game_state.enemies) == 1 and game_state.player_health > 0.6:
            flank_pos = self._calculate_flank_position(
                nearest.position, player_pos, game_state.walls
            )
            decisions.append(TacticalDecision(
                tactic=Tactic.FLANK,
                target_position=flank_pos,
                priority=0.5 + (0.3 if nearest.health_estimate < 0.5 else 0),
                reasoning="Flank for advantage",
                execution_params={
                    "approach_stealthily": True,
                    "burst_damage": True
                }
            ))
        
        # Decision 5: Circle strafe (for 1v1)
        if len(game_state.enemies) == 1:
            decisions.append(TacticalDecision(
                tactic=Tactic.CIRCLE_STRAFE,
                target_position=nearest.position,
                priority=0.5,
                reasoning="Dodge enemy shots",
                execution_params={
                    "clockwise": random.random() < 0.5,
                    "radius": max(100, nearest.distance * 0.8)
                }
            ))
        
        return sorted(decisions, key=lambda d: -d.priority)
    
    def evaluate_retreat(
        self,
        game_state
    ) -> List[TacticalDecision]:
        """
        Evaluate retreat options when in danger.
        """
        decisions = []
        
        if not game_state.player_position:
            return decisions
        
        player_pos = game_state.player_position
        
        # Decision 1: Defensive retreat to safe position
        retreat_point = self._find_safe_retreat_point(
            player_pos,
            game_state.enemies,
            game_state.safe_bushes,
            game_state.walls
        )
        
        if retreat_point:
            decisions.append(TacticalDecision(
                tactic=Tactic.RETREAT_DEFENSIVE,
                target_position=retreat_point,
                priority=0.9 if game_state.player_health < 0.3 else 0.7,
                reasoning="Retreat to safety",
                execution_params={
                    "shoot_while_retreating": game_state.player_health > 0.4,
                    "use_abilities": True,
                    "juke": True
                }
            ))
        
        # Decision 2: Take cover behind walls
        if game_state.walls:
            cover_pos = self._find_cover_position(player_pos, game_state.enemies, game_state.walls)
            if cover_pos:
                decisions.append(TacticalDecision(
                    tactic=Tactic.TAKE_COVER,
                    target_position=cover_pos,
                    priority=0.8,
                    reasoning="Use cover",
                    execution_params={
                        "peek_shoot": True,
                        "wait_for_heal": True
                    }
                ))
        
        # Decision 3: Aggressive retreat (trading kills if possible)
        if game_state.biggest_threat and game_state.biggest_threat.health_estimate < 0.3:
            decisions.append(TacticalDecision(
                tactic=Tactic.RETREAT_AGGRESSIVE,
                target_position=game_state.biggest_threat.position,
                priority=0.6,
                reasoning="Try to trade kill before dying",
                execution_params={
                    "all_in": True,
                    "use_super": True
                }
            ))
        
        return sorted(decisions, key=lambda d: -d.priority)
    
    def evaluate_recovery(
        self,
        game_state
    ) -> List[TacticalDecision]:
        """
        Evaluate recovery options (healing, repositioning).
        """
        decisions = []
        
        if not game_state.player_position:
            return decisions
        
        # Decision 1: Heal up in safe bush
        if game_state.safe_bushes and game_state.player_health < 1.0:
            best_bush = min(
                game_state.safe_bushes,
                key=lambda b: self._distance(game_state.player_position, b.center)
            )
            decisions.append(TacticalDecision(
                tactic=Tactic.HEAL_UP,
                target_position=best_bush.center,
                priority=0.8,
                reasoning="Heal in safe bush",
                execution_params={
                    "wait_for_full": True,
                    "stay_hidden": True
                }
            ))
        
        # Decision 2: Hold position if safe
        if game_state.danger_score < 0.2 and game_state.player_health < 1.0:
            decisions.append(TacticalDecision(
                tactic=Tactic.HOLD_POSITION,
                target_position=None,
                priority=0.6,
                reasoning="Wait for natural healing",
                execution_params={
                    "duration": 3.0,
                    "scan_for_enemies": True
                }
            ))
        
        return sorted(decisions, key=lambda d: -d.priority)
    
    def evaluate_search(
        self,
        game_state
    ) -> List[TacticalDecision]:
        """
        Evaluate search/patrol options when no enemies visible.
        """
        decisions = []
        
        if not game_state.player_position:
            return decisions
        
        player_pos = game_state.player_position
        
        # Decision 1: Patrol common areas
        patrol_points = self._generate_patrol_points(player_pos, game_state.bushes)
        if patrol_points:
            decisions.append(TacticalDecision(
                tactic=Tactic.PATROL,
                target_position=patrol_points[0],
                priority=0.5,
                reasoning="Search for enemies",
                execution_params={
                    "patrol_points": patrol_points,
                    "check_bushes": True,
                    "move_unpredictably": True
                }
            ))
        
        # Decision 2: Ambush position
        if game_state.bushes:
            ambush_bush = max(
                game_state.bushes,
                key=lambda b: self._count_nearby_enemy_spawns(b, game_state)
            )
            decisions.append(TacticalDecision(
                tactic=Tactic.AMBUSH,
                target_position=ambush_bush.center,
                priority=0.6,
                reasoning="Set up ambush",
                execution_params={
                    "wait_for_enemy": True,
                    "first_shot_advantage": True
                }
            ))
        
        return sorted(decisions, key=lambda d: -d.priority)
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_optimal_position(
        self,
        player_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        walls: List
    ) -> Optional[Tuple[float, float]]:
        """Calculate optimal firing position."""
        dx = target_pos[0] - player_pos[0]
        dy = target_pos[1] - player_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist == 0:
            return None
        
        # Scale to optimal range
        scale = self.optimal_range / dist
        optimal_x = target_pos[0] - dx * scale
        optimal_y = target_pos[1] - dy * scale
        
        return (optimal_x, optimal_y)
    
    def _find_cover_position(
        self,
        player_pos: Tuple[float, float],
        enemies: List,
        walls: List
    ) -> Optional[Tuple[float, float]]:
        """Find position behind cover."""
        if not enemies or not walls:
            return None
        
        # Simplified: find wall between player and nearest enemy
        nearest = min(enemies, key=lambda e: self._distance(player_pos, e.position))
        
        for wall in walls:
            # Check if wall is between player and enemy
            wall_center = wall.center
            
            # Simple check: wall should be closer to enemy than player
            dist_to_enemy = self._distance(wall_center, nearest.position)
            dist_to_player = self._distance(wall_center, player_pos)
            
            if dist_to_enemy < dist_to_player:
                return (wall_center[0], wall_center[1] + 50)  # Slightly offset
        
        return None
    
    def _generate_patrol_points(
        self,
        player_pos: Tuple[float, float],
        bushes: List
    ) -> List[Tuple[float, float]]:
        """Generate patrol points around the map."""
        points = [bush.center for bush in bushes[:3]]  # Top 3 bushes
        
        if not points:
            # Generate some random points around player
            for _ in range(3):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(200, 400)
                x = player_pos[0] + math.cos(angle) * dist
                y = player_pos[1] + math.sin(angle) * dist
                points.append((x, y))
        
        return points
    
    def _count_nearby_enemy_spawns(self, bush, game_state) -> int:
        """Estimate how many enemies might pass by this bush."""
        # Simplified heuristic: center bushes are more likely to see traffic
        center_x = 960  # Assuming 1920x1080
        center_y = 540
        
        dist_to_center = self._distance(bush.center, (center_x, center_y))
        return max(0, int(5 - dist_to_center / 200))
