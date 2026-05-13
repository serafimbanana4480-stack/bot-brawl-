"""
State extraction module for Brawl Stars.
Converts raw detections and tracks into structured game state.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .tracker import TrackedObject


class GamePhase(Enum):
    """Current phase of the game."""
    UNKNOWN = "unknown"
    LOBBY = "lobby"
    MATCHMAKING = "matchmaking"
    LOADING = "loading"
    GAMEPLAY = "gameplay"
    END_SCREEN = "end_screen"


class PlayerState(Enum):
    """State of the player character."""
    ALIVE = "alive"
    DEAD = "dead"
    RESPAWNING = "respawning"
    UNKNOWN = "unknown"


@dataclass
class EnemyInfo:
    """Information about an enemy player."""
    track_id: int
    position: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    health_estimate: float  # 0.0 to 1.0
    distance: float
    velocity: Tuple[float, float]
    threat_level: float  # calculated threat score
    visible: bool = True
    in_bush: bool = False
    
    @property
    def is_dangerous(self) -> bool:
        """Check if enemy is considered dangerous."""
        return self.threat_level > 0.6 or self.health_estimate > 0.7


@dataclass
class WallInfo:
    """Information about a wall obstacle."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    blocks_line_of_sight: bool = False


@dataclass
class BushInfo:
    """Information about a bush/hiding spot."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    occupied: bool = False
    enemies_nearby: int = 0


@dataclass
class GameState:
    """Complete structured game state."""
    # Game metadata
    phase: GamePhase = GamePhase.UNKNOWN
    match_time_remaining: Optional[float] = None
    
    # Player info
    player_position: Optional[Tuple[float, float]] = None
    player_bbox: Optional[Tuple[int, int, int, int]] = None
    player_health: float = 1.0
    player_state: PlayerState = PlayerState.UNKNOWN
    player_ammo: int = 3  # Default for most brawlers
    player_super_charged: bool = False
    
    # Environment
    enemies: List[EnemyInfo] = field(default_factory=list)
    walls: List[WallInfo] = field(default_factory=list)
    bushes: List[BushInfo] = field(default_factory=list)
    
    # Strategic info
    nearest_enemy: Optional[EnemyInfo] = None
    lowest_hp_enemy: Optional[EnemyInfo] = None
    biggest_threat: Optional[EnemyInfo] = None
    safe_bushes: List[BushInfo] = field(default_factory=list)
    
    # Danger assessment
    danger_score: float = 0.0  # 0.0 to 1.0
    enemies_in_range: int = 0
    escape_routes: int = 0
    
    @property
    def is_in_danger(self) -> bool:
        """Quick check if player is in danger."""
        return self.danger_score > 0.6 or len(self.enemies) > 2
    
    @property
    def can_engage(self) -> bool:
        """Check if safe to engage."""
        return self.danger_score < 0.4 and self.player_health > 0.5
    
    @property
    def should_retreat(self) -> bool:
        """Check if should retreat."""
        return self.danger_score > 0.7 or self.player_health < 0.3


class StateExtractor:
    """
    Extracts structured game state from vision detections and tracks.
    """
    
    # Class names expected from YOLO model
    CLASS_PLAYER = "player"
    CLASS_ENEMY = "enemy"
    CLASS_WALL = "wall"
    CLASS_BUSH = "bush"
    CLASS_HEALTH_BAR = "health_bar"
    CLASS_AMMO = "ammo"
    CLASS_SUPER = "super"
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        danger_distance: float = 300.0
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.danger_distance = danger_distance
        
    def _calculate_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _estimate_health_from_bar(
        self,
        health_bar_bbox: Tuple[int, int, int, int],
        player_bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Estimate player health from health bar detection.
        Simple heuristic based on bar width relative to player width.
        """
        bar_width = health_bar_bbox[2] - health_bar_bbox[0]
        player_width = player_bbox[2] - player_bbox[0]
        
        if player_width == 0:
            return 1.0
        
        ratio = bar_width / player_width
        # Health bar is typically ~80% of player width at full health
        health = min(1.0, max(0.0, ratio / 0.8))
        return health
    
    def _calculate_threat_level(
        self,
        enemy: TrackedObject,
        player_position: Tuple[float, float],
        player_health: float
    ) -> float:
        """
        Calculate threat level of an enemy.
        Based on distance, enemy health, and velocity.
        """
        distance = self._calculate_distance(enemy.center, player_position)
        
        # Base threat from distance (closer = more threat)
        distance_threat = 1.0 - min(1.0, distance / self.danger_distance)
        
        # Velocity threat (moving toward player)
        velocity_magnitude = np.sqrt(enemy.velocity[0]**2 + enemy.velocity[1]**2)
        velocity_threat = min(1.0, velocity_magnitude / 50.0)
        
        # Health factor (damaged enemies less threatening)
        health_threat = 0.7  # Default if no health info
        
        # Combine factors
        threat = (
            distance_threat * 0.4 +
            velocity_threat * 0.3 +
            health_threat * 0.3
        )
        
        return min(1.0, threat)
    
    def extract_state(
        self,
        tracks: List[TrackedObject],
        raw_detections: Optional[List] = None
    ) -> GameState:
        """
        Extract structured game state from tracking data.
        
        Args:
            tracks: List of tracked objects
            raw_detections: Optional raw YOLO detections for additional info
            
        Returns:
            Structured GameState object
        """
        state = GameState()
        
        # Find player
        player_track = None
        for track in tracks:
            if track.class_name == self.CLASS_PLAYER:
                player_track = track
                break
        
        if player_track:
            state.player_position = player_track.center
            state.player_bbox = player_track.bbox
            state.player_state = PlayerState.ALIVE
            
            # Try to estimate health from health bar detections
            if raw_detections:
                for det in raw_detections:
                    if hasattr(det, 'class_name') and det.class_name == self.CLASS_HEALTH_BAR:
                        # Check if health bar is near player
                        bar_center = (
                            (det.bbox[0] + det.bbox[2]) / 2,
                            (det.bbox[1] + det.bbox[3]) / 2
                        )
                        dist_to_player = self._calculate_distance(bar_center, player_track.center)
                        if dist_to_player < 100:  # Health bar should be close to player
                            state.player_health = self._estimate_health_from_bar(
                                det.bbox, player_track.bbox
                            )
        else:
            state.player_state = PlayerState.DEAD  # Or loading/unknown
        
        # Process enemies
        for track in tracks:
            if track.class_name == self.CLASS_ENEMY:
                if state.player_position:
                    distance = self._calculate_distance(track.center, state.player_position)
                    threat = self._calculate_threat_level(
                        track, state.player_position, state.player_health
                    )
                else:
                    distance = float('inf')
                    threat = 0.0
                
                enemy = EnemyInfo(
                    track_id=track.id,
                    position=track.center,
                    bbox=track.bbox,
                    health_estimate=0.7,  # Default assumption
                    distance=distance,
                    velocity=track.velocity,
                    threat_level=threat,
                    visible=True,
                    in_bush=False  # Would need specific detection
                )
                state.enemies.append(enemy)
        
        # Process walls
        for track in tracks:
            if track.class_name == self.CLASS_WALL:
                wall = WallInfo(
                    track_id=track.id,
                    bbox=track.bbox,
                    center=track.center,
                    blocks_line_of_sight=False  # Would need ray-casting
                )
                state.walls.append(wall)
        
        # Process bushes
        for track in tracks:
            if track.class_name == self.CLASS_BUSH:
                bush = BushInfo(
                    track_id=track.id,
                    bbox=track.bbox,
                    center=track.center,
                    occupied=False,
                    enemies_nearby=0
                )
                state.bushes.append(bush)
        
        # Calculate strategic info
        if state.enemies:
            # Sort by distance
            state.enemies.sort(key=lambda e: e.distance)
            state.nearest_enemy = state.enemies[0]
            
            # Find lowest HP enemy
            state.lowest_hp_enemy = min(state.enemies, key=lambda e: e.health_estimate)
            
            # Find biggest threat
            state.biggest_threat = max(state.enemies, key=lambda e: e.threat_level)
            
            # Count enemies in attack range
            state.enemies_in_range = sum(
                1 for e in state.enemies if e.distance < self.danger_distance
            )
        
        # Find safe bushes (not near enemies)
        for bush in state.bushes:
            if state.player_position:
                dist_to_player = self._calculate_distance(bush.center, state.player_position)
                # Bush is safe if no enemies nearby
                enemies_near = sum(
                    1 for e in state.enemies
                    if self._calculate_distance(bush.center, e.position) < 150
                )
                bush.enemies_nearby = enemies_near
                if enemies_near == 0 and dist_to_player > 50:
                    state.safe_bushes.append(bush)
        
        # Calculate overall danger score
        if state.enemies:
            avg_threat = sum(e.threat_level for e in state.enemies) / len(state.enemies)
            proximity_factor = min(1.0, state.enemies_in_range / 3.0)
            health_factor = 1.0 - state.player_health
            
            state.danger_score = (
                avg_threat * 0.4 +
                proximity_factor * 0.35 +
                health_factor * 0.25
            )
        else:
            state.danger_score = 0.0
        
        # Calculate escape routes (simplified)
        state.escape_routes = len(state.safe_bushes)
        
        return state
    
    def get_state_summary(self, state: GameState) -> Dict:
        """Get a summary dict of state for logging/debugging."""
        return {
            "phase": state.phase.value,
            "player_health": f"{state.player_health:.1%}",
            "player_state": state.player_state.value,
            "num_enemies": len(state.enemies),
            "num_walls": len(state.walls),
            "num_bushes": len(state.bushes),
            "danger_score": f"{state.danger_score:.2f}",
            "in_danger": state.is_in_danger,
            "should_retreat": state.should_retreat,
            "can_engage": state.can_engage,
        }
