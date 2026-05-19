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
    """Information about an enemy player with enriched tracking."""
    track_id: int
    position: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    health_estimate: float  # 0.0 to 1.0
    distance: float
    velocity: Tuple[float, float]
    threat_level: float  # calculated threat score
    visible: bool = True
    in_bush: bool = False
    # NEW enriched fields
    has_super: bool = False  # NEW: enemy super ready
    is_attacking: bool = False  # NEW: currently attacking
    angle: float = 0.0  # NEW: direction in radians
    last_seen: float = 0.0  # NEW: timestamp of last detection
    hp_estimate_confidence: float = 0.5  # NEW: confidence in HP estimate
    
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
    """Complete structured game state with 30+ features for neural policy."""
    # Game metadata
    phase: GamePhase = GamePhase.UNKNOWN
    match_time_remaining: Optional[float] = None
    ocr_match_timer_text: Optional[str] = None
    ocr_match_time_seconds: Optional[float] = None
    ocr_score_text: Optional[str] = None
    ocr_match_score: Optional[Tuple[int, int]] = None
    ocr_ability_texts: Dict[str, str] = field(default_factory=dict)
    ocr_ability_states: Dict[str, Optional[bool]] = field(default_factory=dict)
    
    # Player info - Self-state (8 features)
    player_position: Optional[Tuple[float, float]] = None
    player_bbox: Optional[Tuple[int, int, int, int]] = None
    player_health: float = 1.0
    player_state: PlayerState = PlayerState.UNKNOWN
    player_ammo: int = 3  # Default for most brawlers
    player_super_charged: bool = False
    gadget_ready: bool = False  # NEW: gadget availability
    hypercharge_ready: bool = False  # NEW: hypercharge availability
    cooldown_attack: float = 0.0  # NEW: 0-1 progress
    cooldown_super: float = 0.0  # NEW: 0-1 progress
    is_moving: bool = False  # NEW: movement flag
    is_in_bush: bool = False  # NEW: bush status
    velocity: Tuple[float, float] = (0.0, 0.0)  # NEW: velocity vector
    current_tactic: int = 0  # NEW: high-level tactic encoding
    
    # Environment
    enemies: List[EnemyInfo] = field(default_factory=list)
    walls: List[WallInfo] = field(default_factory=list)
    bushes: List[BushInfo] = field(default_factory=list)
    
    # Spatial features (9 features) - NEW
    dist_nearest_cube: Optional[float] = None  # NEW: distance to nearest power cube
    dist_nearest_cover: Optional[float] = None  # NEW: distance to nearest cover
    dist_nearest_safezone: Optional[float] = None  # NEW: distance to safe zone
    line_of_sight_free: bool = True  # NEW: line of sight to enemy
    safe_direction: Tuple[float, float] = (0.0, 0.0)  # NEW: safe direction vector
    wall_proximity: Dict[str, bool] = field(default_factory=lambda: {"left": False, "right": False, "up": False, "down": False})  # NEW
    bush_nearby: bool = False  # NEW: bush within tactical radius
    projectile_threat: float = 0.0  # NEW: incoming projectile danger score
    objective_pressure: float = 0.0  # NEW: pressure from objective mode
    cover_pressure: float = 0.0  # NEW: pressure to seek cover
    
    # Strategic info
    nearest_enemy: Optional[EnemyInfo] = None
    lowest_hp_enemy: Optional[EnemyInfo] = None
    biggest_threat: Optional[EnemyInfo] = None
    safe_bushes: List[BushInfo] = field(default_factory=list)
    enemy_history: List[EnemyInfo] = field(default_factory=list)  # recent enemy snapshot ordering
    
    # Danger assessment
    danger_score: float = 0.0  # 0.0 to 1.0
    enemies_in_range: int = 0
    escape_routes: int = 0
    dist_nearest_enemy: Optional[float] = None
    
    # Temporal memory (4 features) - NEW
    previous_action: Optional[str] = None  # NEW: last action taken
    previous_position: Optional[Tuple[float, float]] = None  # NEW: previous player position
    time_since_enemy_seen: float = 0.0  # NEW: seconds since last enemy detection
    enemy_last_seen_x: Optional[float] = None  # NEW: last seen enemy offset x
    enemy_last_seen_y: Optional[float] = None  # NEW: last seen enemy offset y
    enemy_last_hp: Optional[float] = None  # NEW: hp ratio when last seen
    enemy_last_super: bool = False  # NEW: enemy super state when last seen
    enemy_last_angle: float = 0.0  # NEW: enemy angle when last seen
    enemy_last_attack: bool = False  # NEW: enemy was attacking when last seen
    
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

    Uses canonical class names from core.class_registry for consistency
    across all vision and decision modules.
    """

    # Import canonical class names from registry
    from core.class_registry import VISUAL_CLASSES

    # Core classes (always available)
    CLASS_PLAYER = "player"
    CLASS_ENEMY = "enemy"

    # Extended classes (schema-dependent)
    CLASS_WALL = "wall"
    CLASS_BUSH = "bush"
    CLASS_HEALTH_BAR = "health_bar"
    CLASS_AMMO = "ammo"
    CLASS_SUPER = "super_area"  # canonical name from registry
    CLASS_CUBEBOX = "cubebox"
    CLASS_POWERUP = "powerup"

    # All canonical class names for reference
    @classmethod
    def get_canonical_classes(cls, schema: str = "full") -> dict:
        """Get canonical class mapping for a schema."""
        from core.class_registry import get_schema
        return get_schema(schema)
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        danger_distance: float = 300.0,
        ocr_detector: Optional[object] = None
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.danger_distance = danger_distance
        self.ocr_detector = ocr_detector
        
        # Initialize HUD feature extractors
        try:
            from vision.feature_extractors import HUDFeatureExtractor
            self.hud_extractor = HUDFeatureExtractor()
        except ImportError:
            self.hud_extractor = None
        
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
    
    def _normalize_class_name(self, name: str) -> str:
        """Normalize class name to canonical form using registry."""
        from core.class_registry import get_canonical
        return get_canonical(name)

    def extract_state(
        self,
        tracks: List[TrackedObject],
        raw_detections: Optional[List] = None,
        screenshot: Optional = None
    ) -> GameState:
        """
        Extract structured game state from tracking data.

        Uses canonical class names from core.class_registry for consistent
        matching across different model outputs.

        Args:
            tracks: List of tracked objects
            raw_detections: Optional raw YOLO detections for additional info

        Returns:
            Structured GameState object
        """
        from core.class_registry import get_canonical
        import time

        state = GameState()
        current_time = time.time()

        # Find player (normalized class name matching)
        player_track = None
        for track in tracks:
            canonical_name = get_canonical(track.class_name)
            if canonical_name == self.CLASS_PLAYER:
                player_track = track
                break
        
        if player_track:
            state.player_position = player_track.center
            state.player_bbox = player_track.bbox
            state.player_state = PlayerState.ALIVE
            
            # Extract HUD features if screenshot available
            if self.hud_extractor and screenshot is not None:
                import time
                hud_features = self.hud_extractor.extract_all(
                    screenshot,
                    player_bbox=player_track.bbox,
                    player_position=player_track.center,
                    timestamp=time.time()
                )
                state.gadget_ready = hud_features.gadget_ready
                state.hypercharge_ready = hud_features.hypercharge_ready
                state.cooldown_attack = hud_features.cooldown_attack

            # Optional OCR enrichment for HUD text like timer/score/ability indicators.
            if self.ocr_detector and screenshot is not None:
                try:
                    if hasattr(self.ocr_detector, "extract_hud_text"):
                        ocr_hud = self.ocr_detector.extract_hud_text(screenshot)
                        state.ocr_match_timer_text = ocr_hud.get("match_timer_text")
                        state.ocr_match_time_seconds = ocr_hud.get("match_time_remaining")
                        state.ocr_score_text = ocr_hud.get("score_text")
                        state.ocr_ability_texts = ocr_hud.get("ability_texts", {}) or {}
                        state.ocr_match_score = ocr_hud.get("match_score")
                        state.ocr_ability_states = ocr_hud.get("ability_states", {}) or {}
                        if state.match_time_remaining is None:
                            state.match_time_remaining = ocr_hud.get("match_time_remaining")
                except Exception:
                    # OCR is best-effort; keep state extraction resilient.
                    pass
                state.cooldown_super = hud_features.cooldown_super
                state.velocity = hud_features.velocity
                state.is_moving = abs(hud_features.velocity[0]) > 2.0 or abs(hud_features.velocity[1]) > 2.0
            
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
        
        # Process enemies (normalized class name matching)
        for track in tracks:
            canonical_name = get_canonical(track.class_name)
            if canonical_name == self.CLASS_ENEMY:
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
                    in_bush=False,  # Would need specific detection
                    last_seen=track.last_seen,
                    hp_estimate_confidence=min(1.0, 0.35 + 0.05 * min(track.hits, 6)),
                )
                state.enemies.append(enemy)

        # Process walls (normalized class name matching)
        for track in tracks:
            canonical_name = get_canonical(track.class_name)
            if canonical_name == self.CLASS_WALL:
                wall = WallInfo(
                    track_id=track.id,
                    bbox=track.bbox,
                    center=track.center,
                    blocks_line_of_sight=False  # Would need ray-casting
                )
                state.walls.append(wall)

        # Process bushes (normalized class name matching)
        for track in tracks:
            canonical_name = get_canonical(track.class_name)
            if canonical_name == self.CLASS_BUSH:
                bush = BushInfo(
                    track_id=track.id,
                    bbox=track.bbox,
                    center=track.center,
                    occupied=False,
                    enemies_nearby=0,
                )
                state.bushes.append(bush)

        # Aggregate spatial/tactical features when we know the player position
        if state.player_position:
            if state.enemies:
                nearest = min(state.enemies, key=lambda e: e.distance)
                state.dist_nearest_enemy = nearest.distance
                state.nearest_enemy = nearest
                state.time_since_enemy_seen = max(0.0, current_time - nearest.last_seen)
                state.enemy_last_seen_x = nearest.position[0] - state.player_position[0]
                state.enemy_last_seen_y = nearest.position[1] - state.player_position[1]
                state.enemy_last_hp = nearest.health_estimate
                state.enemy_last_super = nearest.has_super
                state.enemy_last_angle = nearest.angle
                state.enemy_last_attack = nearest.is_attacking
                state.enemy_history = sorted(
                    state.enemies,
                    key=lambda e: e.last_seen,
                    reverse=True,
                )[:3]

                # Find lowest HP enemy
                state.lowest_hp_enemy = min(state.enemies, key=lambda e: e.health_estimate)

                # Find biggest threat
                state.biggest_threat = max(state.enemies, key=lambda e: e.threat_level)

                # Count enemies in attack range
                state.enemies_in_range = sum(
                    1 for e in state.enemies if e.distance < self.danger_distance
                )

            if state.walls:
                nearest_wall = min(
                    state.walls,
                    key=lambda w: self._calculate_distance(w.center, state.player_position)
                )
                state.dist_nearest_cover = self._calculate_distance(nearest_wall.center, state.player_position)
                px, py = state.player_position
                wx, wy = nearest_wall.center
                state.wall_proximity = {
                    "left": wx < px,
                    "right": wx > px,
                    "up": wy < py,
                    "down": wy > py,
                }

            if state.enemies:
                # Safe direction: average vector away from enemies weighted by inverse distance
                sx, sy, sw = 0.0, 0.0, 0.0
                for enemy in state.enemies:
                    dx = state.player_position[0] - enemy.position[0]
                    dy = state.player_position[1] - enemy.position[1]
                    dist = max(enemy.distance, 1.0)
                    weight = 1.0 / dist
                    sx += dx * weight
                    sy += dy * weight
                    sw += weight
                if sw > 0:
                    norm = max((sx * sx + sy * sy) ** 0.5, 1e-6)
                    state.safe_direction = (sx / norm, sy / norm)

                # Simple line-of-sight heuristic: blocked if there is a wall closer to player than nearest enemy
                if state.walls:
                    nearest_enemy_dist = min(e.distance for e in state.enemies)
                    nearest_wall_dist = min(
                        self._calculate_distance(w.center, state.player_position)
                        for w in state.walls
                    )
                    state.line_of_sight_free = nearest_wall_dist > nearest_enemy_dist

            cube_like = [
                track for track in tracks
                if get_canonical(track.class_name) in {self.CLASS_CUBEBOX, self.CLASS_POWERUP}
            ]
            if cube_like:
                state.dist_nearest_cube = min(
                    self._calculate_distance(track.center, state.player_position)
                    for track in cube_like
                )

            if state.bushes:
                nearest_bush = min(
                    state.bushes,
                    key=lambda b: self._calculate_distance(b.center, state.player_position)
                )
                dist_to_nearest_bush = self._calculate_distance(nearest_bush.center, state.player_position)
                state.bush_nearby = dist_to_nearest_bush < 180
                state.is_in_bush = dist_to_nearest_bush < 40
                for bush in state.bushes:
                    dist_to_player = self._calculate_distance(bush.center, state.player_position)
                    enemies_near = sum(
                        1 for e in state.enemies
                        if self._calculate_distance(bush.center, e.position) < 150
                    )
                    bush.enemies_nearby = enemies_near
                    if enemies_near == 0 and dist_to_player > 50:
                        state.safe_bushes.append(bush)

                state.cover_pressure = 1.0 if state.safe_bushes else 0.0

            state.projectile_threat = min(
                1.0,
                sum(
                    1 for det in (raw_detections or [])
                    if hasattr(det, 'class_name') and get_canonical(det.class_name) in {"bullet_enemy", "incoming_threat"}
                ) / 5.0,
            )
        
        # Calculate overall danger score
        if state.enemies:
            avg_threat = sum(e.threat_level for e in state.enemies) / len(state.enemies)
            proximity_factor = min(1.0, state.enemies_in_range / 3.0)
            health_factor = 1.0 - state.player_health
            projectile_factor = state.projectile_threat
            
            state.danger_score = (
                avg_threat * 0.4 +
                proximity_factor * 0.35 +
                health_factor * 0.2 +
                projectile_factor * 0.05
            )
        else:
            state.danger_score = 0.0

        # Tactical objective encoding after danger score is known
        if state.danger_score > 0.7:
            state.current_tactic = 1  # retreat
        elif state.safe_bushes:
            state.current_tactic = 3  # hide / take cover
        elif state.dist_nearest_cube is not None:
            state.current_tactic = 4  # collect objective
        elif state.enemies:
            state.current_tactic = 2  # chase / engage
        
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
