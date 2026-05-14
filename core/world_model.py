"""
core/world_model.py

Persistent World Model for Brawl Stars bot.

Solves the "reactive architecture" problem by maintaining spatial and temporal
memory across frames. The bot no longer forgets everything each frame.

Features:
- Enemy memory: tracks last-seen positions even when enemies disappear into bushes
- Danger zones: areas where the bot recently took damage
- Safe routes: validated retreat paths
- Map control: which areas are contested/safe
- Power cube memory: remembers cube locations even when not visible
- Pressure zones: areas under enemy influence
"""

import time
import math
import logging
import threading
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    SAFE = "safe"
    CONTESTED = "contested"
    DANGEROUS = "dangerous"
    DEADLY = "deadly"


@dataclass
class EnemyMemory:
    """Persistent memory of a single enemy."""
    track_id: int
    last_known_position: Tuple[float, float]
    last_seen_time: float
    estimated_velocity: Tuple[float, float] = (0.0, 0.0)
    health_estimate: float = 1.0
    brawler_name: str = "unknown"
    times_seen: int = 1
    last_damage_dealt_to_us: float = 0.0
    last_damage_time: float = 0.0
    suspected_in_bush: bool = False
    bush_position: Optional[Tuple[float, float]] = None

    @property
    def is_active(self) -> bool:
        """Enemy is considered active if seen within last 5 seconds."""
        return (time.time() - self.last_seen_time) < 5.0

    @property
    def is_recently_seen(self) -> bool:
        """Seen within last 2 seconds — high confidence position."""
        return (time.time() - self.last_seen_time) < 2.0

    @property
    def confidence(self) -> float:
        """Position confidence: 1.0 (just seen) → 0.0 (5s+ ago)."""
        elapsed = time.time() - self.last_seen_time
        if elapsed >= 5.0:
            return 0.0
        return max(0.0, 1.0 - elapsed / 5.0)

    def predict_position(self, current_time: Optional[float] = None) -> Tuple[float, float]:
        """Predict current position based on last known position + velocity."""
        now = current_time or time.time()
        dt = now - self.last_seen_time
        # Velocity decays over time (enemy may have changed direction)
        decay = max(0.0, 1.0 - dt / 3.0)
        px = self.last_known_position[0] + self.estimated_velocity[0] * dt * decay
        py = self.last_known_position[1] + self.estimated_velocity[1] * dt * decay
        return (px, py)


@dataclass
class DangerZone:
    """An area where the bot recently took damage."""
    position: Tuple[float, float]
    radius: float
    damage_amount: float
    timestamp: float
    source_enemy_id: Optional[int] = None

    @property
    def is_active(self) -> bool:
        return (time.time() - self.timestamp) < 8.0

    @property
    def threat_level(self) -> float:
        """Decaying threat level: 1.0 → 0.0 over 8 seconds."""
        elapsed = time.time() - self.timestamp
        if elapsed >= 8.0:
            return 0.0
        return self.damage_amount * max(0.0, 1.0 - elapsed / 8.0)


@dataclass
class PowerCubeMemory:
    """Memory of a power cube location."""
    position: Tuple[float, float]
    last_seen_time: float
    collected: bool = False
    confidence: float = 1.0

    @property
    def is_available(self) -> bool:
        return not self.collected and (time.time() - self.last_seen_time) < 30.0


@dataclass
class MapZone:
    """A zone on the map with control status."""
    grid_x: int
    grid_y: int
    zone_type: ZoneType = ZoneType.SAFE
    last_updated: float = 0.0
    enemy_presence_count: int = 0
    visits: int = 0


class WorldModel:
    """
    Persistent world model that maintains spatial and temporal memory.

    This is the single source of truth for the bot's understanding of the
    game world. All decision modules should query this instead of raw detections.

    Thread-safe for concurrent access from inference and decision threads.
    """

    # Grid resolution for spatial queries (pixels per cell)
    CELL_SIZE = 40

    # Memory decay parameters
    ENEMY_MEMORY_TTL = 10.0  # Seconds to remember enemy after last seen
    DANGER_ZONE_TTL = 8.0
    CUBE_MEMORY_TTL = 30.0

    def __init__(self, map_width: int = 1280, map_height: int = 720):
        self._lock = threading.RLock()
        self.map_width = map_width
        self.map_height = map_height
        self.grid_cols = max(1, map_width // self.CELL_SIZE)
        self.grid_rows = max(1, map_height // self.CELL_SIZE)

        # Enemy memory: track_id → EnemyMemory
        self.enemies: Dict[int, EnemyMemory] = {}

        # Danger zones: list of recent damage events
        self.danger_zones: List[DangerZone] = []

        # Power cube memory
        self.power_cubes: Dict[str, PowerCubeMemory] = {}

        # Map control grid
        self.map_grid: Dict[Tuple[int, int], MapZone] = {}

        # Safe routes: validated retreat paths
        self.safe_routes: List[List[Tuple[float, float]]] = []

        # Bush memory: positions of detected bushes
        self.known_bushes: List[Tuple[float, float, float, float]] = []  # (x, y, w, h)

        # Player state tracking
        self.player_position: Optional[Tuple[float, float]] = None
        self.player_health: float = 1.0
        self.player_brawler: str = "unknown"
        self.player_ammo: int = 3
        self.player_super_charged: bool = False

        # Match phase tracking
        self.match_start_time: Optional[float] = None
        self.match_phase: str = "early"  # early, mid, late

        # Stats
        self.total_damage_taken: float = 0.0
        self.total_damage_dealt: float = 0.0
        self.kills_this_match: int = 0
        self.deaths_this_match: int = 0

        logger.info("[WORLD_MODEL] Initialized %dx%d grid (%dx%d cells)",
                     map_width, map_height, self.grid_cols, self.grid_rows)

    def update_enemy(self, track_id: int, position: Tuple[float, float],
                     velocity: Tuple[float, float] = (0.0, 0.0),
                     health: float = 1.0, brawler_name: str = "unknown"):
        """Update or create enemy memory entry."""
        with self._lock:
            now = time.time()
            if track_id in self.enemies:
                mem = self.enemies[track_id]
                # Smooth velocity estimate (EMA)
                alpha = 0.6
                mem.estimated_velocity = (
                    alpha * velocity[0] + (1 - alpha) * mem.estimated_velocity[0],
                    alpha * velocity[1] + (1 - alpha) * mem.estimated_velocity[1],
                )
                mem.last_known_position = position
                mem.last_seen_time = now
                mem.health_estimate = health
                mem.brawler_name = brawler_name
                mem.times_seen += 1
                mem.suspected_in_bush = False
            else:
                self.enemies[track_id] = EnemyMemory(
                    track_id=track_id,
                    last_known_position=position,
                    last_seen_time=now,
                    estimated_velocity=velocity,
                    health_estimate=health,
                    brawler_name=brawler_name,
                )

            # Update map grid for this position
            gx, gy = self._pos_to_grid(position)
            if (gx, gy) in self.map_grid:
                self.map_grid[(gx, gy)].enemy_presence_count += 1
                self.map_grid[(gx, gy)].zone_type = ZoneType.DANGEROUS
                self.map_grid[(gx, gy)].last_updated = now
            else:
                self.map_grid[(gx, gy)] = MapZone(
                    grid_x=gx, grid_y=gy,
                    zone_type=ZoneType.DANGEROUS,
                    last_updated=now,
                    enemy_presence_count=1,
                )

    def mark_enemy_in_bush(self, track_id: int, bush_position: Tuple[float, float]):
        """Mark enemy as suspected in a bush (disappeared near bush)."""
        with self._lock:
            if track_id in self.enemies:
                self.enemies[track_id].suspected_in_bush = True
                self.enemies[track_id].bush_position = bush_position

    def record_damage_taken(self, position: Tuple[float, float],
                            damage: float, source_enemy_id: Optional[int] = None):
        """Record that the bot took damage at a position — creates danger zone."""
        with self._lock:
            self.danger_zones.append(DangerZone(
                position=position,
                radius=150.0,
                damage_amount=damage,
                timestamp=time.time(),
                source_enemy_id=source_enemy_id,
            ))
            self.total_damage_taken += damage

            # Mark grid cell as dangerous
            gx, gy = self._pos_to_grid(position)
            if (gx, gy) not in self.map_grid:
                self.map_grid[(gx, gy)] = MapZone(grid_x=gx, grid_y=gy)
            self.map_grid[(gx, gy)].zone_type = ZoneType.DEADLY
            self.map_grid[(gx, gy)].last_updated = time.time()

    def record_damage_dealt(self, damage: float):
        """Record damage dealt to enemies."""
        with self._lock:
            self.total_damage_dealt += damage

    def update_power_cube(self, cube_id: str, position: Tuple[float, float],
                          collected: bool = False):
        """Update power cube memory."""
        with self._lock:
            if cube_id in self.power_cubes:
                self.power_cubes[cube_id].collected = collected
                self.power_cubes[cube_id].last_seen_time = time.time()
                self.power_cubes[cube_id].confidence = 1.0
            else:
                self.power_cubes[cube_id] = PowerCubeMemory(
                    position=position,
                    last_seen_time=time.time(),
                    collected=collected,
                )

    def update_player(self, position: Tuple[float, float], health: float = 1.0,
                      brawler: str = "unknown", ammo: int = 3,
                      super_charged: bool = False):
        """Update player state."""
        with self._lock:
            self.player_position = position
            self.player_health = health
            self.player_brawler = brawler
            self.player_ammo = ammo
            self.player_super_charged = super_charged

    def update_bushes(self, bushes: List[Tuple[float, float, float, float]]):
        """Update known bush positions from detection."""
        with self._lock:
            self.known_bushes = bushes

    def start_match(self):
        """Reset world model for new match."""
        with self._lock:
            self.match_start_time = time.time()
            self.match_phase = "early"
            self.total_damage_taken = 0.0
            self.total_damage_dealt = 0.0
            self.kills_this_match = 0
            self.deaths_this_match = 0
            self.danger_zones.clear()
            self.power_cubes.clear()
            self.safe_routes.clear()

    def end_match(self):
        """Mark match as ended."""
        with self._lock:
            self.match_start_time = None
            self.match_phase = "ended"

    def update_match_phase(self):
        """Update match phase based on elapsed time (Showdown-focused)."""
        with self._lock:
            if self.match_start_time is None:
                return
            elapsed = time.time() - self.match_start_time
            if elapsed < 60:
                self.match_phase = "early"
            elif elapsed < 150:
                self.match_phase = "mid"
            else:
                self.match_phase = "late"

    def get_active_enemies(self) -> List[EnemyMemory]:
        """Get all enemies with active memory (seen within TTL)."""
        with self._lock:
            now = time.time()
            return [
                e for e in self.enemies.values()
                if (now - e.last_seen_time) < self.ENEMY_MEMORY_TTL
            ]

    def get_nearby_enemies(self, position: Tuple[float, float],
                           max_distance: float = 400.0) -> List[EnemyMemory]:
        """Get enemies near a position, including predicted positions for unseen enemies."""
        with self._lock:
            result = []
            now = time.time()
            for enemy in self.enemies.values():
                if (now - enemy.last_seen_time) > self.ENEMY_MEMORY_TTL:
                    continue
                pos = enemy.predict_position(now) if not enemy.is_recently_seen else enemy.last_known_position
                dist = self._distance(position, pos)
                if dist <= max_distance:
                    result.append(enemy)
            return result

    def get_danger_at(self, position: Tuple[float, float]) -> float:
        """Calculate cumulative danger level at a position from all active danger zones."""
        with self._lock:
            danger = 0.0
            for dz in self.danger_zones:
                if not dz.is_active:
                    continue
                dist = self._distance(position, dz.position)
                if dist < dz.radius:
                    # Danger falls off with distance from zone center
                    falloff = 1.0 - (dist / dz.radius)
                    danger += dz.threat_level * falloff
            return danger

    def get_pressure_at(self, position: Tuple[float, float]) -> float:
        """Calculate enemy pressure at a position (influence field from all enemies)."""
        with self._lock:
            pressure = 0.0
            now = time.time()
            for enemy in self.enemies.values():
                if (now - enemy.last_seen_time) > self.ENEMY_MEMORY_TTL:
                    continue
                pos = enemy.predict_position(now)
                dist = self._distance(position, pos)
                if dist < 1.0:
                    dist = 1.0
                # Inverse square falloff, weighted by confidence and health
                influence = enemy.confidence * (1.0 + (1.0 - enemy.health_estimate)) / (dist * dist) * 10000
                pressure += influence
            return pressure

    def get_safest_direction(self, current_pos: Tuple[float, float],
                             num_directions: int = 16) -> Tuple[float, float]:
        """Find the direction with lowest pressure from current position.

        Returns a unit vector pointing toward the safest direction.
        """
        best_angle = 0.0
        best_score = float('inf')
        step_distance = 100.0

        for i in range(num_directions):
            angle = (2 * math.pi * i) / num_directions
            test_x = current_pos[0] + math.cos(angle) * step_distance
            test_y = current_pos[1] + math.sin(angle) * step_distance

            # Clamp to map bounds
            test_x = max(0, min(self.map_width, test_x))
            test_y = max(0, min(self.map_height, test_y))

            # Score = pressure + danger (lower is better)
            pressure = self.get_pressure_at((test_x, test_y))
            danger = self.get_danger_at((test_x, test_y))
            score = pressure + danger * 2.0

            if score < best_score:
                best_score = score
                best_angle = angle

        return (math.cos(best_angle), math.sin(best_angle))

    def get_nearest_power_cube(self, position: Tuple[float, float]) -> Optional[PowerCubeMemory]:
        """Find the nearest available power cube."""
        with self._lock:
            best = None
            best_dist = float('inf')
            for cube in self.power_cubes.values():
                if not cube.is_available:
                    continue
                dist = self._distance(position, cube.position)
                if dist < best_dist:
                    best_dist = dist
                    best = cube
            return best

    def get_nearest_bush(self, position: Tuple[float, float],
                         away_from: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
        """Find the nearest bush, optionally preferring direction away from a threat."""
        with self._lock:
            best = None
            best_score = float('inf')
            for bx, by, bw, bh in self.known_bushes:
                center = (bx + bw / 2, by + bh / 2)
                dist = self._distance(position, center)

                score = dist
                if away_from is not None:
                    # Prefer bushes in the direction away from threat
                    bush_dir = (center[0] - position[0], center[1] - position[1])
                    away_dir = (position[0] - away_from[0], position[1] - away_from[1])
                    alignment = self._dot_product(
                        self._normalize(bush_dir),
                        self._normalize(away_dir)
                    )
                    if alignment > 0:
                        score *= (1.0 - alignment * 0.5)  # Up to 50% bonus for alignment

                if score < best_score:
                    best_score = score
                    best = center
            return best

    def should_retreat(self) -> bool:
        """Determine if the bot should retreat based on world state."""
        with self._lock:
            if self.player_position is None:
                return False

            # Low health
            if self.player_health < 0.3:
                return True

            # High pressure at current position
            pressure = self.get_pressure_at(self.player_position)
            if pressure > 5.0 and self.player_health < 0.6:
                return True

            # Multiple enemies nearby
            nearby = self.get_nearby_enemies(self.player_position, 250.0)
            if len(nearby) >= 2 and self.player_health < 0.5:
                return True

            return False

    def get_match_phase_recommendation(self) -> Dict:
        """Get strategic recommendations based on current match phase."""
        with self._lock:
            self.update_match_phase()

            if self.match_phase == "early":
                return {
                    "phase": "early",
                    "priority": "farm",
                    "aggression": 0.3,
                    "cube_priority": 0.9,
                    "retreat_threshold": 0.4,
                    "description": "Farm power cubes, avoid fights",
                }
            elif self.match_phase == "mid":
                return {
                    "phase": "mid",
                    "priority": "control",
                    "aggression": 0.5,
                    "cube_priority": 0.5,
                    "retreat_threshold": 0.35,
                    "description": "Control zone, pick favorable fights",
                }
            else:  # late
                return {
                    "phase": "late",
                    "priority": "survive",
                    "aggression": 0.2,
                    "cube_priority": 0.2,
                    "retreat_threshold": 0.5,
                    "description": "Survive, avoid unnecessary risks",
                }

    def decay(self):
        """Remove expired entries from all memory stores. Call periodically."""
        with self._lock:
            now = time.time()

            # Decay enemies
            expired = [
                tid for tid, e in self.enemies.items()
                if (now - e.last_seen_time) > self.ENEMY_MEMORY_TTL
            ]
            for tid in expired:
                del self.enemies[tid]

            # Decay danger zones
            self.danger_zones = [dz for dz in self.danger_zones if dz.is_active]

            # Decay power cubes
            expired_cubes = [
                cid for cid, c in self.power_cubes.items()
                if (now - c.last_seen_time) > self.CUBE_MEMORY_TTL
            ]
            for cid in expired_cubes:
                del self.power_cubes[cid]

            # Decay map grid
            for key, zone in list(self.map_grid.items()):
                if (now - zone.last_updated) > 15.0:
                    # Revert dangerous zones to contested, contested to safe
                    if zone.zone_type == ZoneType.DEADLY:
                        zone.zone_type = ZoneType.DANGEROUS
                    elif zone.zone_type == ZoneType.DANGEROUS:
                        zone.zone_type = ZoneType.CONTESTED
                    elif zone.zone_type == ZoneType.CONTESTED:
                        zone.zone_type = ZoneType.SAFE
                    zone.last_updated = now

    def get_summary(self) -> Dict:
        """Get a summary of the world model state for debugging."""
        with self._lock:
            return {
                "active_enemies": len(self.get_active_enemies()),
                "total_enemies_tracked": len(self.enemies),
                "danger_zones": len(self.danger_zones),
                "power_cubes_available": sum(1 for c in self.power_cubes.values() if c.is_available),
                "known_bushes": len(self.known_bushes),
                "player_health": self.player_health,
                "match_phase": self.match_phase,
                "total_damage_taken": self.total_damage_taken,
                "total_damage_dealt": self.total_damage_dealt,
            }

    # --- Internal helpers ---

    def _pos_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        gx = int(position[0] // self.CELL_SIZE)
        gy = int(position[1] // self.CELL_SIZE)
        return (max(0, min(self.grid_cols - 1, gx)),
                max(0, min(self.grid_rows - 1, gy)))

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _normalize(v: Tuple[float, float]) -> Tuple[float, float]:
        length = math.sqrt(v[0] ** 2 + v[1] ** 2)
        if length < 0.001:
            return (0.0, 0.0)
        return (v[0] / length, v[1] / length)

    @staticmethod
    def _dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        return v1[0] * v2[0] + v1[1] * v2[1]
