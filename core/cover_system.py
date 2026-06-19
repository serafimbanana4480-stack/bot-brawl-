"""
core/cover_system.py

Cover System with Line-of-Sight Raycasting.

Solves the "cover system without walls" problem. The existing rules.py
takes a `walls` parameter but never checks wall collision for cover.
This module provides real cover evaluation using the OccupancyGrid.

Features:
- Line-of-sight raycasting through the occupancy grid
- Cover position scoring (distance, safety, escape routes)
- Dynamic cover updates as walls are detected
- Bush cover evaluation (partial concealment)
- Cover memory: remember good cover positions across frames
- Escape route evaluation from cover positions
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CoverType(Enum):
    WALL = "wall"       # Full cover (blocks all shots)
    BUSH = "bush"       # Partial cover (hides you, doesn't block shots)
    NONE = "none"       # No cover


@dataclass
class CoverPosition:
    """A evaluated cover position."""
    x: float
    y: float
    cover_type: CoverType = CoverType.NONE
    cover_score: float = 0.0        # Overall quality (0-1)
    los_blocked_to: list[int] = None  # Enemy track_ids blocked by this cover
    los_exposed_to: list[int] = None  # Enemy track_ids that can still see us
    escape_routes: int = 0          # Number of viable escape directions
    distance: float = 0.0          # Distance from player
    has_nearby_cube: bool = False
    has_nearby_bush: bool = False   # Adjacent bush for continued hiding

    def __post_init__(self):
        if self.los_blocked_to is None:
            self.los_blocked_to = []
        if self.los_exposed_to is None:
            self.los_exposed_to = []


class CoverSystem:
    """
    Real cover evaluation system using OccupancyGrid raycasting.

    Evaluates potential cover positions by:
    1. Raycasting from each enemy to the cover position (line-of-sight)
    2. Scoring based on how many enemies are blocked
    3. Evaluating escape routes from the cover position
    4. Considering proximity to player and other tactical factors

    Integrates with:
    - OccupancyGrid: for wall/bush positions and raycasting
    - PressureMap: for threat-aware cover selection
    - WorldModel: for cube/bush memory
    """

    # Search radius for cover positions (pixels)
    SEARCH_RADIUS = 400

    # Grid step for cover search (pixels)
    SEARCH_STEP = 40

    # Maximum cover positions to evaluate per frame
    MAX_EVALUATIONS = 30

    # Cover memory duration (seconds)
    COVER_MEMORY_DURATION = 10.0

    def __init__(self, occupancy_grid=None, pressure_map=None):
        self._occupancy_grid = occupancy_grid
        self._pressure_map = pressure_map

        # Cover memory: positions that were good cover recently
        self._cover_memory: dict[tuple[int, int], tuple[float, CoverType]] = {}
        self._last_search_time: float = 0.0
        self._cached_results: list[CoverPosition] = []
        self._lock = threading.RLock()

        logger.info("[COVER_SYSTEM] Initialized")

    def set_occupancy_grid(self, grid):
        """Set reference to OccupancyGrid."""
        self._occupancy_grid = grid

    def set_pressure_map(self, map):
        """Set reference to PressureMap."""
        self._pressure_map = map

    def find_best_cover(self, player_pos: tuple[float, float],
                        enemies: list[dict],
                        max_distance: float = None) -> CoverPosition | None:
        """
        Find the best cover position near the player.

        Args:
            player_pos: Player position in pixels (x, y)
            enemies: List of enemy dicts with x, y, track_id
            max_distance: Maximum distance to search (default SEARCH_RADIUS)

        Returns:
            Best CoverPosition or None if no cover available
        """
        if not self._occupancy_grid:
            return None

        max_dist = max_distance or self.SEARCH_RADIUS
        now = time.time()

        with self._lock:
            # Throttle: don't re-search every frame
            if now - self._last_search_time < 0.3 and self._cached_results:
                # Re-score cached results with current enemy positions
                return self._select_best_cached(player_pos, enemies)

            self._last_search_time = now

            # Generate candidate positions in search radius
            candidates = self._generate_candidates(player_pos, max_dist)

            if not candidates:
                return None

            # Evaluate each candidate
            evaluated = []
            for cx, cy in candidates[:self.MAX_EVALUATIONS]:
                cover = self._evaluate_position(cx, cy, player_pos, enemies)
                if cover and cover.cover_score > 0.1:
                    evaluated.append(cover)

            # Cache results
            self._cached_results = evaluated

            if not evaluated:
                return None

            # Return best
            evaluated.sort(key=lambda c: c.cover_score, reverse=True)
            best = evaluated[0]

            # Store in cover memory
            grid_pos = (int(cx / 20), int(cy / 20))
            self._cover_memory[grid_pos] = (now, best.cover_type)

            return best

    def has_cover_at(self, position: tuple[float, float],
                     enemies: list[dict]) -> tuple[bool, CoverType]:
        """
        Check if a specific position has cover from enemies.

        Returns (has_cover, cover_type).
        """
        if not self._occupancy_grid:
            return (False, CoverType.NONE)

        x, y = position
        blocked = 0
        exposed = 0

        for enemy in enemies:
            ex, ey = enemy.get("x", 0), enemy.get("y", 0)
            if self._occupancy_grid.has_line_of_sight((ex, ey), (x, y)):
                exposed += 1
            else:
                blocked += 1

        if blocked > 0 and exposed == 0:
            return (True, CoverType.WALL)
        elif blocked > 0:
            return (True, CoverType.BUSH)  # Partial cover
        else:
            return (False, CoverType.NONE)

    def get_escape_direction(self, cover_pos: tuple[float, float],
                             enemies: list[dict]) -> tuple[float, float] | None:
        """
        Get the best escape direction from a cover position.

        Considers which directions have continued cover or are away from enemies.
        """
        if not self._occupancy_grid:
            return None

        cx, cy = cover_pos
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        best_dir = None
        best_score = float('-inf')

        for dx, dy in directions:
            # Look 3 steps ahead in this direction
            nx = cx + dx * 120
            ny = cy + dy * 120

            score = 0.0

            # Check if next position has cover
            has_cover, _ = self.has_cover_at((nx, ny), enemies)
            if has_cover:
                score += 0.5  # Continued cover is great

            # Check if position is walkable
            gx, gy = self._occupancy_grid._pixel_to_grid(nx, ny)
            if self._occupancy_grid.is_walkable(gx, gy):
                score += 0.3
            else:
                score -= 1.0  # Can't go there

            # Prefer direction away from enemies
            for enemy in enemies:
                ex, ey = enemy.get("x", 0), enemy.get("y", 0)
                dist_current = math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
                dist_next = math.sqrt((nx - ex) ** 2 + (ny - ey) ** 2)
                if dist_next > dist_current:
                    score += 0.1  # Moving away from enemy

            # Pressure map: prefer low-pressure direction
            if self._pressure_map:
                pressure = self._pressure_map.get_pressure_at(nx, ny)
                score -= pressure * 0.2

            if score > best_score:
                best_score = score
                length = math.sqrt(dx * dx + dy * dy)
                best_dir = (dx / length, dy / length)

        return best_dir

    def get_stats(self) -> dict:
        """Get cover system statistics."""
        with self._lock:
            return {
                "cached_positions": len(self._cached_results),
                "cover_memory_size": len(self._cover_memory),
                "last_search_age_ms": round(
                    (time.time() - self._last_search_time) * 1000
                ) if self._last_search_time > 0 else -1,
            }

    # --- Internal ---

    def _generate_candidates(self, player_pos: tuple[float, float],
                              max_dist: float) -> list[tuple[float, float]]:
        """Generate candidate cover positions in search radius."""
        if not self._occupancy_grid:
            return []

        px, py = player_pos
        candidates = []

        # Search in expanding rings
        for r in range(self.SEARCH_STEP, int(max_dist) + 1, self.SEARCH_STEP):
            # Sample positions on the ring
            num_samples = max(8, int(2 * math.pi * r / self.SEARCH_STEP))
            for i in range(num_samples):
                angle = 2 * math.pi * i / num_samples
                cx = px + r * math.cos(angle)
                cy = py + r * math.sin(angle)

                # Check if position is walkable
                gx, gy = self._occupancy_grid._pixel_to_grid(cx, cy)
                if self._occupancy_grid.is_walkable(gx, gy):
                    # Check if there's a wall or bush adjacent
                    cell = self._occupancy_grid.get_cell_type(gx, gy)
                    if cell in (1, 2):  # WALL or BUSH
                        # Position itself is cover
                        candidates.append((cx, cy))
                    else:
                        # Check adjacent cells for walls/bushes
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            adj_cell = self._occupancy_grid.get_cell_type(gx + dx, gy + dy)
                            if adj_cell in (1, 2):
                                candidates.append((cx, cy))
                                break

        # Add positions from cover memory
        now = time.time()
        for (gx, gy), (timestamp, _cover_type) in list(self._cover_memory.items()):
            if now - timestamp < self.COVER_MEMORY_DURATION:
                px_mem = (gx + 0.5) * 20
                py_mem = (gy + 0.5) * 20
                candidates.append((px_mem, py_mem))

        # Remove duplicates (within 20px)
        unique = []
        for cx, cy in candidates:
            is_dup = any(math.sqrt((cx - ux) ** 2 + (cy - uy) ** 2) < 20
                         for ux, uy in unique)
            if not is_dup:
                unique.append((cx, cy))

        # Sort by distance to player (prefer closer)
        unique.sort(key=lambda p: math.sqrt(
            (p[0] - player_pos[0]) ** 2 + (p[1] - player_pos[1]) ** 2
        ))

        return unique

    def _evaluate_position(self, cx: float, cy: float,
                            player_pos: tuple[float, float],
                            enemies: list[dict]) -> CoverPosition | None:
        """Evaluate a single position as cover."""
        if not self._occupancy_grid:
            return None

        # Determine cover type
        gx, gy = self._occupancy_grid._pixel_to_grid(cx, cy)
        cell = self._occupancy_grid.get_cell_type(gx, gy)

        # Check adjacent cells for walls
        has_adjacent_wall = False
        has_adjacent_bush = False
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = self._occupancy_grid.get_cell_type(gx + dx, gy + dy)
            if adj == 1:  # WALL
                has_adjacent_wall = True
            elif adj == 2:  # BUSH
                has_adjacent_bush = True

        if cell == 1:  # On a wall — not walkable
            return None

        cover_type = CoverType.NONE
        if has_adjacent_wall:
            cover_type = CoverType.WALL
        elif cell == 2 or has_adjacent_bush:
            cover_type = CoverType.BUSH

        if cover_type == CoverType.NONE:
            return None

        # Evaluate line-of-sight to each enemy
        blocked = []
        exposed = []
        for enemy in enemies:
            ex, ey = enemy.get("x", 0), enemy.get("y", 0)
            tid = enemy.get("track_id", 0)
            if self._occupancy_grid.has_line_of_sight((ex, ey), (cx, cy)):
                exposed.append(tid)
            else:
                blocked.append(tid)

        # Score the position
        score = 0.0

        # Blocking enemies is the primary goal
        if blocked:
            score += len(blocked) * 0.3

        # Wall cover is better than bush
        if cover_type == CoverType.WALL:
            score += 0.3
        else:
            score += 0.1

        # Exposed enemies penalize
        if exposed:
            score -= len(exposed) * 0.2

        # Distance: prefer closer to player
        dist = math.sqrt((cx - player_pos[0]) ** 2 + (cy - player_pos[1]) ** 2)
        if dist < 100:
            score += 0.2
        elif dist < 200:
            score += 0.1
        elif dist > 350:
            score -= 0.1

        # Escape routes
        escape_dirs = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = gx + dx * 3, gy + dy * 3
            if self._occupancy_grid.is_walkable(nx, ny):
                escape_dirs += 1
        score += escape_dirs * 0.05

        # Pressure: prefer low-pressure areas
        if self._pressure_map:
            pressure = self._pressure_map.get_pressure_at(cx, cy)
            score -= pressure * 0.1

        score = max(0.0, min(1.0, score))

        return CoverPosition(
            x=cx, y=cy,
            cover_type=cover_type,
            cover_score=score,
            los_blocked_to=blocked,
            los_exposed_to=exposed,
            escape_routes=escape_dirs,
            distance=dist,
            has_nearby_bush=has_adjacent_bush,
        )

    def _select_best_cached(self, player_pos: tuple[float, float],
                              enemies: list[dict]) -> CoverPosition | None:
        """Re-score cached positions with current enemy positions."""
        if not self._cached_results:
            return None

        best = None
        best_score = 0.0

        for cover in self._cached_results:
            # Re-evaluate LOS with current enemies
            blocked = []
            exposed = []
            if self._occupancy_grid:
                for enemy in enemies:
                    ex, ey = enemy.get("x", 0), enemy.get("y", 0)
                    tid = enemy.get("track_id", 0)
                    if self._occupancy_grid.has_line_of_sight((ex, ey), (cover.x, cover.y)):
                        exposed.append(tid)
                    else:
                        blocked.append(tid)

            # Quick re-score
            score = len(blocked) * 0.3 - len(exposed) * 0.2
            if cover.cover_type == CoverType.WALL:
                score += 0.3

            dist = math.sqrt((cover.x - player_pos[0]) ** 2 +
                              (cover.y - player_pos[1]) ** 2)
            if dist < 200:
                score += 0.1

            if score > best_score:
                best_score = score
                best = CoverPosition(
                    x=cover.x, y=cover.y,
                    cover_type=cover.cover_type,
                    cover_score=score,
                    los_blocked_to=blocked,
                    los_exposed_to=exposed,
                    escape_routes=cover.escape_routes,
                    distance=dist,
                )

        return best
