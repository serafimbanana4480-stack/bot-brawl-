"""
core/occupancy_grid.py

Occupancy Grid + Real A* Pathfinding for Brawl Stars bot.

Solves the "no spatial awareness" problem by building a grid-based map
from YOLO detections and using it for real pathfinding.

Grid cells:
  0 = WALKABLE (open space)
  1 = WALL (obstacle, impassable)
  2 = BUSH (hiding spot, walkable)
  3 = DANGER (recent damage zone)
  4 = POISON (shrinking zone in Showdown)

Features:
- Real A* pathfinding with diagonal movement
- Line-of-sight raycasting for cover evaluation
- Flow field generation for efficient multi-target pathfinding
- Dynamic updates from YOLO detections
- Integration with WorldModel for danger zones
"""

import heapq
import logging
import math
import threading
import time
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class CellType(IntEnum):
    WALKABLE = 0
    WALL = 1
    BUSH = 2
    DANGER = 3
    POISON = 4


class OccupancyGrid:
    """
    Grid-based spatial representation of the Brawl Stars map.

    Built from YOLO detections (walls, bushes) and updated dynamically
    with danger zones from the WorldModel.
    """

    # Cell size in pixels — smaller = more precise but slower pathfinding
    CELL_SIZE = 20

    # A* movement cost modifiers
    DANGER_COST_MULTIPLIER = 3.0
    BUSH_COST_MULTIPLIER = 0.8  # Bushes are slightly preferred (cover)

    def __init__(self, map_width: int = 1280, map_height: int = 720):
        self.map_width = map_width
        self.map_height = map_height
        self.cols = max(1, map_width // self.CELL_SIZE)
        self.rows = max(1, map_height // self.CELL_SIZE)

        # The grid itself
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Confidence grid: how confident are we about each cell's type
        self.confidence = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Timestamp of last update per cell
        self.last_updated = np.zeros((self.rows, self.cols), dtype=np.float64)

        self._lock = threading.RLock()

        logger.info("[OCCUPANCY] Grid initialized: %dx%d cells (%dx%d px)",
                     self.cols, self.rows, map_width, map_height)

    def update_from_detections(self, detections: list[dict],
                               screenshot_shape: tuple[int, int] | None = None):
        """Update grid from YOLO detection results.

        Args:
            detections: List of detection dicts with class_name, x, y, width, height
            screenshot_shape: Optional (height, width) of the screenshot
        """
        with self._lock:
            now = time.time()

            # Reset wall/bush cells that haven't been confirmed recently
            # (walls don't move, but detection noise can create false walls)
            stale_mask = (now - self.last_updated) > 5.0
            uncertain = stale_mask & (self.confidence < 0.5)
            self.grid[uncertain & (self.grid == CellType.WALL)] = CellType.WALKABLE
            self.confidence[uncertain] = 0.0

            for det in detections:
                class_name = det.get("class_name", "").lower()
                x, y = det.get("x", 0), det.get("y", 0)
                w, h = det.get("width", 0), det.get("height", 0)
                conf = det.get("confidence", 0.5)

                if class_name in ("wall", "obstacle"):
                    self._fill_region(x, y, w, h, CellType.WALL, conf, now)
                elif class_name in ("bush", "grass"):
                    self._fill_region(x, y, w, h, CellType.BUSH, conf, now)
                elif class_name == "powerup":
                    # Power cubes are on walkable cells
                    pass

    def update_danger_zones(self, danger_zones: list[tuple[float, float, float]]):
        """Update danger cells from WorldModel danger zones.

        Args:
            danger_zones: List of (x, y, threat_level) tuples
        """
        with self._lock:
            for x, y, _threat in danger_zones:
                gx, gy = self._pixel_to_grid(x, y)
                radius_cells = max(1, int(150 / self.CELL_SIZE))  # 150px danger radius

                for dy in range(-radius_cells, radius_cells + 1):
                    for dx in range(-radius_cells, radius_cells + 1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.cols and 0 <= ny < self.rows:
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist <= radius_cells:
                                if self.grid[ny, nx] == CellType.WALKABLE:
                                    self.grid[ny, nx] = CellType.DANGER

    def mark_poison_zone(self, x: int, y: int, width: int, height: int):
        """Mark a region as poison (shrinking zone in Showdown)."""
        with self._lock:
            self._fill_region(x, y, width, height, CellType.POISON, 1.0, time.time())

    def is_walkable(self, gx: int, gy: int) -> bool:
        """Check if a grid cell is walkable (not wall, not poison)."""
        if gx < 0 or gx >= self.cols or gy < 0 or gy >= self.rows:
            return False
        return self.grid[gy, gx] != CellType.WALL and self.grid[gy, gx] != CellType.POISON

    def get_cell_type(self, gx: int, gy: int) -> CellType:
        """Get the type of a grid cell."""
        if gx < 0 or gx >= self.cols or gy < 0 or gy >= self.rows:
            return CellType.WALL
        return CellType(self.grid[gy, gx])

    def a_star(self, start_px: tuple[float, float],
               goal_px: tuple[float, float],
               avoid_danger: bool = True) -> list[tuple[float, float]] | None:
        """
        Real A* pathfinding from start to goal in pixel coordinates.

        Args:
            start_px: Start position in pixels (x, y)
            goal_px: Goal position in pixels (x, y)
            avoid_danger: Whether to penalize danger cells

        Returns:
            List of pixel-coordinate waypoints, or None if no path found
        """
        with self._lock:
            start = self._pixel_to_grid(start_px[0], start_px[1])
            goal = self._pixel_to_grid(goal_px[0], goal_px[1])

            if not self.is_walkable(goal[0], goal[1]):
                # Goal is in a wall — find nearest walkable cell
                goal = self._find_nearest_walkable(goal)
                if goal is None:
                    return None

            if not self.is_walkable(start[0], start[1]):
                start = self._find_nearest_walkable(start)
                if start is None:
                    return None

            # A* implementation
            open_set = []
            heapq.heappush(open_set, (0.0, start))
            came_from: dict[tuple[int, int], tuple[int, int]] = {}
            g_score: dict[tuple[int, int], float] = {start: 0.0}
            f_score: dict[tuple[int, int], float] = {start: self._heuristic(start, goal)}
            closed_set: set[tuple[int, int]] = set()

            # 8-directional movement: (dx, dy, cost)
            neighbors = [
                (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
            ]

            max_iterations = self.cols * self.rows  # Safety limit
            iterations = 0

            while open_set and iterations < max_iterations:
                iterations += 1
                _, current = heapq.heappop(open_set)

                if current == goal:
                    # Reconstruct path in pixel coordinates
                    path = self._reconstruct_path(came_from, current)
                    return [self._grid_to_pixel(gx, gy) for gx, gy in path]

                if current in closed_set:
                    continue
                closed_set.add(current)

                for dx, dy, move_cost in neighbors:
                    nx, ny = current[0] + dx, current[1] + dy
                    neighbor = (nx, ny)

                    if neighbor in closed_set:
                        continue
                    if not self.is_walkable(nx, ny):
                        continue

                    # Check diagonal movement doesn't cut corners
                    if dx != 0 and dy != 0:
                        if not self.is_walkable(current[0] + dx, current[1]) or \
                           not self.is_walkable(current[0], current[1] + dy):
                            continue

                    # Movement cost with modifiers
                    cost = move_cost
                    cell = self.get_cell_type(nx, ny)
                    if cell == CellType.DANGER and avoid_danger:
                        cost *= self.DANGER_COST_MULTIPLIER
                    elif cell == CellType.BUSH:
                        cost *= self.BUSH_COST_MULTIPLIER

                    tentative_g = g_score[current] + cost

                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

            # No path found
            logger.debug("[OCCUPANCY] A* no path found from %s to %s", start, goal)
            return None

    def has_line_of_sight(self, start_px: tuple[float, float],
                         end_px: tuple[float, float]) -> bool:
        """
        Check if there is a clear line of sight between two points.

        Uses Bresenham-style raycasting through the grid.
        Returns False if any WALL cell blocks the ray.
        """
        with self._lock:
            x0, y0 = self._pixel_to_grid(start_px[0], start_px[1])
            x1, y1 = self._pixel_to_grid(end_px[0], end_px[1])

            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                # Check current cell
                if 0 <= x0 < self.cols and 0 <= y0 < self.rows:
                    if self.grid[y0, x0] == CellType.WALL:
                        return False
                else:
                    return False  # Out of bounds

                if x0 == x1 and y0 == y1:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

            return True

    def find_cover_position(self, player_px: tuple[float, float],
                            threat_px: tuple[float, float],
                            max_distance: float = 300.0) -> tuple[float, float] | None:
        """
        Find the nearest position that has cover (wall or bush) between it
        and the threat, blocking line of sight.

        Args:
            player_px: Player position in pixels
            threat_px: Threat position in pixels
            max_distance: Maximum distance to search

        Returns:
            Pixel coordinates of cover position, or None
        """
        with self._lock:
            player_grid = self._pixel_to_grid(player_px[0], player_px[1])
            max_cells = int(max_distance / self.CELL_SIZE)

            best_pos = None
            best_score = float('inf')

            # Search in expanding radius
            for radius in range(1, max_cells + 1):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if abs(dx) != radius and abs(dy) != radius:
                            continue  # Only check perimeter of each radius

                        gx = player_grid[0] + dx
                        gy = player_grid[1] + dy

                        if not self.is_walkable(gx, gy):
                            continue

                        pos_px = self._grid_to_pixel(gx, gy)

                        # Must have wall between this position and threat
                        if not self.has_line_of_sight(pos_px, threat_px):
                            # Cover found! Score by distance (closer = better)
                            dist = math.sqrt(dx * dx + dy * dy)
                            if dist < best_score:
                                best_score = dist
                                best_pos = pos_px

                if best_pos is not None:
                    return best_pos

            return None

    def get_flow_field(self, goal_px: tuple[float, float]) -> np.ndarray | None:
        """
        Generate a flow field toward the goal.

        Each cell contains a direction vector pointing toward the next
        cell on the optimal path to the goal. This is more efficient
        than A* when many agents need to path to the same target.

        Returns:
            (rows, cols, 2) array of (dx, dy) direction vectors, or None
        """
        with self._lock:
            goal = self._pixel_to_grid(goal_px[0], goal_px[1])
            if not self.is_walkable(goal[0], goal[1]):
                goal = self._find_nearest_walkable(goal)
                if goal is None:
                    return None

            # BFS from goal outward
            flow = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
            visited = np.zeros((self.rows, self.cols), dtype=bool)
            distance = np.full((self.rows, self.cols), float('inf'), dtype=np.float32)

            queue = [goal]
            visited[goal[1], goal[0]] = True
            distance[goal[1], goal[0]] = 0.0

            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]

            while queue:
                current = queue.pop(0)
                cx, cy = current

                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows and not visited[ny, nx]:
                        if self.is_walkable(nx, ny):
                            visited[ny, nx] = True
                            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                            cell = self.get_cell_type(nx, ny)
                            if cell == CellType.DANGER:
                                move_cost *= self.DANGER_COST_MULTIPLIER
                            distance[ny, nx] = distance[cy, cx] + move_cost
                            queue.append((nx, ny))

            # Build flow field: each cell points toward neighbor with lowest distance
            for gy in range(self.rows):
                for gx in range(self.cols):
                    if not self.is_walkable(gx, gy) or (gx, gy) == goal:
                        continue

                    best_neighbor = None
                    best_dist = distance[gy, gx]

                    for dx, dy in neighbors:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.cols and 0 <= ny < self.rows:
                            if self.is_walkable(nx, ny) and distance[ny, nx] < best_dist:
                                best_dist = distance[ny, nx]
                                best_neighbor = (nx, ny)

                    if best_neighbor is not None:
                        dx = best_neighbor[0] - gx
                        dy = best_neighbor[1] - gy
                        length = math.sqrt(dx * dx + dy * dy)
                        if length > 0:
                            flow[gy, gx] = [dx / length, dy / length]

            return flow

    def get_grid_stats(self) -> dict:
        """Return grid statistics."""
        with self._lock:
            total = self.rows * self.cols
            walls = int(np.sum(self.grid == CellType.WALL))
            bushes = int(np.sum(self.grid == CellType.BUSH))
            danger = int(np.sum(self.grid == CellType.DANGER))
            walkable = total - walls - int(np.sum(self.grid == CellType.POISON))
            return {
                "total_cells": total,
                "walkable": walkable,
                "walls": walls,
                "bushes": bushes,
                "danger": danger,
                "wall_percent": round(walls / max(1, total) * 100, 1),
            }

    # --- Internal helpers ---

    def _fill_region(self, x: int, y: int, w: int, h: int,
                     cell_type: CellType, confidence: float, timestamp: float):
        """Fill a rectangular region of the grid with a cell type."""
        gx1, gy1 = self._pixel_to_grid(x, y)
        gx2, gy2 = self._pixel_to_grid(x + w, y + h)

        for gy in range(gy1, gy2 + 1):
            for gx in range(gx1, gx2 + 1):
                if 0 <= gx < self.cols and 0 <= gy < self.rows:
                    # Only overwrite if new confidence is higher
                    if confidence > self.confidence[gy, gx]:
                        self.grid[gy, gx] = cell_type
                        self.confidence[gy, gx] = confidence
                        self.last_updated[gy, gx] = timestamp

    def _pixel_to_grid(self, px: float, py: float) -> tuple[int, int]:
        gx = int(px / self.CELL_SIZE)
        gy = int(py / self.CELL_SIZE)
        return (max(0, min(self.cols - 1, gx)),
                max(0, min(self.rows - 1, gy)))

    def _grid_to_pixel(self, gx: int, gy: int) -> tuple[float, float]:
        """Convert grid coordinates to pixel coordinates (center of cell)."""
        px = (gx + 0.5) * self.CELL_SIZE
        py = (gy + 0.5) * self.CELL_SIZE
        return (px, py)

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        """Octile distance heuristic for A* (allows diagonal movement)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (1.414 - 1) * min(dx, dy)

    def _reconstruct_path(self, came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
        """Reconstruct A* path from came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _find_nearest_walkable(self, pos: tuple[int, int],
                                max_search: int = 20) -> tuple[int, int] | None:
        """Find the nearest walkable cell to a position."""
        for radius in range(1, max_search + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if self.is_walkable(nx, ny):
                        return (nx, ny)
        return None
