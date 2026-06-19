"""
core/pressure_map.py

Pressure Map and Influence Map for spatial threat awareness.

Solves the "no pressure/influence maps" problem by computing:
- Pressure Map: Where is the bot under threat? (enemy proximity + damage potential)
- Influence Map: Which areas are controlled by which team?
- Threat Gradient: Direction to move to reduce pressure

These maps are updated every frame from tracked enemy positions
and used by UtilityAI, navigation, and kiting engine.

Integration:
- OccupancyGrid: walls block influence propagation
- WorldModel: enemy memory feeds into pressure calculation
- UtilityAI: pressure/danger scores come from here
- Navigation: threat gradient used for retreat direction
"""

import logging
import math
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class PressureMap:
    """
    Computes a spatial pressure map from enemy positions.

    Pressure at each point = sum of enemy threat contributions.
    Each enemy contributes pressure inversely proportional to distance,
    modified by their threat level (brawler type, health, super status).

    The map is a low-resolution grid (e.g., 64x36) for performance.
    """

    # Map resolution (cells)
    GRID_COLS = 64
    GRID_ROWS = 36

    # Enemy threat radius in pixels
    THREAT_RADIUS = 500.0

    # Threat falloff: pressure = threat / (1 + (dist/falloff)^2)
    THREAT_FALLOFF = 200.0

    # Brawler threat levels (0-1 scale)
    BRAWLER_THREAT = {
        # Assassins - high close-range threat
        "edgar": 0.9, "leon": 0.85, "mortis": 0.9, "fang": 0.8,
        # Tanks - sustained threat
        "el_primo": 0.7, "bull": 0.75, "frank": 0.65, "rosa": 0.6,
        # Damage dealers - high ranged threat
        "colt": 0.75, "piper": 0.8, "brock": 0.7, "nita": 0.55,
        "shelly": 0.7, "dynamike": 0.6, "tick": 0.55,
        # Controllers - zone threat
        "sprout": 0.5, "penny": 0.55, "jessie": 0.5, "lou": 0.55,
        # Supports - lower direct threat
        "poco": 0.3, "healer": 0.3, "byron": 0.4,
    }

    DEFAULT_THREAT = 0.55

    def __init__(self, map_width: int = 1280, map_height: int = 720):
        self.map_width = map_width
        self.map_height = map_height
        self.cell_w = map_width / self.GRID_COLS
        self.cell_h = map_height / self.GRID_ROWS

        # Pressure grid (float, 0.0 = safe, high = dangerous)
        self.pressure = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)

        # Influence grid: positive = friendly control, negative = enemy control
        self.influence = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)

        # Threat gradient: direction to move to reduce pressure
        self.gradient_x = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)
        self.gradient_y = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.float32)

        # OccupancyGrid reference (for wall blocking)
        self._occupancy_grid = None

        self._lock = threading.RLock()
        self._last_update = 0.0

        logger.info("[PRESSURE_MAP] Initialized %dx%d grid (%.1fx%.1f px/cell)",
                     self.GRID_COLS, self.GRID_ROWS, self.cell_w, self.cell_h)

    def set_occupancy_grid(self, grid):
        """Set reference to OccupancyGrid for wall-aware calculations."""
        self._occupancy_grid = grid

    def update(self, enemies: list[dict], allies: list[dict] | None = None,
               player_pos: tuple[float, float] | None = None):
        """
        Recompute pressure and influence maps from enemy/ally positions.

        Args:
            enemies: List of enemy dicts with x, y, class_name, health, has_super
            allies: Optional list of ally dicts (same format)
            player_pos: Player position in pixels (x, y)
        """
        with self._lock:
            self.pressure.fill(0.0)
            self.influence.fill(0.0)

            # Compute enemy pressure
            for enemy in enemies:
                ex, ey = enemy.get("x", 0), enemy.get("y", 0)
                brawler = enemy.get("class_name", "").lower().replace(" ", "_")
                health = enemy.get("health", 1.0)
                has_super = enemy.get("has_super", False)

                threat = self.BRAWLER_THREAT.get(brawler, self.DEFAULT_THREAT)

                # Health modifier: low-health enemies are less threatening
                threat *= (0.3 + 0.7 * health)

                # Super modifier: enemy with super is much more dangerous
                if has_super:
                    threat *= 1.5

                self._add_pressure_source(ex, ey, threat, is_enemy=True)

            # Compute ally influence (negative pressure = safety)
            if allies:
                for ally in allies:
                    ax, ay = ally.get("x", 0), ally.get("y", 0)
                    # Allies provide safety influence
                    self._add_pressure_source(ax, ay, 0.3, is_enemy=False)

            # Player position also provides some influence
            if player_pos:
                self._add_pressure_source(player_pos[0], player_pos[1], 0.2, is_enemy=False)

            # Compute gradient (direction of decreasing pressure)
            self._compute_gradient()

            self._last_update = time.time()

    def get_pressure_at(self, x: float, y: float) -> float:
        """Get pressure value at a pixel position."""
        gx, gy = self._pixel_to_grid(x, y)
        with self._lock:
            if 0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS:
                return float(self.pressure[gy, gx])
            return 0.0

    def get_influence_at(self, x: float, y: float) -> float:
        """Get influence value at a pixel position. Positive=friendly, negative=enemy."""
        gx, gy = self._pixel_to_grid(x, y)
        with self._lock:
            if 0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS:
                return float(self.influence[gy, gx])
            return 0.0

    def get_retreat_direction(self, x: float, y: float) -> tuple[float, float]:
        """
        Get the direction to move to reduce pressure.

        Returns normalized (dx, dy) vector pointing away from pressure.
        Use this for kiting and retreat movement.
        """
        gx, gy = self._pixel_to_grid(x, y)
        with self._lock:
            if 0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS:
                return (float(self.gradient_x[gy, gx]),
                        float(self.gradient_y[gy, gx]))
            return (0.0, 0.0)

    def get_safest_direction(self, x: float, y: float,
                             preferred_dir: tuple[float, float] | None = None) -> tuple[float, float]:
        """
        Get the safest movement direction, optionally biased toward a preferred direction.

        Combines the pressure gradient with a preferred direction (e.g., toward cover).
        """
        gx, gy = self._pixel_to_grid(x, y)

        with self._lock:
            if not (0 <= gx < self.GRID_COLS and 0 <= gy < self.GRID_ROWS):
                return (0.0, 0.0) if preferred_dir is None else preferred_dir

            # Sample pressure in 8 directions
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]

            best_dir = (0.0, 0.0)
            best_score = float('inf')

            for dx, dy in directions:
                # Look 3 cells ahead in this direction
                nx = gx + dx * 3
                ny = gy + dy * 3
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    score = float(self.pressure[ny, nx])

                    # Bias toward preferred direction
                    if preferred_dir is not None:
                        alignment = (dx * preferred_dir[0] + dy * preferred_dir[1])
                        score -= alignment * 0.3  # Bonus for alignment

                    if score < best_score:
                        best_score = score
                        length = math.sqrt(dx * dx + dy * dy)
                        best_dir = (dx / length, dy / length)

            return best_dir

    def get_pressure_stats(self) -> dict:
        """Get pressure map statistics."""
        with self._lock:
            return {
                "max_pressure": float(np.max(self.pressure)),
                "avg_pressure": float(np.mean(self.pressure)),
                "enemy_control_pct": float(np.sum(self.influence < 0) /
                                           max(1, self.GRID_ROWS * self.GRID_COLS) * 100),
                "safe_cells_pct": float(np.sum(self.pressure < 0.1) /
                                         max(1, self.GRID_ROWS * self.GRID_COLS) * 100),
                "last_update_age_ms": round((time.time() - self._last_update) * 1000, 0)
                    if self._last_update > 0 else -1,
            }

    def get_pressure_grid(self) -> np.ndarray:
        """Get a copy of the pressure grid for visualization."""
        with self._lock:
            return self.pressure.copy()

    def get_influence_grid(self) -> np.ndarray:
        """Get a copy of the influence grid."""
        with self._lock:
            return self.influence.copy()

    # --- Internal methods ---

    def _add_pressure_source(self, x: float, y: float, threat: float,
                              is_enemy: bool = True):
        """Add a pressure/influence source at position (x, y)."""
        cx, cy = self._pixel_to_grid(x, y)
        radius_cells = int(self.THREAT_RADIUS / max(self.cell_w, self.cell_h))

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    # Distance in pixels
                    dist_px = math.sqrt(
                        (dx * self.cell_w) ** 2 + (dy * self.cell_h) ** 2
                    )

                    if dist_px <= self.THREAT_RADIUS:
                        # Quadratic falloff
                        falloff = 1.0 / (1.0 + (dist_px / self.THREAT_FALLOFF) ** 2)
                        contribution = threat * falloff

                        if is_enemy:
                            self.pressure[ny, nx] += contribution
                            self.influence[ny, nx] -= contribution
                        else:
                            self.influence[ny, nx] += contribution * 0.5

    def _compute_gradient(self):
        """Compute pressure gradient (direction of steepest descent)."""
        # Use Sobel-like operator for gradient
        for gy in range(1, self.GRID_ROWS - 1):
            for gx in range(1, self.GRID_COLS - 1):
                # X gradient
                gx_val = (float(self.pressure[gy, gx + 1]) -
                          float(self.pressure[gy, gx - 1])) / 2.0
                # Y gradient
                gy_val = (float(self.pressure[gy + 1, gx]) -
                          float(self.pressure[gy - 1, gx])) / 2.0

                # Gradient points toward increasing pressure
                # We want direction of DECREASING pressure (retreat direction)
                length = math.sqrt(gx_val ** 2 + gy_val ** 2)
                if length > 0.001:
                    self.gradient_x[gy, gx] = -gx_val / length
                    self.gradient_y[gy, gx] = -gy_val / length
                else:
                    self.gradient_x[gy, gx] = 0.0
                    self.gradient_y[gy, gx] = 0.0

    def _pixel_to_grid(self, px: float, py: float) -> tuple[int, int]:
        gx = int(px / self.cell_w)
        gy = int(py / self.cell_h)
        return (max(0, min(self.GRID_COLS - 1, gx)),
                max(0, min(self.GRID_ROWS - 1, gy)))
