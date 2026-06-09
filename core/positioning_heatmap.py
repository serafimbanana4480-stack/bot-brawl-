"""
core/positioning_heatmap.py

Mapa de Calor (Heatmap) de Posicionamento.

Analisa onde o bot (e inimigos) passam mais tempo no mapa.
Usa um grid 2D que acumula "tempo de permanência" por célula.

Aplicações:
- Identificar zonas de risco (onde o bot morre mais)
- Otimizar pathfinding (evitar zonas de risco)
- Analisar padrões de movimento do bot (anti-detecção)
- Detectar se o bot fica "preso" numa zona
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


class PositioningHeatmap:
    """
    Heatmap espacial de posicionamento.

    Grid: células de NxN pixels (ex: 50x50px = 38x21 células em 1920x1080).
    Cada célula acumula tempo de permanência.
    """

    def __init__(
        self,
        map_width: int = 1920,
        map_height: int = 1080,
        cell_size: int = 50,
        max_history: int = 1000,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.cell_size = cell_size
        self.grid_w = (map_width + cell_size - 1) // cell_size
        self.grid_h = (map_height + cell_size - 1) // cell_size

        # Heatmaps
        self.bot_heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.enemy_heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.death_heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Tracking temporal
        self._bot_position_history: deque = deque(maxlen=max_history)
        self._last_bot_pos: Optional[Tuple[int, int]] = None
        self._last_update_time = 0.0

        # Zonas de risco calculadas
        self._danger_zones: List[Tuple[int, int, float]] = []  # cx, cy, radius

    def _pos_to_cell(self, x: int, y: int) -> Tuple[int, int]:
        """Converte coordenadas pixel para célula do grid."""
        cx = min(x // self.cell_size, self.grid_w - 1)
        cy = min(y // self.cell_size, self.grid_h - 1)
        return cx, cy

    def update_bot_position(self, x: int, y: int, dt: float = 1.0):
        """
        Atualiza heatmap com nova posição do bot.
        dt: tempo desde última atualização (segundos).
        """
        cx, cy = self._pos_to_cell(x, y)

        # Acumular tempo na célula
        self.bot_heatmap[cy, cx] += dt

        # Registrar histórico
        self._bot_position_history.append({
            "x": x, "y": y, "cx": cx, "cy": cy, "timestamp": time.time(), "dt": dt,
        })

        self._last_bot_pos = (x, y)
        self._last_update_time = time.time()

    def update_enemy_positions(self, positions: List[Tuple[int, int]]):
        """Atualiza heatmap de inimigos."""
        for x, y in positions:
            cx, cy = self._pos_to_cell(x, y)
            self.enemy_heatmap[cy, cx] += 1.0

    def record_death(self, x: int, y: int):
        """Registra morte na posição."""
        cx, cy = self._pos_to_cell(x, y)
        self.death_heatmap[cy, cx] += 10.0  # Peso alto para mortes
        logger.debug("[HEATMAP] Morte registrada em célula (%d, %d)", cx, cy)

    def compute_danger_zones(self, threshold_ratio: float = 0.8) -> List[Dict]:
        """
        Computa zonas de risco baseado no heatmap de mortes + inimigos.
        Retorna lista de zonas com centro e raio.
        """
        # Combinar heatmaps
        combined = self.death_heatmap * 2.0 + self.enemy_heatmap
        if combined.max() == 0:
            return []

        # Normalizar
        normalized = combined / combined.max()

        # Encontrar células acima do threshold
        danger_cells = np.argwhere(normalized > threshold_ratio)

        zones = []
        for cy, cx in danger_cells:
            # Calcular raio baseado na densidade local
            local_density = normalized[max(0, cy-1):cy+2, max(0, cx-1):cx+2].mean()
            radius = int(self.cell_size * (1 + local_density * 2))

            px = cx * self.cell_size + self.cell_size // 2
            py = cy * self.cell_size + self.cell_size // 2

            zones.append({
                "x": px, "y": py, "radius": radius,
                "danger_score": float(normalized[cy, cx]),
                "deaths": float(self.death_heatmap[cy, cx]),
                "enemy_presence": float(self.enemy_heatmap[cy, cx]),
            })

        self._danger_zones = zones
        return zones

    def is_in_danger_zone(self, x: int, y: int, safety_margin: float = 1.0) -> bool:
        """Verifica se posição está numa zona de risco."""
        for zone in self._danger_zones:
            dx = x - zone["x"]
            dy = y - zone["y"]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < zone["radius"] * safety_margin:
                return True
        return False

    def get_least_visited_escape(self, x: int, y: int, max_distance: int = 300) -> Optional[Tuple[int, int]]:
        """
        Encontra direção de escape para zona menos visitada.
        Útil quando o bot está numa zona de risco.
        """
        best_score = float("inf")
        best_dir = None

        # Amostrar direções em círculo
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            dx = int(np.cos(angle) * max_distance)
            dy = int(np.sin(angle) * max_distance)
            tx = x + dx
            ty = y + dy

            if 0 <= tx < self.map_width and 0 <= ty < self.map_height:
                cx, cy = self._pos_to_cell(tx, ty)
                score = self.bot_heatmap[cy, cx] + self.death_heatmap[cy, cx] * 5
                if score < best_score:
                    best_score = score
                    best_dir = (tx, ty)

        return best_dir

    def export_visualization(self, output_path: Path):
        """Exporta heatmap como imagem PNG para visualização."""
        try:
            import cv2

            # Combinar heatmaps em 3 canais
            bot_norm = self.bot_heatmap / (self.bot_heatmap.max() + 1e-8)
            enemy_norm = self.enemy_heatmap / (self.enemy_heatmap.max() + 1e-8)
            death_norm = self.death_heatmap / (self.death_heatmap.max() + 1e-8)

            # Escalar para tamanho do mapa
            def upscale(grid):
                return cv2.resize(grid, (self.map_width, self.map_height), interpolation=cv2.INTER_LINEAR)

            img = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
            img[:, :, 0] = (upscale(enemy_norm) * 255).astype(np.uint8)   # Blue = enemies
            img[:, :, 1] = (upscale(bot_norm) * 255).astype(np.uint8)     # Green = bot
            img[:, :, 2] = (upscale(death_norm) * 255).astype(np.uint8)  # Red = deaths

            cv2.imwrite(str(output_path), img)
            logger.info("[HEATMAP] Visualização exportada: %s", output_path)
        except (ImportError, ModuleNotFoundError, ValueError, TypeError, RuntimeError, cv2.error) as e:
            logger.warning("[HEATMAP] Erro ao exportar visualização: %s", e)

    def get_stats(self) -> Dict[str, any]:
        """Retorna estatísticas do heatmap."""
        total_bot_time = float(self.bot_heatmap.sum())
        total_enemies = float(self.enemy_heatmap.sum())
        total_deaths = float(self.death_heatmap.sum())

        # Calcular entropia do posicionamento (quanto mais espalhado = maior entropia)
        bot_probs = self.bot_heatmap / (total_bot_time + 1e-8)
        entropy = -np.sum(bot_probs * np.log(bot_probs + 1e-8))

        return {
            "grid_size": (self.grid_w, self.grid_h),
            "cell_size": self.cell_size,
            "total_bot_time": round(total_bot_time, 1),
            "total_enemy_presence": round(total_enemies, 1),
            "total_deaths": round(total_deaths, 1),
            "positioning_entropy": round(float(entropy), 2),
            "danger_zones": len(self._danger_zones),
            "history_points": len(self._bot_position_history),
        }
