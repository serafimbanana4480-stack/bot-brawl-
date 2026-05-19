"""
world_model_integration.py

Integração do WorldModel com o sistema de combate.
Liga as deteções do play.py ao WorldModel para memória espacial persistente.

Problema resolvido:
- WorldModel existe mas não é atualizado pelo PlayLogic
- Este módulo adiciona a ponte entre deteções e memória espacial
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

logger = logging.getLogger(__name__)


def center(bbox: List[int]) -> Tuple[float, float]:
    """Calcula centro de bounding box [x1, y1, x2, y2]."""
    if len(bbox) >= 4:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return (float(bbox[0]), float(bbox[1]))


class WorldModelIntegrator:
    """
    Integrador que conecta WorldModel ao sistema de combate.

    Responsabilidades:
    - Atualizar WorldModel com posições de inimigos a cada frame
    - Manter histórico de velocidades dos inimigos
    - Marcar inimigos em bushes
    - Atualizar zonas de perigo quando tomar dano
    - Guardar posições de power cubes
    """

    def __init__(self, world_model, map_width: int = 1920, map_height: int = 1080):
        self.world_model = world_model
        self.map_width = map_width
        self.map_height = map_height

        # Histórico de posições por track_id
        self._enemy_history: Dict[int, deque] = {}

        # Track IDs consecutivos (para identificar inimigos)
        self._next_track_id = 0
        self._enemy_positions: Dict[int, Tuple[float, float]] = {}

    def update_from_detections(
        self,
        enemies: List[List[int]],
        player: Optional[List[int]] = None,
        power_cubes: Optional[List[List[int]]] = None,
        bushes: Optional[List[List[int]]] = None,
        walls: Optional[List[List[int]]] = None,
        player_hp: float = 1.0
    ):
        """
        Atualiza WorldModel com as deteções do frame atual.

        Args:
            enemies: Lista de bounding boxes de inimigos [[x1,y1,x2,y2], ...]
            player: Bounding box do jogador [x1,y1,x2,y2]
            power_cubes: Lista de bounding boxes de power cubes
            bushes: Lista de bounding boxes de bushes
            walls: Lista de bounding boxes de paredes
            player_hp: HP atual do jogador (0.0-1.0)
        """
        if self.world_model is None:
            return

        try:
            # Atualizar timestamp do início do match se ainda não definido
            if self.world_model.match_start_time is None:
                self.world_model.match_start_time = time.time()

            # Atualizar posição do jogador
            if player:
                player_center = center(player)
                self.world_model.player_position = player_center
                self.world_model.player_health = player_hp

            # Atualizar inimigos
            current_ids = set()
            for enemy_bbox in enemies:
                enemy_center = center(enemy_bbox)

                # Tentar associar com histórico (mesma posição ≈ mesmo ID)
                track_id = self._find_matching_track(enemy_center)

                if track_id is None:
                    track_id = self._next_track_id
                    self._next_track_id += 1

                current_ids.add(track_id)

                # Calcular velocidade se tivermos histórico
                velocity = (0.0, 0.0)
                if track_id in self._enemy_positions:
                    prev_pos = self._enemy_positions[track_id]
                    dt = 0.1  # Assume ~100ms entre frames
                    velocity = (
                        (enemy_center[0] - prev_pos[0]) / dt,
                        (enemy_center[1] - prev_pos[1]) / dt
                    )

                self._enemy_positions[track_id] = enemy_center

                # Guardar histórico para próximo frame
                if track_id not in self._enemy_history:
                    self._enemy_history[track_id] = deque(maxlen=5)
                self._enemy_history[track_id].append((time.time(), enemy_center))

                # Atualizar WorldModel
                self.world_model.update_enemy(
                    track_id=track_id,
                    position=enemy_center,
                    velocity=velocity,
                    health=1.0,  # Não temos HP por inimigo ainda
                    brawler_name="enemy"
                )

            # Remover inimigos que desapareceram
            for track_id in list(self._enemy_positions.keys()):
                if track_id not in current_ids:
                    # Verificar se pode estar num bush próximo
                    if bushes:
                        for bush in bushes:
                            bush_center = center(bush)
                            enemy_pos = self._enemy_positions[track_id]
                            dist = ((enemy_pos[0] - bush_center[0])**2 +
                                    (enemy_pos[1] - bush_center[1])**2)**0.5
                            if dist < 100:  # Enemy desapareceu perto de bush
                                self.world_model.mark_enemy_in_bush(track_id, bush_center)

            # Atualizar power cubes
            if power_cubes:
                for cube_bbox in power_cubes:
                    cube_center = center(cube_bbox)
                    cube_key = f"{int(cube_center[0]//50)}_{int(cube_center[1]//50)}"
                    self.world_model.power_cubes[cube_key] = {
                        "position": cube_center,
                        "last_seen": time.time(),
                        "collected": False
                    }

            # Atualizar bushes conhecidos
            if bushes:
                self.world_model.known_bushes = [center(b) for b in bushes]

            # Atualizar match phase
            elapsed = time.time() - (self.world_model.match_start_time or time.time())
            if elapsed < 30:
                self.world_model.match_phase = "early"
            elif elapsed < 90:
                self.world_model.match_phase = "mid"
            else:
                self.world_model.match_phase = "late"

        except Exception as e:
            logger.warning(f"[WORLD_INTEGRATION] Erro ao atualizar WorldModel: {e}")

    def _find_matching_track(self, position: Tuple[float, float]) -> Optional[int]:
        """Encontra track_id existente para esta posição ou None se nova."""
        min_dist = 80  # Distância mínima para considerar mesma pessoa

        for track_id, prev_pos in self._enemy_positions.items():
            dist = ((position[0] - prev_pos[0])**2 +
                    (position[1] - prev_pos[1])**2)**0.5
            if dist < min_dist:
                return track_id

        return None

    def record_damage(self, position: Tuple[float, float], damage: float, enemy_id: Optional[int] = None):
        """Regista que o jogador tomou dano."""
        if self.world_model:
            self.world_model.record_damage_taken(position, damage, enemy_id)

    def get_safe_zones(self) -> List[Tuple[float, float]]:
        """Retorna lista de zonas seguras para当前位置."""
        if not self.world_model:
            return []

        safe = []
        player_pos = self.world_model.player_position or (self.map_width // 2, self.map_height // 2)

        # Encontrar células safe que não são dangerous
        for (gx, gy), zone in self.world_model.map_grid.items():
            if zone.zone_type.value == "safe":
                cell_center = (
                    gx * self.world_model.CELL_SIZE + self.world_model.CELL_SIZE // 2,
                    gy * self.world_model.CELL_SIZE + self.world_model.CELL_SIZE // 2
                )
                safe.append(cell_center)

        return safe

    def get_enemy_predictions(self) -> Dict[int, Tuple[float, float]]:
        """Retorna posições previstas dos inimigos."""
        if not self.world_model:
            return {}

        predictions = {}
        for track_id, enemy_mem in self.world_model.enemies.items():
            if enemy_mem.is_active:
                predictions[track_id] = enemy_mem.predict_position()

        return predictions

    def get_cube_positions(self) -> List[Tuple[float, float]]:
        """Retorna posições de power cubes ainda disponíveis."""
        if not self.world_model:
            return []

        available = []
        for cube_data in self.world_model.power_cubes.values():
            if not cube_data.get("collected", True):
                if time.time() - cube_data.get("last_seen", 0) < 30:
                    available.append(cube_data["position"])

        return available

    def reset(self):
        """Limpa todo o estado para novo match."""
        self._enemy_history.clear()
        self._enemy_positions.clear()
        self._next_track_id = 0
        if self.world_model:
            self.world_model.match_start_time = None
            self.world_model.enemies.clear()
            self.world_model.danger_zones.clear()