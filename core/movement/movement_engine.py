"""
movement_engine.py

Movement logic extracted from play.py.
Provides MovementEngineMixin with player movement, pathfinding,
and environmental interaction.
"""

import time
import math
import numpy as np
import logging
import random
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple

# Utilitarios de humanizacao
from pylaai_real.humanization_utils import human_delay, jitter_value, HumanPauseSimulator

logger = logging.getLogger(__name__)

try:
    from tracker import EnemyTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

try:
    from realtime_logs import get_log_manager
    log_manager = get_log_manager()
except ImportError:
    log_manager = None

# Phase 10: Occupancy Grid (A* pathfinding)
try:
    from core.occupancy_grid import OccupancyGrid
    HAS_OCCUPANCY_GRID = True
except ImportError:
    HAS_OCCUPANCY_GRID = False
    OccupancyGrid = None


class MovementEngineMixin:
    """Mixin providing movement and navigation logic."""

    def _distance(self, box1, box2):
        # box1/box2 can be: bbox [x1,y1,x2,y2], center-point (cx,cy), or tuple
        if isinstance(box1, (list, tuple)) and len(box1) >= 4:
            c1 = ((box1[0]+box1[2])//2, (box1[1]+box1[3])//2)
        elif isinstance(box1, (list, tuple)) and len(box1) >= 2:
            c1 = (box1[0], box1[1])
        else:
            c1 = (0, 0)

        if isinstance(box2, (list, tuple)) and len(box2) >= 4:
            c2 = ((box2[0]+box2[2])//2, (box2[1]+box2[3])//2)
        elif isinstance(box2, (list, tuple)) and len(box2) >= 2:
            c2 = (box2[0], box2[1])
        else:
            c2 = (0, 0)

        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def _find_player(self, detections):
        """
        Find player in detections using unified class registry.

        Uses core.class_registry for consistent name normalization.
        Eliminates hardcoded variants (Player, player, self, teammate, person).
        """
        from core.class_registry import get_by_type

        logger.debug(f"[PLAYER] Procurando player em detecções: {list(detections.keys())}")

        # Use unified registry for consistent lookup
        player = get_by_type(detections, "player", first_only=True)

        if player is not None:
            logger.info(f"[PLAYER] Player encontrado: bbox={player}")
            logger.debug(f"[PLAYER] Coordenadas: x={player[0]}, y={player[1]}, w={player[2]-player[0]}, h={player[3]-player[1]}")
            return player

        logger.error("[PLAYER] Player não encontrado em detecções")
        return None

    def _find_bushes(self, detections):
        """Find bushes to hide in using unified class registry."""
        from core.class_registry import get_by_type
        return get_by_type(detections, "bush", first_only=False)

    def _find_power_cubes(self, detections):
        """Find power cubes using unified class registry."""
        from core.class_registry import get_by_type
        # cubebox is the canonical name for power cubes
        return get_by_type(detections, "cubebox", first_only=False)

    def _execute_movement(self, key):
        if not key:
            logger.debug("[COMBAT] Nenhuma tecla de movimento fornecida")
            return
        logger.debug(f"[MOVEMENT] Iniciando movimento: key={key}")
        if not self.emulator_controller:
            logger.warning("[COMBAT] EmulatorController não disponível para executar movimento")
            return
        # Usar coordenadas dinâmicas do joystick se disponíveis
        if self.movement and hasattr(self.movement, 'joystick_center_x'):
            jx = self.movement.joystick_center_x
            jy = self.movement.joystick_center_y
        else:
            jx, jy = 192, 810
        dist = 100
        dx, dy = 0, 0
        if 'W' in key: dy = -dist
        if 'S' in key: dy = dist
        if 'A' in key: dx = -dist
        if 'D' in key: dx = dist
        logger.debug(f"[MOVEMENT] Posição do joystick base: ({jx}, {jy})")
        logger.debug(f"[MOVEMENT] Deslocamento calculado: dx={dx}, dy={dy}")
        logger.debug(f"[MOVEMENT] Coordenada final do swipe: ({jx+dx}, {jy+dy})")
        logger.debug(f"[MOVEMENT] Duração do swipe: 150ms")
        logger.debug(f"[COMBAT] Executando swipe de joystick: ({jx}, {jy}) -> ({jx+dx}, {jy+dy}), key={key}")
        try:
            self.emulator_controller.ensure_window_active()
            self.emulator_controller.swipe_scaled(jx, jy, jx+dx, jy+dy, duration=150)
            logger.info(f"[MOVEMENT] Swipe executado: ({jx}, {jy}) -> ({jx+dx}, {jy+dy})")
            logger.debug("[COMBAT] Movimento executado")
        except Exception as e:
            logger.error(f"[MOVEMENT] Erro ao executar movimento: key={key}, error={e}")
            logger.error(f"[MOVEMENT] EmulatorController disponível: {self.emulator_controller is not None}")

    def _is_in_bush(self, player_bbox, bush_bbox):
        """Check if player bbox intersects with bush bbox."""
        if not player_bbox or not bush_bbox or len(player_bbox) < 4 or len(bush_bbox) < 4:
            return False
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = bush_bbox
        return not (px2 < bx1 or px1 > bx2 or py2 < by1 or py1 > by2)

