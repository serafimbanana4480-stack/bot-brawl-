"""
ability_manager.py

Ability (super/gadget) logic extracted from play.py.
Provides AbilityManagerMixin with super and gadget management.
"""

import time
import math
import numpy as np
import logging
import random
from typing import Optional, List, Dict, Tuple

# Utilitarios de humanizacao
from pylaai_real.humanization_utils import human_delay, jitter_value, HumanPauseSimulator

logger = logging.getLogger(__name__)

try:
    from realtime_logs import get_log_manager
    log_manager = get_log_manager()
except ImportError:
    log_manager = None


class AbilityManagerMixin:
    """Mixin providing super and gadget management logic."""

    def _manage_abilities(self, player, enemies):
        """Usa Super e Gadgets estrategicamente"""
        logger.debug(f"[COMBAT] Avaliando uso de habilidades: {len(enemies)} inimigos")

        # Coordenadas dinâmicas dos botões
        if self.movement and hasattr(self.movement, 'window_w'):
            w, h = self.movement.window_w, self.movement.window_h
        else:
            w, h = self._get_safe_resolution()

        super_btn_x = round(w * 0.75)
        super_btn_y = round(h * 0.69)
        gadget_btn_x = round(w * 0.78)
        gadget_btn_y = round(h * 0.58)

        # Se houver 2+ inimigos perto, usar Super
        if len(enemies) >= 2 and self._distance(player, enemies[0]) < 300:
            if self.emulator_controller:
                logger.info("[COMBAT] SUPER ATIVADO! Multidao detectada.")
                logger.debug(f"[COMBAT] Executando tap de Super em ({super_btn_x}, {super_btn_y})")
                self.emulator_controller.ensure_window_active()
                self.emulator_controller.tap_scaled(super_btn_x, super_btn_y)
                self.last_combat_snapshot = {**self.last_combat_snapshot, "super_taken": True}
            else:
                logger.debug("[COMBAT] Super disponivel mas EmulatorController nao disponivel")
        else:
            logger.debug(f"[COMBAT] Condicoes para Super nao atendidas: enemies={len(enemies)}, dist={self._distance(player, enemies[0]) if enemies else 'N/A'}")

        # Usar Gadget quando há inimigo próximo e agressividade alta
        brawler_strategy = self.get_brawler_strategy()
        if enemies and brawler_strategy.get("has_gadget", False):
            closest_dist = min(self._distance(player, e) for e in enemies)
            if closest_dist < 400:
                if self.emulator_controller:
                    logger.info(f"[COMBAT] GADGET ATIVADO! Inimigo a {closest_dist:.0f}px.")
                    self.emulator_controller.ensure_window_active()
                    self.emulator_controller.tap_scaled(gadget_btn_x, gadget_btn_y)
                    self.last_combat_snapshot = {**self.last_combat_snapshot, "gadget_taken": True}

