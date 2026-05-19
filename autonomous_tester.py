"""
autonomous_tester.py

Sistema de testes e diagnósticos autônomos para o bot.
Verifica a saúde do bot periodicamente e aplica correções automáticas.
"""

import time
import logging

logger = logging.getLogger(__name__)


class AutonomousTester:
    """Testa e diagnostica o bot automaticamente."""

    def __init__(self, state_manager=None):
        self.state_manager = state_manager
        self._last_check_time = 0
        self._check_interval = 10.0  # segundos entre checks
        self._stuck_counter = 0
        self._last_state = None
        self._last_state_time = 0

    def periodic_check(self):
        """Verificação periódica de saúde do bot."""
        now = time.time()
        if now - self._last_check_time < self._check_interval:
            return
        self._last_check_time = now

        if self.state_manager is None:
            return

        sm = self.state_manager
        current_state = getattr(sm, 'current_state', 'unknown')

        # Check 1: Stuck in same state for too long
        if current_state == self._last_state:
            stuck_time = now - self._last_state_time
            if stuck_time > 30:
                logger.warning(f"[AUTOTEST] Bot preso em '{current_state}' há {stuck_time:.0f}s")
                self._apply_stuck_recovery(sm, current_state)
        else:
            self._last_state = current_state
            self._last_state_time = now
            self._stuck_counter = 0

        # Check 2: In game but no combat actions
        if current_state == 'in_game':
            last_action = getattr(sm, '_last_combat_action_time', 0)
            if now - last_action > 5:
                logger.warning(f"[AUTOTEST] Bot em jogo mas sem ações há {now - last_action:.0f}s")
                self._force_combat_action(sm)

        # Check 3: Play logic available
        play = getattr(sm, 'play', None)
        if play is None:
            logger.warning("[AUTOTEST] Play logic não disponível!")

        # Check 4: Screenshot working
        screenshot = getattr(sm, 'screenshot', None)
        if screenshot:
            try:
                img = screenshot.take()
                if img is None:
                    logger.warning("[AUTOTEST] Screenshot retornou None!")
            except Exception as e:
                logger.warning(f"[AUTOTEST] Erro no screenshot: {e}")

        logger.info(f"[AUTOTEST] Check completo - estado: {current_state}")

    def _apply_stuck_recovery(self, sm, state):
        """Aplica recovery quando bot está preso."""
        logger.info(f"[AUTOTEST] Aplicando recovery para estado '{state}'")
        if state in ('loading', 'matchmaking'):
            # Force in_game
            sm.current_state = 'in_game'
            sm.state_start_time = time.time()
            if hasattr(sm, '_remember_known_state'):
                sm._remember_known_state('in_game')
            if hasattr(sm, '_forced_in_game_time'):
                sm._forced_in_game_time = time.time()
            logger.info("[AUTOTEST] Forçado in_game para sair de loading/matchmaking")
        elif state == 'unknown':
            # Go to lobby
            sm.current_state = 'lobby'
            sm.state_start_time = time.time()
            if hasattr(sm, '_remember_known_state'):
                sm._remember_known_state('lobby')
            logger.info("[AUTOTEST] Forçado lobby para sair de unknown")
        elif state == 'lobby':
            # Try pressing play directly
            emu = getattr(sm, 'emulator_controller', None)
            if emu:
                try:
                    w, h = sm._get_window_size()
                    play_x = round(w * 0.9119)
                    play_y = round(h * 0.9122)
                    emu.tap_scaled(play_x, play_y)
                    logger.info(f"[AUTOTEST] Clicou Play em ({play_x},{play_y}) para sair de lobby")
                except Exception as e:
                    logger.warning(f"[AUTOTEST] Falha ao clicar Play: {e}")

    def _force_combat_action(self, sm):
        """Força uma ação de combate."""
        emu = getattr(sm, 'emulator_controller', None)
        movement = getattr(sm, 'movement', None)
        if emu is None:
            return

        try:
            joy_x = getattr(movement, 'joystick_center_x', 192) if movement else 192
            joy_y = getattr(movement, 'joystick_center_y', 810) if movement else 810

            import random
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.randint(100, 250)
            target_x = int(joy_x + distance * __import__('math').cos(angle))
            target_y = int(joy_y + distance * __import__('math').sin(angle))
            emu.swipe_scaled(joy_x, joy_y, target_x, target_y, duration=200)
            logger.info(f"[AUTOTEST] Forçado movimento: ({joy_x},{joy_y}) -> ({target_x},{target_y})")

            # Attack
            w, h = sm._get_window_size()
            atk_x = round(w * 0.90)
            atk_y = round(h * 0.82)
            emu.tap_scaled(atk_x, atk_y)
            logger.info(f"[AUTOTEST] Forçado ataque em ({atk_x},{atk_y})")

            if hasattr(sm, '_last_combat_action_time'):
                sm._last_combat_action_time = time.time()
        except Exception as e:
            logger.warning(f"[AUTOTEST] Falha ao forçar ação: {e}")
