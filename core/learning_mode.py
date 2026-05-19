"""
core/learning_mode.py

Orquestra o Modo Teste Aprendizagem:
- Navega do lobby para a Training Cave
- Executa partidas contra bots
- Regista métricas via LearningMetricsCollector
- Imprime sumário ao terminar

Integração:
    StateManager chama este controller nos handlers de lobby e in_game_learning.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports para evitar circular dependencies
try:
    from core.learning_metrics import LearningMetricsCollector
    HAS_LEARNING_METRICS = True
except ImportError:
    HAS_LEARNING_METRICS = False
    LearningMetricsCollector = None


try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None


class LearningModeController:
    """
    Controlador do modo de aprendizagem contra bots (Training Cave).

    Responsabilidades:
    1. Navegar lobby → Training Cave
    2. Gerenciar ciclo de partidas (início, frames, fim, reinício)
    3. Delegar recolha de métricas ao LearningMetricsCollector
    """

    def __init__(
        self,
        lobby_automator,
        emulator_controller,
        screenshot_taker,
        state_finder,
        play_logic,
        metrics_collector=None,
        max_matches: int = 5,
        match_timeout_seconds: float = 300.0,
        frame_interval: float = 0.1,
    ):
        self.lobby = lobby_automator
        self.emulator_controller = emulator_controller
        self.screenshot = screenshot_taker
        self.state_finder = state_finder
        self.play = play_logic
        self.max_matches = max_matches
        self.match_timeout_seconds = match_timeout_seconds
        self.frame_interval = frame_interval

        # Metrics collector (cria automaticamente se não fornecido)
        if metrics_collector is not None:
            self.metrics = metrics_collector
        elif HAS_LEARNING_METRICS:
            self.metrics = LearningMetricsCollector()
        else:
            self.metrics = None
            logger.warning("[LEARNING_MODE] LearningMetricsCollector não disponível")

        self._current_match_index = 0
        self._match_start_time: Optional[float] = None
        self._in_training_cave = False
        self._match_active = False

    # ------------------------------------------------------------------
    # Navegação: Lobby → Training Cave
    # ------------------------------------------------------------------

    def enter_training_cave(self) -> bool:
        """Navega do lobby para dentro da Training Cave."""
        logger.info("[LEARNING_MODE] A entrar na Training Cave...")

        if self.lobby is None:
            logger.error("[LEARNING_MODE] LobbyAutomator não disponível")
            return False

        screenshot = self._take_screenshot()
        if screenshot is None:
            logger.error("[LEARNING_MODE] Screenshot falhou ao entrar na Training Cave")
            return False

        # Usa o TrainingCaveNavigator via LobbyAutomator
        if hasattr(self.lobby, 'enter_training_cave'):
            try:
                result = self.lobby.enter_training_cave(
                    screenshot=screenshot,
                    window_size=self._get_window_size(),
                )
                if result and getattr(result, 'success', False):
                    logger.info("[LEARNING_MODE] Training Cave entrada com sucesso (method=%s)",
                                getattr(result, 'method_used', 'unknown'))
                    self._in_training_cave = True
                    time.sleep(2.0)  # Esperar load
                    return True
                else:
                    logger.warning("[LEARNING_MODE] enter_training_cave retornou insucesso")
            except Exception as e:
                logger.warning("[LEARNING_MODE] Erro ao entrar na Training Cave: %s", e)
        else:
            logger.warning("[LEARNING_MODE] LobbyAutomator não tem enter_training_cave")

        # Fallback manual simplificado: menu → clique em Treino → Train
        return self._manual_enter_training_cave()

    def _manual_enter_training_cave(self) -> bool:
        """Fallback manual caso o LobbyAutomator expandido não esteja disponível."""
        if self.emulator_controller is None:
            return False

        w, h = self._get_window_size()
        # Menu (canto superior esquerdo)
        self.emulator_controller.tap_scaled(int(w * 0.06), int(h * 0.08))
        time.sleep(1.0)
        # Coordenadas típicas do botão Training Cave
        self.emulator_controller.tap_scaled(int(w * 0.50), int(h * 0.55))
        time.sleep(1.5)
        # Botão Train
        self.emulator_controller.tap_scaled(int(w * 0.50), int(h * 0.88))
        time.sleep(2.0)
        self._in_training_cave = True
        logger.info("[LEARNING_MODE] Fallback manual para Training Cave executado")
        return True

    def exit_training_cave(self) -> bool:
        """Sai da Training Cave e volta ao lobby."""
        if self.lobby is not None and hasattr(self.lobby, 'exit_training_cave'):
            try:
                result = self.lobby.exit_training_cave(window_size=self._get_window_size())
                if result and getattr(result, 'success', False):
                    self._in_training_cave = False
                    return True
            except Exception as e:
                logger.warning("[LEARNING_MODE] Erro ao sair da Training Cave: %s", e)

        # Fallback manual
        if self.emulator_controller:
            w, h = self._get_window_size()
            self.emulator_controller.tap_scaled(int(w * 0.94), int(h * 0.06))  # pause
            time.sleep(0.5)
            self.emulator_controller.tap_scaled(int(w * 0.50), int(h * 0.55))  # exit
            time.sleep(0.5)
            self.emulator_controller.tap_scaled(int(w * 0.60), int(h * 0.58))  # confirm
            time.sleep(0.5)
            self.emulator_controller.keyevent(4)  # ESC
            self._in_training_cave = False
            return True
        return False

    # ------------------------------------------------------------------
    # Ciclo de partida
    # ------------------------------------------------------------------

    def start_match(self, brawler: str = "unknown") -> None:
        """Inicia métricas para uma nova partida na Training Cave."""
        self._current_match_index += 1
        self._match_start_time = time.time()
        self._match_active = True
        if self.metrics:
            self.metrics.start_match(brawler)
        logger.info("[LEARNING_MODE] Partida %s/%s iniciada | Brawler: %s",
                    self._current_match_index, self.max_matches, brawler)

    def run_frame(self, screenshot: np.ndarray) -> Any:
        """Executa um único frame de gameplay e regista métricas."""
        if self.play is None:
            logger.warning("[LEARNING_MODE] PlayLogic não disponível")
            return None

        result = self.play.play_round(screenshot)

        # Notificar métricas
        if self.metrics:
            enemies = 0
            player = False
            action = str(result) if result else None

            # Tentar extrair deteções do resultado ou do PlayLogic
            if hasattr(self.play, 'last_combat_snapshot'):
                snap = self.play.last_combat_snapshot
                enemies = snap.get('enemies', 0)
                player = snap.get('player') is not None

            self.metrics.log_frame(
                enemies_detected=enemies,
                player_detected=player,
                action_taken=action,
            )

        return result

    def is_match_ended(self, screenshot: np.ndarray) -> Tuple[bool, str]:
        """
        Verifica se a partida na Training Cave terminou.

        Retorna (ended, reason) onde reason pode ser 'timeout', 'restart_screen', 'manual'.
        """
        # 1. Timeout
        if self._match_start_time is not None:
            elapsed = time.time() - self._match_start_time
            if elapsed > self.match_timeout_seconds:
                logger.info("[LEARNING_MODE] Timeout de partida (%.0fs)", elapsed)
                return True, "timeout"

        # 2. Verificar se apareceu o botão de restart (indica morte/fim)
        if self._detect_restart_button(screenshot):
            return True, "restart_screen"

        # 3. Se o state_finder indicar que saímos do jogo
        if self.state_finder is not None:
            try:
                state = self.state_finder.get_state(screenshot)
                if state in ('lobby', 'end', 'unknown'):
                    logger.info("[LEARNING_MODE] Estado mudou para '%s' — partida terminada", state)
                    return True, f"state_changed_to_{state}"
            except Exception:
                pass

        return False, ""

    def end_match(self, reason: str = "completed") -> None:
        """Finaliza a partida atual e regista o resultado."""
        self._match_active = False
        duration = 0.0
        if self._match_start_time is not None:
            duration = time.time() - self._match_start_time

        if self.metrics:
            self.metrics.end_match(result=reason, duration=duration)
        logger.info("[LEARNING_MODE] Partida %s finalizada: %s | Duração: %.1fs",
                    self._current_match_index, reason, duration)

    def restart_training(self) -> bool:
        """Reinicia o treino na Training Cave (após morte/fim de partida)."""
        if self.lobby is not None and hasattr(self.lobby, 'restart_training'):
            try:
                ok = self.lobby.restart_training(window_size=self._get_window_size())
                if ok:
                    logger.info("[LEARNING_MODE] Treino reiniciado via LobbyAutomator")
                    time.sleep(2.0)
                    return True
            except Exception as e:
                logger.warning("[LEARNING_MODE] Falha em restart_training: %s", e)

        # Fallback: pause → restart
        if self.emulator_controller:
            w, h = self._get_window_size()
            self.emulator_controller.tap_scaled(int(w * 0.94), int(h * 0.06))  # pause
            time.sleep(0.5)
            self.emulator_controller.tap_scaled(int(w * 0.50), int(h * 0.50))  # restart
            time.sleep(2.0)
            logger.info("[LEARNING_MODE] Treino reiniciado via fallback manual")
            return True
        return False

    def should_continue(self) -> bool:
        """Verifica se ainda deve continuar o modo de aprendizagem."""
        return self._current_match_index < self.max_matches

    def start_learning_mode(self, max_matches: Optional[int] = None) -> bool:
        """Ativa o modo de aprendizagem dinamicamente (pode ser chamado pela dashboard)."""
        if max_matches is not None:
            self.max_matches = max_matches
        self._current_match_index = 0
        logger.info("[LEARNING_MODE] Modo de aprendizagem ativado (max_matches=%s)", self.max_matches)
        return True

    def stop_learning_mode(self) -> bool:
        """Desativa o modo de aprendizagem e volta ao lobby."""
        self._match_active = False
        self._in_training_cave = False
        self.exit_training_cave()
        logger.info("[LEARNING_MODE] Modo de aprendizagem desativado")
        return True

    def get_live_metrics(self) -> Dict[str, Any]:
        """Retorna métricas live para a dashboard."""
        metrics_data: Dict[str, Any] = {
            "active": self._match_active or self._in_training_cave,
            "current_match": self._current_match_index,
            "max_matches": self.max_matches,
            "current_brawler": "unknown",
            "match_duration_seconds": 0.0,
            "detections_enemies": 0,
            "detections_player": 0,
            "actions_attack": 0,
            "actions_move": 0,
            "actions_super": 0,
            "actions_none": 0,
            "kills": 0,
            "deaths": 0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "accuracy_percent": 0.0,
            "frames_history": [],
        }

        if self.metrics and self.metrics.current_match:
            cm = self.metrics.current_match
            metrics_data["current_brawler"] = cm.brawler
            metrics_data["detections_enemies"] = cm.detections_enemies
            metrics_data["detections_player"] = cm.detections_player
            metrics_data["actions_attack"] = cm.actions_attack
            metrics_data["actions_move"] = cm.actions_move
            metrics_data["actions_super"] = cm.actions_super
            metrics_data["kills"] = cm.kills
            metrics_data["deaths"] = cm.deaths
            metrics_data["damage_dealt"] = round(cm.damage_dealt, 1)
            metrics_data["damage_taken"] = round(cm.damage_taken, 1)

            # Duração do match atual
            if self._match_start_time is not None:
                metrics_data["match_duration_seconds"] = round(time.time() - self._match_start_time, 1)

            # Precisão estimada: kills / ações de ataque * 100
            total_actions = cm.actions_attack + cm.actions_move + cm.actions_super
            if total_actions > 0 and cm.actions_attack > 0:
                metrics_data["accuracy_percent"] = round(
                    (cm.kills / max(1, cm.actions_attack)) * 100, 1
                )

            # Histórico de frames para gráficos
            metrics_data["frames_history"] = self.metrics.get_frame_history(limit=120)

        return metrics_data

    def print_summary(self) -> None:
        """Imprime sumário da sessão de aprendizagem."""
        if self.metrics:
            self.metrics.print_summary()
        else:
            logger.info("[LEARNING_MODE] Nenhum metrics collector disponível para sumário.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _take_screenshot(self) -> Optional[np.ndarray]:
        if self.screenshot is not None and hasattr(self.screenshot, 'take'):
            try:
                return self.screenshot.take()
            except Exception as e:
                logger.debug("[LEARNING_MODE] Screenshot falhou: %s", e)
        return None

    def _get_window_size(self) -> Tuple[int, int]:
        """Retorna resolução da janela (default 1920x1080)."""
        if self.lobby and hasattr(self.lobby, 'lobby_config'):
            cfg = self.lobby.lobby_config
            return getattr(cfg, 'w', 1920), getattr(cfg, 'h', 1080)
        return 1920, 1080

    def _detect_restart_button(self, screenshot: np.ndarray) -> bool:
        """Heurística simples para detetar botão de restart na Training Cave."""
        if not HAS_CV2 or screenshot is None or screenshot.size == 0:
            return False
        try:
            h, w = screenshot.shape[:2]
            # Região central onde o botão Restart aparece
            roi = screenshot[int(h * 0.45):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
            if roi.size == 0:
                return False
            # Procurar por cor branca/cinzenta predominante (fundo do botão)
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            mean_val = gray.mean()
            # Se a região central for muito clara e com pouca variância, pode ser o menu de pause/restart
            if mean_val > 180 and gray.std() < 60:
                return True
        except Exception:
            pass
        return False
