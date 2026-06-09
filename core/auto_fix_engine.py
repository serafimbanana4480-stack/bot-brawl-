"""
auto_fix_engine.py

Motor de auto-diagnóstico e recuperação automática para o bot.
Funciona em background, analisando periodicamente o estado do jogo e aplicando
 correções quando deteta bloqueios, loops infinitos, ou deteção falhada.

Principais funcionalidades:
- Deadlock detection: deteta quando o bot está parado no mesmo estado
- Screenshot validation: verifica se os screenshots são válidos
- Color space detection: deteta e corrige RGB vs BGR
- Smart recovery: aplica ações de recuperação baseadas no contexto
- Performance metrics: regista tempos de resposta e taxas de sucesso
"""

import time
import logging
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

import numpy as np

from core.screenshot_analyzer import ScreenshotAnalyzer, ScreenshotAnalysis

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Snapshot de saúde do bot num momento específico."""
    timestamp: float
    state: str
    screenshot_valid: bool
    detector_working: bool
    lobby_detected: bool
    in_game_detected: bool
    action_taken: Optional[str] = None
    recovery_level: int = 0  # 0=normal, 1=soft, 2=medium, 3=hard
    details: Dict = field(default_factory=dict)


class AutoFixEngine:
    """
    Motor de auto-diagnóstico e recuperação automática.

    Usa um sistema de níveis de recuperação:
    - Nível 0: Normal, monitorização apenas
    - Nível 1: Soft recovery (cliques adicionais, pequenos ajustes)
    - Nível 2: Medium recovery (ESC, retry, mudança de estratégia)
    - Nível 3: Hard recovery (reset completo, forçar estado)
    """

    def __init__(
        self,
        screenshot_func: Optional[Callable[[], Optional[np.ndarray]]] = None,
        click_func: Optional[Callable[[int, int], None]] = None,
        key_func: Optional[Callable[[str], None]] = None,
        state_detector: Optional[Any] = None,
        emulator_controller: Optional[Any] = None,
    ):
        self.screenshot_func = screenshot_func
        self.click_func = click_func
        self.key_func = key_func
        self.state_detector = state_detector
        self.emulator_controller = emulator_controller

        self.analyzer = ScreenshotAnalyzer()
        self.health_history: deque = deque(maxlen=100)
        self.running = False

        # Configurações de timeout
        self.lobby_timeout = 60.0       # Max tempo no lobby antes de recovery
        self.loading_timeout = 30.0    # Max tempo em loading
        self.unknown_timeout = 45.0    # Max tempo em unknown
        self.in_game_min_time = 5.0    # Min tempo em in_game (anti-oscilação)

        # Contadores de estado
        self.state_durations: Dict[str, float] = {}
        self.last_known_good_state: Optional[str] = None
        self.last_known_good_time: float = time.time()

        # Estatísticas
        self.total_cycles = 0
        self.deadlocks_detected = 0
        self.recoveries_applied = 0
        self.successful_recoveries = 0

    def tick(self, current_state: str) -> Optional[str]:
        """
        Executa um ciclo de diagnóstico.

        Args:
            current_state: Estado atual do StateManager

        Returns:
            Estado forçado se recovery foi aplicado, None se normal.
        """
        self.total_cycles += 1
        now = time.time()

        # Atualizar duração do estado atual
        if current_state not in self.state_durations:
            self.state_durations[current_state] = now

        duration_in_state = now - self.state_durations.get(current_state, now)

        # === PASSO 1: Capturar e analisar screenshot ===
        screenshot = None
        analysis = None
        if self.screenshot_func:
            try:
                screenshot = self.screenshot_func()
                analysis = self.analyzer.analyze(screenshot)
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning(f"[AUTOFIX] Erro ao analisar screenshot: {e}")

        # === PASSO 2: Verificar se detector está a funcionar ===
        detector_working = self._check_detector_working(screenshot, current_state)

        # === PASSO 3: Detetar deadlocks e bloqueios ===
        recovery_action = None
        forced_state = None

        # 3a. Screenshot inválido (preto, branco, frozen)
        if analysis and not analysis.valid:
            issues = analysis.issues
            if "screenshot_nearly_black" in issues:
                recovery_action = "screenshot_black"
                forced_state = self._handle_black_screenshot()
            elif "screenshot_frozen" in issues:
                recovery_action = "screenshot_frozen"
                forced_state = self._handle_frozen_screenshot()

        # 3b. Stuck no lobby (não consegue pressionar play)
        elif current_state == "lobby" and duration_in_state > self.lobby_timeout:
            recovery_action = "lobby_timeout"
            forced_state = self._handle_lobby_timeout(screenshot, analysis)

        # 3c. Stuck em loading
        elif current_state == "loading" and duration_in_state > self.loading_timeout:
            recovery_action = "loading_timeout"
            forced_state = self._handle_loading_timeout()

        # 3d. Stuck em unknown
        elif current_state == "unknown" and duration_in_state > self.unknown_timeout:
            recovery_action = "unknown_timeout"
            forced_state = self._handle_unknown_timeout(screenshot)

        # 3e. Detector não funciona (ex: RGB/BGR trocado)
        elif not detector_working and duration_in_state > 10.0:
            recovery_action = "detector_broken"
            forced_state = self._handle_detector_broken(screenshot, analysis)

        # 3f. Detector "cego": retorna unknown mas screenshot mostra jogo visivel
        elif (current_state == "unknown" and duration_in_state > 15.0 and
              analysis and analysis.valid and not detector_working):
            recovery_action = "detector_blind"
            forced_state = self._handle_detector_blind(screenshot, analysis)

        # 3g. Anti-oscilação: se alternar rapidamente entre lobby e in_game
        elif self._is_oscillating(current_state):
            recovery_action = "anti_oscillation"
            forced_state = self._handle_oscillation(current_state)

        # === PASSO 4: Registrar snapshot ===
        snapshot = HealthSnapshot(
            timestamp=now,
            state=current_state,
            screenshot_valid=analysis.valid if analysis else False,
            detector_working=detector_working,
            lobby_detected=analysis.region_health.get("play_button_yellow", 0) > 0.5 if analysis else False,
            in_game_detected=current_state == "in_game",
            action_taken=recovery_action,
            recovery_level=self._get_recovery_level(recovery_action),
            details={
                "duration_in_state": duration_in_state,
                "color_space": analysis.color_space if analysis else "unknown",
                "avg_brightness": analysis.avg_brightness if analysis else 0.0,
            }
        )
        self.health_history.append(snapshot)

        if recovery_action:
            self.deadlocks_detected += 1
            self.recoveries_applied += 1
            logger.warning(f"[AUTOFIX] Recovery aplicado: {recovery_action} -> {forced_state}")
            # Reset state duration for new state
            if forced_state:
                self.state_durations = {forced_state: now}
            return forced_state

        # Estado normal
        if current_state in ("in_game", "lobby"):
            self.last_known_good_state = current_state
            self.last_known_good_time = now

        return None

    def _check_detector_working(self, screenshot: Optional[np.ndarray], current_state: str) -> bool:
        """
        Verifica se o detector de estado está a produzir resultados plausíveis.
        """
        if screenshot is None or self.state_detector is None:
            return True  # Não podemos verificar, assumir OK

        try:
            result = self.state_detector.detect(screenshot)
            if result.state == "unknown" and result.confidence < 0.1:
                # Pode estar quebrado, mas também pode ser um estado desconhecido real
                # Verificar com screenshot analyzer
                is_lobby, _ = self.analyzer.is_lobby_likely(screenshot)
                if is_lobby and result.state != "lobby":
                    logger.warning(f"[AUTOFIX] Detector falhou: analyzer diz lobby, detector diz {result.state}")
                    return False
            return True
        except (FileNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.warning(f"[AUTOFIX] Erro ao verificar detector: {e}")
            return False

    def _get_recovery_level(self, action: Optional[str]) -> int:
        levels = {
            "screenshot_black": 2,
            "screenshot_frozen": 2,
            "lobby_timeout": 2,
            "loading_timeout": 2,
            "unknown_timeout": 3,
            "detector_broken": 3,
            "detector_blind": 2,
            "anti_oscillation": 1,
        }
        return levels.get(action, 0)

    def _is_oscillating(self, current_state: str) -> bool:
        if len(self.health_history) < 10:
            return False
        recent = list(self.health_history)[-10:]
        states = [s.state for s in recent]
        # Se alternar entre 2 estados mais de 6 vezes em 10 ticks
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        return transitions > 6

    def _handle_black_screenshot(self) -> Optional[str]:
        """Screenshot preto: tentar re-capturar ou reiniciar captura."""
        logger.warning("[AUTOFIX] Screenshot preto detectado. Tentando recovery...")
        # Esperar um pouco e tentar novamente
        time.sleep(1.0)
        if self.key_func:
            self.key_func("esc")
        return None  # Não forçar estado, apenas tentar novamente

    def _handle_frozen_screenshot(self) -> Optional[str]:
        """Screenshot congelado: o emulador pode estar travado."""
        logger.warning("[AUTOFIX] Screenshot congelado. Enviando pequeno input para desbloquear...")
        if self.click_func:
            # Clicar numa área neutra
            self.click_func(100, 100)
        return None

    def _handle_lobby_timeout(self, screenshot: Optional[np.ndarray], analysis: Optional[ScreenshotAnalysis]) -> Optional[str]:
        """Preso no lobby: forçar clique no play com múltiplas estratégias."""
        logger.warning("[AUTOFIX] Timeout no lobby. Aplicando recovery inteligente...")

        if not screenshot or not analysis:
            return None

        h, w = screenshot.shape[:2]

        # Estratégia 1: Verificar se há popup bloqueando
        if analysis.region_health.get("center_brightness", 255) < 120:
            logger.info("[AUTOFIX] Possível popup detectado. Clicando centro...")
            if self.click_func:
                self.click_func(w // 2, h // 2)
            time.sleep(0.5)
            if self.key_func:
                self.key_func("esc")
            return None

        # Estratégia 2: O detector pode estar a reportar lobby mas o botão Play
        # não foi clicado com sucesso. Forçar clique direto nas coordenadas.
        lobby_prob, lobby_conf = self.analyzer.is_lobby_likely(screenshot)
        if lobby_prob:
            # Usar coordenadas do botão Play baseadas na análise visual
            play_x = int(w * 0.9419)
            play_y = int(h * 0.8949)
            logger.info(f"[AUTOFIX] Forçando clique no Play ({play_x}, {play_y})")
            if self.click_func:
                self.click_func(play_x, play_y)
            time.sleep(2.0)
            return None

        # Estratégia 3: Se não parece lobby, forçar reset para unknown e depois lobby
        logger.info("[AUTOFIX] Lobby não detectado visualmente. Forçando reset...")
        return "unknown"

    def _handle_loading_timeout(self) -> Optional[str]:
        """Preso em loading: forçar in_game."""
        logger.warning("[AUTOFIX] Timeout em loading. Forçando in_game...")
        return "in_game"

    def _handle_unknown_timeout(self, screenshot: Optional[np.ndarray]) -> Optional[str]:
        """Preso em unknown: tentar detetar visualmente ou resetar."""
        logger.warning("[AUTOFIX] Timeout em unknown. Aplicando recovery...")

        if screenshot is not None:
            is_lobby, _ = self.analyzer.is_lobby_likely(screenshot)
            if is_lobby:
                logger.info("[AUTOFIX] Analyzer confirma lobby. Forçando estado lobby.")
                return "lobby"

        # Fallback: clicar centro para fechar popup, depois lobby
        if self.click_func:
            self.click_func(960, 540)  # Centro 1920x1080
        time.sleep(0.5)
        if self.key_func:
            self.key_func("esc")
        time.sleep(0.5)
        return "lobby"

    def _handle_detector_broken(self, screenshot: Optional[np.ndarray], analysis: Optional[ScreenshotAnalysis]) -> Optional[str]:
        """
        O detector de estado está a falhar (ex: RGB vs BGR).
        Aplicar correções conhecidas.
        """
        logger.warning("[AUTOFIX] Detector de estado parece quebrado. Aplicando correções...")

        if analysis and analysis.color_space == "bgr":
            logger.error("[AUTOFIX] DETETADO: screenshots estão em BGR em vez de RGB!")
            # Não podemos corrigir o passado, mas podemos informar
            # No futuro, o wrapper deve converter para RGB antes de passar ao detector

        # Forçar estado baseado na análise visual
        if screenshot is not None:
            is_lobby, conf = self.analyzer.is_lobby_likely(screenshot)
            if is_lobby:
                logger.info(f"[AUTOFIX] Analyzer diz lobby (conf={conf:.2f}). Forçando lobby.")
                return "lobby"

            # Verificar se é in_game pelo joystick escuro + attack button visível
            h, w = screenshot.shape[:2]
            joy_region = screenshot[int(h*0.55):int(h*0.90), 0:int(w*0.20)]
            attack_region = screenshot[int(h*0.65):h, int(w*0.80):w]
            if joy_region.size > 0 and attack_region.size > 0:
                joy_dark = np.mean(joy_region) < 80
                attack_visible = np.std(attack_region) > 20
                if joy_dark and attack_visible:
                    logger.info("[AUTOFIX] Analyzer diz in_game. Forçando in_game.")
                    return "in_game"

        return "lobby"  # Fallback mais seguro

    def _handle_detector_blind(self, screenshot: Optional[np.ndarray], analysis: Optional[ScreenshotAnalysis]) -> Optional[str]:
        """
        O detector retorna 'unknown' mas o screenshot parece valido.
        Tentar usar heuristicas avancadas para determinar o estado.
        """
        logger.warning("[AUTOFIX] Detector 'cego' - screenshot valido mas detector retorna unknown. "
                       "Aplicando heuristicas avancadas...")

        if screenshot is None:
            return "lobby"

        h, w = screenshot.shape[:2]

        # Heuristica 1: Verificar se ha um botao Play visivel usando SmartPlayButtonDetector
        try:
            from pylaai_real.lobby_navigator import SmartPlayButtonDetector
            detector = SmartPlayButtonDetector(self.images_path)
            result = detector.find_play_button(screenshot)
            if result.found and result.coords:
                logger.info(f"[AUTOFIX] SmartPlayButtonDetector encontrou botao Play em {result.coords}. "
                           "Forcando estado lobby.")
                # Se temos click_func, clicar no botao encontrado
                if self.click_func and result.coords:
                    self.click_func(*result.coords)
                    time.sleep(1.5)
                return "lobby"
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug(f"[AUTOFIX] SmartPlayButtonDetector falhou no blind recovery: {e}")

        # Heuristica 2: Se o centro for escuro mas houver elementos coloridos na parte inferior,
        # pode ser o lobby com um fundo escuro (ex: menu de evento)
        bottom_area = screenshot[int(h*0.7):h, :]
        bottom_std = float(np.std(bottom_area))
        if bottom_std > 30:
            logger.info("[AUTOFIX] Area inferior tem conteudo variado. Possivelmente lobby com menu aberto.")
            if self.click_func:
                # Tentar clicar no centro para fechar qualquer menu/popup
                self.click_func(w // 2, h // 2)
                time.sleep(0.5)
            return "lobby"

        # Heuristica 3: Se houver muitos pixels escuros no centro (como nas screenshots de debug),
        # pode ser uma tela de transicao. Forcar lobby para tentar avancar.
        center_brightness = float(np.mean(screenshot[h//2-100:h//2+100, w//2-100:w//2+100]))
        if center_brightness < 30:
            logger.info("[AUTOFIX] Centro muito escuro. Possivelmente tela de transicao. "
                       "Forcando lobby e clicando centro.")
            if self.click_func:
                self.click_func(w // 2, h // 2)
                time.sleep(0.5)
            return "lobby"

        return "lobby"  # Fallback seguro

    def _handle_oscillation(self, current_state: str) -> Optional[str]:
        """Anti-oscilação: manter o estado mais provável."""
        logger.warning("[AUTOFIX] Anti-oscilação ativada. Mantendo estado estável.")
        # Manter o último estado conhecido bom
        if self.last_known_good_state:
            return self.last_known_good_state
        return current_state

    def get_status(self) -> Dict:
        """Retorna status atual do motor de auto-fix."""
        return {
            "total_cycles": self.total_cycles,
            "deadlocks_detected": self.deadlocks_detected,
            "recoveries_applied": self.recoveries_applied,
            "successful_recoveries": self.successful_recoveries,
            "last_known_good_state": self.last_known_good_state,
            "health_history_count": len(self.health_history),
        }

    def reset(self):
        """Reseta o motor para novo ciclo."""
        self.state_durations.clear()
        self.health_history.clear()
        self.last_known_good_state = None
        self.last_known_good_time = time.time()
