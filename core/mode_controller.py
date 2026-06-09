"""
core/mode_controller.py

Orquestra os três modos operacionais do bot:
- training:   Modo Teste (Training Cave)
- farm:         Executar Bot normalmente (partidas PvP/PvE)
- learn:        RL Online + Recolha de Dados

Garante que apenas um modo está ativo de cada vez.
Expõe métricas live para a dashboard.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModeStatus:
    active_mode: Optional[str] = None
    training_active: bool = False
    farm_active: bool = False
    learn_active: bool = False
    session_start: float = field(default_factory=time.time)
    matches_completed: int = 0
    matches_target: int = 0
    current_brawler: str = "unknown"
    session_duration_seconds: float = 0.0
    message: str = ""


class ModeController:
    """
    Controlador central de modos.
    Interface unificada para a dashboard iniciar/parar qualquer modo.
    """

    VALID_MODES = {"training", "farm", "learn"}

    def __init__(
        self,
        wrapper_ref: Optional[Any] = None,
        learning_mode_controller: Optional[Any] = None,
        rl_engine: Optional[Any] = None,
        live_collector: Optional[Any] = None,
        esp_overlay: Optional[Any] = None,
    ):
        self.wrapper = wrapper_ref
        self.learning_mode_controller = learning_mode_controller
        self.rl_engine = rl_engine
        self.live_collector = live_collector
        self.esp_overlay = esp_overlay

        self._status = ModeStatus()
        self._lock = False

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start_mode(self, mode: str, config: Optional[Dict] = None) -> bool:
        config = config or {}
        if mode not in self.VALID_MODES:
            logger.error("[MODE_CTRL] Modo invalido: %s", mode)
            return False
        if self._status.active_mode and self._status.active_mode != mode:
            logger.info("[MODE_CTRL] Parando modo anterior '%s' para iniciar '%s'", self._status.active_mode, mode)
            self.stop_mode(self._status.active_mode)

        try:
            if mode == "training":
                return self._start_training(config)
            elif mode == "farm":
                return self._start_farm(config)
            elif mode == "learn":
                return self._start_learn(config)
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[MODE_CTRL] Falha ao iniciar %s: %s", mode, e)
            self._status.message = f"Erro: {e}"
            return False
        return False

    def stop_mode(self, mode: Optional[str] = None) -> bool:
        if mode is None:
            mode = self._status.active_mode
        if mode is None:
            return True
        try:
            if mode == "training":
                self._stop_training()
            elif mode == "farm":
                self._stop_farm()
            elif mode == "learn":
                self._stop_learn()
            self._status.active_mode = None
            return True
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[MODE_CTRL] Falha ao parar %s: %s", mode, e)
            return False

    def stop_all(self):
        for m in list(self.VALID_MODES):
            self.stop_mode(m)
        self._status = ModeStatus()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _start_training(self, config: Dict) -> bool:
        max_matches = config.get("max_matches", 5)
        brawler = config.get("brawler", None)
        logger.info("[MODE_CTRL] Iniciando Modo Treinamento (max_matches=%s)", max_matches)

        if self.wrapper and hasattr(self.wrapper, 'toggle_learning_mode'):
            ok = self.wrapper.toggle_learning_mode(enabled=True, max_matches=max_matches)
            if not ok:
                return False

        self._status.active_mode = "training"
        self._status.training_active = True
        self._status.session_start = time.time()
        self._status.matches_target = max_matches
        self._status.current_brawler = brawler or "auto"
        self._status.message = "Treinamento ativo"
        return True

    def _stop_training(self):
        logger.info("[MODE_CTRL] Parando Modo Treinamento")
        if self.wrapper and hasattr(self.wrapper, 'toggle_learning_mode'):
            self.wrapper.toggle_learning_mode(enabled=False)
        self._status.training_active = False
        self._status.message = "Treinamento parado"

    # ------------------------------------------------------------------
    # Farm
    # ------------------------------------------------------------------

    def _start_farm(self, config: Dict) -> bool:
        max_matches = config.get("max_matches", 0)
        max_time = config.get("max_time_minutes", 0)
        brawler = config.get("brawler", None)
        game_mode = config.get("game_mode", None)
        logger.info("[MODE_CTRL] Iniciando Modo Farm (matches=%s, time=%s, mode=%s)", max_matches, max_time, game_mode)

        if self.wrapper:
            # Inicia o bot normalmente
            if hasattr(self.wrapper, 'start'):
                self.wrapper.start()
            # Configura opcoes
            if game_mode and hasattr(self.wrapper, 'preferred_mode'):
                self.wrapper.preferred_mode = game_mode
            if brawler and hasattr(self.wrapper, 'current_brawler'):
                self.wrapper.current_brawler = brawler

        self._status.active_mode = "farm"
        self._status.farm_active = True
        self._status.session_start = time.time()
        self._status.matches_target = max_matches
        self._status.current_brawler = brawler or "auto"
        self._status.message = "Farm ativo"
        return True

    def _stop_farm(self):
        logger.info("[MODE_CTRL] Parando Modo Farm")
        if self.wrapper and hasattr(self.wrapper, 'stop'):
            self.wrapper.stop()
        self._status.farm_active = False
        self._status.message = "Farm parado"

    # ------------------------------------------------------------------
    # Learn (RL + Data Collection)
    # ------------------------------------------------------------------

    def _start_learn(self, config: Dict) -> bool:
        rl_enabled = config.get("rl_enabled", True)
        collection_enabled = config.get("collection_enabled", True)
        max_matches = config.get("max_matches", 10)
        logger.info("[MODE_CTRL] Iniciando Modo Aprender (rl=%s, collect=%s)", rl_enabled, collection_enabled)

        # Start RL engine
        if rl_enabled and self.rl_engine and hasattr(self.rl_engine, 'start'):
            try:
                self.rl_engine.start()
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[MODE_CTRL] RL start falhou: %s", e)

        # Start live collector
        if collection_enabled and self.live_collector and hasattr(self.live_collector, 'start'):
            try:
                self.live_collector.start()
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[MODE_CTRL] Collector start falhou: %s", e)

        # Start farm in background for RL to observe
        if self.wrapper and hasattr(self.wrapper, 'start'):
            self.wrapper.start()

        self._status.active_mode = "learn"
        self._status.learn_active = True
        self._status.session_start = time.time()
        self._status.matches_target = max_matches
        self._status.message = "Aprendizagem ativa"
        return True

    def _stop_learn(self):
        logger.info("[MODE_CTRL] Parando Modo Aprender")
        if self.rl_engine and hasattr(self.rl_engine, 'stop'):
            try:
                self.rl_engine.stop()
            except (RuntimeError, AttributeError, OSError):
                pass
        if self.live_collector and hasattr(self.live_collector, 'stop'):
            try:
                self.live_collector.stop()
            except (RuntimeError, AttributeError, OSError):
                pass
        if self.wrapper and hasattr(self.wrapper, 'stop'):
            self.wrapper.stop()
        self._status.learn_active = False
        self._status.message = "Aprendizagem parada"

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        s = self._status
        s.session_duration_seconds = time.time() - s.session_start if s.active_mode else 0.0
        return {
            "active_mode": s.active_mode,
            "training_active": s.training_active,
            "farm_active": s.farm_active,
            "learn_active": s.learn_active,
            "session_duration_seconds": round(s.session_duration_seconds, 1),
            "matches_completed": s.matches_completed,
            "matches_target": s.matches_target,
            "current_brawler": s.current_brawler,
            "message": s.message,
        }

    def update_match_count(self, increment: int = 1):
        self._status.matches_completed += increment

    def is_any_active(self) -> bool:
        return self._status.active_mode is not None
