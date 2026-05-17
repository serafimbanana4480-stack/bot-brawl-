"""
core/v2_integration.py

Integrador Unificado v2.1 — conecta todos os módulos estratégicos ao ciclo
principal do Soberana Omega SEM modificar profundamente wrapper.py.

Design:
- Singleton lazy-initialized que orquestra módulos v2.1
- Hook-based: registra callbacks no ciclo principal via monkey-patch seguro
- Graceful: se um módulo falha, os outros continuam
- Config-driven: ativa/desativa via config.json

Integrações:
1. DegradationManager → ajusta qualidade a cada ciclo
2. EventStore → registra eventos de lifecycle
3. RateLimiter → decide se pode jogar antes de cada partida
4. GameStateCheckpointer → salva estado a cada 30s
5. DistributedTracing → trace do ciclo principal
6. BrawlerAdaptiveController → adapta quando brawler muda
"""

import time
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class V2IntegrationConfig:
    """Configuração do integrador v2.1."""
    enabled: bool = True
    enable_degradation_manager: bool = True
    enable_event_store: bool = True
    enable_rate_limiter: bool = True
    enable_game_state_checkpoint: bool = True
    enable_distributed_tracing: bool = False  # Off by default (overhead)
    enable_brawler_adaptive: bool = True
    enable_replay_analyzer: bool = True
    enable_multi_objective_rl: bool = False  # Experimental
    enable_smart_frame_skip: bool = True
    enable_alert_system: bool = True
    account_id: str = "default_account"
    checkpoint_interval: float = 30.0


class V2Integrator:
    """
    Orquestra todos os módulos v2.1.

    Uso:
        integrator = V2Integrator.from_config(wrapper_instance, config)
        integrator.on_cycle_start()  # no inicio de cada ciclo
        integrator.on_cycle_end()    # no fim de cada ciclo
    """

    _instance: Optional["V2Integrator"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        wrapper: Optional[Any] = None,
        config: Optional[V2IntegrationConfig] = None,
    ):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.wrapper = wrapper
        self.config = config or V2IntegrationConfig()
        self._cycle_count = 0
        self._last_checkpoint_time = 0.0
        self._last_brawler: Optional[str] = None

        # Módulos lazy-initialized
        self._degradation_mgr: Optional[Any] = None
        self._event_store: Optional[Any] = None
        self._rate_limiter: Optional[Any] = None
        self._checkpointer: Optional[Any] = None
        self._tracer: Optional[Any] = None
        self._brawler_ctrl: Optional[Any] = None
        self._replay_analyzer: Optional[Any] = None
        self._moo: Optional[Any] = None
        self._frame_skipper: Optional[Any] = None
        self._alert_system: Optional[Any] = None

        self._init_modules()
        logger.info("[V2_INTEGRATOR] Inicializado (enabled=%s)", self.config.enabled)

    def _init_modules(self):
        """Inicializa módulos de forma lazy e segura."""
        cfg = self.config

        if cfg.enable_degradation_manager:
            try:
                from core.degradation_manager import DegradationManager
                self._degradation_mgr = DegradationManager()
            except Exception as e:
                logger.warning("[V2] DegradationManager indisponível: %s", e)

        if cfg.enable_event_store:
            try:
                from core.event_store import EventStore, DomainEventType
                self._event_store = EventStore()
                self._event_store.append(
                    DomainEventType.SESSION_STARTED,
                    self.config.account_id,
                    "session",
                    {"config": cfg.__dict__},
                )
            except Exception as e:
                logger.warning("[V2] EventStore indisponível: %s", e)

        if cfg.enable_rate_limiter:
            try:
                from core.rate_limiter import IntelligentRateLimiter
                self._rate_limiter = IntelligentRateLimiter()
                self._rate_limiter.register_account(cfg.account_id)
            except Exception as e:
                logger.warning("[V2] RateLimiter indisponível: %s", e)

        if cfg.enable_game_state_checkpoint:
            try:
                from core.game_state_checkpoint import GameStateCheckpointer
                self._checkpointer = GameStateCheckpointer(
                    checkpoint_interval=cfg.checkpoint_interval,
                )
            except Exception as e:
                logger.warning("[V2] GameStateCheckpointer indisponível: %s", e)

        if cfg.enable_distributed_tracing:
            try:
                from core.distributed_tracing import Tracer
                self._tracer = Tracer()
            except Exception as e:
                logger.warning("[V2] Tracer indisponível: %s", e)

        if cfg.enable_brawler_adaptive:
            try:
                from decision.brawler_adaptive_controller import BrawlerAdaptiveController
                self._brawler_ctrl = BrawlerAdaptiveController()
            except Exception as e:
                logger.warning("[V2] BrawlerAdaptiveController indisponível: %s", e)

        if cfg.enable_replay_analyzer:
            try:
                from core.replay_failure_analyzer import ReplayFailureAnalyzer
                self._replay_analyzer = ReplayFailureAnalyzer()
            except Exception as e:
                logger.warning("[V2] ReplayFailureAnalyzer indisponível: %s", e)

        if cfg.enable_multi_objective_rl:
            try:
                from decision.multi_objective_rl import MultiObjectiveOptimizer
                self._moo = MultiObjectiveOptimizer()
            except Exception as e:
                logger.warning("[V2] MultiObjectiveOptimizer indisponível: %s", e)

        if cfg.enable_smart_frame_skip:
            try:
                from core.smart_frame_skipper import SmartFrameSkipper
                self._frame_skipper = SmartFrameSkipper()
            except Exception as e:
                logger.warning("[V2] SmartFrameSkipper indisponível: %s", e)

        if cfg.enable_alert_system:
            try:
                from core.alert_system import AlertSystem
                self._alert_system = AlertSystem()
            except Exception as e:
                logger.warning("[V2] AlertSystem indisponível: %s", e)

    # ------------------------------------------------------------------
    # Ciclo principal hooks
    # ------------------------------------------------------------------

    def on_cycle_start(self) -> bool:
        """
        Chamado no início de cada ciclo do monitor loop.
        Retorna False se o ciclo deve ser abortado (ex: rate limit).
        """
        if not self.config.enabled:
            return True

        self._cycle_count += 1
        start_time = time.time()

        # Tracing
        if self._tracer:
            self._cycle_span = self._tracer.start_span("monitor_cycle")

        # Rate limiting — verificar se pode jogar
        if self._rate_limiter:
            if not self._rate_limiter.should_play(self.config.account_id):
                logger.info("[V2] Rate limiter: pausa forçada")
                if self._event_store:
                    from core.event_store import DomainEventType
                    self._event_store.append(
                        DomainEventType.BREAK_STARTED, self.config.account_id, "session",
                        {"reason": "rate_limit"},
                    )
                return False

        # Degradation — ajustar qualidade
        if self._degradation_mgr:
            self._degradation_mgr.check_health_and_degrade()

        # Frame skip — decidir se roda inferência neste ciclo
        if self._frame_skipper:
            should_skip = not self._frame_skipper.should_process_frame(
                frame_counter=self._cycle_count,
                current_state=self._get_wrapper_state(),
                degradation_mode=self._degradation_mgr.mode.value if self._degradation_mgr else "full_quality",
            )
            if should_skip:
                return True  # Skip inference but continue cycle

        # Brawler adaptation
        current_brawler = self._get_current_brawler()
        if current_brawler and current_brawler != self._last_brawler:
            if self._brawler_ctrl:
                self._brawler_ctrl.set_brawler(current_brawler)
                self._last_brawler = current_brawler
                if self._event_store:
                    from core.event_store import DomainEventType
                    self._event_store.append(
                        DomainEventType.BRAWLER_SELECTED,
                        self.config.account_id,
                        "session",
                        {"brawler": current_brawler},
                    )

        return True

    def on_cycle_end(self, cycle_duration: float = 0.0):
        """Chamado no fim de cada ciclo do monitor loop."""
        if not self.config.enabled:
            return

        now = time.time()

        # Tracing
        if self._tracer and hasattr(self, "_cycle_span") and self._cycle_span:
            self._tracer.finish_span(self._cycle_span, tags={"duration_ms": round(cycle_duration * 1000, 2)})
            self._cycle_span = None

        # Checkpointing
        if self._checkpointer and (now - self._last_checkpoint_time) >= self.config.checkpoint_interval:
            self._maybe_checkpoint()
            self._last_checkpoint_time = now

        # Record cycle metrics in rate limiter
        if self._rate_limiter and self.wrapper and hasattr(self.wrapper, "safety"):
            # Record match starts/ends if state changed
            pass

        # Alert system
        if self._alert_system:
            self._alert_system.check_alerts(
                cycle_duration=cycle_duration,
                wrapper_state=self._get_wrapper_state(),
                degradation_status=self._degradation_mgr.get_status() if self._degradation_mgr else {},
            )

    def on_match_start(self, brawler: str, map_name: str):
        """Chamado quando uma partida inicia."""
        if self._event_store:
            from core.event_store import DomainEventType
            self._event_store.append(
                DomainEventType.MATCH_STARTED,
                self.config.account_id,
                "match",
                {"brawler": brawler, "map": map_name},
            )
        if self._rate_limiter:
            self._rate_limiter.record_match_start(self.config.account_id)
        if self._brawler_ctrl:
            self._brawler_ctrl.set_brawler(brawler)

    def on_match_end(self, result: str, brawler: str, map_name: str, metrics: Optional[Dict] = None):
        """Chamado quando uma partida termina."""
        if self._event_store:
            from core.event_store import DomainEventType
            self._event_store.append(
                DomainEventType.MATCH_ENDED,
                self.config.account_id,
                "match",
                {"result": result, "brawler": brawler, "map": map_name, "metrics": metrics or {}},
            )
        if self._rate_limiter:
            self._rate_limiter.record_match_end(self.config.account_id, result)
        if self._brawler_ctrl:
            self._brawler_ctrl.update_from_match_result(brawler, result, metrics)

    def on_player_died(self, context: Dict[str, Any]):
        """Chamado quando o jogador morre."""
        if self._event_store:
            from core.event_store import DomainEventType
            self._event_store.append(
                DomainEventType.PLAYER_DIED,
                self.config.account_id,
                "match",
                context,
            )

    def on_action_taken(self, action: str, context: Dict[str, Any]):
        """Chamado quando uma ação é executada."""
        if self._event_store:
            from core.event_store import DomainEventType
            self._event_store.append(
                DomainEventType.ACTION_TAKEN,
                self.config.account_id,
                "match",
                {"action": action, **context},
            )

    def on_error(self, error: str, component: str = ""):
        """Chamado quando ocorre um erro."""
        if self._event_store:
            from core.event_store import DomainEventType
            self._event_store.append(
                DomainEventType.ERROR_OCCURRED,
                self.config.account_id,
                "system",
                {"error": error, "component": component},
            )
        if self._degradation_mgr:
            self._degradation_mgr.record_error(component, "error")

    # ------------------------------------------------------------------
    # Dashboard API
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Retorna dados para o dashboard."""
        data = {
            "v2_enabled": self.config.enabled,
            "cycle_count": self._cycle_count,
        }
        if self._degradation_mgr:
            data["degradation"] = self._degradation_mgr.get_status()
        if self._rate_limiter:
            data["rate_limiter"] = self._rate_limiter.get_account_status(self.config.account_id)
        if self._checkpointer:
            data["checkpointer"] = self._checkpointer.get_stats()
        if self._brawler_ctrl:
            data["brawler_adaptive"] = self._brawler_ctrl.get_status()
        if self._alert_system:
            data["alerts"] = self._alert_system.get_active_alerts()
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_wrapper_state(self) -> str:
        """Retorna estado atual do wrapper."""
        if self.wrapper and hasattr(self.wrapper, "state_manager") and self.wrapper.state_manager:
            return getattr(self.wrapper.state_manager, "current_state", "unknown")
        return "unknown"

    def _get_current_brawler(self) -> Optional[str]:
        """Retorna brawler atual do wrapper."""
        if self.wrapper and hasattr(self.wrapper, "lobby") and self.wrapper.lobby:
            return getattr(self.wrapper.lobby, "current_brawler", None)
        if self.wrapper and hasattr(self.wrapper, "brawler_selector") and self.wrapper.brawler_selector:
            return getattr(self.wrapper.brawler_selector, "current_brawler", None)
        return None

    def _maybe_checkpoint(self):
        """Salva checkpoint se possível."""
        if not self._checkpointer or not self.wrapper:
            return
        try:
            state = self._get_wrapper_state()
            brawler = self._get_current_brawler()
            spatial = None
            rl_state = None

            if self.wrapper and hasattr(self.wrapper, "play_logic") and self.wrapper.play_logic:
                snap = getattr(self.wrapper.play_logic, "last_combat_snapshot", {})
                if snap:
                    from core.game_state_checkpoint import SpatialSnapshot, RLStateSnapshot
                    spatial = SpatialSnapshot(
                        player_position=snap.get("player_pos"),
                        player_hp=snap.get("player_hp", 1.0),
                        enemy_positions=snap.get("enemies", []),
                    )
                    rl_state = RLStateSnapshot(
                        epsilon=getattr(self.wrapper, "epsilon", 0.1),
                    )

            self._checkpointer.maybe_checkpoint(
                current_state=state,
                brawler=brawler,
                spatial=spatial,
                rl_state=rl_state,
                force=False,
            )
        except Exception as e:
            logger.warning("[V2] Checkpoint falhou: %s", e)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, wrapper: Any, config_dict: Optional[Dict] = None) -> "V2Integrator":
        """Cria integrador a partir de config.json."""
        cfg = V2IntegrationConfig()
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        return cls(wrapper=wrapper, config=cfg)

    @classmethod
    def get_instance(cls) -> Optional["V2Integrator"]:
        """Retorna instância singleton."""
        return cls._instance

    def shutdown(self):
        """Desliga integrador graciosamente."""
        logger.info("[V2_INTEGRATOR] Shutdown iniciado")
        if self._event_store:
            try:
                from core.event_store import DomainEventType
                self._event_store.append(
                    DomainEventType.SESSION_ENDED,
                    self.config.account_id,
                    "session",
                    {"cycles": self._cycle_count},
                )
            except Exception:
                pass
        V2Integrator._instance = None
