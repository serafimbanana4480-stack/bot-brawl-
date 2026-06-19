"""
wrapper.py

Wrapper que integra PylaAI real com Safety System, Humanizacao e Error Recovery.
Este e o ponto de entrada principal para usar o PylaAI melhorado.

Fixes Applied:
- Error #10: EmulatorController now integrated (replaces raw ScreenshotTaker)
- Error #12: Model loading prefers trained models (main_info.pt, brawler_id.pt)
- Error #28: Reads from centralized config.json
- Phase 9: Error Recovery, State Recovery, AutoCalibrator, OCR, Debug Visualizer
- Refactor: God Class extracted into core/subsystems/ (backward-compatible facade)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from core.plugin_system import PluginManager
from core.subsystems import (
    DecisionSubsystem,
    EmulatorSubsystem,
    LearningSubsystem,
    SafetySubsystem,
    UISubsystem,
    VisionSubsystem,
)
from humanization import HumanizationConfig  # noqa: F401
from pylaai_real.lobby_automator import BrawlerConfig, BrawlerQueue

# Preserve lazy imports for backward compatibility
from safety_system import SafetyConfig  # noqa: F401

# Initialize plugin manager and discover optional components
_plugin_manager = PluginManager()
try:
    import plugins.learning_mode_plugin  # noqa: F401
    import plugins.orchestrator_plugin  # noqa: F401
    _plugin_manager.discover()
    _plugin_manager.load_all()
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"[WRAPPER] Plugin discovery failed: {e}")

HAS_ORCHESTRATOR = _plugin_manager.is_available("orchestrator")
HAS_LEARNING_MODE = _plugin_manager.is_available("learning_mode")
create_orchestrator = _plugin_manager.get("orchestrator") if HAS_ORCHESTRATOR else None
LearningModeController = _plugin_manager.get("learning_mode") if HAS_LEARNING_MODE else None

logger = logging.getLogger(__name__)

# --- SOVERANA FIX 2026-06-19: optional omega_qa auto-improver ---
_omega_qa_available = False
_omega_qa_integration = None
try:
    from omega_qa.integration import QAIntegration
    _omega_qa_available = True
except Exception as e:
    logger.debug(f"[WRAPPER] omega_qa unavailable: {e}")
# --- end fix ---

# --- SOVERANA FIX 2026-06-19: optional adversarial humanizer ---
_adversarial_humanizer = None
try:
    from core.adversarial_humanization import (
        AdversarialHumanizationConfig,
        AdversarialHumanizer,
    )
    _adversarial_humanizer_class = AdversarialHumanizer
except Exception as e:
    logger.debug(f"[WRAPPER] AdversarialHumanizer unavailable: {e}")
    _adversarial_humanizer_class = None
# --- end fix ---
_omega_qa_available = False
_omega_qa_integration = None
try:
    from omega_qa.integration import QAIntegration
    _omega_qa_available = True
except Exception as e:
    logger.debug(f"[WRAPPER] omega_qa unavailable: {e}")
# --- end fix ---

_DEFAULT_INSTALL_PATH = Path(os.getenv("PYLAAI_INSTALL_PATH", str(Path(__file__).parent / "pylaai_workspace")))
_BOT_ROOT = Path(__file__).parent


def _load_central_config() -> dict:
    config_path = _BOT_ROOT / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}


class PylaAIEnhanced:
    """
    PylaAI com melhorias — agora um facade delegando para subsystems.
    Todos os atributos e métodos públicos são preservados para compatibilidade.
    """

    HEARTBEAT_TIMEOUT = 30.0
    MAX_UNKNOWN_STATE_DURATION = 60.0

    def __init__(
        self,
        install_path: Path = _DEFAULT_INSTALL_PATH,
        safety_config: Any | None = None,
        humanization_config: Any | None = None,
        diagnostic_mode: bool | None = None,
        enable_recording: bool = False,
        enable_error_recovery: bool = True,
        learning_mode: bool = False,
    ):
        logger.info("[WRAPPER] Inicializando PylaAIEnhanced")
        self.install_path = install_path
        self.images_path = _BOT_ROOT / "images"
        self.models_path = _BOT_ROOT / "models"
        self.central_config = _load_central_config()
        self.diagnostic_mode = diagnostic_mode if diagnostic_mode is not None else bool(
            self.central_config.get("diagnostic_mode", False)
            or os.getenv("PYLAAI_DIAGNOSTIC", "0") == "1"
        )
        self.learning_mode = learning_mode
        logger.info(f"[WRAPPER] Diagnostic mode: {self.diagnostic_mode}")
        logger.info(f"[WRAPPER] Learning mode: {self.learning_mode}")

        self.recording_enabled = enable_recording or bool(
            self.central_config.get("enable_recording", False)
        )
        self.auto_retrain_enabled = bool(
            self.central_config.get("auto_retrain_enabled", False)
        )
        logger.info(f"[WRAPPER] Gameplay recording: {'enabled' if self.recording_enabled else 'disabled'}")
        logger.info(f"[WRAPPER] Auto-retraining: {'enabled' if self.auto_retrain_enabled else 'disabled'}")

        # Brawler queue (shared across subsystems)
        self.brawler_queue = BrawlerQueue()
        queue_config = self.central_config.get("brawler_queue", [])
        if queue_config:
            for bcfg in queue_config:
                self.brawler_queue.add_brawler(BrawlerConfig(
                    name=bcfg.get("name", "colt"),
                    current_trophies=bcfg.get("current_trophies", 0),
                    target_trophies=bcfg.get("target_trophies", 500),
                    target_wins=bcfg.get("target_wins", 10),
                    priority=bcfg.get("priority", 1),
                    enabled=bcfg.get("enabled", True),
                    game_mode=bcfg.get("game_mode", None),
                ))
            logger.info(
                f"[WRAPPER] {len(queue_config)} brawlers carregados do config.json: "
                f"{[b.get('name') for b in queue_config]}"
            )
        else:
            self.brawler_queue.add_brawler(BrawlerConfig(
                name="colt", current_trophies=0, target_trophies=400,
                target_wins=10, priority=1, enabled=True, game_mode=None,
            ))
            logger.info("[WRAPPER] Brawler 'colt' adicionado à fila (padrão)")

        # Subsystems (in dependency order)
        self.safety_subsystem = SafetySubsystem(
            self, safety_config, humanization_config, enable_error_recovery
        )
        self.emulator_subsystem = EmulatorSubsystem(
            self, self.central_config, self.images_path, self.models_path
        )
        self.vision_subsystem = VisionSubsystem(
            self, self.models_path, self.images_path, self.diagnostic_mode, self.central_config
        )
        self.decision_subsystem = DecisionSubsystem(
            self, self.central_config, self.install_path, self.images_path, self.models_path, self.brawler_queue
        )
        self.learning_subsystem = LearningSubsystem(self, self.central_config)
        self.ui_subsystem = UISubsystem(self, self.central_config, self.diagnostic_mode)

        # Phase v2.1: Strategic Integrator
        self.v2_integrator: Any | None = None
        try:
            from core.v2_integration import V2IntegrationConfig, V2Integrator

            v2_config = V2IntegrationConfig()
            v2_config.account_id = self.central_config.get("account_id", "default_account")
            v2_config.enable_multi_objective_rl = self.central_config.get("enable_moo", False)
            self.v2_integrator = V2Integrator.from_config(self, v2_config.__dict__)
            logger.info("[WRAPPER] V2 Strategic Integrator inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] V2 Integrator indisponível: {e}")

        # Phase 10: State Persistence
        self.state_persistence: Any | None = None
        try:
            from state_persistence import StatePersistence

            self.state_persistence = StatePersistence()
            logger.info("[WRAPPER] State Persistence inicializado")
        except Exception as e:
            logger.warning(f"[WRAPPER] State Persistence indisponível: {e}")

        # Orchestrator (opt-in)
        self.orchestrator: Any | None = None
        self.orchestrator_thread: threading.Thread | None = None
        self.use_orchestrator = bool(self.central_config.get("use_orchestrator", False))
        if self.use_orchestrator and HAS_ORCHESTRATOR:
            try:
                logger.info("[WRAPPER] Creating BotOrchestrator via factory...")
                self.orchestrator = create_orchestrator(
                    install_path=self.install_path,
                    config=self.central_config,
                    safety_config=safety_config,
                    humanization_config=humanization_config,
                )
                logger.info("[WRAPPER] BotOrchestrator created (will be initialized on start)")
            except Exception as e:
                logger.error(f"[WRAPPER] Failed to create BotOrchestrator: {e}")
                self.orchestrator = None
                self.use_orchestrator = False
        elif self.use_orchestrator:
            logger.warning("[WRAPPER] use_orchestrator=True but orchestrator not available")
            self.use_orchestrator = False

        # Threading / state
        self._running_lock = threading.Lock()
        self._running = False
        self.stop_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.state_thread: threading.Thread | None = None
        self.session_start: float | None = None
        self.matches_played = 0
        self._paused = False

        # --- SOVERANA FIX 2026-06-19: attach omega_qa auto-improver ---
        self.qa_integration: Any | None = None
        self.qa_enabled: bool = bool(self.central_config.get("omega_qa_enabled", True))
        if self.qa_enabled and _omega_qa_available:
            try:
                self.qa_integration = QAIntegration(
                    wrapper_instance=self,
                    config={"cycle_interval_seconds": 30.0, "min_global_score": 7.0},
                )
                logger.info("[WRAPPER] omega_qa auto-improver attached (8 agents)")
            except Exception as e:
                logger.warning(f"[WRAPPER] omega_qa attach failed: {e}")
                self.qa_integration = None
        # --- end fix ---

    @property
    def running(self) -> bool:
        with self._running_lock:
            return self._running

    @running.setter
    def running(self, value: bool):
        with self._running_lock:
            self._running = value

    # ------------------------------------------------------------------
    # Setup / Start / Stop
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        try:
            logger.info("Iniciando setup do PylaAI Enhanced...")
            logger.info(f"Diagnostic mode: {'enabled' if self.diagnostic_mode else 'disabled'}")

            self.install_path.mkdir(parents=True, exist_ok=True)
            self.images_path.mkdir(parents=True, exist_ok=True)
            self.models_path.mkdir(parents=True, exist_ok=True)
            (_BOT_ROOT / "data").mkdir(parents=True, exist_ok=True)
            (_BOT_ROOT / "logs").mkdir(parents=True, exist_ok=True)

            self.safety_subsystem.setup()
            if not self.emulator_subsystem.setup():
                return False
            self.safety_subsystem.post_emulator_setup()
            if not self.vision_subsystem.setup():
                return False
            if not self.decision_subsystem.setup():
                return False
            self.learning_subsystem.setup()
            self.ui_subsystem.setup()

            logger.info("Setup completo!")
            return True
        except Exception as e:
            logger.error(f"Erro no setup: {e}", exc_info=True)
            return False

    def start(self) -> bool:
        logger.info("[WRAPPER] Chamando start()")
        if self.running:
            logger.warning("[WRAPPER] Bot já está em execução!")
            return False

        self._install_signal_handlers()
        logger.info("[WRAPPER] Iniciando bot PylaAI Enhanced...")

        if not self.learning_mode and self.anti_ban and not self.anti_ban.should_start_match():
            logger.warning("[WRAPPER] Anti-ban: início bloqueado por schedule/padrão")
            return False

        if not self.learning_mode:
            trophy_status = self.safety.check_trophy_limit(self._get_current_trophies())
            if not trophy_status["can_play"]:
                logger.error(f"[WRAPPER] Safety block: {trophy_status['message']}")
                return False
            if trophy_status.get("warning"):
                logger.warning(f"[WRAPPER] Safety warning: {trophy_status['message']}")

        self.safety.start_session()
        self.session_start = time.time()

        # --- SOVERANA FIX 2026-06-19: start omega_qa monitoring ---
        if self.qa_integration:
            try:
                self.qa_integration.start_monitoring()
            except Exception as e:
                logger.warning(f"[WRAPPER] omega_qa start failed: {e}")
        # --- end fix ---

        if self.state_manager and self.brawler_queue:
            current = self.brawler_queue.get_current()
            if current:
                self.state_manager.current_brawler = current.name

        self.running = True
        self.stop_event.clear()

        if self.use_orchestrator and self.orchestrator:
            logger.info("[WRAPPER] Starting in ORCHESTRATOR mode")
            try:
                self.orchestrator.initialize()
                # Wire non-blocking vision pipeline: orchestrator reads latest snapshot from VisionSubsystem thread
                if hasattr(self.orchestrator.vision, 'set_snapshot_source'):
                    self.orchestrator.vision.set_snapshot_source(self.vision_subsystem.get_latest_snapshot)
                    logger.info("[WRAPPER] VisionAdapter wired to VisionSubsystem (non-blocking)")
                logger.info("[WRAPPER] BotOrchestrator initialized")
            except Exception as e:
                logger.error(f"[WRAPPER] BotOrchestrator initialization failed: {e}")
                self.use_orchestrator = False

        self.ui_subsystem.start()
        self.learning_subsystem.start()
        self.decision_subsystem.start()
        self.vision_subsystem.start()
        self.emulator_subsystem.start()
        self.safety_subsystem.start()

        if self.use_orchestrator and self.orchestrator:
            self.orchestrator_thread = threading.Thread(
                target=self._orchestrator_loop, daemon=True, name="orchestrator"
            )
            self.orchestrator_thread.start()
            logger.info("[WRAPPER] Orchestrator thread started")
        else:
            logger.info("[WRAPPER] Starting in LEGACY mode")
            if self.state_manager:
                self.state_thread = threading.Thread(
                    target=self.state_manager.run, daemon=True, name="state-manager"
                )
                self.state_thread.start()
            else:
                logger.warning("[WRAPPER] state_manager not available")

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="safety-monitor"
        )
        self.monitor_thread.start()

        logger.info("[WRAPPER] PylaAI Enhanced iniciado com sucesso!")
        return True

    def stop(self):
        logger.info("[WRAPPER] Chamando stop()")
        if not self.running:
            logger.debug("[WRAPPER] Bot não está em execução")
            return False

        self.running = False
        self.stop_event.set()

        for hook in getattr(self, "_shutdown_hooks", []):
            try:
                hook()
            except Exception as e:
                logger.warning(f"[WRAPPER] Shutdown hook failed: {e}")

        # --- SOVERANA FIX 2026-06-19: stop omega_qa monitoring ---
        if getattr(self, "qa_integration", None):
            try:
                self.qa_integration.stop_monitoring()
            except Exception as e:
                logger.warning(f"[WRAPPER] omega_qa stop failed: {e}")
        # --- end fix ---

        self.ui_subsystem.cleanup()
        self.learning_subsystem.cleanup()
        self.decision_subsystem.cleanup()
        self.vision_subsystem.cleanup()
        self.emulator_subsystem.cleanup()
        self.safety_subsystem.cleanup()

        if self.learning_mode and getattr(self, "learning_mode_controller", None):
            try:
                self.learning_mode_controller.print_summary()
            except Exception as e:
                logger.warning(f"[WRAPPER] Falha ao imprimir sumário de learning mode: {e}")

        for thr, name in [
            (self.orchestrator_thread, "orchestrator"),
            (self.state_thread, "state-manager"),
            (self.monitor_thread, "safety-monitor"),
        ]:
            if thr and thr.is_alive():
                thr.join(timeout=5)
                if thr.is_alive():
                    logger.warning(f"[WRAPPER] Thread {name} não terminou em 5s")
                else:
                    logger.debug(f"[WRAPPER] Thread {name} terminou")

        logger.info("[WRAPPER] PylaAI Enhanced parado com sucesso!")
        return True

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _install_signal_handlers(self):
        try:
            def _shutdown_handler(signum, frame):
                sig_name = signal.Signals(signum).name
                logger.info(f"[WRAPPER] Received {sig_name}, initiating graceful shutdown...")
                self.stop()

            signal.signal(signal.SIGINT, _shutdown_handler)
            signal.signal(signal.SIGTERM, _shutdown_handler)
            logger.debug("[WRAPPER] Signal handlers installed (SIGINT, SIGTERM)")
        except (OSError, ValueError) as e:
            logger.debug(f"[WRAPPER] Cannot install signal handlers: {e}")

    def _monitor_loop(self):
        # TODO: full extraction of monitor_loop into SafetySubsystem is partial;
        # some cross-subsystem orchestration remains here for backward compatibility.
        self.safety_subsystem.run_monitor_loop(self, self.stop_event)

    def _orchestrator_loop(self):
        logger.info("[WRAPPER] Orchestrator loop started")
        try:
            self.orchestrator.run()
        except Exception as e:
            logger.error(f"[WRAPPER] Orchestrator loop crashed: {e}")
        finally:
            logger.info("[WRAPPER] Orchestrator loop ended")
            if self.running:
                logger.warning("[WRAPPER] Orchestrator stopped but wrapper still running — shutting down")
                self.stop()

    # ------------------------------------------------------------------
    # Health / hooks
    # ------------------------------------------------------------------

    def record_heartbeat(self):
        self.safety_subsystem.record_heartbeat()

    def check_health(self) -> dict:
        return self.safety_subsystem.check_health()

    def register_shutdown_hook(self, hook: callable):
        if not hasattr(self, "_shutdown_hooks"):
            self._shutdown_hooks = []
        self._shutdown_hooks.append(hook)

    # ------------------------------------------------------------------
    # Public control methods (backward compatible)
    # ------------------------------------------------------------------

    def pause(self) -> bool:
        if not self.running:
            logger.warning("[WRAPPER] Bot nao esta em execucao, nada para pausar")
            return False
        if self._paused:
            return True
        self._paused = True
        if self.state_manager:
            self.state_manager.pause()
        logger.info("[WRAPPER] Bot PAUSADO — threads mantidas vivas")
        return True

    def resume(self) -> bool:
        if not self._paused:
            return True
        self._paused = False
        if self.state_manager:
            self.state_manager.resume()
        logger.info("[WRAPPER] Bot RETOMADO")
        return True

    def update_config(self, key: str, value) -> bool:
        try:
            self.central_config[key] = value
            config_path = _BOT_ROOT / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    full_config = json.load(f)
                full_config[key] = value
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(full_config, f, indent=2, ensure_ascii=False)
            logger.info(f"[WRAPPER] Config atualizado: {key}={value}")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao atualizar config {key}: {e}")
            return False

    def set_brawler(self, name: str) -> bool:
        if not self.brawler_queue:
            return False
        try:
            for i, b in enumerate(self.brawler_queue.brawlers):
                if b.name.lower() == name.lower():
                    if not getattr(b, "enabled", True):
                        logger.warning(f"[WRAPPER] Brawler bloqueado/desativado recusado: {b.name}")
                        return False
                    self.brawler_queue.current_index = i
                    if self.state_manager:
                        self.state_manager.current_brawler = b.name
                    logger.info(f"[WRAPPER] Brawler forçado: {b.name}")
                    return True
            logger.warning(f"[WRAPPER] Brawler nao encontrado na fila: {name}")
            return False
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao set brawler {name}: {e}")
            return False

    def get_available_brawler_names(self) -> list[str]:
        if not self.brawler_queue:
            return []
        return [b.name for b in self.brawler_queue.brawlers if getattr(b, "enabled", True)]

    def update_queue(self, queue_data: list[dict]) -> bool:
        if not self.brawler_queue:
            return False
        try:
            self.brawler_queue.brawlers = []
            self.brawler_queue.current_index = 0
            for item in queue_data:
                cfg = BrawlerConfig(
                    name=item.get("name", "colt"),
                    current_trophies=item.get("current_trophies", 0),
                    target_trophies=item.get("target_trophies", 350),
                    target_wins=item.get("target_wins", 10),
                    priority=item.get("priority", 1),
                    enabled=item.get("enabled", True),
                    game_mode=item.get("game_mode", None),
                )
                self.brawler_queue.add_brawler(cfg)
            logger.info(f"[WRAPPER] Fila de brawlers atualizada: {len(queue_data)} brawlers")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao atualizar fila: {e}")
            return False

    def toggle_system(self, system_name: str, enabled: bool) -> bool:
        try:
            if system_name == "rl_engine":
                if self.online_learner:
                    self.online_learner.enabled = enabled
                else:
                    return False
            elif system_name == "humanization":
                if self.humanization:
                    self.humanization.enabled = enabled
                else:
                    return False
            elif system_name == "anti_ban":
                if self.anti_ban:
                    self.anti_ban.enabled = enabled
                else:
                    return False
            elif system_name == "error_recovery":
                if self.error_recovery:
                    self.error_recovery.enabled = enabled
                else:
                    return False
            elif system_name == "recording":
                if enabled and not self.recording_enabled:
                    self.recording_enabled = True
                    self.start_recording()
                elif not enabled and self.recording_enabled:
                    self.recording_enabled = False
                    self.stop_recording()
            elif system_name == "auto_tuner":
                self.auto_tuning_enabled = enabled
            elif system_name == "learning_mode":
                self.toggle_learning_mode(enabled)
            else:
                logger.warning(f"[WRAPPER] Sistema desconhecido: {system_name}")
                return False
            logger.info(f"[WRAPPER] {system_name} {'ativado' if enabled else 'desativado'}")
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao toggle {system_name}: {e}")
            return False

    def toggle_learning_mode(self, enabled: bool = True, max_matches: int = 5) -> bool:
        try:
            if enabled:
                if self.learning_mode_controller is None and HAS_LEARNING_MODE:
                    screenshot_source = self.emulator_controller or self.screenshot
                    self.learning_mode_controller = LearningModeController(
                        lobby_automator=self.lobby,
                        emulator_controller=self.emulator_controller,
                        screenshot_taker=screenshot_source,
                        state_finder=self.state_finder,
                        play_logic=self.play_logic,
                        max_matches=max_matches,
                    )
                if self.learning_mode_controller:
                    self.learning_mode_controller.start_learning_mode(max_matches=max_matches)
                    self.learning_mode = True
                    if self.state_manager:
                        self.state_manager.learning_mode_controller = self.learning_mode_controller
                    logger.info("[WRAPPER] Modo de aprendizagem ATIVADO via dashboard")
                    return True
            else:
                if self.learning_mode_controller:
                    self.learning_mode_controller.stop_learning_mode()
                self.learning_mode = False
                if self.state_manager:
                    self.state_manager.learning_mode_controller = None
                logger.info("[WRAPPER] Modo de aprendizagem DESATIVADO via dashboard")
                return True
            return False
        except Exception as e:
            logger.error(f"[WRAPPER] Falha ao toggle learning mode: {e}")
            return False

    def execute_action(self, action_name: str, **kwargs) -> bool:
        try:
            if action_name == "force_goto_lobby":
                if self.state_manager:
                    self.state_manager.current_state = "lobby"
                return True
            elif action_name == "force_click_play":
                if self.lobby and hasattr(self.lobby, "_click"):
                    w, h = self._get_safe_resolution()
                    self.lobby._click(round(w * 0.9119), round(h * 0.9122))
                return True
            elif action_name == "force_attack":
                if self.play_logic and hasattr(self.play_logic, "_execute_attack"):
                    w, h = self._get_safe_resolution()
                    self.play_logic._execute_attack((round(w * 0.6), round(h * 0.5)))
                return True
            elif action_name == "force_super":
                if self.emulator_controller:
                    w, h = self._get_safe_resolution()
                    self.emulator_controller.tap_scaled(round(w / 2), round(h / 2))
                return True
            elif action_name == "force_collect_cube":
                if self.play_logic and hasattr(self.play_logic, "_collect_power_cubes"):
                    self.play_logic._collect_power_cubes([])
                return True
            elif action_name == "screenshot":
                if self.screenshot:
                    img = self.screenshot.take()
                    logger.info(f"[WRAPPER] screenshot capturado {img.shape if img is not None else 'None'}")
                return True
            elif action_name == "back_press":
                if self.emulator_controller:
                    self.emulator_controller.keyevent(4)
                return True
            else:
                logger.warning(f"[WRAPPER] Acao desconhecida: {action_name}")
                return False
        except Exception as e:
            logger.error(f"[WRAPPER] Falha na acao {action_name}: {e}")
            return False

    def toggle_esp(self, enabled: bool | None = None) -> bool:
        if not self.esp_overlay:
            return False
        return self.esp_overlay.toggle(enabled)

    def start_mode(self, mode: str, config: dict | None = None) -> bool:
        if not self.mode_controller:
            return False
        return self.mode_controller.start_mode(mode, config)

    def stop_mode(self, mode: str | None = None) -> bool:
        if not self.mode_controller:
            return False
        return self.mode_controller.stop_mode(mode)

    def get_mode_status(self) -> dict:
        if not self.mode_controller:
            return {"available": False}
        return self.mode_controller.get_status()

    def get_detection_snapshot(self) -> dict:
        return self.ui_subsystem.get_detection_snapshot()

    def get_rl_metrics(self) -> dict:
        return self.ui_subsystem.get_rl_metrics()

    def get_system_status(self) -> dict:
        return self.ui_subsystem.get_system_status()

    def start_recording(self) -> bool:
        if not self.gameplay_recorder:
            return False
        try:
            self.gameplay_recorder.start()
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to start recording: {e}")
            return False

    def stop_recording(self) -> bool:
        if not self.gameplay_recorder:
            return False
        try:
            self.gameplay_recorder.stop()
            return True
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to stop recording: {e}")
            return False

    def record_performance_metric(self, metric_type: str, **kwargs):
        self.learning_subsystem.record_performance_metric(self, metric_type, **kwargs)

    def check_retrain_trigger(self) -> tuple[bool, str]:
        return self.learning_subsystem.check_retrain_trigger(self)

    def trigger_retrain(self) -> bool:
        return self.learning_subsystem.trigger_retrain(self)

    def add_brawler_to_queue(
        self, name: str, current_trophies: int = 0, target_trophies: int = 350,
        target_wins: int = 10, priority: int = 1, game_mode: str | None = None,
    ):
        cfg = BrawlerConfig(
            name=name, current_trophies=current_trophies,
            target_trophies=target_trophies, target_wins=target_wins,
            priority=priority, game_mode=game_mode,
        )
        self.brawler_queue.add_brawler(cfg)
        logger.info(f"Brawler adicionado: {name}")

    def get_queue(self) -> list[dict]:
        return self.brawler_queue.get_queue()

    def get_status(self) -> dict:
        if hasattr(self, "ui_subsystem") and self.ui_subsystem is not None:
            return self.ui_subsystem.get_status(self)
        # Legacy fallback for tests that construct PylaAIEnhanced via __new__
        import time

        def _safe_get(obj_name: str, attr_name: str, default: Any) -> Any:
            """Safely get attribute from optional subsystem."""
            obj = getattr(self, obj_name, None)
            if obj is None:
                return default
            return getattr(obj, attr_name, default)

        def _safe_call(obj_name: str, method_name: str, default: Any) -> Any:
            """Safely call method on optional subsystem."""
            obj = getattr(self, obj_name, None)
            if obj is None:
                return default
            method = getattr(obj, method_name, None)
            if method is None:
                return default
            try:
                return method()
            except Exception:
                return default

        return {
            "running": getattr(self, "running", False),
            "current_state": _safe_get("state_manager", "current_state", "unknown"),
            "current_brawler": (
                self.brawler_queue.get_current().name if self.brawler_queue.get_current() else None
            ),
            "matches_played": getattr(self, "matches_played", 0),
            "session_duration_minutes": (time.time() - getattr(self, "session_start", time.time())) / 60.0,
            "window_active": _safe_call("emulator_controller", "get_status_snapshot", {}).get("window_active", False),
            "window_title": _safe_call("emulator_controller", "get_status_snapshot", {}).get("window_title", ""),
            "diagnostics": {
                "diagnostic_mode": getattr(self, "diagnostic_mode", False),
                "lobby": _safe_call("lobby", "get_diagnostic_report", {}),
                "screen_state": (
                    _safe_get("state_manager", "screen_automation", None).get_current_state_name()
                    if _safe_get("state_manager", "screen_automation", None) is not None
                    else "unknown"
                ),
                "progress": _safe_call("progress", "get_stats", {}),
                "combat": _safe_call("play_logic", "get_last_combat_snapshot", {}),
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_safe_resolution(self):
        if hasattr(self, "resolution_manager") and self.resolution_manager is not None:
            try:
                return self.resolution_manager.actual_resolution
            except Exception:
                pass
        if self.play_logic and self.play_logic.movement:
            return (self.play_logic.movement.window_w, self.play_logic.movement.window_h)
        return (1920, 1080)

    def _get_current_trophies(self) -> int:
        current = self.brawler_queue.get_current()
        if current:
            return current.current_trophies
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Soberana Omega Brawl Stars Bot")
    parser.add_argument("--setup-only", action="store_true", help="Run setup and exit")
    parser.add_argument("--learning-mode", action="store_true", help="Modo teste aprendizagem")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    bot = PylaAIEnhanced(learning_mode=args.learning_mode)
    if not bot.setup():
        logger.error("Setup failed; see logs above.")
        sys.exit(1)
    if args.setup_only:
        logger.info("Setup complete (--setup-only).")
        return
    if not bot.start():
        logger.error("Start failed; see logs above.")
        sys.exit(1)
    try:
        while bot.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Interrupt received, stopping...")
    finally:
        bot.stop()
