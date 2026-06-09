"""
core/systems/infrastructure.py

Encapsulates all infrastructure / I/O subsystems:
- EmulatorController (ADB + window control)
- StateManager, LobbyAutomator, MatchController, ProgressObserver
- DashboardServer
- Gameplay recording
- BrawlerQueue

Interface: init(), start(), stop(), status(), health_check()
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig
from pylaai_real.progress_observer import ProgressObserver
from match_controller import MatchController

logger = logging.getLogger(__name__)


class InfrastructureSystem:
    """Cohesive infrastructure subsystem with graceful degradation."""

    def __init__(
        self,
        install_path: Path,
        images_path: Path,
        central_config: Dict[str, Any],
        diagnostic_mode: bool = False,
    ):
        self.install_path = install_path
        self.images_path = images_path
        self.central_config = central_config
        self.diagnostic_mode = diagnostic_mode

        # Components
        self.emulator_controller: Optional[Any] = None
        self.lobby: Optional[LobbyAutomator] = None
        self.match_controller: Optional[MatchController] = None
        self.progress: Optional[ProgressObserver] = None
        self.brawler_queue: Optional[BrawlerQueue] = None
        self.dashboard: Optional[Any] = None
        self.gameplay_recorder: Optional[Any] = None
        self.recording_dir: Path = Path(__file__).parent.parent.parent / "recordings"
        self.mode_controller: Optional[Any] = None
        self.state_persistence: Optional[Any] = None
        self.resolution_manager: Optional[Any] = None

        self._running = False
        self._threads: List[threading.Thread] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        screenshot_source: Optional[Any] = None,
        play_logic: Optional[Any] = None,
        unified_detector: Optional[Any] = None,
        ocr_detector: Optional[Any] = None,
        auto_fix: Optional[Any] = None,
        online_learner: Optional[Any] = None,
        learning_mode_controller: Optional[Any] = None,
        brawler_selector: Optional[Any] = None,
        data_collector: Optional[Any] = None,
        reward_bridge: Optional[Any] = None,
        observability: Optional[Any] = None,
        safety: Optional[Any] = None,
        humanization: Optional[Any] = None,
        window_w: int = 1920,
        window_h: int = 1080,
    ) -> bool:
        """Initialize infrastructure components."""
        success = True

        # EmulatorController
        try:
            from emulator_controller import EmulatorController, EmulatorConfig
            from emulator_detector import get_emulator_detector

            emu_cfg = self.central_config.get("emulator", {})
            emu_type = emu_cfg.get("type", "bluestacks").lower()
            detector = get_emulator_detector()
            emulators = detector.detect_all() if detector else []

            if emulators:
                best = emulators[0]
                cfg = EmulatorConfig(
                    name=best.type,
                    adb_port=best.adb_port or 5555,
                    window_title=best.window_title or best.name,
                )
            else:
                cfg = EmulatorConfig.for_bluestacks() if emu_type == "bluestacks" else EmulatorConfig.for_ldplayer()

            self.emulator_controller = EmulatorController(cfg, safety_system=safety, humanization_system=humanization)
            if self.emulator_controller.connect():
                logger.info("[INFRA] EmulatorController connected: %s", cfg.name)
            else:
                logger.warning("[INFRA] EmulatorController connection failed")
                self.emulator_controller = None
        except (ImportError, ModuleNotFoundError, ConnectionError, TimeoutError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("[INFRA] EmulatorController unavailable: %s", e)
            self.emulator_controller = None

        # ResolutionManager
        try:
            from core.resolution_manager import ResolutionManager
            window_title = getattr(self.emulator_controller.config, "window_title", "auto") if self.emulator_controller else "auto"
            self.resolution_manager = ResolutionManager(window_title=window_title)
            self.resolution_manager.detect()
            window_w, window_h = self.resolution_manager.actual_resolution
            logger.info("[INFRA] ResolutionManager active: %sx%s", window_w, window_h)
        except (ImportError, ModuleNotFoundError, FileNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug("[INFRA] ResolutionManager unavailable: %s", e)

        # ProgressObserver
        try:
            self.progress = ProgressObserver()
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.warning("[INFRA] ProgressObserver unavailable: %s", e)

        # BrawlerQueue
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
            logger.info("[INFRA] %s brawlers loaded from config", len(queue_config))
        else:
            self.brawler_queue.add_brawler(BrawlerConfig(name="colt"))
            logger.info("[INFRA] Default brawler 'colt' added")

        # MatchController
        try:
            self.match_controller = MatchController(self.install_path)
            self.match_controller.brawler_queue = self.brawler_queue
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.warning("[INFRA] MatchController unavailable: %s", e)

        # LobbyAutomator
        try:
            self.lobby = LobbyAutomator(
                self.brawler_queue,
                self.emulator_controller,
                diagnostic_mode=self.diagnostic_mode,
                play_logic=play_logic,
                window_w=window_w,
                window_h=window_h,
                images_path=str(self.images_path),
            )
            if unified_detector:
                self.lobby.set_state_detector(unified_detector)
            if screenshot_source and hasattr(screenshot_source, "take"):
                self.lobby.set_screenshot_func(screenshot_source.take)
        except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[INFRA] LobbyAutomator init failed: %s", e)
            success = False
        except Exception as e:
            logger.exception("[INFRA] Unexpected LobbyAutomator init error: %s", e)
            raise

        # Dashboard
        if self.central_config.get("dashboard_enabled", True):
            try:
                from pylaai_real.dashboard_server import DashboardServer
                self.dashboard = DashboardServer(port=self.central_config.get("dashboard_port", 8765))
                logger.info("[INFRA] Dashboard server initialized (port %s)", self.central_config.get("dashboard_port", 8765))
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning("[INFRA] Dashboard server unavailable: %s", e)

        # ModeController
        try:
            from core.mode_controller import ModeController
            self.mode_controller = ModeController(
                wrapper_ref=None,  # set externally if needed
                learning_mode_controller=learning_mode_controller,
                rl_engine=online_learner,
            )
            logger.info("[INFRA] ModeController initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[INFRA] ModeController unavailable: %s", e)

        # StatePersistence
        try:
            from state_persistence import StatePersistence
            self.state_persistence = StatePersistence()
            logger.info("[INFRA] StatePersistence initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[INFRA] StatePersistence unavailable: %s", e)

        return success

    def start(self) -> bool:
        self._running = True
        if self.dashboard:
            try:
                self.dashboard.start(daemon=True)
                logger.info("[INFRA] Dashboard server started")
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[INFRA] Failed to start dashboard: %s", e)
        return True

    def stop(self) -> bool:
        self._running = False
        if self.dashboard:
            try:
                self.dashboard.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug("[INFRA] Failed to stop dashboard: %s", e)
        if self.match_controller:
            try:
                self.match_controller.history.save()
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
                logger.warning("[INFRA] Failed to save match history: %s", e)
        return True

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "emulator_controller_ok": self.emulator_controller is not None,
            "lobby_ok": self.lobby is not None,
            "match_controller_ok": self.match_controller is not None,
            "dashboard_ok": self.dashboard is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        issues = []
        if self.emulator_controller is None:
            issues.append("no_emulator_controller")
        return {"healthy": len(issues) == 0, "issues": issues}
