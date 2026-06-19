"""
core/subsystems/decision_subsystem.py

DecisionSubsystem: play logic, movement, brawler selection, lobby automator,
state manager, and advanced decision modules (world model, pressure map, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)


class DecisionSubsystem:
    """Manages gameplay decisions, state management, and AI modules."""

    def __init__(
        self,
        wrapper: PylaAIEnhanced,
        central_config: dict,
        install_path: Path,
        images_path: Path,
        models_path: Path,
        brawler_queue: Any,
    ):
        self.wrapper = wrapper
        self.central_config = central_config
        self.install_path = install_path
        self.images_path = images_path
        self.models_path = models_path
        self.brawler_queue = brawler_queue
        self.state_finder: Any | None = None
        self.state_manager: Any | None = None
        self.lobby: Any | None = None
        self.progress: Any | None = None
        self.play_logic: Any | None = None
        self.match_controller: Any | None = None
        self.movement: Any | None = None
        self.brawler_selector: Any | None = None
        self.auto_tuner: Any | None = None
        self.central_coordinator: Any | None = None
        self.world_model: Any | None = None
        self.pressure_map: Any | None = None
        self.behavioral_profile: Any | None = None
        self.cover_system: Any | None = None
        self.utility_ai: Any | None = None
        self.sticky_target: Any | None = None
        self.intent_system: Any | None = None
        self.enemy_intention: Any | None = None
        self.meta_awareness: Any | None = None
        self.lobby_fsm: Any | None = None
        self.auto_fix: Any | None = None
        self.screen_auto: Any | None = None
        self.online_learner: Any | None = None
        self._auto_tuning_enabled = bool(
            central_config.get("auto_tuning_enabled", False)
            or __import__("os").getenv("PYLAAI_AUTO_TUNING", "0") == "1"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Initialize all decision and state management components."""
        from decision.brawler_selector import BrawlerSelector
        from match_controller import MatchController
        from pylaai_real.lobby_automator import LobbyAutomator
        from pylaai_real.movement import Movement
        from pylaai_real.progress_observer import ProgressObserver
        from pylaai_real.state_finder import StateFinder

        self.state_finder = StateFinder(self.images_path)
        self.progress = ProgressObserver()

        window_w, window_h = self._get_resolution()

        # Brawler selector
        if bool(self.central_config.get("brawler_selection_enabled", True)):
            self.brawler_selector = BrawlerSelector()
            logger.info("[WRAPPER] Brawler selector initialized")

        # Auto-tuner
        self.match_controller = MatchController(self.install_path)
        self.match_controller.brawler_queue = self.brawler_queue
        if self._auto_tuning_enabled:
            from auto_tuner import AutoTuner

            self.auto_tuner = AutoTuner(self.match_controller)
            logger.info("[WRAPPER] Auto-tuner inicializado")

        # Advanced core modules
        self._init_advanced_modules()

        # Movement
        emulator_controller = getattr(self.wrapper, "emulator_controller", None)
        self.movement = Movement(
            emulator_controller=emulator_controller,
            window_w=window_w,
            window_h=window_h,
        )

        # Online learner
        try:
            from dataset.collector import GameplayCollector
            from pylaai_real.rl_engine import OnlineLearner

            reward_bridge = getattr(self.wrapper, "reward_bridge", None)
            data_collector = getattr(self.wrapper, "data_collector", None)
            # Usa GameplayCollector se data_collection_mode estiver ativo
            gameplay_collector = None
            try:
                import json
                from pathlib import Path
                config_path = Path("config.json")
                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    if cfg.get("rl", {}).get("data_collection_mode", False):
                        gameplay_collector = data_collector if isinstance(data_collector, GameplayCollector) else GameplayCollector()
            except Exception:
                pass
            self.online_learner = OnlineLearner(
                reward_bridge=reward_bridge,
                gameplay_collector=gameplay_collector,
                enabled=True,
            )
            logger.info("[WRAPPER] OnlineLearner (RL + ELO + DataCollection) inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] OnlineLearner nao disponivel: {e}")
            self.online_learner = None

        # PlayLogic
        detect_main = getattr(self.wrapper, "detect_main", None)
        detect_enemies = getattr(self.wrapper, "detect_enemies", None)
        self.play_logic = self._create_play_logic(
            detect_main, detect_enemies, self.movement, emulator_controller
        )

        # LobbyAutomator
        self.lobby = LobbyAutomator(
            self.brawler_queue,
            emulator_controller,
            diagnostic_mode=getattr(self.wrapper, "diagnostic_mode", False),
            play_logic=self.play_logic,
            window_w=window_w,
            window_h=window_h,
            images_path=str(self.images_path),
        )

        # AutoFixEngine
        try:
            from core.auto_fix_engine import AutoFixEngine

            screenshot_source = getattr(self.wrapper, "emulator_subsystem", None)
            screenshot_func = None
            if screenshot_source:
                screenshot_func = screenshot_source.get_screenshot_source()
                if screenshot_func and hasattr(screenshot_func, "take"):
                    screenshot_func = screenshot_func.take
            self.auto_fix = AutoFixEngine(
                screenshot_func=screenshot_func,
                click_func=emulator_controller.tap_scaled if emulator_controller else None,
                key_func=lambda k: emulator_controller.keyevent(
                    {"esc": 4, "enter": 66, "home": 3}.get(k.lower(), 4)
                )
                if emulator_controller
                else None,
                state_detector=getattr(self.wrapper, "unified_detector", None),
                emulator_controller=emulator_controller,
            )
            logger.info("[WRAPPER] AutoFixEngine inicializado")
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning(f"[WRAPPER] AutoFixEngine indisponível: {e}")

        # ScreenAutomation (for hints only, NOT started as thread)
        self.screen_auto = self._create_screen_auto()

        # StateManager
        screenshot_source = getattr(self.wrapper, "emulator_subsystem", None)
        screenshot_taker = screenshot_source.get_screenshot_source() if screenshot_source else None
        data_collector = getattr(self.wrapper, "data_collector", None)
        observability = getattr(self.wrapper, "observability", None)
        unified_detector = getattr(self.wrapper, "unified_detector", None)
        ocr_detector = getattr(self.wrapper, "ocr_detector", None)
        learning_mode_controller = getattr(self.wrapper, "learning_mode_controller", None)

        from pylaai_real.state_manager import StateManager

        self.state_manager = StateManager(
            screenshot_taker=screenshot_taker,
            state_finder=self.state_finder,
            lobby_automator=self.lobby,
            progress_observer=self.progress,
            play_logic=self.play_logic,
            match_controller=self.match_controller,
            emulator_controller=emulator_controller,
            screen_automation=self.screen_auto,
            movement=self.movement,
            diagnostic_mode=getattr(self.wrapper, "diagnostic_mode", False),
            reward_bridge=getattr(self.wrapper, "reward_bridge", None),
            data_collector=data_collector,
            brawler_selector=self.brawler_selector,
            observability=observability,
            unified_state_detector=unified_detector,
            ocr_detector=ocr_detector,
            rl_engine=self.online_learner,
            learning_mode_controller=learning_mode_controller,
            auto_fix_engine=self.auto_fix,
        )

        # Dashboard bridge
        dashboard = getattr(self.wrapper, "dashboard", None)
        if dashboard:
            self.state_manager._dashboard_bridge = dashboard.bridge
            dashboard.set_wrapper(self.wrapper)

        # Lobby connections
        if self.lobby:
            self.lobby.set_state_detector(unified_detector)
            self.lobby.set_screenshot_func(screenshot_taker.take if screenshot_taker else None)
            self.lobby.set_screen_automation(self.screen_auto)
            self.lobby.set_diagnostic_mode(getattr(self.wrapper, "diagnostic_mode", False))

        if self.screen_auto:
            logger.info("ScreenAutomation created for hints only (NOT started as thread)")
        else:
            logger.info("ScreenAutomation not available, using UnifiedStateDetector only")

        # Sync back
        self.wrapper.state_finder = self.state_finder
        self.wrapper.state_manager = self.state_manager
        self.wrapper.lobby = self.lobby
        self.wrapper.progress = self.progress
        self.wrapper.play_logic = self.play_logic
        self.wrapper.match_controller = self.match_controller
        self.wrapper.movement = self.movement
        self.wrapper.brawler_selector = self.brawler_selector
        self.wrapper.auto_tuner = self.auto_tuner
        self.wrapper.central_coordinator = self.central_coordinator
        self.wrapper.world_model = self.world_model
        self.wrapper.pressure_map = self.pressure_map
        self.wrapper.behavioral_profile = self.behavioral_profile
        self.wrapper.cover_system = self.cover_system
        self.wrapper.utility_ai = self.utility_ai
        self.wrapper.sticky_target = self.sticky_target
        self.wrapper.intent_system = self.intent_system
        self.wrapper.enemy_intention = self.enemy_intention
        self.wrapper.meta_awareness = self.meta_awareness
        self.wrapper.online_learner = self.online_learner
        self.wrapper.auto_fix = self.auto_fix
        return True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if self.state_manager:
            try:
                self.state_manager.stop()
                logger.debug("[CLEANUP] Parando state_manager")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.warning(f"[CLEANUP] Falha ao parar state_manager: {e}")
        if self.state_manager and getattr(self.state_manager, "screen_automation", None):
            try:
                self.state_manager.screen_automation.stop()
                logger.debug("[CLEANUP] Parando screen_automation")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar screen_automation: {e}")
        if self.play_logic:
            try:
                if hasattr(self.play_logic, "stop"):
                    self.play_logic.stop()
                logger.debug("[CLEANUP] Parando play_logic")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar play_logic: {e}")
        if self.behavioral_profile:
            try:
                self.behavioral_profile.save()
                logger.info("[CLEANUP] Behavioral profile saved")
            except (FileNotFoundError, PermissionError, OSError, RuntimeError, AttributeError) as e:
                logger.warning(f"[CLEANUP] Falha ao salvar behavioral_profile: {e}")
        if self.match_controller:
            try:
                self.match_controller.history.save()
                logger.info("[CLEANUP] Match history saved")
            except (FileNotFoundError, PermissionError, OSError, RuntimeError, AttributeError) as e:
                logger.warning(f"[CLEANUP] Falha ao salvar match_history: {e}")
        if self.brawler_selector:
            try:
                self.brawler_selector._save_stats()
                logger.info("[CLEANUP] Brawler selector stats saved")
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
                logger.warning(f"[CLEANUP] Falha ao salvar brawler_selector: {e}")

    def cleanup(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_resolution(self):
        if hasattr(self.wrapper, "emulator_subsystem"):
            return self.wrapper.emulator_subsystem.get_safe_resolution()
        return (1920, 1080)

    def _init_advanced_modules(self) -> None:
        try:
            from core.central_coordinator import CentralCoordinator

            self.central_coordinator = CentralCoordinator()
            logger.info("[WRAPPER] Central Coordinator inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            if isinstance(e, ImportError):
                logger.warning(f"[WRAPPER] CentralCoordinator indisponível (não instalado): {e}")
            else:
                logger.error(f"[WRAPPER] CentralCoordinator ERRO: {e}")

        try:
            from core.world_model import WorldModel

            self.world_model = WorldModel()
            logger.info("[WRAPPER] World Model inicializado")
        except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning(f"[WRAPPER] World Model indisponível: {e}")

        try:
            from core.pressure_map import PressureMap

            self.pressure_map = PressureMap()
            logger.info("[WRAPPER] Pressure Map inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Pressure Map indisponível: {e}")

        try:
            from core.behavioral_profile import BehavioralProfile

            self.behavioral_profile = BehavioralProfile()
            logger.info("[WRAPPER] Behavioral Profile inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Behavioral Profile indisponível: {e}")

        try:
            from core.cover_system import CoverSystem

            self.cover_system = CoverSystem()
            logger.info("[WRAPPER] Cover System inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Cover System indisponível: {e}")

        try:
            from decision.utility_ai import UtilityAI

            self.utility_ai = UtilityAI()
            logger.info("[WRAPPER] Utility AI inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Utility AI indisponível: {e}")

        try:
            from decision.sticky_target import StickyTarget

            self.sticky_target = StickyTarget()
            logger.info("[WRAPPER] Sticky Target inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Sticky Target indisponível: {e}")

        try:
            from decision.intent_system import IntentSystem

            self.intent_system = IntentSystem()
            logger.info("[WRAPPER] Intent System inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Intent System indisponível: {e}")

        try:
            from decision.enemy_intention import EnemyIntentionPredictor

            self.enemy_intention = EnemyIntentionPredictor()
            logger.info("[WRAPPER] Enemy Intention Predictor inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Enemy Intention Predictor indisponível: {e}")

        try:
            from decision.meta_awareness import MetaAwareness

            self.meta_awareness = MetaAwareness()
            logger.info("[WRAPPER] Meta Awareness inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Meta Awareness indisponível: {e}")

    def _create_play_logic(self, detect_main, detect_enemies, movement, emulator_controller):
        from pylaai_real.play import PlayLogic

        return PlayLogic(
            detect_main=detect_main,
            detect_enemies=detect_enemies,
            movement=movement,
            humanization=getattr(self.wrapper, "humanization", None),
            emulator_controller=emulator_controller,
            rl_engine=self.online_learner.q_learning if self.online_learner else None,
            central_coordinator=self.central_coordinator,
            world_model=self.world_model,
            pressure_map=self.pressure_map,
            enemy_intention=self.enemy_intention,
            meta_awareness=self.meta_awareness,
            cover_system=self.cover_system,
            world_model_integrator=getattr(self.wrapper, "world_model_integrator", None),
        )

    def _create_screen_auto(self):
        screenshot = getattr(self.wrapper, "screenshot", None)
        emulator_controller = getattr(self.wrapper, "emulator_controller", None)
        try:
            from pylaai_real.screen_automation import ScreenAutomation

            if screenshot and screenshot.window_handle:
                import win32gui

                rect = win32gui.GetWindowRect(screenshot.window_handle)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                return ScreenAutomation(w, h, rect[0], rect[1])
            elif emulator_controller:
                cfg = emulator_controller.config
                return ScreenAutomation(cfg.resolution[0], cfg.resolution[1], 0, 0)
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning(f"ScreenAutomation init failed: {e}")
        return None
