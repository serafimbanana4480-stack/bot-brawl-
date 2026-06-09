"""
core/subsystems/ui_subsystem.py

UISubsystem: dashboard, diagnostic overlay, ESP overlay, mode controller,
RL metrics, debug visualizer integration, and system status aggregation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)


class UISubsystem:
    """Manages UI components: dashboard, overlays, diagnostics, and status."""

    def __init__(
        self,
        wrapper: "PylaAIEnhanced",
        central_config: dict,
        diagnostic_mode: bool,
    ):
        self.wrapper = wrapper
        self.central_config = central_config
        self.diagnostic_mode = diagnostic_mode
        self.dashboard: Optional[Any] = None
        self.diagnostic_overlay: Optional[Any] = None
        self.esp_overlay: Optional[Any] = None
        self.mode_controller: Optional[Any] = None
        self.rl_metrics_collector: Optional[Any] = None
        self.overlay_enabled = bool(
            diagnostic_mode or __import__("os").getenv("PYLAAI_OVERLAY", "1") == "1"
        )
        self._init_dashboard()
        self._init_mode_controller()
        self._init_rl_metrics()
        self._init_esp()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Initialize UI components that depend on other subsystems."""
        # Debug integration
        debug_visualizer = getattr(self.wrapper, "debug_visualizer", None)
        if debug_visualizer:
            try:
                from pylaai_real.debug_visualizer import DebugIntegration

                self.wrapper.debug_integration = DebugIntegration(self.wrapper)
                logger.info("[WRAPPER] Debug Integration configurado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] Debug Integration indisponível: {e}")

        # Learning mode controller setup if active
        learning_mode = getattr(self.wrapper, "learning_mode", False)
        if learning_mode:
            try:
                from core.learning_mode import LearningModeController

                screenshot_source = getattr(self.wrapper, "emulator_subsystem", None)
                screenshot_taker = (
                    screenshot_source.get_screenshot_source() if screenshot_source else None
                )
                self.wrapper.learning_mode_controller = LearningModeController(
                    lobby_automator=getattr(self.wrapper, "lobby", None),
                    emulator_controller=getattr(self.wrapper, "emulator_controller", None),
                    screenshot_taker=screenshot_taker,
                    state_finder=getattr(self.wrapper, "state_finder", None),
                    play_logic=getattr(self.wrapper, "play_logic", None),
                    max_matches=self.central_config.get("learning_max_matches", 5),
                    match_timeout_seconds=self.central_config.get("learning_match_timeout", 300.0),
                )
                logger.info("[WRAPPER] LearningModeController inicializado")
            except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.error(f"[WRAPPER] LearningModeController init falhou: {e}")
                self.wrapper.learning_mode_controller = None
        else:
            self.wrapper.learning_mode_controller = None

        # Sync back
        self.wrapper.dashboard = self.dashboard
        self.wrapper.diagnostic_overlay = self.diagnostic_overlay
        self.wrapper.esp_overlay = self.esp_overlay
        self.wrapper.mode_controller = self.mode_controller
        self.wrapper.rl_metrics_collector = self.rl_metrics_collector
        return True

    def start(self) -> None:
        if self.dashboard:
            try:
                self.dashboard.start(daemon=True)
                logger.info(
                    "[WRAPPER] Dashboard server iniciado — aceda em http://localhost:%s",
                    self.dashboard.port,
                )
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning(f"[WRAPPER] Falha ao iniciar dashboard: {e}")

        if self.overlay_enabled and self.diagnostic_overlay is None:
            try:
                from diagnostic_overlay import DiagnosticOverlay

                self.diagnostic_overlay = DiagnosticOverlay(self.wrapper.get_status)
                self.diagnostic_overlay.start()
                logger.info("[WRAPPER] Diagnostic overlay iniciado")
            except (ImportError, ModuleNotFoundError, ConnectionError, RuntimeError, AttributeError, OSError) as e:
                logger.warning(f"[WRAPPER] Diagnostic overlay indisponível: {e}")
                self.diagnostic_overlay = None

        self.wrapper.dashboard = self.dashboard
        self.wrapper.diagnostic_overlay = self.diagnostic_overlay

    def stop(self) -> None:
        if self.diagnostic_overlay:
            try:
                self.diagnostic_overlay.stop()
                logger.debug("[CLEANUP] Parando diagnostic overlay")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar diagnostic_overlay: {e}")
        if self.dashboard:
            try:
                self.dashboard.stop()
                logger.info("[CLEANUP] Dashboard server parado")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar dashboard: {e}")

    def cleanup(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_dashboard(self) -> None:
        if not bool(self.central_config.get("dashboard_enabled", True)):
            return
        try:
            from pylaai_real.dashboard_server import DashboardServer

            self.dashboard = DashboardServer(
                port=self.central_config.get("dashboard_port", 8765)
            )
            logger.info(
                "[WRAPPER] Dashboard server inicializado (porta %s)",
                self.central_config.get("dashboard_port", 8765),
            )
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Dashboard server indisponível: {e}")

    def _init_mode_controller(self) -> None:
        try:
            from core.mode_controller import ModeController

            self.mode_controller = ModeController(
                wrapper_ref=self.wrapper,
                learning_mode_controller=getattr(self.wrapper, "learning_mode_controller", None),
                rl_engine=getattr(self.wrapper, "online_learner", None),
            )
            logger.info("[WRAPPER] ModeController inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] ModeController init falhou: {e}")

    def _init_rl_metrics(self) -> None:
        try:
            from core.learning_rl_metrics import LearningRLMetricsCollector

            self.rl_metrics_collector = LearningRLMetricsCollector()
            logger.info("[WRAPPER] LearningRLMetricsCollector inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] LearningRLMetricsCollector init falhou: {e}")

    def _init_esp(self) -> None:
        try:
            from core.esp_overlay import ESPOverlay

            emulator_title = "BlueStacks"
            emulator_controller = getattr(self.wrapper, "emulator_controller", None)
            if emulator_controller and hasattr(emulator_controller, "window_title"):
                emulator_title = emulator_controller.window_title
            self.esp_overlay = ESPOverlay(window_title=emulator_title)
            logger.info("[WRAPPER] ESPOverlay inicializado")
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning(f"[WRAPPER] ESPOverlay init falhou: {e}")

    # ------------------------------------------------------------------
    # Status / queries
    # ------------------------------------------------------------------

    def get_system_status(self) -> Dict:
        """Return detailed system status for dashboard."""
        status = {
            "paused": getattr(self.wrapper, "_paused", False),
            "running": self.wrapper.running,
            "systems": {
                "rl_engine": {
                    "enabled": (
                        getattr(self.wrapper.online_learner, "enabled", False)
                        if getattr(self.wrapper, "online_learner", None)
                        else False
                    ),
                    "available": getattr(self.wrapper, "online_learner", None) is not None,
                },
                "humanization": {
                    "enabled": (
                        getattr(self.wrapper.humanization, "enabled", True)
                        if getattr(self.wrapper, "humanization", None)
                        else False
                    ),
                    "available": getattr(self.wrapper, "humanization", None) is not None,
                },
                "anti_ban": {
                    "enabled": (
                        getattr(self.wrapper.anti_ban, "enabled", True)
                        if getattr(self.wrapper, "anti_ban", None)
                        else False
                    ),
                    "available": getattr(self.wrapper, "anti_ban", None) is not None,
                },
                "error_recovery": {
                    "enabled": (
                        getattr(self.wrapper.error_recovery, "enabled", True)
                        if getattr(self.wrapper, "error_recovery", None)
                        else False
                    ),
                    "available": getattr(self.wrapper, "error_recovery", None) is not None,
                },
                "recording": {
                    "enabled": getattr(self.wrapper, "recording_enabled", False),
                    "available": getattr(self.wrapper, "gameplay_recorder", None) is not None,
                },
                "auto_tuner": {
                    "enabled": getattr(self.wrapper, "auto_tuning_enabled", False),
                    "available": getattr(self.wrapper, "auto_tuner", None) is not None,
                },
                "data_collector": {
                    "enabled": getattr(self.wrapper, "data_collector", None) is not None,
                    "available": getattr(self.wrapper, "data_collector", None) is not None,
                },
                "learning_mode": {
                    "enabled": getattr(self.wrapper, "learning_mode", False),
                    "available": True,
                    "active": (
                        getattr(self.wrapper.learning_mode_controller, "_match_active", False)
                        if getattr(self.wrapper, "learning_mode_controller", None)
                        else False
                    ),
                },
                "mode_controller": {
                    "available": self.mode_controller is not None,
                    "active_mode": (
                        self.mode_controller.get_status().get("active_mode")
                        if self.mode_controller
                        else None
                    ),
                },
                "esp_overlay": {
                    "available": self.esp_overlay is not None,
                    "enabled": self.esp_overlay.enabled if self.esp_overlay else False,
                },
                "rl_metrics": {
                    "available": self.rl_metrics_collector is not None,
                },
            },
        }
        return status

    def get_detection_snapshot(self) -> Dict:
        """Return raw vision detections."""
        detections = []
        vision_stats = {}
        try:
            detect_main = getattr(self.wrapper, "detect_main", None)
            if detect_main and hasattr(detect_main, "get_raw_detections"):
                detections = detect_main.get_raw_detections()
            if detect_main and hasattr(detect_main, "get_vision_stats"):
                vision_stats = detect_main.get_vision_stats()
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug(f"[WRAPPER] Detection snapshot error: {e}")
        return {"detections": detections, "vision_stats": vision_stats}

    def get_rl_metrics(self) -> Dict:
        """Return RL metrics for dashboard."""
        if self.rl_metrics_collector:
            return self.rl_metrics_collector.get_metrics()
        online_learner = getattr(self.wrapper, "online_learner", None)
        if online_learner and hasattr(online_learner, "rl_bridge") and online_learner.rl_bridge:
            return online_learner.rl_bridge.get_stats()
        if online_learner and hasattr(online_learner, "q_learning"):
            return online_learner.q_learning.get_live_metrics()
        return {}

    # ------------------------------------------------------------------
    # Dashboard update (extracted from wrapper.py)
    # ------------------------------------------------------------------

    def update_dashboard(self, wrapper: Any) -> None:
        if not wrapper.dashboard:
            return
        try:
            if wrapper.learning_mode and wrapper.learning_mode_controller:
                try:
                    lm_data = wrapper.learning_mode_controller.get_live_metrics()
                    wrapper.dashboard.bridge.update(
                        learning_mode_active=lm_data.get("active", False),
                        learning_current_match=lm_data.get("current_match", 0),
                        learning_max_matches=lm_data.get("max_matches", 0),
                        learning_kills=lm_data.get("kills", 0),
                        learning_detections=lm_data.get("detections_enemies", 0),
                        learning_accuracy=lm_data.get("accuracy_percent", 0.0),
                        learning_damage=lm_data.get("damage_dealt", 0.0),
                        learning_survival_seconds=lm_data.get("match_duration_seconds", 0.0),
                        learning_brawler=lm_data.get("current_brawler", "unknown"),
                    )
                except Exception as e:
                    logger.debug(f"[WRAPPER] Learning mode dashboard update error: {e}")

            wrapper.dashboard.update_from_wrapper(wrapper)

            try:
                snap = self.get_detection_snapshot()
                wrapper.dashboard.bridge.update(
                    detections=snap.get("detections", []),
                    vision_stats=snap.get("vision_stats", {}),
                )
                if wrapper.esp_overlay and wrapper.esp_overlay.enabled:
                    player_pos = None
                    target_pos = None
                    if wrapper.play_logic and hasattr(wrapper.play_logic, "last_combat_snapshot"):
                        pcs = wrapper.play_logic.last_combat_snapshot
                        if pcs:
                            player = pcs.get("player")
                            if player and len(player) >= 4:
                                player_pos = (int((player[0] + player[2]) / 2), int((player[1] + player[3]) / 2))
                            enemies = pcs.get("enemies", [])
                            if enemies and len(enemies[0]) >= 4:
                                e = enemies[0]
                                target_pos = (int((e[0] + e[2]) / 2), int((e[1] + e[3]) / 2))
                    wrapper.esp_overlay.update_detections(
                        snap.get("detections", []),
                        player_pos=player_pos,
                        target_pos=target_pos,
                    )
            except Exception as e:
                logger.debug(f"[WRAPPER] Detection dashboard update error: {e}")

            try:
                rl_data = self.get_rl_metrics()
                if rl_data:
                    wrapper.dashboard.bridge.update(
                        rl_active=rl_data.get("active", False),
                        rl_engine_type=rl_data.get("engine_type", ""),
                        rl_epsilon=rl_data.get("epsilon", 0.0),
                        rl_q_table_size=rl_data.get("q_table_size", 0),
                        rl_total_updates=rl_data.get("total_updates", 0),
                        rl_last_reward=rl_data.get("last_reward", 0.0),
                        rl_episode_reward=rl_data.get("episode_reward", 0.0),
                        rl_policy_loss=rl_data.get("policy_loss", 0.0),
                        rl_value_loss=rl_data.get("value_loss", 0.0),
                        rl_entropy=rl_data.get("entropy", 0.0),
                        rl_buffer_size=rl_data.get("buffer_size", 0),
                    )
            except Exception as e:
                logger.debug(f"[WRAPPER] RL metrics dashboard update error: {e}")

            try:
                if wrapper.mode_controller:
                    mode_status = wrapper.mode_controller.get_status()
                    wrapper.dashboard.bridge.update(
                        active_mode=mode_status.get("active_mode"),
                        mode_training=mode_status.get("training_active", False),
                        mode_farm=mode_status.get("farm_active", False),
                        mode_learn=mode_status.get("learn_active", False),
                        mode_matches_completed=mode_status.get("matches_completed", 0),
                        mode_matches_target=mode_status.get("matches_target", 0),
                        mode_current_brawler=mode_status.get("current_brawler", "unknown"),
                        mode_session_duration=mode_status.get("session_duration_seconds", 0.0),
                    )
            except Exception as e:
                logger.debug(f"[WRAPPER] Mode status dashboard update error: {e}")

            if (
                wrapper.state_manager
                and wrapper.state_manager.current_state == "in_game"
                and wrapper.dashboard.recorder
                and getattr(wrapper.state_manager, "_last_screenshot", None) is not None
            ):
                try:
                    import cv2
                    import numpy as np

                    img = wrapper.state_manager._last_screenshot
                    if isinstance(img, np.ndarray) and img.size > 0:
                        thumb = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 50])
                        b64 = __import__("base64").b64encode(buf).decode("ascii")
                        wrapper.dashboard.bridge.update(screenshot_b64=b64)
                        wrapper.dashboard.record_replay_frame(
                            screenshot=thumb,
                            state=wrapper.state_manager.current_state,
                            action=getattr(wrapper.play_logic, "_last_action", "idle") if wrapper.play_logic else "idle",
                            enemies=getattr(wrapper.play_logic, "_last_enemies", 0) if wrapper.play_logic else 0,
                        )
                except Exception as e:
                    logger.debug(f"[WRAPPER] Replay frame capture failed: {e}")
        except Exception as e:
            logger.debug(f"[WRAPPER] Dashboard update error: {e}")

    # ------------------------------------------------------------------
    # get_status (extracted from wrapper.py)
    # ------------------------------------------------------------------

    def get_status(self, wrapper: Any) -> Dict:
        logger.debug("[WRAPPER] get_status() chamado")
        session_duration = 0.0
        if wrapper.session_start:
            session_duration = time.time() - wrapper.session_start

        window_snapshot = None
        if wrapper.emulator_controller and hasattr(wrapper.emulator_controller, "get_status_snapshot"):
            try:
                window_snapshot = wrapper.emulator_controller.get_status_snapshot()
            except Exception:
                pass

        combat_snapshot = None
        if wrapper.play_logic and hasattr(wrapper.play_logic, "get_last_combat_snapshot"):
            try:
                combat_snapshot = wrapper.play_logic.get_last_combat_snapshot()
            except Exception:
                pass

        lobby_diagnostic = None
        if wrapper.lobby and hasattr(wrapper.lobby, "get_diagnostic_report"):
            try:
                lobby_diagnostic = wrapper.lobby.get_diagnostic_report()
            except Exception:
                pass

        screen_state = None
        if (
            wrapper.state_manager
            and wrapper.state_manager.screen_automation
            and hasattr(wrapper.state_manager.screen_automation, "get_current_state_name")
        ):
            try:
                screen_state = wrapper.state_manager.screen_automation.get_current_state_name()
            except Exception:
                pass

        tracker_stats = None
        if wrapper.play_logic and hasattr(wrapper.play_logic, "enemy_tracker") and wrapper.play_logic.enemy_tracker:
            try:
                tracker_stats = wrapper.play_logic.enemy_tracker.get_stats()
            except Exception:
                pass

        current_map = None
        if wrapper.play_logic and hasattr(wrapper.play_logic, "movement") and wrapper.play_logic.movement:
            try:
                current_map = wrapper.play_logic.movement.current_map
            except Exception:
                pass

        return {
            "running": wrapper.running,
            "current_state": wrapper.state_manager.current_state if wrapper.state_manager else "unknown",
            "last_known_state": getattr(wrapper.state_manager, "last_known_state", "unknown") if wrapper.state_manager else "unknown",
            "unknown_streak": getattr(wrapper.state_manager, "unknown_streak", 0) if wrapper.state_manager else 0,
            "last_unknown_hint": getattr(wrapper.state_manager, "last_unknown_hint", None) if wrapper.state_manager else None,
            "queue": wrapper.get_queue(),
            "safety": wrapper.safety.get_status() if wrapper.safety else None,
            "session_duration_minutes": session_duration / 60,
            "matches_played": wrapper.matches_played,
            "current_brawler": wrapper.brawler_queue.get_current().name if wrapper.brawler_queue.get_current() else None,
            "current_map": current_map,
            "emulator_controller_active": wrapper.emulator_controller is not None,
            "window_active": window_snapshot.get("window_active") if isinstance(window_snapshot, dict) else None,
            "window_title": window_snapshot.get("window_title") if isinstance(window_snapshot, dict) else None,
            "models_loaded": wrapper.detect_main is not None,
            "diagnostic_overlay_active": bool(getattr(wrapper, "diagnostic_overlay", None)),
            "diagnostics": {
                "diagnostic_mode": wrapper.diagnostic_mode,
                "lobby": lobby_diagnostic,
                "screen_state": screen_state,
                "progress": wrapper.progress.get_stats() if wrapper.progress else None,
                "match": wrapper.match_controller.get_session_stats() if wrapper.match_controller else None,
                "combat": combat_snapshot,
            },
            "tracker_stats": tracker_stats,
            "orchestrator": {
                "enabled": getattr(wrapper, "use_orchestrator", False),
                "state": wrapper.orchestrator.get_status() if getattr(wrapper, "orchestrator", None) else None,
            },
            "error_recovery": wrapper.error_recovery.get_stats() if wrapper.error_recovery else {"enabled": False},
            "state_recovery": wrapper.state_recovery.get_recovery_status() if wrapper.state_recovery else {"is_recovering": False},
            "auto_calibrator": {
                "enabled": wrapper.auto_calibrator is not None,
                "cached_coords": len(wrapper.auto_calibrator.coords_cache) if wrapper.auto_calibrator else 0,
            },
            "ocr_detector": wrapper.ocr_detector.get_detection_stats() if wrapper.ocr_detector else {"reader_available": False},
            "debug_visualizer": {
                "enabled": wrapper.debug_visualizer is not None,
                "running": wrapper.debug_visualizer.is_running if wrapper.debug_visualizer else False,
            },
        }
