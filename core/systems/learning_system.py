"""
core/systems/learning_system.py

Encapsulates all learning / RL / data subsystems:
- DataCollector, RewardBridge
- MetaLearningSystem
- LearningModeController
- RL Metrics Collector
- AutoTuner, PerformanceMonitor, RetrainOrchestrator

Interface: init(), start(), stop(), status(), health_check()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LearningSystem:
    """Cohesive learning subsystem with graceful degradation."""

    def __init__(
        self,
        install_path: Path,
        models_path: Path,
        central_config: dict[str, Any],
    ):
        self.install_path = install_path
        self.models_path = models_path
        self.central_config = central_config

        # Components
        self.data_collector: Any | None = None
        self.reward_bridge: Any | None = None
        self.meta_learning: Any | None = None
        self.learning_mode_controller: Any | None = None
        self.rl_metrics_collector: Any | None = None
        self.auto_tuner: Any | None = None
        self.performance_monitor: Any | None = None
        self.retrain_orchestrator: Any | None = None

        self._running = False
        self._recording_enabled = bool(central_config.get("enable_recording", False))
        self._auto_retrain_enabled = bool(central_config.get("auto_retrain_enabled", False))
        self._auto_tuning_enabled = bool(central_config.get("auto_tuning_enabled", False))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        match_controller: Any | None = None,
        play_logic: Any | None = None,
        lobby: Any | None = None,
        emulator_controller: Any | None = None,
        screenshot_source: Any | None = None,
        state_finder: Any | None = None,
    ) -> bool:
        """Initialize learning components."""
        success = True

        # DataCollector + RewardBridge
        try:
            from dataset.collector import GameplayCollector
            self.data_collector = GameplayCollector()
            logger.info("[LEARNING] Data collector initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[LEARNING] Data collector unavailable: %s", e)

        try:
            from core.reward_bridge import RewardBridge
            self.reward_bridge = RewardBridge(data_collector=self.data_collector)
            logger.info("[LEARNING] Reward bridge initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[LEARNING] Reward bridge unavailable: %s", e)

        # MetaLearning
        try:
            from meta_learning import MetaLearningSystem
            self.meta_learning = MetaLearningSystem()
            logger.info("[LEARNING] Meta-Learning System initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[LEARNING] MetaLearning unavailable: %s", e)

        # LearningModeController
        try:
            from core.learning_mode import LearningModeController
            self.learning_mode_controller = LearningModeController(
                lobby_automator=lobby,
                emulator_controller=emulator_controller,
                screenshot_taker=screenshot_source,
                state_finder=state_finder,
                play_logic=play_logic,
                max_matches=self.central_config.get("learning_max_matches", 5),
                match_timeout_seconds=self.central_config.get("learning_match_timeout", 300.0),
            )
            logger.info("[LEARNING] LearningModeController initialized")
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug("[LEARNING] LearningModeController unavailable: %s", e)

        # RL Metrics
        try:
            from core.learning_rl_metrics import LearningRLMetricsCollector
            self.rl_metrics_collector = LearningRLMetricsCollector()
            logger.info("[LEARNING] RL Metrics Collector initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[LEARNING] RL Metrics Collector unavailable: %s", e)

        # AutoTuner
        if self._auto_tuning_enabled and match_controller:
            try:
                from auto_tuner import AutoTuner
                self.auto_tuner = AutoTuner(match_controller)
                logger.info("[LEARNING] Auto-tuner initialized")
            except (ImportError, ModuleNotFoundError, TypeError) as e:
                logger.debug("[LEARNING] Auto-tuner unavailable: %s", e)

        # Auto-retrain system
        if self._auto_retrain_enabled:
            try:
                from training.retrain import PerformanceMonitor, RetrainOrchestrator, RetrainTrigger
                logs_dir = Path(__file__).parent.parent.parent / "logs" / "performance"
                self.performance_monitor = PerformanceMonitor(log_dir=logs_dir)
                trigger_config = self.central_config.get("retrain_triggers", {})
                triggers = RetrainTrigger(
                    min_matches_before_retrain=trigger_config.get("min_matches", 10),
                    win_rate_threshold=trigger_config.get("win_rate_threshold", 0.4),
                    min_detection_accuracy=trigger_config.get("min_detection_accuracy", 0.7),
                    max_false_positive_rate=trigger_config.get("max_false_positive_rate", 0.2),
                    decision_accuracy_threshold=trigger_config.get("decision_accuracy_threshold", 0.6),
                    max_days_without_retrain=trigger_config.get("max_days", 7),
                    min_new_samples=trigger_config.get("min_new_samples", 500),
                )
                dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "raw"
                self.retrain_orchestrator = RetrainOrchestrator(
                    monitor=self.performance_monitor,
                    trigger_conditions=triggers,
                    dataset_dir=dataset_dir,
                    models_dir=self.models_path,
                )
                logger.info("[LEARNING] Auto-retrain system initialized")
            except (ImportError, ModuleNotFoundError, FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, OSError) as e:
                logger.debug("[LEARNING] Auto-retrain system unavailable: %s", e)

        return success

    def start(self) -> bool:
        self._running = True
        if self.data_collector:
            try:
                self.data_collector.start_episode()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[LEARNING] Failed to start data collector episode: %s", e)
        return True

    def stop(self) -> bool:
        self._running = False
        if self.data_collector:
            try:
                self.data_collector.flush()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[LEARNING] Failed to flush data collector: %s", e)
        if self.reward_bridge:
            try:
                self.reward_bridge.reset()
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.debug("[LEARNING] Failed to reset reward bridge: %s", e)
        if self.meta_learning and hasattr(self.meta_learning, "save"):
            try:
                self.meta_learning.save()
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.debug("[LEARNING] Failed to save meta_learning: %s", e)
        return True

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "data_collector_ok": self.data_collector is not None,
            "reward_bridge_ok": self.reward_bridge is not None,
            "meta_learning_ok": self.meta_learning is not None,
            "learning_mode_ok": self.learning_mode_controller is not None,
            "rl_metrics_ok": self.rl_metrics_collector is not None,
            "auto_tuner_ok": self.auto_tuner is not None,
            "auto_retrain_ok": self.retrain_orchestrator is not None,
        }

    def health_check(self) -> dict[str, Any]:
        issues = []
        if self.data_collector is None:
            issues.append("no_data_collector")
        return {"healthy": len(issues) == 0, "issues": issues}
