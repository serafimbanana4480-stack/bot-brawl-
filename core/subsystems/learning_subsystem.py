"""
core/subsystems/learning_subsystem.py

LearningSubsystem: online learning, data collection, model retraining,
meta-learning, gameplay recording, and performance monitoring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)

_BOT_ROOT = Path(__file__).parent.parent.parent


class LearningSubsystem:
    """Manages learning, data collection, recording, and retraining."""

    def __init__(
        self,
        wrapper: PylaAIEnhanced,
        central_config: dict,
    ):
        self.wrapper = wrapper
        self.central_config = central_config
        self.data_collector: Any | None = None
        self.reward_bridge: Any | None = None
        self.meta_learning: Any | None = None
        self.gameplay_recorder: Any | None = None
        self.performance_monitor: Any | None = None
        self.retrain_orchestrator: Any | None = None
        self.recording_enabled = bool(
            central_config.get("enable_recording", False)
        )
        self.auto_retrain_enabled = bool(
            central_config.get("auto_retrain_enabled", False)
        )
        self.recording_dir = _BOT_ROOT / "recordings"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Initialize data collection, reward bridge, recording, and retraining."""
        # Data collector
        try:
            from dataset.collector import GameplayCollector

            self.data_collector = GameplayCollector()
            logger.info("[WRAPPER] Data collector inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Data collector indisponível: {e}")

        # Reward bridge
        try:
            from core.reward_bridge import RewardBridge

            self.reward_bridge = RewardBridge(data_collector=self.data_collector)
            logger.info("[WRAPPER] Reward bridge inicializado")
        except ImportError as e:
            logger.warning(f"[WRAPPER] RewardBridge indisponível (não instalado): {e}")
        except ModuleNotFoundError as e:
            logger.error(f"[WRAPPER] RewardBridge ERRO (Q-Learning vai usar heurísticas): {e}")

        # Meta-learning
        try:
            from meta_learning import MetaLearningSystem
            self.meta_learning = MetaLearningSystem()
            logger.info("[WRAPPER] Meta-Learning System inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] Meta-Learning System indisponível: {e}")

        # Gameplay recording
        if self.recording_enabled:
            self._init_gameplay_recorder()

        # Auto-retrain
        if self.auto_retrain_enabled:
            self._init_auto_retrain_system()

        # Sync back
        self.wrapper.data_collector = self.data_collector
        self.wrapper.reward_bridge = self.reward_bridge
        self.wrapper.meta_learning = self.meta_learning
        self.wrapper.gameplay_recorder = self.gameplay_recorder
        self.wrapper.performance_monitor = self.performance_monitor
        self.wrapper.retrain_orchestrator = self.retrain_orchestrator
        self.wrapper.recording_enabled = self.recording_enabled
        self.wrapper.auto_retrain_enabled = self.auto_retrain_enabled
        self.wrapper.recording_dir = self.recording_dir
        return True

    def start(self) -> None:
        if self.recording_enabled and self.gameplay_recorder:
            try:
                logger.debug("[WRAPPER] Iniciando gameplay recording")
                self.gameplay_recorder.start()
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error(f"[WRAPPER] Failed to start recording: {e}")
        if self.data_collector is not None:
            try:
                self.data_collector.start_episode()
                logger.info("[WRAPPER] Data collector sessão iniciada")
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning(f"[WRAPPER] Falha ao iniciar data_collector: {e}")

    def stop(self) -> None:
        if self.data_collector is not None:
            try:
                self.data_collector.flush()
                logger.info("[CLEANUP] Data collector flushed")
            except (FileNotFoundError, PermissionError, OSError, RuntimeError, AttributeError) as e:
                logger.warning(f"[CLEANUP] Falha ao flush data_collector: {e}")

        online_learner = getattr(self.wrapper, "online_learner", None)
        if online_learner is not None:
            try:
                online_learner.save()
                stats = online_learner.get_stats()
                logger.info(f"[CLEANUP] OnlineLearner salvo: {stats}")
            except (FileNotFoundError, PermissionError, OSError, RuntimeError, AttributeError) as e:
                logger.warning(f"[CLEANUP] Falha ao salvar OnlineLearner: {e}")

        if self.reward_bridge is not None:
            try:
                summary = self.reward_bridge.get_session_summary()
                logger.info(f"[CLEANUP] Reward session summary: {summary}")
                self.reward_bridge.reset()
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
                logger.warning(f"[CLEANUP] Falha ao reset reward_bridge: {e}")

        if self.recording_enabled and self.gameplay_recorder:
            try:
                logger.debug("[CLEANUP] Parando gameplay recording")
                self.gameplay_recorder.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar recording: {e}")

    def cleanup(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Recording / Retraining
    # ------------------------------------------------------------------

    def _init_gameplay_recorder(self) -> bool:
        try:
            from automation.gameplay_recorder import GameplayRecorder

            emulator_controller = getattr(self.wrapper, "emulator_controller", None)
            adb_id = None
            if emulator_controller and emulator_controller.adb:
                adb_id = emulator_controller.adb.device_id
            if not adb_id:
                logger.warning("[WRAPPER] No ADB connection available, gameplay recording disabled")
                return False

            self.recording_dir.mkdir(parents=True, exist_ok=True)
            self.gameplay_recorder = GameplayRecorder(
                adb_id=adb_id,
                adb_path=emulator_controller.adb.adb_path if emulator_controller else None,
                output_dir=self.recording_dir,
                fps=10,
                compress=True,
            )
            logger.info("[WRAPPER] Gameplay recorder initialized successfully")
            return True
        except ImportError as e:
            logger.warning(f"[WRAPPER] GameplayRecorder not available: {e}")
            return False
        except (ModuleNotFoundError, FileNotFoundError, PermissionError, ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as e:
            logger.error(f"[WRAPPER] Failed to initialize gameplay recorder: {e}")
            return False

    def _init_auto_retrain_system(self) -> bool:
        try:
            from training.retrain import PerformanceMonitor, RetrainOrchestrator, RetrainTrigger

            logs_dir = _BOT_ROOT / "logs" / "performance"
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

            dataset_dir = _BOT_ROOT / "dataset" / "raw"
            models_dir = _BOT_ROOT / "models"
            self.retrain_orchestrator = RetrainOrchestrator(
                monitor=self.performance_monitor,
                trigger_conditions=triggers,
                dataset_dir=dataset_dir,
                models_dir=models_dir,
            )
            self.retrain_orchestrator.on_retrain_start = self._on_retrain_start
            self.retrain_orchestrator.on_retrain_complete = self._on_retrain_complete
            self.retrain_orchestrator.on_retrain_failed = self._on_retrain_failed

            logger.info("[WRAPPER] Auto-retrain system initialized successfully")
            return True
        except ImportError as e:
            logger.warning(f"[WRAPPER] Auto-retrain system not available: {e}")
            return False
        except (ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"[WRAPPER] Failed to initialize auto-retrain system: {e}")
            return False

    def _on_retrain_start(self):
        logger.info("[WRAPPER] Retraining started - pausing bot operations")
        state_manager = getattr(self.wrapper, "state_manager", None)
        if state_manager:
            state_manager.pause()

    def _on_retrain_complete(self, new_model_path: Path):
        logger.info(f"[WRAPPER] Retraining completed - new model: {new_model_path}")
        # Ask vision subsystem to reload
        vision = getattr(self.wrapper, "vision_subsystem", None)
        if vision:
            vision._load_trained_models()
        state_manager = getattr(self.wrapper, "state_manager", None)
        if state_manager:
            state_manager.resume()

    def _on_retrain_failed(self, error: str):
        logger.error(f"[WRAPPER] Retraining failed: {error}")

    # ------------------------------------------------------------------
    # Performance / retraining helpers (extracted from wrapper.py)
    # ------------------------------------------------------------------

    def record_performance_metric(self, wrapper: PylaAIEnhanced, metric_type: str, **kwargs):
        pm = getattr(wrapper, "performance_monitor", None)
        if not pm:
            return
        try:
            if metric_type == "kill":
                pm.record_kill()
            elif metric_type == "death":
                pm.record_death()
            elif metric_type == "damage":
                pm.record_damage(dealt=kwargs.get("dealt", 0), taken=kwargs.get("taken", 0))
            elif metric_type == "match_result":
                pm.record_match_result(
                    won=kwargs.get("won", False),
                    survival_time=kwargs.get("survival_time", 0),
                )
            elif metric_type == "decision":
                pm.record_decision(was_good=kwargs.get("good", True))
            elif metric_type == "detection":
                pm.update_detection_metrics(
                    accuracy=kwargs.get("accuracy", 0.0),
                    tracking=kwargs.get("tracking", 0.0),
                    false_positive=kwargs.get("false_positive", 0.0),
                )
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to record performance metric: {e}")

    def check_retrain_trigger(self, wrapper: PylaAIEnhanced) -> tuple[bool, str]:
        if not getattr(wrapper, "retrain_orchestrator", None):
            return False, "Auto-retrain not enabled"
        try:
            return wrapper.retrain_orchestrator.should_retrain()
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to check retrain trigger: {e}")
            return False, "Error checking trigger"

    def trigger_retrain(self, wrapper: PylaAIEnhanced) -> bool:
        if not getattr(wrapper, "retrain_orchestrator", None):
            logger.warning("[WRAPPER] Auto-retrain not enabled")
            return False
        try:
            return wrapper.retrain_orchestrator.trigger_retrain()
        except Exception as e:
            logger.error(f"[WRAPPER] Failed to trigger retrain: {e}")
            return False
