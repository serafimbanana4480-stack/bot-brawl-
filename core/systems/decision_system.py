"""
core/systems/decision_system.py

Encapsulates all decision-making subsystems:
- PlayLogic, Movement, UtilityAI
- BrawlerSelector
- OnlineLearner (RL)
- Tactical modules: CoverSystem, EnemyIntentionPredictor, MetaAwareness, etc.

Interface: init(), start(), stop(), status(), health_check()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pylaai_real.play import PlayLogic
from pylaai_real.movement import Movement
from decision.brawler_selector import BrawlerSelector

logger = logging.getLogger(__name__)


class DecisionSystem:
    """Cohesive decision subsystem with graceful degradation."""

    def __init__(
        self,
        central_config: Dict[str, Any],
        images_path: Path,
        models_path: Path,
    ):
        self.central_config = central_config
        self.images_path = images_path
        self.models_path = models_path

        # Components
        self.play_logic: Optional[PlayLogic] = None
        self.movement: Optional[Any] = None
        self.online_learner: Optional[Any] = None
        self.brawler_selector: Optional[BrawlerSelector] = None
        self.utility_ai: Optional[Any] = None
        self.sticky_target: Optional[Any] = None
        self.intent_system: Optional[Any] = None
        self.enemy_intention: Optional[Any] = None
        self.meta_awareness: Optional[Any] = None
        self.cover_system: Optional[Any] = None
        self.central_coordinator: Optional[Any] = None
        self.world_model: Optional[Any] = None
        self.pressure_map: Optional[Any] = None
        self.world_model_integrator: Optional[Any] = None
        self.behavioral_profile: Optional[Any] = None

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        detect_main: Optional[Any] = None,
        detect_enemies: Optional[Any] = None,
        emulator_controller: Optional[Any] = None,
        humanization: Optional[Any] = None,
        window_w: int = 1920,
        window_h: int = 1080,
    ) -> bool:
        """Initialize decision components."""
        success = True

        # Brawler selector
        if self.central_config.get("brawler_selection_enabled", True):
            try:
                self.brawler_selector = BrawlerSelector()
                logger.info("[DECISION] Brawler selector initialized")
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.warning("[DECISION] Brawler selector unavailable: %s", e)

        # Movement
        try:
            self.movement = Movement(
                emulator_controller=emulator_controller,
                window_w=window_w,
                window_h=window_h,
            )
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[DECISION] Movement init failed: %s", e)
            success = False
        except Exception as e:
            logger.exception("[DECISION] Unexpected Movement init error: %s", e)
            raise

        # OnlineLearner (RL)
        try:
            from pylaai_real.rl_engine import OnlineLearner
            from dataset.collector import GameplayCollector

            reward_bridge = getattr(self, "reward_bridge", None)
            gameplay_collector = None
            try:
                import json
                from pathlib import Path
                config_path = Path("config.json")
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    if cfg.get("rl", {}).get("data_collection_mode", False):
                        gameplay_collector = GameplayCollector()
            except Exception:
                pass
            self.online_learner = OnlineLearner(
                reward_bridge=reward_bridge,
                gameplay_collector=gameplay_collector,
                enabled=True,
            )
            logger.info("[DECISION] OnlineLearner (RL + ELO + DataCollection) initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.warning("[DECISION] OnlineLearner unavailable: %s", e)

        # Tactical / advanced modules (optional)
        optional_modules = [
            ("core.central_coordinator", "CentralCoordinator", "central_coordinator"),
            ("core.world_model", "WorldModel", "world_model"),
            ("core.pressure_map", "PressureMap", "pressure_map"),
            ("core.cover_system", "CoverSystem", "cover_system"),
            ("core.behavioral_profile", "BehavioralProfile", "behavioral_profile"),
            ("decision.utility_ai", "UtilityAI", "utility_ai"),
            ("decision.sticky_target", "StickyTarget", "sticky_target"),
            ("decision.intent_system", "IntentSystem", "intent_system"),
            ("decision.enemy_intention", "EnemyIntentionPredictor", "enemy_intention"),
            ("decision.meta_awareness", "MetaAwareness", "meta_awareness"),
        ]
        for module_name, class_name, attr_name in optional_modules:
            try:
                mod = __import__(module_name, fromlist=[class_name])
                cls = getattr(mod, class_name)
                setattr(self, attr_name, cls())
                logger.info("[DECISION] %s initialized", class_name)
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.debug("[DECISION] %s unavailable: %s", class_name, e)

        # World Model Integrator
        try:
            from world_model_integration import WorldModelIntegrator
            self.world_model_integrator = WorldModelIntegrator(world_model=self.world_model)
            logger.info("[DECISION] World Model Integrator initialized")
        except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug("[DECISION] WorldModelIntegrator unavailable: %s", e)

        # PlayLogic (core combat engine)
        try:
            rl_engine = getattr(self.online_learner, "q_learning", None) if self.online_learner else None
            self.play_logic = PlayLogic(
                detect_main=detect_main,
                detect_enemies=detect_enemies,
                movement=self.movement,
                humanization=humanization,
                emulator_controller=emulator_controller,
                rl_engine=rl_engine,
                central_coordinator=self.central_coordinator,
                world_model=self.world_model,
                pressure_map=self.pressure_map,
                enemy_intention=self.enemy_intention,
                meta_awareness=self.meta_awareness,
                cover_system=self.cover_system,
                world_model_integrator=self.world_model_integrator,
            )
            logger.info("[DECISION] PlayLogic initialized")
        except (FileNotFoundError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.error("[DECISION] PlayLogic init failed: %s", e)
            success = False
        except Exception as e:
            logger.exception("[DECISION] Unexpected PlayLogic init error: %s", e)
            raise

        return success

    def start(self) -> bool:
        self._running = True
        return True

    def stop(self) -> bool:
        self._running = False
        if self.play_logic and hasattr(self.play_logic, "stop"):
            try:
                self.play_logic.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug("[DECISION] Failed to stop play_logic: %s", e)
        if self.online_learner:
            try:
                self.online_learner.save()
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
                logger.warning("[DECISION] Failed to save OnlineLearner: %s", e)
        if self.behavioral_profile and hasattr(self.behavioral_profile, "save"):
            try:
                self.behavioral_profile.save()
            except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError) as e:
                logger.warning("[DECISION] Failed to save behavioral_profile: %s", e)
        return True

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "play_logic_ok": self.play_logic is not None,
            "movement_ok": self.movement is not None,
            "online_learner_ok": self.online_learner is not None,
            "brawler_selector_ok": self.brawler_selector is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        issues = []
        if self.play_logic is None:
            issues.append("no_play_logic")
        return {"healthy": len(issues) == 0, "issues": issues}
