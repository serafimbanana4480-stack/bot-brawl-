"""Learning-related plugins: Collector, RewardBridge, MetaLearning, OnlineLearner, etc."""

import sys
from pathlib import Path

from core.plugin_system import IPlugin, PluginRegistry


@PluginRegistry
class CollectorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "collector"

    def is_available(self) -> bool:
        try:
            from dataset.collector import GameplayCollector

            self._cls = GameplayCollector
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class RewardBridgePlugin(IPlugin):
    @property
    def name(self) -> str:
        return "reward_bridge"

    def is_available(self) -> bool:
        try:
            from core.reward_bridge import RewardBridge

            self._cls = RewardBridge
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class MetaLearningPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "meta_learning"

    def is_available(self) -> bool:
        try:
            from meta_learning import MetaLearningSystem
            self._cls = MetaLearningSystem
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class OnlineLearnerPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "online_learner"

    def is_available(self) -> bool:
        try:
            from pylaai_real.rl_engine import OnlineLearner

            self._cls = OnlineLearner
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class LearningModePlugin(IPlugin):
    @property
    def name(self) -> str:
        return "learning_mode"

    def is_available(self) -> bool:
        try:
            from core.learning_mode import LearningModeController

            self._cls = LearningModeController
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class StatePersistencePlugin(IPlugin):
    @property
    def name(self) -> str:
        return "state_persistence"

    def is_available(self) -> bool:
        try:
            from state_persistence import StatePersistence

            self._cls = StatePersistence
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class WorldModelPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "world_model"

    def is_available(self) -> bool:
        try:
            from core.world_model import WorldModel

            self._cls = WorldModel
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class WorldModelIntegratorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "world_model_integrator"

    def is_available(self) -> bool:
        try:
            from world_model_integration import WorldModelIntegrator

            self._cls = WorldModelIntegrator
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls
