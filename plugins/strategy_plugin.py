"""Strategy and decision plugins: CentralCoordinator, PressureMap, BehavioralProfile, etc."""

from core.plugin_system import IPlugin, PluginRegistry


@PluginRegistry
class CentralCoordinatorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "central_coordinator"

    def is_available(self) -> bool:
        try:
            from core.central_coordinator import CentralCoordinator

            self._cls = CentralCoordinator
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class PressureMapPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "pressure_map"

    def is_available(self) -> bool:
        try:
            from core.pressure_map import PressureMap

            self._cls = PressureMap
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class BehavioralProfilePlugin(IPlugin):
    @property
    def name(self) -> str:
        return "behavioral_profile"

    def is_available(self) -> bool:
        try:
            from core.behavioral_profile import BehavioralProfile

            self._cls = BehavioralProfile
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class CoverSystemPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "cover_system"

    def is_available(self) -> bool:
        try:
            from core.cover_system import CoverSystem

            self._cls = CoverSystem
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class UtilityAIPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "utility_ai"

    def is_available(self) -> bool:
        try:
            from decision.utility_ai import UtilityAI

            self._cls = UtilityAI
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class StickyTargetPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "sticky_target"

    def is_available(self) -> bool:
        try:
            from decision.sticky_target import StickyTarget

            self._cls = StickyTarget
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class IntentSystemPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "intent_system"

    def is_available(self) -> bool:
        try:
            from decision.intent_system import IntentSystem

            self._cls = IntentSystem
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class EnemyIntentionPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "enemy_intention"

    def is_available(self) -> bool:
        try:
            from decision.enemy_intention import EnemyIntentionPredictor

            self._cls = EnemyIntentionPredictor
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class MetaAwarenessPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "meta_awareness"

    def is_available(self) -> bool:
        try:
            from decision.meta_awareness import MetaAwareness

            self._cls = MetaAwareness
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class V2IntegratorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "v2_integrator"

    def is_available(self) -> bool:
        try:
            from core.v2_integration import V2Integrator, V2IntegrationConfig

            self._integrator_cls = V2Integrator
            self._config_cls = V2IntegrationConfig
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return {"integrator": self._integrator_cls, "config": self._config_cls}


@PluginRegistry
class OrchestratorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "orchestrator"

    def is_available(self) -> bool:
        try:
            from core.factory import create_orchestrator
            from core.orchestrator import BotOrchestrator

            self._factory = create_orchestrator
            self._cls = BotOrchestrator
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return {"factory": self._factory, "class": self._cls}
