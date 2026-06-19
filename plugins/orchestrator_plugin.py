"""Plugin for the hexagonal orchestrator subsystem."""

from core.plugin_system import IPlugin


class OrchestratorPlugin(IPlugin):
    """Provides BotOrchestrator when core.orchestrator is available."""

    @property
    def name(self) -> str:
        return "orchestrator"

    def is_available(self) -> bool:
        try:
            from core.factory import create_orchestrator  # noqa: F401
            from core.orchestrator import BotOrchestrator  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def initialize(self, **kwargs):
        from core.factory import create_orchestrator
        return create_orchestrator


# Auto-register when module is imported
from core.plugin_system import PluginRegistry  # noqa: E402

PluginRegistry(OrchestratorPlugin)
