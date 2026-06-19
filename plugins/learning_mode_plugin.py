"""Plugin for the learning mode controller."""

from core.plugin_system import IPlugin


class LearningModePlugin(IPlugin):
    """Provides LearningModeController when available."""

    @property
    def name(self) -> str:
        return "learning_mode"

    def is_available(self) -> bool:
        try:
            from core.learning_mode import LearningModeController  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def initialize(self, **kwargs):
        from core.learning_mode import LearningModeController
        return LearningModeController


from core.plugin_system import PluginRegistry  # noqa: E402

PluginRegistry(LearningModePlugin)
