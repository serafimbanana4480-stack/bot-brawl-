"""UI-related plugins: Dashboard, ESPOverlay, ModeController, RLMetrics, DebugVisualizer, Observability."""

from pathlib import Path

from core.plugin_system import IPlugin, PluginRegistry


@PluginRegistry
class DashboardPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "dashboard"

    def is_available(self) -> bool:
        try:
            from pylaai_real.dashboard_server import DashboardServer

            self._cls = DashboardServer
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class ESPOverlayPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "esp_overlay"

    def is_available(self) -> bool:
        try:
            from core.esp_overlay import ESPOverlay

            self._cls = ESPOverlay
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class ModeControllerPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "mode_controller"

    def is_available(self) -> bool:
        try:
            from core.mode_controller import ModeController

            self._cls = ModeController
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class RLMetricsPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "rl_metrics"

    def is_available(self) -> bool:
        try:
            from core.learning_rl_metrics import LearningRLMetricsCollector

            self._cls = LearningRLMetricsCollector
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class DebugVisualizerPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "debug_visualizer"

    def is_available(self) -> bool:
        try:
            from pylaai_real.debug_visualizer import DebugVisualizer

            self._cls = DebugVisualizer
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class DebugIntegrationPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "debug_integration"

    def is_available(self) -> bool:
        try:
            from pylaai_real.debug_visualizer import DebugIntegration

            self._cls = DebugIntegration
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class DebugModePlugin(IPlugin):
    @property
    def name(self) -> str:
        return "debug_mode"

    def is_available(self) -> bool:
        try:
            from pylaai_real.debug_visualizer import DebugMode

            self._cls = DebugMode
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class ObservabilityPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "observability"

    def is_available(self) -> bool:
        try:
            from core.observability import ObservabilityCollector

            self._collector_cls = ObservabilityCollector
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        install_path = kwargs.get("install_path")
        metrics_dir = (
            install_path / "observability" if install_path else Path("observability")
        )
        return self._collector_cls(max_events=2000, metrics_dir=metrics_dir)
