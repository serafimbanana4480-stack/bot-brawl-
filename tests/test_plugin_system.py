"""Unit tests for the plugin system."""

import pytest
from unittest.mock import MagicMock

from core.plugin_system import (
    IPlugin,
    PluginManager,
    PluginRegistry,
    clear_global_registry,
    get_global_registry,
)


class TestIPlugin:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            IPlugin()

    def test_subclass_must_implement_name(self):
        class BadPlugin(IPlugin):
            def is_available(self):
                return True

        with pytest.raises(TypeError):
            BadPlugin()

    def test_subclass_must_implement_is_available(self):
        class BadPlugin(IPlugin):
            @property
            def name(self):
                return "bad"

        with pytest.raises(TypeError):
            BadPlugin()


class TestPluginManager:
    def setup_method(self):
        self.manager = PluginManager()

    def test_register_plugin(self):
        class DummyPlugin(IPlugin):
            @property
            def name(self):
                return "dummy"

            def is_available(self):
                return True

        self.manager.register(DummyPlugin)
        assert len(self.manager._registry) == 1

    def test_register_non_plugin_raises(self):
        with pytest.raises(TypeError):
            self.manager.register(str)

    def test_load_all_initializes_available_plugins(self):
        class AvailablePlugin(IPlugin):
            @property
            def name(self):
                return "available"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                return "instance"

        self.manager.register(AvailablePlugin)
        results = self.manager.load_all()
        assert results == {"available": "instance"}
        assert self.manager.get("available") == "instance"

    def test_load_all_skips_unavailable_plugins(self):
        class UnavailablePlugin(IPlugin):
            @property
            def name(self):
                return "unavailable"

            def is_available(self):
                return False

        self.manager.register(UnavailablePlugin)
        results = self.manager.load_all()
        assert "unavailable" not in results
        assert self.manager.get("unavailable") is None

    def test_load_all_graceful_on_init_failure(self):
        class BrokenPlugin(IPlugin):
            @property
            def name(self):
                return "broken"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                raise RuntimeError("boom")

        self.manager.register(BrokenPlugin)
        results = self.manager.load_all()
        assert "broken" not in results

    def test_is_available(self):
        class MaybePlugin(IPlugin):
            @property
            def name(self):
                return "maybe"

            def is_available(self):
                return True

        self.manager.register(MaybePlugin)
        self.manager.load_all()
        assert self.manager.is_available("maybe") is True

    def test_is_available_missing_plugin(self):
        assert self.manager.is_available("missing") is False

    def test_shutdown_all(self):
        class ShutdownPlugin(IPlugin):
            @property
            def name(self):
                return "shutdown_test"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                return MagicMock()

            def shutdown(self):
                self._shutdown_called = True

        self.manager.register(ShutdownPlugin)
        self.manager.load_all()
        self.manager.shutdown_all()
        assert self.manager.get("shutdown_test") is None

    def test_priority_sorting(self):
        class HighPriority(IPlugin):
            priority = 10

            @property
            def name(self):
                return "high"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                return "high_inst"

        class LowPriority(IPlugin):
            priority = 100

            @property
            def name(self):
                return "low"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                return "low_inst"

        self.manager.register(LowPriority)
        self.manager.register(HighPriority)
        self.manager.load_all()
        assert list(self.manager._instances.keys()) == ["high", "low"]

    def test_kwargs_cascade(self):
        class FirstPlugin(IPlugin):
            @property
            def name(self):
                return "first"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                return {"data": 42}

        class SecondPlugin(IPlugin):
            @property
            def name(self):
                return "second"

            def is_available(self):
                return True

            def initialize(self, **kwargs):
                first = kwargs.get("first")
                return first["data"] * 2 if first else 0

        self.manager.register(FirstPlugin)
        self.manager.register(SecondPlugin)
        results = self.manager.load_all()
        assert results["second"] == 84


class TestPluginRegistry:
    def setup_method(self):
        clear_global_registry()

    def teardown_method(self):
        clear_global_registry()

    def test_decorator_registers_plugin(self):
        @PluginRegistry
        class TestPlugin(IPlugin):
            @property
            def name(self):
                return "test"

            def is_available(self):
                return True

        assert len(get_global_registry()) == 1
        assert get_global_registry()[0] is TestPlugin

    def test_discover_populates_manager(self):
        @PluginRegistry
        class AutoPlugin(IPlugin):
            @property
            def name(self):
                return "auto"

            def is_available(self):
                return True

        manager = PluginManager()
        manager.discover()
        assert len(manager._registry) == 1

    def test_decorator_rejects_non_plugin(self):
        with pytest.raises(TypeError):

            @PluginRegistry
            class NotAPlugin:
                pass


class TestLazyImportPlugins:
    """Tests for the actual plugin wrappers in plugins/."""

    def test_yolo_plugin_imports(self):
        from plugins.vision_plugin import YOLOPlugin

        plugin = YOLOPlugin()
        assert plugin.name == "yolo"
        avail = plugin.is_available()
        assert isinstance(avail, bool)

    def test_anti_ban_plugin_imports(self):
        from plugins.safety_plugin import AntiBanPlugin

        plugin = AntiBanPlugin()
        assert plugin.name == "anti_ban"
        avail = plugin.is_available()
        assert isinstance(avail, bool)

    def test_collector_plugin_imports(self):
        from plugins.learning_plugin import CollectorPlugin

        plugin = CollectorPlugin()
        assert plugin.name == "collector"
        avail = plugin.is_available()
        assert isinstance(avail, bool)

    def test_dashboard_plugin_imports(self):
        from plugins.ui_plugin import DashboardPlugin

        plugin = DashboardPlugin()
        assert plugin.name == "dashboard"
        avail = plugin.is_available()
        assert isinstance(avail, bool)

    def test_central_coordinator_plugin_imports(self):
        from plugins.strategy_plugin import CentralCoordinatorPlugin

        plugin = CentralCoordinatorPlugin()
        assert plugin.name == "central_coordinator"
        avail = plugin.is_available()
        assert isinstance(avail, bool)

    def test_orchestrator_plugin_returns_namespace(self):
        from plugins.strategy_plugin import OrchestratorPlugin

        plugin = OrchestratorPlugin()
        assert plugin.name == "orchestrator"
        avail = plugin.is_available()
        assert isinstance(avail, bool)
        if avail:
            ns = plugin.initialize()
            assert "factory" in ns
            assert "class" in ns
