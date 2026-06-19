"""
Plugin system for Brawl Stars Bot.
Replaces the fragile HAS_XXX lazy import pattern with a proper plugin architecture.
"""

import abc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class IPlugin(abc.ABC):
    """Abstract base class for all plugins."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the unique plugin identifier."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying optional dependency is importable."""

    def initialize(self, **kwargs) -> Any:
        """Create and return the plugin instance or class. Called once during load."""
        return None

    def shutdown(self) -> None:
        """Clean up resources. Called during shutdown."""
        return None


class PluginManager:
    """Manages registration, loading, and querying of plugins."""

    def __init__(self):
        self._registry: list[type[IPlugin]] = []
        self._plugins: dict[str, IPlugin] = {}
        self._instances: dict[str, Any] = {}

    def register(self, plugin_class: type[IPlugin]) -> type[IPlugin]:
        """Register a plugin class."""
        if not issubclass(plugin_class, IPlugin):
            raise TypeError(f"{plugin_class.__name__} must inherit from IPlugin")
        self._registry.append(plugin_class)
        return plugin_class

    def discover(self) -> None:
        """Auto-discover all plugins decorated with @PluginRegistry."""
        from core.plugin_system import get_global_registry

        for plugin_class in get_global_registry():
            if plugin_class not in self._registry:
                self._registry.append(plugin_class)

    def load_all(self, **kwargs) -> dict[str, Any]:
        """Initialize all registered plugins that are available."""
        results: dict[str, Any] = {}
        sorted_plugins = sorted(
            self._registry,
            key=lambda p: getattr(p, "priority", 50),
        )
        for plugin_class in sorted_plugins:
            plugin = plugin_class()
            name = plugin.name
            self._plugins[name] = plugin
            if plugin.is_available():
                try:
                    instance = plugin.initialize(**kwargs)
                    if instance is not None:
                        self._instances[name] = instance
                        results[name] = instance
                        kwargs[name] = instance
                        logger.info(f"[PLUGIN] Loaded '{name}'")
                    else:
                        logger.debug(
                            f"[PLUGIN] '{name}' available but returned no instance"
                        )
                except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                    logger.warning(f"[PLUGIN] Failed to initialize '{name}': {e}")
            else:
                logger.debug(f"[PLUGIN] '{name}' not available")
        return results

    def get(self, name: str) -> Any | None:
        """Get initialized plugin instance/class by name."""
        return self._instances.get(name)

    def is_available(self, name: str) -> bool:
        """Check if a plugin is available (importable)."""
        plugin = self._plugins.get(name)
        return plugin.is_available() if plugin else False

    def shutdown_all(self) -> None:
        """Shutdown all plugins in reverse load order."""
        for name in reversed(list(self._plugins.keys())):
            try:
                self._plugins[name].shutdown()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.warning(f"[PLUGIN] Shutdown error for '{name}': {e}")
        self._instances.clear()


_GLOBAL_REGISTRY: list[type[IPlugin]] = []


def PluginRegistry(cls: type[IPlugin]) -> type[IPlugin]:  # noqa: N802
    """Class decorator for auto-discoverable plugins."""
    if not issubclass(cls, IPlugin):
        raise TypeError(f"{cls.__name__} must inherit from IPlugin")
    _GLOBAL_REGISTRY.append(cls)
    return cls


def get_global_registry() -> list[type[IPlugin]]:
    """Return a copy of the global plugin registry."""
    return list(_GLOBAL_REGISTRY)


def clear_global_registry() -> None:
    """Clear the global registry. Useful for testing."""
    _GLOBAL_REGISTRY.clear()
