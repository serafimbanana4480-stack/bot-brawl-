"""
Core module for Brawl Stars bot.
Provides orchestration and main execution loop.
"""

# Lazy imports to avoid heavy dependencies on submodule imports
def __getattr__(name):
    if name in ("BrawlStarsOrchestrator", "BotConfig", "create_bot_orchestrator"):
        from .orchestrator import BrawlStarsOrchestrator, BotConfig, create_bot_orchestrator
        if name == "BrawlStarsOrchestrator":
            return BrawlStarsOrchestrator
        if name == "BotConfig":
            return BotConfig
        if name == "create_bot_orchestrator":
            return create_bot_orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BrawlStarsOrchestrator",
    "BotConfig",
    "create_bot_orchestrator",
]
