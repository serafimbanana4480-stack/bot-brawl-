"""
Automation system for Brawl Stars Bot.
Provides rule engine for automated monitoring and actions.
"""

from .rule_engine import RuleEngine, Trigger, Action, Rule
from .bot_monitoring import BotMonitoringAutomation
from .code_quality import CodeQualityAutomation
from .model_download import ModelDownloadAutomation
from .emulator_monitoring import EmulatorMonitoringAutomation

__all__ = [
    'RuleEngine',
    'Trigger',
    'Action',
    'Rule',
    'BotMonitoringAutomation',
    'CodeQualityAutomation',
    'ModelDownloadAutomation',
    'EmulatorMonitoringAutomation',
]
