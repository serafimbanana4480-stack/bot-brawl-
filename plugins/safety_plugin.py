"""Safety-related plugins: AntiBan, ErrorRecovery, StateRecovery, AutoFix."""

from pathlib import Path

from core.plugin_system import IPlugin, PluginRegistry


@PluginRegistry
class AntiBanPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "anti_ban"

    def is_available(self) -> bool:
        try:
            from pylaai_real.anti_ban_advanced import AdvancedAntiBanSystem

            self._advanced_cls = AdvancedAntiBanSystem
            self._use_advanced = True
            return True
        except Exception:
            self._advanced_cls = None
            self._use_advanced = False

        try:
            from core.anti_ban import AntiBanSystem

            self._basic_cls = AntiBanSystem
            return True
        except Exception:
            self._basic_cls = None
            return False

    def initialize(self, **kwargs):
        if self._use_advanced:
            return self._advanced_cls({"enabled": True})
        return self._basic_cls()


@PluginRegistry
class ErrorRecoveryPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "error_recovery"

    def is_available(self) -> bool:
        try:
            from core.error_recovery import ErrorRecoverySystem

            self._cls = ErrorRecoverySystem
            return True
        except Exception:
            try:
                import importlib.util as _ilu

                _er_path = Path(__file__).parent.parent / "core" / "error_recovery.py"
                _spec = _ilu.spec_from_file_location("error_recovery", _er_path)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                self._cls = _mod.ErrorRecoverySystem
                return True
            except Exception:
                return False

    def initialize(self, **kwargs):
        return self._cls(
            enable_auto_recovery=True,
            max_recovery_attempts=3,
            global_circuit_breaker=True,
        )


@PluginRegistry
class ErrorRecoveryIntegrationPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "error_recovery_integration"

    def is_available(self) -> bool:
        try:
            from core.error_recovery import ErrorRecoveryIntegration

            self._cls = ErrorRecoveryIntegration
            return True
        except Exception:
            try:
                import importlib.util as _ilu

                _er_path = Path(__file__).parent.parent / "core" / "error_recovery.py"
                _spec = _ilu.spec_from_file_location("error_recovery", _er_path)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                self._cls = _mod.ErrorRecoveryIntegration
                return True
            except Exception:
                return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class StateRecoveryPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "state_recovery"

    def is_available(self) -> bool:
        try:
            from pylaai_real.state_recovery import StateRecoverySystem

            self._cls = StateRecoverySystem
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class AutoFixPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "auto_fix"

    def is_available(self) -> bool:
        try:
            from core.auto_fix_engine import AutoFixEngine

            self._cls = AutoFixEngine
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls
