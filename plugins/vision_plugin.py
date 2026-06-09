"""Vision-related plugins: YOLO, OCR, AutoCalibrator, ResolutionManager."""

from pathlib import Path

from core.plugin_system import IPlugin, PluginRegistry


@PluginRegistry
class YOLOPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "yolo"

    def is_available(self) -> bool:
        try:
            from ultralytics import YOLO

            self._cls = YOLO
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls


@PluginRegistry
class OCRDetectorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "ocr_detector"

    def is_available(self) -> bool:
        try:
            from pylaai_real.ocr_state_detector import OCRStateDetector

            self._cls = OCRStateDetector
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls()


@PluginRegistry
class AutoCalibratorPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "auto_calibrator"

    def is_available(self) -> bool:
        try:
            from pylaai_real.auto_calibrator import AutoCalibrator

            self._cls = AutoCalibrator
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        bot_root = kwargs.get("bot_root") or Path(__file__).parent.parent
        return self._cls(templates_dir=bot_root / "images" / "templates")


@PluginRegistry
class ResolutionManagerPlugin(IPlugin):
    @property
    def name(self) -> str:
        return "resolution_manager"

    def is_available(self) -> bool:
        try:
            from core.resolution_manager import ResolutionManager

            self._cls = ResolutionManager
            return True
        except Exception:
            return False

    def initialize(self, **kwargs):
        return self._cls
