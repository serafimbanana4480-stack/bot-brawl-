"""
core/systems/vision_system.py

Encapsulates all vision-related subsystems:
- Screenshot capture (ScreenshotTaker, EmulatorController wrapper)
- Object detection (Detect / YOLO)
- State detection (UnifiedStateDetector, OCRStateDetector)
- Diagnostic overlay & ESP

Interface: init(), start(), stop(), status(), health_check()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pylaai_real.screenshot_taker import ScreenshotTaker
from pylaai_real.detect import Detect
from pylaai_real.unified_state_detector import UnifiedStateDetector
from diagnostic_overlay import DiagnosticOverlay

logger = logging.getLogger(__name__)


class VisionSystem:
    """Cohesive vision subsystem with graceful degradation."""

    def __init__(
        self,
        images_path: Path,
        models_path: Path,
        central_config: Dict[str, Any],
        diagnostic_mode: bool = False,
    ):
        self.images_path = images_path
        self.models_path = models_path
        self.central_config = central_config
        self.diagnostic_mode = diagnostic_mode

        # Components (populated during setup)
        self.screenshot: Optional[ScreenshotTaker] = None
        self.detect_main: Optional[Detect] = None
        self.detect_enemies: Optional[Detect] = None
        self.unified_detector: Optional[UnifiedStateDetector] = None
        self.ocr_detector: Optional[Any] = None
        self.diagnostic_overlay: Optional[DiagnosticOverlay] = None
        self.esp_overlay: Optional[Any] = None

        self._overlay_enabled = bool(
            diagnostic_mode or self.central_config.get("debug_visualizer", False)
        )
        self._window_w = 1920
        self._window_h = 1080
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(
        self,
        emulator_controller: Optional[Any] = None,
        window_w: int = 1920,
        window_h: int = 1080,
    ) -> bool:
        """Initialize vision components."""
        self._window_w = window_w
        self._window_h = window_h
        success = True

        # ScreenshotTaker (Win32 fallback when no emulator controller)
        if emulator_controller is not None:
            window_title = getattr(emulator_controller.config, "window_title", "BlueStacks App Player")
        else:
            window_title = self.central_config.get("emulator", {}).get("window_title", "BlueStacks App Player")

        try:
            self.screenshot = ScreenshotTaker(window_title=window_title)
            if not self.screenshot.find_window():
                logger.warning("[VISION] ScreenshotTaker window not found")
                self.screenshot = None
        except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.warning("[VISION] ScreenshotTaker unavailable: %s", e)
            self.screenshot = None

        # UnifiedStateDetector
        try:
            self.unified_detector = UnifiedStateDetector(
                self.images_path, window_w=window_w, window_h=window_h,
                ocr_detector=self.ocr_detector,
            )
            logger.info("[VISION] UnifiedStateDetector initialized (%sx%s)", window_w, window_h)
        except (ValueError, TypeError, RuntimeError, AttributeError) as e:
            logger.error("[VISION] UnifiedStateDetector failed: %s", e)
            success = False
        except Exception as e:
            logger.exception("[VISION] Unexpected UnifiedStateDetector error: %s", e)
            raise

        # OCR State Detector (optional)
        try:
            from pylaai_real.ocr_state_detector import OCRStateDetector
            self.ocr_detector = OCRStateDetector()
            logger.info("[VISION] OCR State Detector initialized")
        except (ImportError, ModuleNotFoundError, TypeError) as e:
            logger.debug("[VISION] OCR State Detector unavailable: %s", e)

        # Diagnostic overlay
        if self._overlay_enabled:
            try:
                self.diagnostic_overlay = DiagnosticOverlay(status_func=lambda: {})
                logger.info("[VISION] Diagnostic overlay initialized")
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.debug("[VISION] Diagnostic overlay unavailable: %s", e)

        # ESP Overlay
        try:
            from core.esp_overlay import ESPOverlay
            emulator_title = getattr(emulator_controller, "window_title", "BlueStacks") if emulator_controller else "BlueStacks"
            self.esp_overlay = ESPOverlay(window_title=emulator_title)
            logger.info("[VISION] ESPOverlay initialized")
        except (ImportError, ModuleNotFoundError, ConnectionError, ValueError, TypeError, RuntimeError, OSError) as e:
            logger.debug("[VISION] ESPOverlay unavailable: %s", e)

        return success

    def start(self) -> bool:
        """Start vision threads / overlays."""
        self._running = True
        if self.diagnostic_overlay:
            try:
                self.diagnostic_overlay.start()
            except (ConnectionError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[VISION] Failed to start diagnostic overlay: %s", e)
        return True

    def stop(self) -> bool:
        """Stop vision components."""
        self._running = False
        if self.diagnostic_overlay:
            try:
                self.diagnostic_overlay.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug("[VISION] Failed to stop diagnostic overlay: %s", e)
        if self.esp_overlay:
            try:
                self.esp_overlay.stop()
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug("[VISION] Failed to stop ESP overlay: %s", e)
        return True

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self) -> bool:
        """Load YOLO / trained models."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("[VISION] ultralytics not installed")
            return False

        model_loaded = False
        candidates = [
            ("latest_trained", self._find_latest_trained()),
            ("brawlstars_yolov8_8class", self.models_path / "brawlstars_yolov8_8class.pt"),
            ("brawlstars_yolov8", self.models_path / "brawlstars_yolov8.pt"),
            ("main_info", self.models_path / "main_info.pt"),
            ("yolov8n", self.models_path / "yolov8n.pt"),
        ]

        for name, path in candidates:
            if path is None or not path.exists():
                continue
            try:
                real_model = YOLO(str(path))
                self.detect_main = Detect(model=real_model, conf=0.40)
                self.detect_enemies = self.detect_main
                logger.info("[VISION] Loaded model: %s", path.name)
                model_loaded = True
                break
            except (FileNotFoundError, ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.warning("[VISION] Failed to load %s: %s", name, e)

        if not model_loaded:
            logger.error("[VISION] NO models loaded. Vision will not work.")
        return model_loaded

    def _find_latest_trained(self) -> Optional[Path]:
        try:
            runs = sorted(self.models_path.glob("yolo/*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            return runs[0] if runs else None
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError):
            return None

    # ------------------------------------------------------------------
    # Status / Health
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "screenshot_ok": self.screenshot is not None,
            "detector_ok": self.detect_main is not None,
            "unified_detector_ok": self.unified_detector is not None,
            "ocr_ok": self.ocr_detector is not None,
            "overlay_ok": self.diagnostic_overlay is not None,
            "esp_ok": self.esp_overlay is not None,
        }

    def health_check(self) -> Dict[str, Any]:
        issues = []
        if self.detect_main is None:
            issues.append("no_detector")
        if self.screenshot is None:
            issues.append("no_screenshot")
        return {"healthy": len(issues) == 0, "issues": issues}

    def get_detection_snapshot(self) -> Dict[str, Any]:
        detections = []
        vision_stats = {}
        try:
            if self.detect_main and hasattr(self.detect_main, "get_raw_detections"):
                detections = self.detect_main.get_raw_detections()
            if self.detect_main and hasattr(self.detect_main, "get_vision_stats"):
                vision_stats = self.detect_main.get_vision_stats()
        except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
            logger.debug("[VISION] Detection snapshot error: %s", e)
        return {"detections": detections, "vision_stats": vision_stats}

    def update_resolution(self, w: int, h: int) -> None:
        self._window_w = w
        self._window_h = h
        if self.unified_detector and hasattr(self.unified_detector, "update_resolution"):
            try:
                self.unified_detector.update_resolution(w, h)
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError) as e:
                logger.error("[VISION] Failed to update UnifiedStateDetector: %s", e)
            except Exception as e:
                logger.exception("[VISION] Unexpected update_resolution error: %s", e)
                raise
