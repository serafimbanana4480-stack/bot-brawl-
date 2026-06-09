"""
core/subsystems/vision_subsystem.py

VisionSubsystem: YOLO model loading, state detection, screenshot analysis,
AutoCalibrator, OCR, and debug visualizer setup.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from wrapper import PylaAIEnhanced

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class VisionSubsystem:
    """Manages vision models, detectors, and visual analysis."""

    def __init__(
        self,
        wrapper: "PylaAIEnhanced",
        models_path: Path,
        images_path: Path,
        diagnostic_mode: bool,
        central_config: dict,
    ):
        self.wrapper = wrapper
        self.models_path = models_path
        self.images_path = images_path
        self.diagnostic_mode = diagnostic_mode
        self.central_config = central_config
        self.detect_main: Optional[Any] = None
        self.detect_enemies: Optional[Any] = None
        self.unified_detector: Optional[Any] = None
        self.ocr_detector: Optional[Any] = None
        self.auto_calibrator: Optional[Any] = None
        self.debug_visualizer: Optional[Any] = None
        self.debug_integration: Optional[Any] = None
        self._debug_mode_enabled = bool(
            central_config.get("debug_visualizer", False)
            or __import__("os").getenv("PYLAAI_DEBUG_VISUAL", "0") == "1"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Load vision models and initialize detectors."""
        self._load_trained_models()

        # Phase 9: AutoCalibrator
        try:
            from pylaai_real.auto_calibrator import AutoCalibrator

            self.auto_calibrator = AutoCalibrator(
                templates_dir=self.images_path / "templates"
            )
            logger.info("[WRAPPER] AutoCalibrator inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] AutoCalibrator indisponível: {e}")

        # Phase 9: OCR State Detector
        try:
            from pylaai_real.ocr_state_detector import OCRStateDetector

            self.ocr_detector = OCRStateDetector()
            logger.info("[WRAPPER] OCR State Detector inicializado")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"[WRAPPER] OCR State Detector indisponível: {e}")

        # Phase 9: Debug Visualizer
        if self._debug_mode_enabled:
            try:
                from pylaai_real.debug_visualizer import DebugVisualizer, DebugMode

                self.debug_visualizer = DebugVisualizer(mode=DebugMode.DETAILED)
                logger.info("[WRAPPER] Debug Visualizer inicializado")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[WRAPPER] Debug Visualizer indisponível: {e}")

        # Initialize UnifiedStateDetector (needs window size from wrapper)
        window_w, window_h = self._get_resolution()
        try:
            from pylaai_real.unified_state_detector import UnifiedStateDetector

            self.unified_detector = UnifiedStateDetector(
                self.images_path,
                window_w=window_w,
                window_h=window_h,
                ocr_detector=self.ocr_detector,
            )
            logger.info(f"[WRAPPER] UnifiedStateDetector inicializado ({window_w}x{window_h})")
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"[WRAPPER] UnifiedStateDetector init falhou: {e}")
            return False

        # Sync back to wrapper for backward compatibility
        self.wrapper.detect_main = self.detect_main
        self.wrapper.detect_enemies = self.detect_enemies
        self.wrapper.unified_detector = self.unified_detector
        self.wrapper.ocr_detector = self.ocr_detector
        self.wrapper.auto_calibrator = self.auto_calibrator
        self.wrapper.debug_visualizer = self.debug_visualizer
        self.wrapper.debug_integration = self.debug_integration
        return True

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if self.debug_visualizer:
            try:
                self.debug_visualizer.stop()
                logger.info("[CLEANUP] Debug Visualizer parado")
            except (RuntimeError, AttributeError, OSError) as e:
                logger.debug(f"[CLEANUP] Falha ao parar debug visualizer: {e}")

    def cleanup(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_trained_models(self) -> bool:
        if not HAS_YOLO:
            logger.error("ultralytics not installed. pip install ultralytics")
            return False

        trained_models = {
            "latest_trained": None,
            "brawlstars_yolov8_8class": self.models_path / "brawlstars_yolov8_8class.pt",
            "brawlstars_yolov8": self.models_path / "brawlstars_yolov8.pt",
            "main_info": self.models_path / "main_info.pt",
            "brawler_id": self.models_path / "brawler_id.pt",
        }

        try:
            yolo_runs = sorted(
                self.models_path.glob("yolo/*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if yolo_runs:
                trained_models["latest_trained"] = yolo_runs[0]
                logger.debug(f"[MODEL] Discovered latest trained model: {yolo_runs[0]}")
        except (FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, AttributeError, OSError, IOError):
            pass

        generic_models = {
            "yolov8n": self.models_path / "yolov8n.pt",
            "yolov8m": self.models_path / "yolov8m.pt",
        }

        model_loaded = False

        # Latest trained
        latest_path = trained_models.get("latest_trained")
        if latest_path and latest_path.exists():
            for attempt in range(1, 4):
                try:
                    real_model = YOLO(str(latest_path))
                    try:
                        import torch

                        if torch.cuda.is_available():
                            real_model.to("cuda")
                            logger.info(f"[MODEL] Model moved to GPU: {torch.cuda.get_device_name(0)}")
                    except ImportError:
                        logger.debug("[MODEL] PyTorch/CUDA not available - using CPU")
                    from core.class_registry import get_schema

                    schema = "core"
                    expected = get_schema(schema)
                    actual_classes = set(real_model.names.values()) if hasattr(real_model, "names") else set()
                    expected_classes = set(expected.values())
                    if expected_classes.issubset(actual_classes):
                        from pylaai_real.detect import Detect

                        self.detect_main = Detect(model=real_model, classes=expected, conf=0.40)
                        self.detect_enemies = self.detect_main
                        logger.info(
                            f"[MODEL] Loaded LATEST TRAINED model: {latest_path.name} "
                            f"(classes: {actual_classes}, attempt={attempt})"
                        )
                        model_loaded = True
                        break
                    else:
                        logger.warning(
                            f"[MODEL] Latest trained model missing classes {expected_classes - actual_classes}, "
                            f"got: {actual_classes}"
                        )
                except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                    logger.error(f"[MODEL] Failed to load latest trained model {latest_path} (attempt {attempt}/3): {e}")
                    if attempt < 3:
                        time.sleep(0.5)

        # BrawlStarsBot trained model
        if not model_loaded:
            brawlstars_path = trained_models.get("brawlstars_yolov8_8class") or trained_models.get("brawlstars_yolov8")
            if brawlstars_path and brawlstars_path.exists():
                for attempt in range(1, 4):
                    try:
                        real_model = YOLO(str(brawlstars_path))
                        try:
                            import torch

                            if torch.cuda.is_available():
                                real_model.to("cuda")
                                logger.info(f"[MODEL] Model moved to GPU: {torch.cuda.get_device_name(0)}")
                        except ImportError:
                            logger.debug("[MODEL] PyTorch/CUDA not available - using CPU")
                        from core.class_registry import get_schema, get_canonical

                        schema = "extended" if "8class" in brawlstars_path.name else "core"
                        expected_classes_raw = get_schema(schema).values()
                        expected_classes = {get_canonical(name) for name in expected_classes_raw}
                        actual_classes = set(real_model.names.values()) if hasattr(real_model, "names") else set()
                        if expected_classes.issubset(actual_classes):
                            from pylaai_real.detect import Detect

                            self.detect_main = Detect(model=real_model, classes=get_schema(schema), conf=0.40)
                            self.detect_enemies = self.detect_main
                            logger.info(
                                f"[MODEL] Loaded BrawlStarsBot TRAINED model: {brawlstars_path.name} "
                                f"(classes: {actual_classes}, attempt={attempt})"
                            )
                            model_loaded = True
                            break
                        else:
                            missing = expected_classes - actual_classes
                            logger.warning(f"[MODEL] BrawlStarsBot model missing classes {missing}, got: {actual_classes}")
                            from pylaai_real.detect import Detect

                            self.detect_main = Detect(model=real_model, classes=get_schema(schema), conf=0.40)
                            self.detect_enemies = self.detect_main
                            model_loaded = True
                            break
                    except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                        logger.error(
                            f"[MODEL] Failed to load BrawlStarsBot model {brawlstars_path} (attempt {attempt}/3): {e}"
                        )
                        if attempt < 3:
                            time.sleep(0.5)

        # PylaAI trained models
        if not model_loaded:
            main_model_path = trained_models.get("main_info")
            if main_model_path and main_model_path.exists():
                try:
                    real_model = YOLO(str(main_model_path))
                    from pylaai_real.detect import Detect

                    self.detect_main = Detect(
                        model=real_model,
                        classes={0: "enemy", 1: "player", 2: "teammate"},
                        conf=0.1,
                    )
                    self.detect_enemies = self.detect_main
                    logger.info(f"Loaded PylaAI TRAINED model: {main_model_path.name}")
                    model_loaded = True
                except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                    logger.error(f"Failed to load PylaAI model {main_model_path}: {e}")

        # Fallback generic models
        if not model_loaded:
            for name, path in generic_models.items():
                if path.exists():
                    try:
                        real_model = YOLO(str(path))
                        from pylaai_real.detect import Detect

                        self.detect_main = Detect(
                            model=real_model,
                            classes={0: "person", 1: "brawler", 2: "object"},
                            conf=0.1,
                        )
                        self.detect_enemies = self.detect_main
                        logger.warning(f"Loaded GENERIC model: {path.name} (COCO-80, not trained for Brawl Stars)")
                        model_loaded = True
                        break
                    except (ImportError, ModuleNotFoundError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
                        logger.error(f"Failed to load generic model {path}: {e}")

        # Last resort: download yolov8n
        if not model_loaded:
            try:
                from model_downloader import get_model_downloader

                downloader = get_model_downloader()
                model_path = downloader.get_model_path("yolov8n")
                if not model_path:
                    logger.info("Downloading YOLOv8n as fallback...")
                    res = downloader.download_model("yolov8n")
                    if res.get("success"):
                        model_path = downloader.get_model_path("yolov8n")
                if model_path:
                    real_model = YOLO(str(model_path))
                    from pylaai_real.detect import Detect

                    self.detect_main = Detect(
                        model=real_model,
                        classes={0: "person", 1: "brawler", 2: "object"},
                        conf=0.1,
                    )
                    self.detect_enemies = self.detect_main
                    logger.warning("Using downloaded YOLOv8n with confidence threshold 0.1")
                    model_loaded = True
            except (ImportError, ModuleNotFoundError, FileNotFoundError, PermissionError, ValueError, TypeError, RuntimeError, OSError, IOError) as e:
                logger.error(f"Model download fallback failed: {e}")

        if not model_loaded:
            logger.error("NO models loaded. Vision will not work.")
        return model_loaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_resolution(self):
        if hasattr(self.wrapper, "resolution_manager") and self.wrapper.resolution_manager is not None:
            try:
                return self.wrapper.resolution_manager.actual_resolution
            except (ValueError, TypeError, RuntimeError, AttributeError, OSError):
                pass
        if hasattr(self.wrapper, "emulator_subsystem"):
            return self.wrapper.emulator_subsystem.get_safe_resolution()
        return (1920, 1080)
