"""
vision_engine.py

Sistema de visão computacional YOLOv8/YOLO11 para Brawl Stars.

Supports both YOLOv8 and YOLO11 models with automatic model switching
and YOLO-World fallback for open-vocabulary detection.

NOTE: TensorRT support is implemented but requires:
  - NVIDIA GPU
  - TensorRT SDK installation
  - Trained model in .pt format
  Performance improvements are only achievable after these prerequisites are met.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time

# Import do otimizador criado anteriormente
try:
    from .tensorrt_optimizer import TensorRTOptimizer
except ImportError:
    # Caso corra fora do pacote
    try:
        from tensorrt_optimizer import TensorRTOptimizer
    except ImportError:
        TensorRTOptimizer = None

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Uma deteção de objeto"""
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int


@dataclass
class VisionConfig:
    """Configuração do sistema de visão"""
    confidence_threshold: float = 0.65
    iou_threshold: float = 0.45
    input_size: int = 640
    use_half_precision: bool = True
    max_detections: int = 100
    
    # Configurações TensorRT (Fase 1 - Otimização)
    use_tensorrt: bool = True
    tensorrt_workspace: int = 1 << 30 # 1GB
    
    # Classes específicas do Brawl Stars
    detect_enemies: bool = True
    detect_walls: bool = True
    detect_bushes: bool = True
    detect_powerups: bool = True
    
    # Ajustes de performance
    skip_frames: int = 0
    inference_device: str = "auto"


class YOLOVisionEngine:
    """
    YOLO Vision Engine for Brawl Stars (YOLOv8/YOLO11).

    Loads trained models from model_registry.json, validates classes,
    and runs inference. Supports YOLOv8, YOLO11, and YOLO-World models.
    Supports multiple specialized models with automatic fallback.
    """

    BRAWL_STARS_CLASSES = {
        "enemy", "teammate", "player", "wall", "bush",
        "powerup", "box", "bullet", "super_indicator",
        "health_bar", "joystick", "attack_button"
    }

    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.models: Dict[str, Any] = {}
        self.trt_engines: Dict[str, Any] = {}
        self.device = self._get_device()
        self.frame_count = 0
        self.is_initialized = False
        self.loaded_classes: set = set()
        
        # Model switcher for YOLO11 + YOLO-World
        self.model_switcher = None
        self.use_model_switcher = False

    def _get_device(self) -> str:
        if self.config.inference_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.inference_device

    def _get_valid_models(self, models_dir: Path) -> List[Tuple[str, Path]]:
        """
        Scan models_dir and return only models validated by registry.
        Returns list of (model_key, path) tuples.
        """
        registry_path = models_dir / "model_registry.json"
        if not registry_path.exists():
            logger.warning(f"No model_registry.json found at {registry_path}")
            return []

        try:
            import json
            registry = json.loads(registry_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return []

        valid = []
        for name, meta in registry.get("models", {}).items():
            if meta.get("status") != "valid":
                continue
            path = models_dir / name
            if not path.exists():
                logger.warning(f"Registry lists {name} as valid but file not found")
                continue
            # Check for Brawl Stars classes
            classes = meta.get("classes", [])
            class_set = {c.lower() for c in classes}
            overlap = class_set & self.BRAWL_STARS_CLASSES
            if len(overlap) >= 2:
                valid.append((name, path))
                logger.info(f"Valid model: {name} (classes: {list(overlap)})")
            else:
                logger.warning(f"Model {name} has no Brawl Stars classes: {classes[:5]}")

        return valid

    def _get_pending_models(self, models_dir: Path) -> List[Tuple[str, Path]]:
        """
        Get pending models from registry (YOLO-World, etc.).
        Returns list of (model_key, path) tuples.
        """
        registry_path = models_dir / "model_registry.json"
        if not registry_path.exists():
            return []

        try:
            import json
            registry = json.loads(registry_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return []

        pending = []
        for name, meta in registry.get("pending_models", {}).items():
            if meta.get("status") != "pending":
                continue
            path = models_dir / name
            if path.exists():
                pending.append((name, path))
                logger.info(f"Pending model: {name}")
            else:
                logger.warning(f"Pending model {name} not found at {path}")

        return pending

    def _setup_model_switcher(self) -> None:
        """Setup model switcher if YOLO11 and YOLO-World are available."""
        # Find YOLO11 model
        yolo11_model = None
        yolo_world_model = None

        for name, model in self.models.items():
            if "yolo11" in name.lower():
                yolo11_model = model
            elif "world" in name.lower():
                yolo_world_model = model

        if yolo11_model and yolo_world_model:
            try:
                # Import model switcher
                from training.yolo11.model_switcher import ModelSwitcher
                self.model_switcher = ModelSwitcher(
                    primary_model=yolo11_model,
                    fallback_model=yolo_world_model,
                    confidence_threshold=0.3
                )
                self.use_model_switcher = True
                logger.info("Model switcher enabled (YOLO11 + YOLO-World)")
            except ImportError:
                logger.warning("Model switcher not available, using single model")
                self.use_model_switcher = False

    def load_models(self, models_dir: Path) -> bool:
        """Load validated YOLO models from models directory (YOLOv8, YOLO11, YOLO-World)."""
        try:
            logger.info(f"Loading models from: {models_dir}")

            valid_models = self._get_valid_models(models_dir)
            pending_models = self._get_pending_models(models_dir)

            if not valid_models and not pending_models:
                logger.error("No valid or pending Brawl Stars models found in registry.")
                logger.error("Train a model first: python -m backend.brawl_bot.train_yolo")
                self.is_initialized = False
                return False

            # Load valid models
            for model_name, pt_path in valid_models:
                try:
                    from ultralytics import YOLO
                    model = YOLO(str(pt_path))
                    if self.device == "cuda" and self.config.use_half_precision:
                        model.to(self.device).half()
                    else:
                        model.to(self.device)

                    # Store by first valid class name as key
                    classes = list(model.names.values()) if hasattr(model, "names") else []
                    self.loaded_classes.update(c.lower() for c in classes)
                    self.models[model_name] = model
                    logger.info(f"Loaded {model_name} on {self.device}")

                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")

            # Load pending models (YOLO-World, etc.)
            for model_name, pt_path in pending_models:
                try:
                    from ultralytics import YOLO
                    model = YOLO(str(pt_path))
                    if self.device == "cuda" and self.config.use_half_precision:
                        model.to(self.device).half()
                    else:
                        model.to(self.device)

                    self.models[model_name] = model
                    logger.info(f"Loaded pending model {model_name} on {self.device}")

                except Exception as e:
                    logger.error(f"Failed to load pending model {model_name}: {e}")

            # Setup model switcher if we have both YOLO11 and YOLO-World
            self._setup_model_switcher()

            self.is_initialized = len(self.models) > 0
            if self.is_initialized:
                logger.info(f"Vision engine ready: {len(self.models)} model(s), classes: {sorted(self.loaded_classes & self.BRAWL_STARS_CLASSES)}")
            return self.is_initialized

        except Exception as e:
            logger.error(f"Fatal error loading models: {e}")
            return False

    def detect(self, screenshot: np.ndarray) -> List[Detection]:
        """Run inference and return detections with actual class names from model."""
        if not self.is_initialized:
            return []

        detections = []

        # Use model switcher if available
        if self.use_model_switcher and self.model_switcher:
            try:
                result = self.model_switcher.detect(
                    screenshot,
                    conf_threshold=self.config.confidence_threshold
                )
                
                if result and result.detections:
                    for r in result.detections:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            
                            # Get class name from the model that produced the result
                            if result.model_name and result.model_name in self.models:
                                model = self.models[result.model_name]
                                class_name = model.names.get(cls_id, "unknown") if hasattr(model, "names") else "unknown"
                            else:
                                class_name = "unknown"

                            detections.append(Detection(
                                class_name=class_name,
                                confidence=conf,
                                x=int(x1), y=int(y1),
                                width=int(x2 - x1), height=int(y2 - y1),
                                center_x=int((x1 + x2) / 2), center_y=int((y1 + y2) / 2)
                            ))
                    
                    if result.fallback_used:
                        logger.debug(f"Used fallback model: {result.model_name}")
                    
                    return detections
                
            except Exception as e:
                logger.error(f"Model switcher error: {e}, falling back to standard detection")

        # Standard detection (fallback or if switcher not enabled)
        for model_name, model in self.models.items():
            try:
                results = model(
                    screenshot,
                    conf=self.config.confidence_threshold,
                    device=self.device,
                    verbose=False,
                    iou=self.config.iou_threshold,
                )
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        # Get actual class name from model, not hardcoded key
                        cls_id = int(box.cls[0])
                        class_name = model.names.get(cls_id, "unknown") if hasattr(model, "names") else "unknown"

                        detections.append(Detection(
                            class_name=class_name,
                            confidence=conf,
                            x=int(x1), y=int(y1),
                            width=int(x2 - x1), height=int(y2 - y1),
                            center_x=int((x1 + x2) / 2), center_y=int((y1 + y2) / 2)
                        ))
            except Exception as e:
                logger.error(f"Inference error on {model_name}: {e}")

        return detections
