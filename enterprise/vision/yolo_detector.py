"""YOLOv8 Detector - Real YOLO detection for Brawl Stars (NO MOCK DATA)"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger("yolo_detector")


class YOLODetectionError(Exception):
    """Raised when YOLO detection fails."""
    pass


class YOLOv8Detector:
    """
    Real YOLO detection for Brawl Stars.
    NO MOCK DATA - Requires actual model and ultralytics installed.
    """

    def __init__(self, model_path: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = "auto"):
        if model_path is None:
            model_path = self._find_brawlstars_model()

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._loaded = False
        self.classes = {}

        self._validate_requirements()

    def _validate_requirements(self):
        """Validate that ultralytics is installed and model exists."""
        try:
            import ultralytics
        except ImportError:
            raise YOLODetectionError(
                "Ultralytics not installed. Install with: pip install ultralytics"
            )

        if self.model_path and not os.path.exists(self.model_path):
            raise YOLODetectionError(
                f"Model file not found: {self.model_path}"
            )

    def _find_brawlstars_model(self) -> str:
        """Find the trained Brawl Stars model."""
        possible_paths = [
            "c:/Users/rodri/Desktop/bot brawl/models/brawlstars_yolov8.pt",
            "c:/Users/rodri/Desktop/bot brawl/models/main_info.pt",
            "models/brawlstars_yolov8.pt",
            "models/main_info.pt",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"[YOLOv8Detector] Found model: {path}")
                return path

        raise YOLODetectionError(
            "No Brawl Stars model found. Please train a model or download one."
        )

    def load(self) -> bool:
        """Load the YOLO model. RAISES ERROR if fails - NO MOCK."""
        try:
            from ultralytics import YOLO

            logger.info(f"[YOLOv8Detector] Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)

            if self.device == "auto":
                self.device = self._detect_device()

            self.model.to(self.device)
            logger.info(f"[YOLOv8Detector] Model loaded on device: {self.device}")

            if hasattr(self.model, 'names') and self.model.names:
                self.classes = self.model.names
            else:
                self.classes = {0: "Player", 1: "Enemy", 2: "Bush", 3: "Cubebox"}

            logger.info(f"[YOLOv8Detector] Classes: {self.classes}")
            self._loaded = True
            return True

        except Exception as e:
            raise YOLODetectionError(f"Failed to load YOLO model: {e}")

    def _detect_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except (ImportError, RuntimeError):
            pass
        return "cpu"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using real YOLO model.
        Returns real detections - NO MOCK DATA.
        """
        if not self._loaded:
            raise YOLODetectionError("Model not loaded. Call load() first.")

        if frame is None or len(frame) == 0:
            raise YOLODetectionError("Invalid frame provided")

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                class_name = self.classes.get(cls, f"class_{cls}")

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": class_name,
                })

        logger.debug(f"[YOLOv8Detector] Detected {len(detections)} objects")
        return detections

    def detect_batch(self, frames: List[np.ndarray],
                    batch_size: int = 8) -> List[List[Dict[str, Any]]]:
        """Detect objects in batch of frames. NO MOCK."""
        if not self._loaded:
            raise YOLODetectionError("Model not loaded. Call load() first.")

        if not frames:
            return []

        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            results = self.model(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            for result in results:
                detections = []
                boxes = result.boxes

                for j in range(len(boxes)):
                    box = boxes[j]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    class_name = self.classes.get(cls, f"class_{cls}")

                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": class_name,
                    })

                all_detections.append(detections)

        return all_detections

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "model_path": self.model_path,
            "loaded": self._loaded,
            "device": self.device,
            "classes": self.classes,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }
