"""
semi_supervised_trainer.py

Pseudo-labeling based semi-supervised training helpers.

This module creates pseudo-labels from a teacher model for unlabeled images,
letting the project reduce annotation effort while improving coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import cv2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PseudoLabelConfig:
    confidence_threshold: float = 0.65
    iou_threshold: float = 0.5
    max_images: Optional[int] = None


class SemiSupervisedTrainer:
    """Generate pseudo labels using a teacher model."""

    def __init__(self, teacher_model, config: Optional[PseudoLabelConfig] = None):
        self.teacher_model = teacher_model
        self.config = config or PseudoLabelConfig()

    def _iter_images(self, input_dir: Path) -> Iterable[Path]:
        for path in sorted(input_dir.rglob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                yield path

    def _extract_detections(self, predictions) -> List[Tuple[int, float, float, float, float]]:
        detections: List[Tuple[int, float, float, float, float]] = []
        for result in predictions:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                if conf < self.config.confidence_threshold:
                    continue
                cls_id = int(box.cls[0]) if hasattr(box, "cls") else 0
                xyxy = box.xyxy[0].tolist()
                detections.append((cls_id, *xyxy))
        return detections

    @staticmethod
    def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        return cx, cy, w, h

    def generate_pseudo_labels(self, unlabeled_dir: Path, output_dir: Path) -> Dict[str, int]:
        """Generate YOLO labels for unlabeled images."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"images": 0, "labels": 0, "filtered": 0}
        for idx, image_path in enumerate(self._iter_images(Path(unlabeled_dir))):
            if self.config.max_images is not None and idx >= self.config.max_images:
                break

            image = cv2.imread(str(image_path))
            if image is None:
                stats["filtered"] += 1
                continue

            predictions = self.teacher_model.predict(
                source=image,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                verbose=False,
            )
            detections = self._extract_detections(predictions)
            if not detections:
                stats["filtered"] += 1
                continue

            h, w = image.shape[:2]
            label_path = output_dir / f"{image_path.stem}.txt"
            with open(label_path, "w", encoding="utf-8") as handle:
                for cls_id, x1, y1, x2, y2 in detections:
                    cx, cy, bw, bh = self._xyxy_to_yolo(x1, y1, x2, y2, w, h)
                    handle.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            stats["images"] += 1
            stats["labels"] += len(detections)

        return stats
