"""
professional_training_pipeline.py

Pipeline profissional de treino YOLO para Brawl Stars.
Unifica classes, captura dados reais, auto-labela, e treina modelo.

Classes padronizadas (compatíveis com o modelo actual brawlstars_yolov8.pt):
    0: Player   - Jogador controlado (indicado por seta/joystick)
    1: Bush     - Arbustos (cover vegetal)
    2: Enemy    - Inimigos (brawlers adversários)
    3: Cubebox  - Caixas de gemas/power cubes

Usage:
    python training/professional_training_pipeline.py --capture --label --train --epochs 50
    python training/professional_training_pipeline.py --train-only --epochs 100
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("training_pipeline")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(name)s | %(message)s")

# === STANDARDIZED CLASSES (must match wrapper.py and config.json) ===
STANDARD_CLASSES = {
    0: "Player",
    1: "Bush",
    2: "Enemy",
    3: "Cubebox",
}
NC = len(STANDARD_CLASSES)

# === HSV COLOR RANGES FOR BRAWL STARS ELEMENTS ===
BRAWLSTARS_HSV = {
    # Player: typically has a blue arrow/indicator above
    "player_arrow_blue": ((100, 100, 100), (130, 255, 255)),
    # Enemy: typically has a red arrow/indicator above
    "enemy_arrow_red1": ((0, 100, 100), (10, 255, 255)),
    "enemy_arrow_red2": ((170, 100, 100), (180, 255, 255)),
    # Bush: dark green, low saturation
    "bush_dark_green": ((35, 30, 30), (85, 180, 120)),
    # Cubebox: distinctive purple/blue glow
    "cubebox_purple": ((130, 50, 80), (170, 255, 255)),
    "cubebox_blue": ((100, 80, 80), (130, 255, 255)),
    # Health bars (for finding characters)
    "health_green": ((35, 80, 80), (85, 255, 255)),
    "health_red": ((0, 100, 80), (10, 255, 255)),
    # Attack button (orange/yellow)
    "attack_orange": ((10, 150, 150), (25, 255, 255)),
    # Super button (yellow glow)
    "super_yellow": ((20, 150, 150), (35, 255, 255)),
}


class BrawlStarsAutoLabeler:
    """
    Auto-labeler profissional para Brawl Stars.
    Detecta elementos do jogo usando heurísticas visuais específicas.
    """

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self.min_contour_area = 200
        self.max_contour_area = 80000

    def _hsv_mask(self, image: np.ndarray, lower: Tuple, upper: Tuple) -> np.ndarray:
        """Create HSV mask with optional dual-range for red."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(lower), np.array(upper))

    def _find_contours_filtered(self, mask: np.ndarray,
                                 min_area: int = 200,
                                 max_area: int = 80000,
                                 aspect_min: float = 0.2,
                                 aspect_max: float = 5.0) -> List[Tuple[int, int, int, int]]:
        """Find contours and filter by area and aspect ratio."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            if aspect_min <= aspect <= aspect_max:
                bboxes.append((x, y, w, h))
        return bboxes

    def _to_yolo(self, bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert pixel bbox to YOLO normalized format."""
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        # Clamp to [0, 1]
        return (
            max(0, min(1, x_center)),
            max(0, min(1, y_center)),
            max(0, min(1, norm_w)),
            max(0, min(1, norm_h)),
        )

    def _expand_bbox(self, bbox: Tuple[int, int, int, int], factor: float = 1.5,
                     img_w: int = 640, img_h: int = 640) -> Tuple[int, int, int, int]:
        """Expand bbox by factor to capture full character."""
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        new_w = int(w * factor)
        new_h = int(h * factor)
        new_x = max(0, cx - new_w // 2)
        new_y = max(0, cy - new_h // 2)
        # Clamp to image bounds
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        return (new_x, new_y, new_w, new_h)

    def detect_players_and_enemies(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """
        Detect Player and Enemy by finding health bars + arrow indicators.
        Player has blue arrow, Enemy has red arrow.
        Returns list of (class_id, yolo_bbox).
        """
        h, w = image.shape[:2]
        results = []

        # Find health bars (green strips above characters)
        mask_green = self._hsv_mask(image, *BRAWLSTARS_HSV["health_green"])
        health_bars = self._find_contours_filtered(mask_green, min_area=100, max_area=5000,
                                                     aspect_min=2.0, aspect_max=15.0)

        for hx, hy, hw, hh in health_bars:
            # Look above the health bar for arrow indicator
            arrow_y_start = max(0, hy - 30)
            arrow_y_end = hy
            arrow_x_start = max(0, hx - 10)
            arrow_x_end = min(w, hx + hw + 10)
            arrow_region = image[arrow_y_start:arrow_y_end, arrow_x_start:arrow_x_end]

            if arrow_region.size == 0:
                continue

            # Check for blue arrow (Player)
            mask_blue = self._hsv_mask(arrow_region, *BRAWLSTARS_HSV["player_arrow_blue"])
            blue_pixels = cv2.countNonZero(mask_blue)

            # Check for red arrow (Enemy)
            mask_red1 = self._hsv_mask(arrow_region, *BRAWLSTARS_HSV["enemy_arrow_red1"])
            mask_red2 = self._hsv_mask(arrow_region, *BRAWLSTARS_HSV["enemy_arrow_red2"])
            red_pixels = cv2.countNonZero(mask_red1) + cv2.countNonZero(mask_red2)

            # Determine class based on arrow color
            if blue_pixels > red_pixels and blue_pixels > 5:
                cls_id = 0  # Player
            elif red_pixels > blue_pixels and red_pixels > 5:
                cls_id = 2  # Enemy
            else:
                # No clear arrow - try position heuristic
                # Player is usually near joystick (bottom-left)
                health_center_x = (hx + hw / 2) / w
                health_center_y = (hy + hh / 2) / h
                if health_center_x < 0.3 and health_center_y > 0.6:
                    cls_id = 0  # Player (near joystick)
                else:
                    cls_id = 2  # Assume enemy

            # Expand bbox to capture full character body
            char_bbox = (hx, hy - int(hh * 2), hw, int(hh * 5))
            char_bbox = self._expand_bbox(char_bbox, factor=1.2, img_w=w, img_h=h)
            yolo_bbox = self._to_yolo(char_bbox, w, h)
            results.append((cls_id, yolo_bbox))

        return results

    def detect_bushes(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """Detect bushes by dark green color segmentation."""
        h, w = image.shape[:2]
        mask = self._hsv_mask(image, *BRAWLSTARS_HSV["bush_dark_green"])
        bboxes = self._find_contours_filtered(mask, min_area=500, max_area=30000,
                                                aspect_min=0.3, aspect_max=4.0)
        results = []
        for bbox in bboxes:
            yolo_bbox = self._to_yolo(bbox, w, h)
            results.append((1, yolo_bbox))  # Bush = class 1
        return results

    def detect_cubeboxes(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """Detect gem boxes / power cube boxes by purple/blue glow."""
        h, w = image.shape[:2]
        mask1 = self._hsv_mask(image, *BRAWLSTARS_HSV["cubebox_purple"])
        mask2 = self._hsv_mask(image, *BRAWLSTARS_HSV["cubebox_blue"])
        mask = cv2.bitwise_or(mask1, mask2)
        bboxes = self._find_contours_filtered(mask, min_area=300, max_area=15000,
                                                aspect_min=0.5, aspect_max=2.0)
        results = []
        for bbox in bboxes:
            yolo_bbox = self._to_yolo(bbox, w, h)
            results.append((3, yolo_bbox))  # Cubebox = class 3
        return results

    def auto_label(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """
        Full auto-labeling pipeline.
        Returns list of (class_id, (x_center, y_center, width, height)) in YOLO format.
        """
        all_detections = []

        # Detect characters (Player/Enemy)
        try:
            characters = self.detect_players_and_enemies(image)
            all_detections.extend(characters)
        except Exception as e:
            logger.debug(f"Character detection failed: {e}")

        # Detect bushes
        try:
            bushes = self.detect_bushes(image)
            all_detections.extend(bushes)
        except Exception as e:
            logger.debug(f"Bush detection failed: {e}")

        # Detect cube boxes
        try:
            cubes = self.detect_cubeboxes(image)
            all_detections.extend(cubes)
        except Exception as e:
            logger.debug(f"Cubebox detection failed: {e}")

        # NMS: remove overlapping detections (keep highest priority class)
        all_detections = self._nms(all_detections, iou_threshold=0.4)

        return all_detections

    def _nms(self, detections: List[Tuple[int, Tuple[float, float, float, float]]],
             iou_threshold: float = 0.4) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """Simple Non-Maximum Suppression to remove overlapping bboxes."""
        if not detections:
            return detections

        # Sort by class priority (Player > Enemy > Bush > Cubebox)
        class_priority = {0: 0, 2: 1, 1: 2, 3: 3}
        detections.sort(key=lambda d: class_priority.get(d[0], 99))

        keep = []
        for i, (cls_i, (cx_i, cy_i, w_i, h_i)) in enumerate(detections):
            should_keep = True
            x1_i = cx_i - w_i / 2
            y1_i = cy_i - h_i / 2
            x2_i = cx_i + w_i / 2
            y2_i = cy_i + h_i / 2
            area_i = w_i * h_i

            for cls_k, (cx_k, cy_k, w_k, h_k) in keep:
                x1_k = cx_k - w_k / 2
                y1_k = cy_k - h_k / 2
                x2_k = cx_k + w_k / 2
                y2_k = cy_k + h_k / 2
                area_k = w_k * h_k

                # Calculate IoU
                inter_x1 = max(x1_i, x1_k)
                inter_y1 = max(y1_i, y1_k)
                inter_x2 = min(x2_i, x2_k)
                inter_y2 = min(y2_i, y2_k)

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                iou = inter_area / (area_i + area_k - inter_area + 1e-6)

                if iou > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                keep.append((cls_i, (cx_i, cy_i, w_i, h_i)))

        return keep

    def label_image(self, image_path: str, output_label_path: str) -> int:
        """Label a single image and save YOLO format label file."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot read image: {image_path}")
            return 0

        # Resize to target size
        image = cv2.resize(image, self.target_size)

        detections = self.auto_label(image)

        # Write label file
        with open(output_label_path, 'w') as f:
            for cls_id, (cx, cy, w, h) in detections:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        return len(detections)


class RealDataCapture:
    """Captures real game screenshots from the emulator for training."""

    def __init__(self, window_title: str = "BlueStacks App Player",
                 output_dir: str = "dataset/real_capture"):
        self.window_title = window_title
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_taker = None

    def _init_screenshot(self):
        """Initialize screenshot taker."""
        from pylaai_real.screenshot_taker import ScreenshotTaker
        self.screenshot_taker = ScreenshotTaker(self.window_title)
        if not self.screenshot_taker.find_window():
            logger.error(f"Window '{self.window_title}' not found")
            return False
        return True

    def capture_batch(self, num_frames: int = 100, interval: float = 0.5,
                      target_size: Tuple[int, int] = (640, 640)) -> int:
        """Capture a batch of screenshots from the emulator."""
        if self.screenshot_taker is None:
            if not self._init_screenshot():
                return 0

        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        captured = 0
        existing = len(list(images_dir.glob("*.png")))

        for i in range(num_frames):
            img = self.screenshot_taker.take()
            if img is not None:
                # Save full-res for template extraction
                fullres_path = images_dir / f"fullres_{existing + i:05d}.png"
                cv2.imwrite(str(fullres_path), img)

                # Save resized for YOLO training
                img_resized = cv2.resize(img, target_size)
                train_path = images_dir / f"frame_{existing + i:05d}.png"
                cv2.imwrite(str(train_path), img_resized)

                captured += 1
                if captured % 10 == 0:
                    logger.info(f"Captured {captured}/{num_frames} frames")
            else:
                logger.warning(f"Screenshot failed at frame {i}")

            time.sleep(interval)

        logger.info(f"Capture complete: {captured} frames saved to {images_dir}")
        return captured


def create_standard_data_yaml(dataset_dir: Path) -> Path:
    """Create data.yaml with standardized classes matching the current model."""
    data_config = {
        "path": str(dataset_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": STANDARD_CLASSES,
        "nc": NC,
    }

    yaml_path = dataset_dir / "data.yaml"
    try:
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    except ImportError:
        with open(yaml_path, 'w') as f:
            json.dump(data_config, f, indent=2)

    logger.info(f"Created data.yaml at {yaml_path} with {NC} classes: {list(STANDARD_CLASSES.values())}")
    return yaml_path


def prepare_training_dataset(raw_images_dir: Path, dataset_dir: Path,
                              train_ratio: float = 0.8,
                              labeler: Optional[BrawlStarsAutoLabeler] = None) -> int:
    """
    Prepare YOLO training dataset from raw images.
    Auto-labels images and splits into train/val/test.
    """
    if labeler is None:
        labeler = BrawlStarsAutoLabeler()

    # Create directory structure
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Find all images (various naming patterns)
    image_files = sorted(
        list(raw_images_dir.glob("frame_*.png")) +
        list(raw_images_dir.glob("real_capture_*.png")) +
        list(raw_images_dir.glob("fullres_*.png")) +
        list(raw_images_dir.glob("*.jpg")) +
        list(raw_images_dir.glob("image_*.jpg"))
    )

    if not image_files:
        logger.error(f"No images found in {raw_images_dir}")
        return 0

    logger.info(f"Found {len(image_files)} images to process")

    # Auto-label all images
    total_labels = 0
    labeled_images = []

    for img_path in image_files:
        label_path = img_path.parent / f"{img_path.stem}.txt"
        num_labels = labeler.label_image(str(img_path), str(label_path))
        total_labels += num_labels

        if num_labels > 0:
            labeled_images.append(img_path)
        else:
            # Still include unlabeled images (empty label files)
            labeled_images.append(img_path)

    logger.info(f"Auto-labeling complete: {total_labels} labels across {len(labeled_images)} images")

    # Split into train/val/test
    n = len(labeled_images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + 0.1))

    splits = {
        "train": labeled_images[:train_end],
        "val": labeled_images[train_end:val_end],
        "test": labeled_images[val_end:],
    }

    for split_name, files in splits.items():
        for img_path in files:
            # Copy image
            dst_img = dataset_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy label
            label_path = img_path.parent / f"{img_path.stem}.txt"
            dst_label = dataset_dir / split_name / "labels" / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, dst_label)
            else:
                dst_label.touch()

    logger.info(f"Dataset prepared: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Also merge with existing synthetic data if available
    synthetic_dir = Path("dataset/synthetic_expanded")
    if synthetic_dir.exists():
        _merge_synthetic_data(synthetic_dir, dataset_dir, labeler)

    # Create data.yaml
    create_standard_data_yaml(dataset_dir)

    return total_labels


def _merge_synthetic_data(synthetic_dir: Path, dataset_dir: Path,
                           labeler: BrawlStarsAutoLabeler) -> int:
    """Merge synthetic expanded data into the training dataset, remapping classes."""
    synth_images = sorted((synthetic_dir / "images").glob("*.png"))
    synth_labels = sorted((synthetic_dir / "labels").glob("*.txt"))

    # Class mapping from synthetic (5-class) to standard (4-class)
    # Synthetic: 0=player, 1=enemy, 2=obstacle, 3=powerup, 4=projectile
    # Standard:  0=Player, 1=Bush, 2=Enemy, 3=Cubebox
    CLASS_MAP = {
        0: 0,  # player -> Player
        1: 2,  # enemy -> Enemy
        2: 1,  # obstacle -> Bush (approximate)
        3: 3,  # powerup -> Cubebox (approximate)
        4: -1, # projectile -> skip (no standard class)
    }

    merged = 0
    train_images = dataset_dir / "train" / "images"
    train_labels = dataset_dir / "train" / "labels"

    for img_path in synth_images[:200]:  # Limit to 200 synthetic images
        label_path = synthetic_dir / "labels" / f"{img_path.stem}.txt"

        # Copy image
        dst_img = train_images / f"synth_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        # Remap labels
        dst_label = train_labels / f"synth_{img_path.stem}.txt"
        with open(label_path, 'r') as fin, open(dst_label, 'w') as fout:
            for line in fin:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_cls = int(parts[0])
                    new_cls = CLASS_MAP.get(old_cls, -1)
                    if new_cls >= 0:
                        fout.write(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        merged += 1

    logger.info(f"Merged {merged} synthetic images with class remapping")
    return merged


def train_model(data_yaml: Path, epochs: int = 50, batch_size: int = 16,
                img_size: int = 640, device: str = "cpu",
                pretrained_model: Optional[str] = None) -> Optional[str]:
    """
    Train YOLO model with the prepared dataset.
    Supports fine-tuning from existing model or training from COCO pretrained.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return None

    # Determine base model
    if pretrained_model and Path(pretrained_model).exists():
        logger.info(f"Fine-tuning from existing model: {pretrained_model}")
        model = YOLO(pretrained_model)
    else:
        # Check if we have yolov8n.pt locally
        local_yolov8n = Path("models/yolov8n.pt")
        if local_yolov8n.exists():
            logger.info(f"Training from local YOLOv8n: {local_yolov8n}")
            model = YOLO(str(local_yolov8n))
        else:
            logger.info("Training from YOLOv8n (will download)")
            model = YOLO("yolov8n.pt")

    output_dir = Path("runs/detect/professional_training")

    logger.info(f"Starting training:")
    logger.info(f"  Data: {data_yaml}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch: {batch_size}")
    logger.info(f"  Image size: {img_size}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Classes: {STANDARD_CLASSES}")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        erasing=0.4,
    )

    # Copy best model to models directory
    best_pt = output_dir / "weights" / "best.pt"
    if best_pt.exists():
        dest = Path("models/brawlstars_yolov8.pt")
        backup = Path("models/brawlstars_yolov8_backup.pt")
        if dest.exists():
            shutil.copy2(dest, backup)
            logger.info(f"Backed up existing model to {backup}")
        shutil.copy2(best_pt, dest)
        logger.info(f"New model saved to {dest}")

        # Validate the new model
        _validate_model(dest, data_yaml)

        return str(dest)
    else:
        logger.error("Training completed but best.pt not found!")
        return None


def _validate_model(model_path: Path, data_yaml: Path):
    """Validate the trained model on the test set."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))

        logger.info(f"Model classes: {model.names}")
        logger.info(f"Model nc: {model.model.nc}")

        # Run validation
        metrics = model.val(data=str(data_yaml), split="test")

        logger.info(f"Validation Results:")
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall: {metrics.box.mr:.4f}")

        # Per-class metrics
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            for i, name in STANDARD_CLASSES.items():
                if i < len(metrics.box.maps):
                    logger.info(f"  {name} mAP50: {metrics.box.maps[i]:.4f}")

    except Exception as e:
        logger.error(f"Validation failed: {e}")


def capture_real_templates(output_dir: str = "images"):
    """
    Capture real template images from the emulator for template matching.
    These replace the synthetic placeholder images.
    """
    from pylaai_real.screenshot_taker import ScreenshotTaker

    st = ScreenshotTaker("BlueStacks App Player")
    if not st.find_window():
        logger.error("Window not found for template capture")
        return

    img = st.take()
    if img is None:
        logger.error("Screenshot failed")
        return

    h, w = img.shape[:2]
    logger.info(f"Screenshot for templates: {w}x{h}")

    out = Path(output_dir)

    # Define template regions (1920x1080 coordinates)
    templates = {
        "attack_button.png": (1690, 790, 120, 120),   # x, y, w, h of region
        "super_button.png": (1390, 690, 100, 100),
        "gadget_button.png": (1450, 580, 80, 80),
        "play_button.png": (1730, 910, 150, 110),
        "joystick_area.png": (92, 710, 200, 200),
        "brawler_select.png": (760, 300, 400, 200),
        "thumbs_down.png": (860, 800, 200, 100),
    }

    for name, (tx, ty, tw, th) in templates.items():
        # Scale coordinates if image is not 1920x1080
        sx = w / 1920.0
        sy = h / 1080.0
        rx = int(tx * sx)
        ry = int(ty * sy)
        rw = int(tw * sx)
        rh = int(th * sy)

        # Clamp to image bounds
        rx = max(0, min(w - 1, rx))
        ry = max(0, min(h - 1, ry))
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)

        region = img[ry:ry + rh, rx:rx + rw]
        if region.size > 0:
            # Save the template
            path = out / name
            cv2.imwrite(str(path), region)
            logger.info(f"Saved template: {path} ({region.shape[1]}x{region.shape[0]})")
        else:
            logger.warning(f"Empty region for template {name}")


def main():
    parser = argparse.ArgumentParser(description="Professional Brawl Stars YOLO Training Pipeline")
    parser.add_argument("--capture", action="store_true", help="Capture real screenshots from emulator")
    parser.add_argument("--label", action="store_true", help="Auto-label captured images")
    parser.add_argument("--train", action="store_true", help="Train YOLO model")
    parser.add_argument("--templates", action="store_true", help="Capture real template images")
    parser.add_argument("--train-only", action="store_true", help="Skip capture/label, just train")
    parser.add_argument("--capture-frames", type=int, default=100, help="Number of frames to capture")
    parser.add_argument("--capture-interval", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu or 0 for GPU)")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path for fine-tuning")
    parser.add_argument("--output-dataset", type=str, default="dataset/yolo_v2", help="Output dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.output_dataset)

    # Step 1: Capture real data
    if args.capture:
        logger.info("=" * 60)
        logger.info("STEP 1: Capturing real game screenshots")
        logger.info("=" * 60)
        capturer = RealDataCapture(output_dir="dataset/real_capture")
        capturer.capture_batch(num_frames=args.capture_frames, interval=args.capture_interval)

    # Step 2: Auto-label and prepare dataset
    if args.label:
        logger.info("=" * 60)
        logger.info("STEP 2: Auto-labeling and preparing dataset")
        logger.info("=" * 60)
        labeler = BrawlStarsAutoLabeler()
        raw_dir = Path("dataset/real_capture/images")
        if raw_dir.exists():
            total = prepare_training_dataset(raw_dir, dataset_dir, labeler=labeler)
            logger.info(f"Total labels created: {total}")
        else:
            logger.error(f"Raw images directory not found: {raw_dir}")

    # Step 3: Train model
    if args.train or args.train_only:
        logger.info("=" * 60)
        logger.info("STEP 3: Training YOLO model")
        logger.info("=" * 60)

        data_yaml = dataset_dir / "data.yaml"

        # If train-only and dataset doesn't exist, try existing dataset
        if args.train_only and not data_yaml.exists():
            existing_yaml = Path("dataset/yolo/data.yaml")
            if existing_yaml.exists():
                # Update existing data.yaml with standard classes
                data_yaml = existing_yaml
                create_standard_data_yaml(existing_yaml.parent)
                logger.info(f"Using existing dataset: {data_yaml}")
            else:
                logger.error("No dataset found. Run with --capture --label first.")
                return

        if data_yaml.exists():
            result = train_model(
                data_yaml=data_yaml,
                epochs=args.epochs,
                batch_size=args.batch,
                device=args.device,
                pretrained_model=args.pretrained,
            )
            if result:
                logger.info(f"Training successful! Model saved to: {result}")
            else:
                logger.error("Training failed!")
        else:
            logger.error(f"data.yaml not found at {data_yaml}")

    # Step 4: Capture real templates
    if args.templates:
        logger.info("=" * 60)
        logger.info("STEP 4: Capturing real template images")
        logger.info("=" * 60)
        capture_real_templates()

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
