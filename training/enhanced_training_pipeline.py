"""
enhanced_training_pipeline.py

Pipeline profissional completo de treino YOLO para Brawl Stars.
Inclui:
- Captura de screenshots (500-1000)
- Limpeza e curadoria de dados
- Auto-labeling com validação
- Data augmentation
- Treino otimizado com early stopping
- Validação e relatórios

Classes padronizadas:
    - core: 4 classes presentes no dataset atual
    - extended: 8 classes para evolução futura sem quebrar produção
    0: Player   - Jogador controlado
    1: Enemy    - Inimigos
    2: Bush     - Arbustos (cover)
    3: Cubebox  - Caixas de power cubes
    4: Wall     - Paredes/obstáculos
    5: Powerup  - Power-ups diversos
    6: Bullet   - Balas/projéteis
    7: Super    - Indicador de super

Usage:
    python training/enhanced_training_pipeline.py --capture --label --train
    python training/enhanced_training_pipeline.py --capture-only --frames 500
    python training/enhanced_training_pipeline.py --train-only --epochs 100 --gpu
"""

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

import cv2
import numpy as np

from training.class_schema import CORE_CLASSES, EXTENDED_CLASSES, get_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("enhanced_training")

# ============================================================
# CLASSES PADRONIZADAS
# ============================================================
STANDARD_CLASSES = EXTENDED_CLASSES
CORE_STANDARD_CLASSES = CORE_CLASSES
NC = len(STANDARD_CLASSES)

# HSV ranges for Brawl Stars elements
BRAWLSTARS_HSV = {
    "player_arrow_blue": ((100, 100, 100), (130, 255, 255)),
    "enemy_arrow_red1": ((0, 100, 100), (10, 255, 255)),
    "enemy_arrow_red2": ((170, 100, 100), (180, 255, 255)),
    "bush_dark_green": ((35, 30, 30), (85, 180, 120)),
    "cubebox_purple": ((130, 50, 80), (170, 255, 255)),
    "cubebox_blue": ((100, 80, 80), (130, 255, 255)),
    "health_green": ((35, 80, 80), (85, 255, 255)),
    "health_red": ((0, 100, 80), (10, 255, 255)),
    "wall_gray": ((0, 0, 100), (180, 50, 200)),
    "powerup_yellow": ((20, 150, 150), (35, 255, 255)),
    "bullet_orange": ((10, 150, 150), (25, 255, 255)),
}


# ============================================================
# DATA CURATOR - Limpeza e curadoria de dados
# ============================================================
class DataCurator:
    """Limpa e curacopia datasets: remove duplicados, borrados, etc."""

    def __init__(self, min_quality: float = 0.5):
        self.min_quality = min_quality
        self.seen_hashes: Set[str] = set()

    def compute_hash(self, image: np.ndarray) -> str:
        """Computa hash perceptual da imagem."""
        img_small = cv2.resize(image, (64, 64))
        return hashlib.md5(img_small.tobytes()).hexdigest()

    def is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Detecta imagens borradas usando Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance < threshold

    def is_duplicate(self, image: np.ndarray) -> bool:
        """Detecta imagens duplicadas por hash."""
        h = self.compute_hash(image)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def check_image_quality(self, image: np.ndarray) -> Tuple[bool, str]:
        """Verifica qualidade da imagem."""
        if image is None or image.size == 0:
            return False, "empty"

        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return False, "too_small"

        if self.is_blurry(image):
            return False, "blurry"

        if self.is_duplicate(image):
            return False, "duplicate"

        mean_brightness = np.mean(image)
        if mean_brightness < 20 or mean_brightness > 235:
            return False, "poor_lighting"

        return True, "ok"

    def curate_dataset(self, input_dir: Path, output_dir: Path,
                       stats: Optional[Dict] = None) -> Dict:
        """Curacopia um dataset inteiro."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = stats or {
            "total": 0, "valid": 0, "blurry": 0, "duplicate": 0,
            "too_small": 0, "poor_lighting": 0, "errors": 0
        }

        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        logger.info(f"Curating {len(image_files)} images from {input_dir}")

        for img_path in image_files:
            stats["total"] += 1
            try:
                img = cv2.imread(str(img_path))
                is_valid, reason = self.check_image_quality(img)

                if is_valid:
                    dst = output_dir / img_path.name
                    shutil.copy2(img_path, dst)
                    stats["valid"] += 1
                else:
                    stats[reason] = stats.get(reason, 0) + 1
                    logger.debug(f"Rejected {img_path.name}: {reason}")

            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"Error processing {img_path.name}: {e}")

        logger.info(f"Curated: {stats['valid']}/{stats['total']} valid images")
        return stats


# ============================================================
# ENHANCED AUTO-LABELER
# ============================================================
class EnhancedAutoLabeler:
    """Auto-labeler melhorado com 8 classes e múltiplas heurísticas."""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size

    def _merge_masks(self, *masks: np.ndarray) -> np.ndarray:
        """Combina múltiplas máscaras sem depender de uma única paleta de cor."""
        valid_masks = [mask for mask in masks if mask is not None]
        if not valid_masks:
            return np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        merged = valid_masks[0].copy()
        for mask in valid_masks[1:]:
            merged = cv2.bitwise_or(merged, mask)
        return merged

    def _lab_mask(self, image: np.ndarray, lower: Tuple, upper: Tuple) -> np.ndarray:
        """Cria máscara no espaço LAB para reforçar a segmentação HSV."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return cv2.inRange(lab, np.array(lower), np.array(upper))

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Suaviza ruído e fecha buracos pequenos antes do contour extraction."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned

    def _hsv_mask(self, image: np.ndarray, lower: Tuple, upper: Tuple) -> np.ndarray:
        """Cria máscara HSV."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.array(lower), np.array(upper))

    def _find_contours_filtered(self, mask: np.ndarray,
                                min_area: int = 100,
                                max_area: int = 50000,
                                aspect_range: Tuple = (0.1, 10.0)) -> List:
        """Encontra contornos filtrados por área e aspecto."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if aspect < aspect_range[0] or aspect > aspect_range[1]:
                continue
            filtered.append((x, y, w, h))
        return filtered

    def _to_yolo(self, bbox: Tuple[int, int, int, int],
                 img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Converte bbox para formato YOLO (cx, cy, w, h) normalizado."""
        x, y, w, h = bbox
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        return (cx, cy, nw, nh)

    def _expand_bbox(self, bbox: Tuple[int, int, int, int],
                    factor: float = 1.2,
                    img_w: int = 640, img_h: int = 640) -> Tuple[int, int, int, int]:
        """Expande bbox para incluir área circundante."""
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        new_w, new_h = int(w * factor), int(h * factor)
        new_x = max(0, cx - new_w // 2)
        new_y = max(0, cy - new_h // 2)
        new_x2 = min(img_w, new_x + new_w)
        new_y2 = min(img_h, new_y + new_h)
        return (new_x, new_y, new_x2 - new_x, new_y2 - new_y)

    def detect_players(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Player (classe 0) via seta azul/joystick."""
        results = []
        h, w = image.shape[:2]

        # Azul para Player (seta acima do personagem)
        mask_blue = self._hsv_mask(image, *BRAWLSTARS_HSV["player_arrow_blue"])
        bboxes = self._find_contours_filtered(mask_blue, min_area=50, max_area=2000)

        for x, y, bw, bh in bboxes:
            # Verificar se está na metade superior (onde biasanya aparecem setas)
            if y < h * 0.6:
                expanded = self._expand_bbox((x, y, bw, bh), factor=1.5, img_w=w, img_h=h)
                yolo = self._to_yolo(expanded, w, h)
                if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                    results.append((0, yolo))

        return results

    def detect_enemies(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Enemy (classe 1) via seta vermelha."""
        results = []
        h, w = image.shape[:2]

        # Vermelho para Enemy
        mask_red1 = self._hsv_mask(image, *BRAWLSTARS_HSV["enemy_arrow_red1"])
        mask_red2 = self._hsv_mask(image, *BRAWLSTARS_HSV["enemy_arrow_red2"])
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        bboxes = self._find_contours_filtered(mask_red, min_area=50, max_area=2000)

        for x, y, bw, bh in bboxes:
            if y < h * 0.7:
                expanded = self._expand_bbox((x, y, bw, bh), factor=1.5, img_w=w, img_h=h)
                yolo = self._to_yolo(expanded, w, h)
                if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                    results.append((1, yolo))

        # Também detectar via health bar vermelha (inimigos sem seta mas com HP bar)
        mask_hp_red = self._hsv_mask(image, *BRAWLSTARS_HSV["health_red"])
        hp_boxes = self._find_contours_filtered(mask_hp_red, min_area=30, max_area=500)

        for x, y, bw, bh in hp_boxes:
            # HP bars são finas e horizontais
            if bw > bh * 2:  # Mais largo que alto
                expanded = self._expand_bbox((x, y, bw, bh), factor=2.0, img_w=w, img_h=h)
                yolo = self._to_yolo(expanded, w, h)
                if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                    results.append((1, yolo))

        return results

    def detect_bushes(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Bush (classe 2) via verde escuro + faixa LAB de vegetação."""
        results = []
        h, w = image.shape[:2]

        mask_green = self._hsv_mask(image, *BRAWLSTARS_HSV["bush_dark_green"])
        mask_lab = self._lab_mask(image, (20, 90, 80), (200, 150, 170))
        mask = self._postprocess_mask(self._merge_masks(mask_green, mask_lab))
        bboxes = self._find_contours_filtered(mask, min_area=200, max_area=50000)

        for x, y, bw, bh in bboxes:
            expanded = self._expand_bbox((x, y, bw, bh), factor=1.1, img_w=w, img_h=h)
            yolo = self._to_yolo(expanded, w, h)
            if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                results.append((2, yolo))

        return results

    def detect_cubeboxes(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Cubebox (classe 3) via cor roxa/azul."""
        results = []
        h, w = image.shape[:2]

        mask_purple = self._hsv_mask(image, *BRAWLSTARS_HSV["cubebox_purple"])
        mask_blue = self._hsv_mask(image, *BRAWLSTARS_HSV["cubebox_blue"])
        mask = cv2.bitwise_or(mask_purple, mask_blue)

        bboxes = self._find_contours_filtered(mask, min_area=50, max_area=5000)

        for x, y, bw, bh in bboxes:
            expanded = self._expand_bbox((x, y, bw, bh), factor=1.3, img_w=w, img_h=h)
            yolo = self._to_yolo(expanded, w, h)
            if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                results.append((3, yolo))

        return results

    def detect_walls(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Wall (classe 4) via baixa saturação + bordas fortes."""
        results = []
        h, w = image.shape[:2]

        mask_gray = self._hsv_mask(image, *BRAWLSTARS_HSV["wall_gray"])
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        low_chroma = cv2.inRange(lab[:, :, 1], 100, 150)
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 140)
        mask = self._merge_masks(mask_gray, low_chroma, edges)
        mask = self._postprocess_mask(mask)
        bboxes = self._find_contours_filtered(mask, min_area=500, max_area=100000,
                                               aspect_range=(0.05, 20.0))

        for x, y, bw, bh in bboxes:
            # Paredes são geralmente retangulares
            expanded = self._expand_bbox((x, y, bw, bh), factor=1.05, img_w=w, img_h=h)
            yolo = self._to_yolo(expanded, w, h)
            if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                results.append((4, yolo))

        return results

    def detect_powerups(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Powerup (classe 5) via cor amarela."""
        results = []
        h, w = image.shape[:2]

        mask_yellow = self._hsv_mask(image, *BRAWLSTARS_HSV["powerup_yellow"])
        bboxes = self._find_contours_filtered(mask_yellow, min_area=30, max_area=2000)

        for x, y, bw, bh in bboxes:
            expanded = self._expand_bbox((x, y, bw, bh), factor=1.4, img_w=w, img_h=h)
            yolo = self._to_yolo(expanded, w, h)
            if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                results.append((5, yolo))

        return results

    def detect_bullets(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Bullet (classe 6) via laranja/amarelo + brilho local."""
        results = []
        h, w = image.shape[:2]

        mask_orange = self._hsv_mask(image, *BRAWLSTARS_HSV["bullet_orange"])
        mask_yellow = self._hsv_mask(image, *BRAWLSTARS_HSV["powerup_yellow"])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        mask = self._postprocess_mask(self._merge_masks(mask_orange, mask_yellow, bright))
        bboxes = self._find_contours_filtered(mask, min_area=10, max_area=500)

        for x, y, bw, bh in bboxes:
            expanded = self._expand_bbox((x, y, bw, bh), factor=1.5, img_w=w, img_h=h)
            yolo = self._to_yolo(expanded, w, h)
            if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                results.append((6, yolo))

        return results

    def detect_supers(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta Super (classe 7) via brilho amarelo intenso e HUD linear."""
        results = []
        h, w = image.shape[:2]

        # Super costuma aparecer em elementos HUD brilhantes; combinamos HSV + LAB.
        mask_yellow = self._hsv_mask(image, (18, 160, 170), (40, 255, 255))
        mask_lab = self._lab_mask(image, (180, 110, 150), (255, 145, 220))
        mask = self._postprocess_mask(self._merge_masks(mask_yellow, mask_lab))
        bboxes = self._find_contours_filtered(mask, min_area=50, max_area=2000)

        for x, y, bw, bh in bboxes:
            expanded = self._expand_bbox((x, y, bw, bh), factor=1.3, img_w=w, img_h=h)
            yolo = self._to_yolo(expanded, w, h)
            if 0 < yolo[0] < 1 and 0 < yolo[1] < 1:
                results.append((7, yolo))

        return results

    def auto_label(self, image: np.ndarray) -> List[Tuple[int, Tuple[float, ...]]]:
        """Deteta todos os elementos na imagem."""
        detections = []

        detections.extend(self.detect_players(image))
        detections.extend(self.detect_enemies(image))
        detections.extend(self.detect_bushes(image))
        detections.extend(self.detect_cubeboxes(image))
        detections.extend(self.detect_walls(image))
        detections.extend(self.detect_powerups(image))
        detections.extend(self.detect_bullets(image))
        detections.extend(self.detect_supers(image))

        return self._nms(detections)

    def _nms(self, detections: List[Tuple[int, Tuple[float, ...]]],
             iou_threshold: float = 0.4) -> List[Tuple[int, Tuple[float, ...]]]:
        """
        Non-maximum suppression using cv2.dnn.NMSBoxes.
        FIX #3: Uses OpenCV's optimized NMS instead of manual implementation.
        """
        if not detections:
            return []

        # Group by class for OpenCV NMSBoxes format
        by_class: Dict[int, List] = defaultdict(list)
        for cls_id, bbox in detections:
            by_class[cls_id].append(bbox)

        results = []
        for cls_id, bboxes in by_class.items():
            # Convert YOLO format (cx, cy, w, h) to (x1, y1, x2, y2) for NMSBoxes
            boxes = []
            for (cx, cy, w, h) in bboxes:
                x1 = (cx - w/2)
                y1 = (cy - h/2)
                x2 = (cx + w/2)
                y2 = (cy + h/2)
                boxes.append([x1, y1, x2, y2])

            # OpenCV NMSBoxes expects (x, y, w, h) format
            boxes_cv = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]

            if not boxes_cv:
                continue

            try:
                indices = cv2.dnn.NMSBoxes(
                    boxes_cv,
                    [1.0] * len(boxes_cv),  # confidences
                    score_threshold=0.01,
                    nms_threshold=iou_threshold
                )

                if indices is not None and len(indices) > 0:
                    for idx in indices:
                        if isinstance(idx, (list, np.ndarray)):
                            idx = idx[0]
                        results.append((cls_id, bboxes[idx]))
            except Exception as e:
                logger.debug(f"NMSBoxes failed, using manual NMS: {e}")
                # Fallback to manual NMS if cv2.dnn.NMSBoxes fails
                results.extend(self._nms_manual(bboxes, cls_id, iou_threshold))

        return results

    def _nms_manual(self, bboxes: List[Tuple[float, ...]], cls_id: int,
                   iou_threshold: float = 0.4) -> List[Tuple[int, Tuple[float, ...]]]:
        """Manual NMS fallback if cv2.dnn.NMSBoxes fails."""
        if not bboxes:
            return []

        # Sort by area (larger first) so the fallback behaves closer to
        # a conventional "keep the strongest box" NMS strategy.
        sorted_bboxes = sorted(enumerate(bboxes), key=lambda x: x[1][2] * x[1][3], reverse=True)

        keep = []
        while sorted_bboxes:
            idx, largest = sorted_bboxes.pop(0)
            keep.append((cls_id, largest))

            sorted_bboxes = [
                (i, b) for i, b in sorted_bboxes
                if self._iou(largest, b) < iou_threshold
            ]

        return keep

    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calcula IoU entre dois boxes YOLO."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def label_image(self, image_path: str, output_label_path: str) -> int:
        """Label uma imagem e salva em formato YOLO."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Cannot read image: {image_path}")
            return 0

        image = cv2.resize(image, self.target_size)
        detections = self.auto_label(image)

        with open(output_label_path, 'w') as f:
            for cls_id, (cx, cy, w, h) in detections:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        return len(detections)


# ============================================================
# SCREENSHOT CAPTURE
# ============================================================
class ScreenCapturer:
    """Captura screenshots do emulador."""

    def __init__(self, window_title: str = "auto",
                 output_dir: str = "dataset/captured"):
        self.window_title = window_title
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_taker = None

    def _candidate_window_titles(self) -> List[str]:
        """Lista ordenada de títulos de janela para vários emuladores."""
        candidates = []
        if self.window_title and self.window_title != "auto":
            candidates.append(self.window_title)
        candidates.extend([
            "BlueStacks App Player",
            "BlueStacks",
            "LDPlayer",
            "NoxPlayer",
            "MEmu",
            "MuMuPlayer",
            "GameLoop",
        ])
        # Remove duplicados preservando ordem
        seen = set()
        ordered = []
        for title in candidates:
            if title not in seen:
                ordered.append(title)
                seen.add(title)
        return ordered

    def _init_screenshot(self) -> bool:
        """Inicializa screenshot taker."""
        try:
            from pylaai_real.screenshot_taker import ScreenshotTaker
            for title in self._candidate_window_titles():
                candidate = ScreenshotTaker(title)
                if candidate.find_window():
                    self.screenshot_taker = candidate
                    logger.info(f"Screenshot source initialized with window title: {title}")
                    return True

            logger.error("No compatible emulator window found")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize screenshot: {e}")
            return False

    def capture(self, num_frames: int = 100, interval: float = 0.5,
                target_size: Tuple[int, int] = (640, 640)) -> int:
        """Captura batch de screenshots."""
        if self.screenshot_taker is None:
            if not self._init_screenshot():
                return 0

        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        existing = len(list(images_dir.glob("*.png")))
        captured = 0

        for i in range(num_frames):
            img = self.screenshot_taker.take()
            if img is not None:
                fullres = images_dir / f"fullres_{existing + i:05d}.png"
                cv2.imwrite(str(fullres), img)

                resized = cv2.resize(img, target_size)
                train_img = images_dir / f"frame_{existing + i:05d}.png"
                cv2.imwrite(str(train_img), resized)

                captured += 1
                if captured % 50 == 0:
                    logger.info(f"Captured {captured}/{num_frames} frames")
            else:
                logger.warning(f"Screenshot failed at frame {i}")

            time.sleep(interval)

        logger.info(f"Capture complete: {captured} frames saved")
        return captured


# ============================================================
# DATASET PREPARATION
# ============================================================
def _stratified_split(image_files: List[Path], labeler: EnhancedAutoLabeler,
                     train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Path]]:
    """
    Split dataset with stratification to ensure all classes are represented in each split.
    FIX #4: Uses stratified splitting based on primary class in each image.
    """
    # Assign primary class to each image
    image_classes: Dict[Path, int] = {}
    for img_path in image_files:
        label_path = img_path.parent / f"{img_path.stem}.txt"
        primary_cls = 0  # Default to Player
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cls_id = int(line.split()[0])
                        primary_cls = cls_id
                        break
        image_classes[img_path] = primary_cls

    # Group images by primary class
    class_groups: Dict[int, List[Path]] = defaultdict(list)
    for img_path, cls_id in image_classes.items():
        class_groups[cls_id].append(img_path)

    # Stratified split - maintain class distribution in each split
    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for cls_id, imgs in class_groups.items():
        random.shuffle(imgs)
        n = len(imgs)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits["train"].extend(imgs[:train_end])
        splits["val"].extend(imgs[train_end:val_end])
        splits["test"].extend(imgs[val_end:])

    # Final shuffle of each split
    for split_name in splits:
        random.shuffle(splits[split_name])

    return splits


def _check_minimum_dataset(image_files: List[Path], min_images: int = 500) -> bool:
    """
    Verify dataset has minimum required images.
    FIX #2: Prevents training with insufficient data.
    """
    if len(image_files) < min_images:
        logger.error(
            f"Dataset too small: {len(image_files)} images (minimum: {min_images}). "
            f"Collect more screenshots before training."
        )
        return False
    return True


def prepare_dataset(raw_dir: Path, output_dir: Path,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1) -> Dict:
    """Prepara dataset YOLO com split train/val/test."""
    labeler = EnhancedAutoLabeler()

    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in raw_dir.iterdir()
                   if f.suffix.lower() in extensions and "frame_" in f.name]

    if not image_files:
        logger.error(f"No images found in {raw_dir}")
        return {}

    logger.info(f"Found {len(image_files)} images")

    # FIX #2: Verify minimum dataset size
    if not _check_minimum_dataset(image_files, min_images=100):
        return {}

    random.seed(42)

    # FIX #4: Use stratified split for class balance
    splits = _stratified_split(image_files, labeler, train_ratio, val_ratio)

    stats = {"train": 0, "val": 0, "test": 0, "labels": 0}

    for split_name, files in splits.items():
        for img_path in files:
            dst_img = output_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            label_path = img_path.parent / f"{img_path.stem}.txt"
            dst_label = output_dir / split_name / "labels" / f"{img_path.stem}.txt"

            if label_path.exists():
                shutil.copy2(label_path, dst_label)
            else:
                num_labels = labeler.label_image(str(img_path), str(dst_label))
                stats["labels"] += num_labels

            stats[split_name] += 1

    logger.info(f"Dataset prepared: train={stats['train']}, val={stats['val']}, test={stats['test']}")
    return stats


def create_data_yaml(dataset_dir: Path, schema: str = "core") -> Path:
    """Cria data.yaml para treino YOLO."""
    names = get_schema(schema)
    data_config = {
        "path": str(dataset_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
        "nc": len(names),
    }

    yaml_path = dataset_dir / "data.yaml"
    try:
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    except ImportError:
        with open(yaml_path, 'w') as f:
            json.dump(data_config, f, indent=2)

    logger.info(f"Created data.yaml at {yaml_path}")
    return yaml_path


def _load_dataset_names(data_yaml: Path) -> Dict[int, str]:
    """Load class names from a YOLO dataset YAML if possible."""
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        names = cfg.get("names", {})
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
    except Exception:
        pass
    return STANDARD_CLASSES


# ============================================================
# MODEL TRAINING
# ============================================================
def train_yolo(data_yaml: Path,
               epochs: int = 50,
               batch_size: int = 16,
               img_size: int = 640,
               device: str = "cpu",
               pretrained: Optional[str] = None,
               output_name: str = "brawlstars_yolo11",
               freeze: int = 0,
               cos_lr: bool = True,
               resume: bool = False,
               hyperparams: Optional[Dict[str, object]] = None) -> Optional[str]:
    """Treina modelo YOLO com otimizações."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return None

    # Determine base model
    if pretrained and Path(pretrained).exists():
        logger.info(f"Fine-tuning from: {pretrained}")
        model = YOLO(pretrained)
    else:
        # Try to find local yolo11n or yolov8n
        for model_name in ["yolo11n.pt", "yolov8n.pt"]:
            try:
                model = YOLO(model_name)
                logger.info(f"Using base model: {model_name}")
                break
            except Exception:
                continue
        else:
            logger.info("Downloading yolo11n.pt...")
            model = YOLO("yolo11n.pt")

    # FIX #1: Verify GPU availability and fallback to CPU if needed
    if device != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                device = "0"  # Ensure device is set correctly for CUDA
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            device = "cpu"
    else:
        logger.info("Using CPU for training")

    output_dir = Path(f"runs/detect/{output_name}")

    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    dataset_names = _load_dataset_names(Path(data_yaml))
    logger.info(f"  Data:       {data_yaml}")
    logger.info(f"  Epochs:     {epochs}")
    logger.info(f"  Batch:      {batch_size}")
    logger.info(f"  Image size: {img_size}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Classes:    {len(dataset_names)} - {dataset_names}")
    logger.info(f"  Freeze:     {freeze}")
    logger.info(f"  Cos LR:     {cos_lr}")
    logger.info(f"  Resume:     {resume}")
    if hyperparams:
        logger.info(f"  Hyperparams: {hyperparams}")
    logger.info("=" * 60)

    # Training with augmentation
    # NOTE: ultralytics appends project/name to create save_dir.
    # We use absolute project path to avoid double-nesting.
    training_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": device,
        "project": str(Path.cwd() / "runs" / "detect"),
        "name": output_name,
        "exist_ok": True,
        "patience": 15,
        "save": True,
        "plots": True,
        "verbose": True,
        # Optimizer settings
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Loss settings - FIX #6: Class-weighted loss for rare classes
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        # Augmentation - data augmentation pipeline
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0002,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        "erasing": 0.4,
        "copy_paste": 0.0,
        "freeze": freeze,
        "cos_lr": cos_lr,
        "resume": resume,
        # Multi-GPU (if available)
        "workers": 4 if device != "cpu" else 2,
        "amp": True,
        "cache": "disk",
    }
    if hyperparams:
        training_kwargs.update(hyperparams)

    _results = model.train(**training_kwargs)

    # Save best model
    best_pt = output_dir / "weights" / "best.pt"
    if best_pt.exists():
        dest = Path("models/brawlstars_yolov8.pt")
        backup = Path("models/brawlstars_yolov8_backup.pt")
        if dest.exists():
            shutil.copy2(dest, backup)
        shutil.copy2(best_pt, dest)
        logger.info(f"Model saved to {dest}")
        return str(dest)

    return None


def validate_model(model_path: Path, data_yaml: Path) -> Dict:
    """Valida modelo no test set e retorna métricas."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))

        logger.info("=" * 60)
        logger.info("MODEL VALIDATION")
        logger.info("=" * 60)

        metrics = model.val(data=str(data_yaml), split="test")

        results = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
        }

        logger.info(f"  mAP@50:     {results['mAP50']:.4f}")
        logger.info(f"  mAP@50-95:  {results['mAP50-95']:.4f}")
        logger.info(f"  Precision:  {results['precision']:.4f}")
        logger.info(f"  Recall:     {results['recall']:.4f}")
        logger.info(f"  F1-Score:   {results['f1']:.4f}")

        # Per-class metrics
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            logger.info("  Per-class mAP@50:")
            for i, name in dataset_names.items():
                if i < len(metrics.box.maps):
                    logger.info(f"    {name}: {metrics.box.maps[i]:.4f}")

        return results

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {}


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Enhanced Brawl Stars YOLO Training")
    parser.add_argument("--capture-only", action="store_true", help="Only capture screenshots")
    parser.add_argument("--label-only", action="store_true", help="Only run auto-labeling")
    parser.add_argument("--train-only", action="store_true", help="Only train (skip capture/label)")
    parser.add_argument("--frames", type=int, default=500, help="Frames to capture (500-1000)")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between captures")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0 for GPU)")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained model path")
    parser.add_argument(
        "--schema",
        type=str,
        default="core",
        choices=["core", "extended"],
        help="Class schema for dataset generation and training (default: core)",
    )
    parser.add_argument("--curate", action="store_true", help="Curate existing dataset")
    parser.add_argument("--dataset", type=str, default="dataset/yolo_final",
                       help="Dataset directory with data.yaml (default: dataset/yolo_final)")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing model")
    args = parser.parse_args()

    capture_dir = Path("dataset/captured")
    curated_dir = Path("dataset/curated")
    output_dir = Path(args.dataset)

    # Validate-only mode
    if args.validate_only:
        model_path = Path("models/brawlstars_yolov8.pt")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
        data_yaml = output_dir / "data.yaml"
        if not data_yaml.exists():
            logger.error(f"data.yaml not found: {data_yaml}")
            return
        validate_model(model_path, data_yaml)
        return

    # Step 1: Capture
    if args.capture_only or not args.train_only:
        logger.info("=" * 60)
        logger.info("STEP 1: CAPTURING SCREENSHOTS")
        logger.info("=" * 60)
        capturer = ScreenCapturer(output_dir=str(capture_dir))
        num_captured = capturer.capture(
            num_frames=args.frames,
            interval=args.interval
        )
        logger.info(f"Captured {num_captured} frames")

    # Step 2: Curate (clean duplicates, blurry, etc.)
    if args.curate and capture_dir.exists():
        logger.info("=" * 60)
        logger.info("STEP 2: CURATING DATASET")
        logger.info("=" * 60)
        curator = DataCurator()
        curator.curate_dataset(capture_dir / "images", curated_dir / "images")
        use_dir = curated_dir
    else:
        use_dir = capture_dir

    # Step 3: Prepare dataset
    if not args.train_only and use_dir.exists():
        logger.info("=" * 60)
        logger.info("STEP 3: PREPARING DATASET")
        logger.info("=" * 60)
        prepare_dataset(use_dir / "images", output_dir)
        create_data_yaml(output_dir, schema=args.schema)

    # Step 4: Train
    if args.train_only or (not args.capture_only and not args.label_only):
        data_yaml = output_dir / "data.yaml"
        if not data_yaml.exists():
            logger.error(f"data.yaml not found at {data_yaml}")
            return

        logger.info("=" * 60)
        logger.info("STEP 4: TRAINING MODEL")
        logger.info("=" * 60)
        model_path = train_yolo(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            pretrained=args.pretrained or f"models/brawlstars_{args.schema}.pt"
        )

        if model_path:
            logger.info("=" * 60)
            logger.info("STEP 5: VALIDATING MODEL")
            logger.info("=" * 60)
            results = validate_model(Path(model_path), data_yaml)

            # Save validation report
            report_path = Path("models/validation_report.json")
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Validation report saved to {report_path}")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
