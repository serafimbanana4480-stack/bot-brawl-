"""
vision/game_feature_extractor.py

Extracts game features from screenshots that YOLO doesn't detect:
- Wall/obstacle detection via color analysis + edge detection
- Player HP extraction from health bar pixel analysis
- Bush detection via color clustering
- Match timer extraction

These complement YOLO detections with deterministic pixel-level analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("[FEATURES] OpenCV not available - wall/bush detection disabled")


class GameFeatureExtractor:
    """Extract game features from raw screenshots using pixel analysis."""

    # Brawl Stars color ranges (RGB) - expanded to cover more map themes
    # Note: These ranges are BROAD approximations. Walls in Brawl Stars vary widely
    # by map theme (industrial gray, ice blue, forest green, desert gold, purple arcade, etc.)
    # This detection is BEST-EFFORT only - always pair with confidence scoring.
    WALL_COLOR_RANGES = [
        ((120, 90, 60), (180, 150, 120)),   # Brown/gray (original)
        ((80, 80, 80), (200, 200, 200)),    # Neutral gray
        ((100, 120, 140), (160, 180, 200)),  # Ice blue
        ((60, 100, 60), (100, 140, 100)),    # Forest green
        ((180, 160, 100), (220, 200, 140)),  # Desert gold
        ((140, 100, 160), (180, 140, 200)),  # Purple arcade
    ]

    # Bush color (dark green)
    BUSH_COLOR_LOW = np.array([30, 100, 30])
    BUSH_COLOR_HIGH = np.array([80, 180, 70])

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.w, self.h = resolution

    def _create_wall_mask(self, screenshot: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Create combined wall mask from multiple color ranges.
        Returns (mask, confidence) where confidence is 0.0-1.0 based on
        how much of the image matches wall colors.
        """
        if not HAS_CV2:
            return np.array([]), 0.0

        combined_mask = None
        total_match_pixels = 0

        for low, high in self.WALL_COLOR_RANGES:
            mask = cv2.inRange(screenshot, np.array(low), np.array(high))
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            total_match_pixels += int(np.sum(mask > 0))

        total_pixels = screenshot.shape[0] * screenshot.shape[1]
        confidence = min(1.0, total_match_pixels / (total_pixels * 0.1))  # 10% of image = full confidence

        return combined_mask, confidence

    def detect_walls(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect walls/obstacles using color analysis + edge detection.

        Strategy:
        1. Create combined mask from multiple wall color ranges
        2. Apply morphological operations to connect nearby wall pixels
        3. Find contours and bounding boxes
        4. Filter by minimum size and aspect ratio
        5. Return with confidence score

        Note: This is BEST-EFFORT detection. Map themes vary significantly.
        Use the 'confidence' field to gauge reliability.
        """
        if not HAS_CV2 or screenshot is None or screenshot.size == 0:
            return []

        try:
            wall_mask, color_confidence = self._create_wall_mask(screenshot)

            if wall_mask is None or color_confidence < 0.1:
                return []

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            walls = []
            min_area = 500

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / max(h, 1)
                if aspect < 0.2 or aspect > 5.0:
                    continue

                walls.append({
                    "bbox": [x, y, x + w, y + h],
                    "center": (x + w // 2, y + h // 2),
                    "area": area,
                    "blocks_los": True,
                    "confidence": color_confidence,  # Indicate detection confidence
                })

            if walls:
                logger.debug(f"[WALLS] Detected {len(walls)} wall segments (color_confidence={color_confidence:.2f})")
            return walls

        except Exception as e:
            logger.debug(f"[WALLS] Detection error: {e}")
            return []

    def extract_player_hp(self, screenshot: np.ndarray, player_bbox: Optional[List[int]] = None) -> float:
        """
        Extract player HP from the health bar above the player character.

        Strategy:
        1. Locate HP bar region above player bbox (enlarged search window)
        2. Analyze green/red/blue pixels in the bar
        3. HP ratio = green_pixels / (green_pixels + red_pixels)
        4. Handle shields (blue bars) as potential confusion

        Returns HP as float 0.0-1.0 with 1.0 = full HP or shield-only bar.
        Note: This is BEST-EFFORT. Shields, poison, and burn effects can affect accuracy.
        """
        if screenshot is None or screenshot.size == 0 or player_bbox is None:
            return 1.0

        try:
            x1, y1, x2, y2 = player_bbox
            player_h = y2 - y1

            # HP bar search window - enlarged to handle variable bar sizes (30-60px)
            # Brawl Stars HP bars vary by brawler and game mode
            bar_height = max(30, min(60, player_h))  # Dynamic based on player size
            bar_y_top = max(0, y1 - int(player_h * 0.2) - bar_height)
            bar_y_bot = max(0, y1 - int(player_h * 0.1))
            bar_x_left = max(0, x1 - 10)
            bar_x_right = min(screenshot.shape[1], x2 + 10)

            if bar_y_bot <= bar_y_top or bar_x_right <= bar_x_left:
                return 1.0

            bar_region = screenshot[bar_y_top:bar_y_bot, bar_x_left:bar_x_right]
            if bar_region.size == 0:
                return 1.0

            # Count green (HP), red (damage), and blue (shield) pixels
            # Use more permissive thresholds to catch variations
            green_mask = (
                (bar_region[:, :, 1] > 100) &   # High green
                (bar_region[:, :, 0] < 150) &   # Low red
                (bar_region[:, :, 2] < 150)     # Low blue
            )
            red_mask = (
                (bar_region[:, :, 0] > 100) &  # High red
                (bar_region[:, :, 1] < 150) &   # Low green
                (bar_region[:, :, 2] < 150)     # Low blue
            )
            blue_mask = (
                (bar_region[:, :, 2] > 100) &  # High blue (shield)
                (bar_region[:, :, 0] < 150) &  # Low red
                (bar_region[:, :, 1] < 150)    # Low green
            )

            green_count = int(np.sum(green_mask))
            red_count = int(np.sum(red_mask))
            blue_count = int(np.sum(blue_mask))
            total_colored = green_count + red_count

            # If we only see blue (shield) and no green/red, assume full HP
            # This handles Rosa's shield, 8-Bit's extra life, etc.
            if total_colored == 0 and blue_count > 50:
                logger.debug(f"[HP] Shield-only bar detected (blue={blue_count}), returning 1.0")
                return 1.0

            if total_colored == 0:
                return 1.0

            hp = green_count / total_colored

            # Sanity check: HP extraction should not return values too close to 0
            # unless there's actual damage. A bar that's mostly red means low HP.
            if green_count < 10 and red_count > 100:
                hp = min(hp, 0.2)  # At most 20% if almost all red

            logger.debug(f"[HP] Extracted HP: {hp:.2f} (green={green_count}, red={red_count}, blue={blue_count})")
            return float(np.clip(hp, 0.0, 1.0))

        except Exception as e:
            logger.debug(f"[HP] Extraction error: {e}")
            return 1.0

    def detect_bushes(self, screenshot: np.ndarray) -> List[Dict]:
        """Detect bushes using color analysis."""
        if not HAS_CV2 or screenshot is None or screenshot.size == 0:
            return []

        try:
            mask = cv2.inRange(screenshot, self.BUSH_COLOR_LOW, self.BUSH_COLOR_HIGH)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bushes = []
            min_area = 800

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                bushes.append({
                    "bbox": [x, y, x + w, y + h],
                    "center": (x + w // 2, y + h // 2),
                    "area": area,
                })

            if bushes:
                logger.debug(f"[BUSHES] Detected {len(bushes)} bush patches")
            return bushes

        except Exception as e:
            logger.debug(f"[BUSHES] Detection error: {e}")
            return []

    def extract_match_timer(self, screenshot: np.ndarray) -> Optional[float]:
        """Extract match timer from top of screen. Returns -1.0 if timer detected but value unknown."""
        if screenshot is None or screenshot.size == 0:
            return None

        try:
            h, w = screenshot.shape[:2]
            timer_region = screenshot[int(h * 0.02):int(h * 0.08), int(w * 0.35):int(w * 0.65)]

            if timer_region.size == 0:
                return None

            brightness = np.mean(timer_region, axis=2)
            white_pixels = int(np.sum(brightness > 200))
            total_pixels = brightness.size

            if white_pixels / total_pixels > 0.1:
                return -1.0  # Timer detected, value needs OCR

            return None

        except Exception as e:
            logger.debug(f"[TIMER] Extraction error: {e}")
            return None

    def extract_features(self, screenshot: np.ndarray, player_bbox: Optional[List[int]] = None) -> Dict:
        """Extract all game features from a screenshot."""
        return {
            "walls": self.detect_walls(screenshot),
            "bushes": self.detect_bushes(screenshot),
            "player_hp": self.extract_player_hp(screenshot, player_bbox),
            "match_timer": self.extract_match_timer(screenshot),
        }


# No-op fallback when cv2 is not available
if not HAS_CV2:
    class _NoopFeatureExtractor:
        def __init__(self, resolution=(1920, 1080)):
            self.w, self.h = resolution

        def detect_walls(self, screenshot=None):
            return []

        def extract_player_hp(self, screenshot=None, player_bbox=None):
            return 1.0

        def detect_bushes(self, screenshot=None):
            return []

        def extract_match_timer(self, screenshot=None):
            return None

        def extract_features(self, screenshot=None, player_bbox=None):
            return {"walls": [], "bushes": [], "player_hp": 1.0, "match_timer": None}

    GameFeatureExtractor = _NoopFeatureExtractor  # type: ignore[misc,assignment]
