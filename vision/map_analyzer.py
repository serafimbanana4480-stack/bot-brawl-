"""
map_analyzer.py

Map analysis system for Brawl Stars.

Detects and analyzes map layout, identifying key features like:
- Walls and obstacles
- Bushes and hiding spots
- Power cube locations
- Choke points
- Safe zones
- Spawn points

Features:
- OCR for map name detection
- Layout analysis
- Strategic point identification
- Map categorization
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MapFeatures:
    """Features extracted from a map."""
    name: str
    walls: list[tuple[int, int, int, int]]  # (x, y, w, h)
    bushes: list[tuple[int, int, int, int]]
    power_cubes: list[tuple[int, int]]
    choke_points: list[tuple[int, int]]
    safe_zones: list[tuple[int, int, int, int]]
    spawn_points: list[tuple[int, int]]

    # Layout metrics
    openness: float  # 0.0 (closed) to 1.0 (open)
    symmetry: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0


class MapAnalyzer:
    """
    Analyzes Brawl Stars maps from screenshots.

    Extracts layout information and strategic features.
    """

    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "images"
        self.ocr_reader = None
        self._init_ocr()

        # Map database
        self.map_database: dict[str, MapFeatures] = {}
        self._load_map_database()

    def _init_ocr(self):
        """Initialize OCR for map name detection."""
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR initialized for map detection")
        except ImportError:
            logger.warning("EasyOCR not available, map detection limited")

    def _load_map_database(self):
        """Load pre-analyzed map database."""
        db_path = Path(__file__).parent.parent / "data" / "map_database.json"
        if db_path.exists():
            try:
                with open(db_path) as f:
                    data = json.load(f)

                for name, features in data.items():
                    self.map_database[name] = MapFeatures(
                        name=name,
                        walls=features.get('walls', []),
                        bushes=features.get('bushes', []),
                        power_cubes=features.get('power_cubes', []),
                        choke_points=features.get('choke_points', []),
                        safe_zones=features.get('safe_zones', []),
                        spawn_points=features.get('spawn_points', []),
                        openness=features.get('openness', 0.5),
                        symmetry=features.get('symmetry', 0.5),
                        complexity=features.get('complexity', 0.5),
                    )

                logger.info(f"Loaded map database with {len(self.map_database)} maps")
            except Exception as e:
                logger.error(f"Failed to load map database: {e}")

    def detect_map_name(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Detect map name from frame using OCR.

        Returns:
            (map_name, confidence)
        """
        if self.ocr_reader is None:
            return "", 0.0

        try:
            # Map name typically appears at top center during loading
            h, w = frame.shape[:2]
            map_region = frame[100:200, w//2-300:w//2+300]

            result = self.ocr_reader.ocr(map_region)
            if result and result[0]:
                text = result[0][0][0]
                # Clean up text
                map_name = text.strip().upper()
                if len(map_name) > 3:  # Minimum reasonable length
                    return map_name, 0.7
        except Exception as e:
            logger.debug(f"OCR error for map name: {e}")

        return "", 0.0

    def analyze_layout(self, frame: np.ndarray) -> MapFeatures:
        """
        Analyze map layout from frame.

        Extracts walls, bushes, power cubes, and other features.
        """
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect walls (brown/gray with low saturation)
        wall_lower = np.array([0, 0, 50])
        wall_upper = np.array([30, 80, 150])
        wall_mask = cv2.inRange(hsv, wall_lower, wall_upper)

        # Detect bushes (green)
        bush_lower = np.array([35, 40, 40])
        bush_upper = np.array([85, 255, 255])
        bush_mask = cv2.inRange(hsv, bush_lower, bush_upper)

        # Find contours for walls
        walls = self._extract_features(wall_mask, min_size=50)
        bushes = self._extract_features(bush_mask, min_size=30)

        # Calculate layout metrics
        openness = self._calculate_openness(frame, walls)
        symmetry = self._calculate_symmetry(frame)
        complexity = self._calculate_complexity(walls, bushes)

        # Identify choke points (narrow passages)
        choke_points = self._find_choke_points(walls)

        # Identify safe zones (areas with cover)
        safe_zones = self._find_safe_zones(bushes, walls)

        return MapFeatures(
            name="unknown",
            walls=walls,
            bushes=bushes,
            power_cubes=[],
            choke_points=choke_points,
            safe_zones=safe_zones,
            spawn_points=[],
            openness=openness,
            symmetry=symmetry,
            complexity=complexity,
        )

    def _extract_features(self, mask: np.ndarray, min_size: int = 30) -> list[tuple[int, int, int, int]]:
        """Extract bounding boxes from binary mask."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_size and h > min_size:
                features.append((x, y, w, h))

        return features

    def _calculate_openness(self, frame: np.ndarray, walls: list[tuple[int, int, int, int]]) -> float:
        """Calculate how open the map is (0.0 = closed, 1.0 = open)."""
        h, w = frame.shape[:2]
        total_area = h * w

        # Calculate total wall area
        wall_area = sum(w * h for _, _, w, h in walls)

        # Openness = 1 - (wall_area / total_area)
        openness = 1.0 - (wall_area / total_area)

        return max(0.0, min(1.0, openness))

    def _calculate_symmetry(self, frame: np.ndarray) -> float:
        """Calculate map symmetry (0.0 = asymmetric, 1.0 = symmetric)."""
        h, w = frame.shape[:2]

        # Split image in half and compare
        left = frame[:, :w//2]
        right = cv2.flip(frame[:, w//2:], 1)

        # Calculate difference
        diff = cv2.absdiff(left, right)
        diff_score = np.mean(diff)

        # Normalize (lower diff = higher symmetry)
        symmetry = 1.0 - min(1.0, diff_score / 50.0)

        return symmetry

    def _calculate_complexity(self, walls: list, bushes: list) -> float:
        """Calculate map complexity based on feature count."""
        total_features = len(walls) + len(bushes)

        # Normalize (more features = higher complexity)
        complexity = min(1.0, total_features / 50.0)

        return complexity

    def _find_choke_points(self, walls: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
        """Identify choke points (narrow passages between walls)."""
        # Simplified: find gaps between walls
        choke_points = []

        if len(walls) < 2:
            return choke_points

        # For each pair of walls, check if they form a narrow passage
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                w1 = walls[i]
                w2 = walls[j]

                # Calculate distance
                center1 = (w1[0] + w1[2]//2, w1[1] + w1[3]//2)
                center2 = (w2[0] + w2[2]//2, w2[1] + w2[3]//2)

                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

                # If walls are close but not touching, it might be a choke point
                if 50 < distance < 150:
                    midpoint = ((center1[0] + center2[0])//2, (center1[1] + center2[1])//2)
                    choke_points.append(midpoint)

        return choke_points

    def _find_safe_zones(self, bushes: list, walls: list) -> list[tuple[int, int, int, int]]:
        """Identify safe zones (areas with bush cover)."""
        safe_zones = []

        for bush in bushes:
            x, y, w, h = bush

            # Check if bush is near walls (better cover)
            near_wall = False
            for wall in walls:
                wx, wy, ww, wh = wall
                # Check if bush is adjacent to wall
                if (abs(x - (wx + ww)) < 50 or abs((x + w) - wx) < 50 or
                    abs(y - (wy + wh)) < 50 or abs((y + h) - wy) < 50):
                    near_wall = True
                    break

            if near_wall:
                safe_zones.append(bush)

        return safe_zones

    def get_or_analyze(self, frame: np.ndarray) -> MapFeatures:
        """
        Get map features from database or analyze frame.

        Args:
            frame: Current frame

        Returns:
            MapFeatures object
        """
        # Try to detect map name
        map_name, confidence = self.detect_map_name(frame)

        # Check if map is in database
        if map_name and map_name in self.map_database:
            logger.debug(f"Using cached features for map: {map_name}")
            return self.map_database[map_name]

        # Analyze layout
        features = self.analyze_layout(frame)

        # If we detected a name, update the database
        if map_name:
            features.name = map_name
            self.map_database[map_name] = features
            self._save_map_database()

        return features

    def _save_map_database(self):
        """Save map database to file."""
        db_path = Path(__file__).parent.parent / "data" / "map_database.json"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for name, features in self.map_database.items():
            data[name] = {
                'walls': features.walls,
                'bushes': features.bushes,
                'power_cubes': features.power_cubes,
                'choke_points': features.choke_points,
                'safe_zones': features.safe_zones,
                'spawn_points': features.spawn_points,
                'openness': features.openness,
                'symmetry': features.symmetry,
                'complexity': features.complexity,
            }

        with open(db_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved map database with {len(self.map_database)} maps")


def main():
    """Test map analyzer."""
    logging.basicConfig(level=logging.INFO)

    analyzer = MapAnalyzer()

    # Test with dummy frame
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    features = analyzer.analyze_layout(test_frame)

    print(f"Map features: {features}")
    print(f"Openness: {features.openness:.2f}")
    print(f"Symmetry: {features.symmetry:.2f}")
    print(f"Complexity: {features.complexity:.2f}")


if __name__ == "__main__":
    main()
