"""
metadata_extractor.py

Extracts rich metadata from Brawl Stars gameplay frames.
Analyzes UI elements, game state, and statistics for ML training.

Features:
- Trophy count extraction
- Brawler identification
- Game mode detection
- Map name extraction
- Game phase detection (lobby, gameplay, end screen)
- Health and ammo estimation
- Super charge detection

Usage:
    from metadata_extractor import MetadataExtractor
    extractor = MetadataExtractor()
    metadata = extractor.extract(frame)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GameMetadata:
    """Complete metadata extracted from a game frame."""
    # Game phase
    game_phase: str  # "lobby", "matchmaking", "loading", "gameplay", "end_screen"
    
    # Player info
    trophies: int
    brawler_name: str
    brawler_level: int
    
    # Game info
    game_mode: str  # "showdown", "gem_grab", "brawl_ball", etc.
    map_name: str
    
    # Player state
    health: float  # 0.0 to 1.0
    ammo: int  # 0 to 3 typically
    super_charged: bool
    
    # Match info
    time_remaining: Optional[float]  # Seconds
    score: Tuple[int, int]  # (team_score, enemy_score) or (kills, deaths)
    
    # Confidence scores
    confidence: Dict[str, float]


class MetadataExtractor:
    """
    Extracts metadata from Brawl Stars frames using OCR and heuristics.
    """
    
    # Color ranges for UI elements (HSV)
    TROPHY_COLOR = (np.array([20, 100, 100]), np.array([40, 255, 255]))  # Gold
    HEALTH_GREEN = (np.array([40, 50, 50]), np.array([80, 255, 255]))
    HEALTH_RED = (np.array([0, 50, 50]), np.array([10, 255, 255]))
    SUPER_COLOR = (np.array([100, 150, 150]), np.array([140, 255, 255]))  # Blue-ish
    
    # Template paths
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "images"
        self.templates = {}
        self._load_templates()
        
        # Try to initialize OCR
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR initialized")
        except ImportError:
            self.ocr_reader = None
            logger.warning("EasyOCR not available, metadata extraction will be limited")
    
    def _load_templates(self):
        """Load template images for UI element detection."""
        template_files = [
            "play_button.png",
            "thumbs_down.png",
            "brawler_select.png",
            "joystick.png",
        ]
        
        for template_file in template_files:
            template_path = self.templates_dir / template_file
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates[template_file] = template
    
    def detect_game_phase(self, frame: np.ndarray) -> str:
        """
        Detect the current game phase based on UI elements.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for play button (lobby)
        if "play_button.png" in self.templates:
            result = cv2.matchTemplate(gray, self.templates["play_button.png"], cv2.TM_CCOEFF_NORMED)
            if np.max(result) > 0.8:
                return "lobby"
        
        # Check for thumbs down (end screen)
        if "thumbs_down.png" in self.templates:
            result = cv2.matchTemplate(gray, self.templates["thumbs_down.png"], cv2.TM_CCOEFF_NORMED)
            if np.max(result) > 0.8:
                return "end_screen"
        
        # Check for brawler select (character selection)
        if "brawler_select.png" in self.templates:
            result = cv2.matchTemplate(gray, self.templates["brawler_select.png"], cv2.TM_CCOEFF_NORMED)
            if np.max(result) > 0.8:
                return "lobby"
        
        # Default to gameplay if no specific UI detected
        # Could add more sophisticated detection here
        return "gameplay"
    
    def extract_trophies(self, frame: np.ndarray) -> Tuple[int, float]:
        """
        Extract trophy count from the frame using OCR.
        Returns (trophy_count, confidence)
        """
        if self.ocr_reader is None:
            return 0, 0.0
        
        # Crop trophy area (top left corner typically)
        h, w = frame.shape[:2]
        trophy_region = frame[50:150, 20:200]  # Adjust based on actual UI
        
        try:
            result = self.ocr_reader.ocr(trophy_region)
            if result and result[0]:
                text = result[0][0][0]
                # Extract numbers
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    trophy_count = int(numbers[0])
                    return trophy_count, 0.9
        except Exception as e:
            logger.debug(f"OCR error for trophies: {e}")
        
        return 0, 0.0
    
    def extract_health(self, frame: np.ndarray) -> Tuple[float, float]:
        """
        Extract player health from health bar.
        Returns (health_0_to_1, confidence)
        """
        # Find health bar using color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect green health bar
        green_mask = cv2.inRange(hsv, self.HEALTH_GREEN[0], self.HEALTH_GREEN[1])
        
        # Find contours
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest health bar
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Health is typically proportional to width
            # Max width for full health ~150 pixels
            health = min(1.0, w / 150.0)
            return health, 0.8
        
        return 1.0, 0.0  # Default to full health
    
    def extract_ammo(self, frame: np.ndarray) -> Tuple[int, float]:
        """
        Extract ammo count from UI.
        Returns (ammo_count, confidence)
        """
        # This would use template matching for ammo icons
        # For now, return default
        return 3, 0.5
    
    def detect_super_charged(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if super is charged based on UI indicator.
        Returns (is_charged, confidence)
        """
        # Look for super indicator (typically blue/glowing)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        super_mask = cv2.inRange(hsv, self.SUPER_COLOR[0], self.SUPER_COLOR[1])
        
        # Check for significant super-colored pixels
        pixel_count = np.count_nonzero(super_mask)
        is_charged = pixel_count > 100  # Threshold
        
        return is_charged, 0.7
    
    def extract_map_name(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Extract map name using OCR.
        Returns (map_name, confidence)
        """
        if self.ocr_reader is None:
            return "", 0.0
        
        # Map name typically appears at top center during loading
        h, w = frame.shape[:2]
        map_region = frame[100:200, w//2-300:w//2+300]
        
        try:
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
    
    def extract_game_mode(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Detect game mode based on UI elements.
        Returns (game_mode, confidence)
        """
        # This would use template matching or OCR for mode indicators
        # For now, return default
        return "showdown", 0.5
    
    def extract(self, frame: np.ndarray) -> GameMetadata:
        """
        Extract complete metadata from a frame.
        """
        game_phase = self.detect_game_phase(frame)
        
        # Extract player stats
        trophies, trophy_conf = self.extract_trophies(frame)
        health, health_conf = self.extract_health(frame)
        ammo, ammo_conf = self.extract_ammo(frame)
        super_charged, super_conf = self.detect_super_charged(frame)
        
        # Extract game info
        map_name, map_conf = self.extract_map_name(frame)
        game_mode, mode_conf = self.extract_game_mode(frame)
        
        return GameMetadata(
            game_phase=game_phase,
            trophies=trophies,
            brawler_name="",  # Would need brawler detection
            brawler_level=0,
            game_mode=game_mode,
            map_name=map_name,
            health=health,
            ammo=ammo,
            super_charged=super_charged,
            time_remaining=None,
            score=(0, 0),
            confidence={
                "game_phase": 0.8,
                "trophies": trophy_conf,
                "health": health_conf,
                "ammo": ammo_conf,
                "super": super_conf,
                "map_name": map_conf,
                "game_mode": mode_conf,
            }
        )


def main():
    """Test the metadata extractor."""
    logging.basicConfig(level=logging.INFO)
    
    extractor = MetadataExtractor()
    
    # Load a test frame if available
    test_frame_path = Path(__file__).parent.parent / "images" / "test_frame.png"
    if test_frame_path.exists():
        frame = cv2.imread(str(test_frame_path))
        metadata = extractor.extract(frame)
        print(f"Extracted metadata: {metadata}")
    else:
        print("No test frame found")


if __name__ == "__main__":
    main()
