"""
yolo_world_detector.py

YOLO-World Open-Vocabulary Detector for Brawl Stars.

Provides open-vocabulary detection capabilities using YOLO-World,
allowing detection of objects without specific training.
"""

import logging
from typing import List, Dict, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLOWorldDetector:
    """
    Open-vocabulary detector using YOLO-World.
    
    Allows detection of objects based on text descriptions,
    enabling detection of new objects without retraining.
    """
    
    BRAWL_STARS_PROMPTS = {
        "enemy": ["enemy brawler", "opponent", "foe", "adversary"],
        "teammate": ["teammate brawler", "ally", "friend", "partner"],
        "player": ["my brawler", "my character", "controlled brawler", "player character"],
        "wall": ["wall", "obstacle", "barrier", "block"],
        "bush": ["bush", "vegetation", "hideout", "cover"],
        "powerup": ["powerup", "boost", "item", "collectible"],
        "box": ["box", "crate", "container", "chest"],
        "bullet": ["bullet", "projectile", "attack", "shot"]
    }
    
    def __init__(self, model_path: str = "yolov8s-worldv2.pt", use_all_prompts: bool = True):
        """
        Initialize YOLO-World detector.
        
        Args:
            model_path: Path to YOLO-World model
            use_all_prompts: Whether to use all prompt variations or just primary
        """
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self.use_all_prompts = use_all_prompts
        self.current_classes: List[str] = []
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load the YOLO-World model."""
        try:
            logger.info(f"Loading YOLO-World model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info("YOLO-World model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO-World model: {e}")
            self.is_loaded = False
            return False
    
    def set_classes(self, class_names: List[str]) -> None:
        """
        Set custom vocabulary for detection.
        
        Args:
            class_names: List of class names to detect
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, cannot set classes")
            return
        
        try:
            # Flatten prompts if using all variations
            if self.use_all_prompts:
                prompts = []
                for class_name in class_names:
                    class_lower = class_name.lower()
                    if class_lower in self.BRAWL_STARS_PROMPTS:
                        prompts.extend(self.BRAWL_STARS_PROMPTS[class_lower])
                    else:
                        prompts.append(class_name)
                self.current_classes = prompts
            else:
                self.current_classes = class_names
            
            # Set classes in the model
            self.model.set_classes(self.current_classes)
            logger.info(f"Set {len(self.current_classes)} prompts for detection: {self.current_classes[:5]}...")
            
        except Exception as e:
            logger.error(f"Failed to set classes: {e}")
    
    def detect(self, image, conf_threshold: float = 0.25):
        """
        Detect objects with open-vocabulary.
        
        Args:
            image: Input image (numpy array or path)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            YOLO results object
        """
        if not self.is_loaded:
            logger.warning("Model not loaded, cannot detect")
            return None
        
        try:
            results = self.model(image, conf=conf_threshold, verbose=False)
            return results
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None
    
    def detect_with_specific_classes(self, image, class_names: List[str], conf_threshold: float = 0.25):
        """
        Detect specific classes on-the-fly.
        
        Args:
            image: Input image
            class_names: Classes to detect
            conf_threshold: Confidence threshold
            
        Returns:
            YOLO results object
        """
        # Set classes temporarily
        original_classes = self.current_classes.copy()
        self.set_classes(class_names)
        
        # Detect
        results = self.detect(image, conf_threshold)
        
        # Restore original classes
        self.current_classes = original_classes
        if self.is_loaded:
            try:
                self.model.set_classes(self.current_classes)
            except Exception:
                pass
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "current_classes": len(self.current_classes),
            "class_names": self.current_classes[:10]  # First 10
        }
