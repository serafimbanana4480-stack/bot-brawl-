"""
sam2_wrapper.py

Wrapper for Meta's Segment Anything Model 2 (SAM2).
Provides easy integration with SAM2 for auto-labeling Brawl Stars datasets.

Installation:
    pip install git+https://github.com/facebookresearch/segment-anything-2.git

Usage:
    from sam2_wrapper import SAM2Wrapper
    sam = SAM2Wrapper(model_type="sam2_hiera_large")
    masks = sam.segment(image, points=[(x, y)], labels=[1])
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np

logger = logging.getLogger(__name__)


class SAM2Wrapper:
    """
    Wrapper for Segment Anything Model 2.
    """
    
    AVAILABLE_MODELS = {
        "sam2_hiera_tiny": "facebook/sam2_hiera_tiny.pt",
        "sam2_hiera_small": "facebook/sam2_hiera_small.pt",
        "sam2_hiera_base": "facebook/sam2_hiera_base.pt",
        "sam2_hiera_large": "facebook/sam2_hiera_large.pt",
    }
    
    def __init__(
        self,
        model_type: str = "sam2_hiera_small",
        device: str = "cpu",
        model_path: Optional[Path] = None,
    ):
        """
        Initialize SAM2 wrapper.
        
        Args:
            model_type: Model type to use (see AVAILABLE_MODELS)
            device: Device to run on ("cpu" or "cuda")
            model_path: Local path to model weights (optional)
        """
        self.model_type = model_type
        self.device = device
        self.model_path = model_path
        self.model = None
        self.predictor = None
        
        self._load_model()
    
    def _load_model(self):
        """Load SAM2 model."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Determine model path
            if self.model_path and self.model_path.exists():
                checkpoint = str(self.model_path)
            else:
                # Will download automatically
                checkpoint = self.AVAILABLE_MODELS.get(self.model_type)
                if not checkpoint:
                    raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Build model config
            config = f"configs/sam2/{self.model_type.replace('sam2_', '')}.yaml"
            
            logger.info(f"Loading SAM2 model: {self.model_type}")
            self.model = build_sam2(config, checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            logger.info("SAM2 model loaded successfully")
            
        except ImportError as e:
            logger.error(f"SAM2 not installed: {e}")
            logger.error("Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM2 model: {e}")
            raise
    
    def segment(
        self,
        image: np.ndarray,
        points: Optional[List[Tuple[int, int]]] = None,
        labels: Optional[List[int]] = None,
        boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        multimask_output: bool = False,
    ) -> np.ndarray:
        """
        Segment objects in an image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            points: List of (x, y) point prompts
            labels: List of point labels (1 for foreground, 0 for background)
            boxes: List of bounding boxes (x1, y1, x2, y2)
            multimask_output: Whether to output multiple masks per prompt
            
        Returns:
            masks: (N, H, W) array of binary masks
        """
        if self.predictor is None:
            raise RuntimeError("SAM2 predictor not initialized")
        
        # Convert BGR to RGB if needed
        if image.shape[2] == 3 and image.dtype == np.uint8:
            # Assume BGR (OpenCV default), convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Set image
        self.predictor.set_image(image_rgb)
        
        # Prepare input
        input_point = np.array(points) if points else None
        input_label = np.array(labels) if labels else None
        input_box = np.array(boxes) if boxes else None
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=multimask_output,
        )
        
        return masks
    
    def segment_from_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Segment object from a bounding box prompt.
        
        Args:
            image: Input image (H, W, 3)
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            mask: (H, W) binary mask
        """
        masks = self.segment(image, boxes=[box], multimask_output=False)
        return masks[0] if len(masks) > 0 else np.zeros(image.shape[:2], dtype=np.uint8)
    
    def segment_from_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        label: int = 1,
    ) -> np.ndarray:
        """
        Segment object from a point prompt.
        
        Args:
            image: Input image (H, W, 3)
            point: Point (x, y)
            label: Point label (1 for foreground, 0 for background)
            
        Returns:
            mask: (H, W) binary mask
        """
        masks = self.segment(image, points=[point], labels=[label], multimask_output=False)
        return masks[0] if len(masks) > 0 else np.zeros(image.shape[:2], dtype=np.uint8)
    
    def auto_segment(
        self,
        image: np.ndarray,
        num_points: int = 5,
    ) -> List[np.ndarray]:
        """
        Automatic segmentation using grid of points.
        
        Args:
            image: Input image (H, W, 3)
            num_points: Number of points per dimension for grid
            
        Returns:
            List of (H, W) binary masks
        """
        h, w = image.shape[:2]
        
        # Generate grid of points
        points = []
        for i in range(num_points):
            for j in range(num_points):
                x = int(w * (i + 0.5) / num_points)
                y = int(h * (j + 0.5) / num_points)
                points.append((x, y))
        
        # Segment each point
        masks = []
        for point in points:
            mask = self.segment_from_point(image, point, label=1)
            if mask.sum() > 100:  # Filter very small masks
                masks.append(mask)
        
        # Merge overlapping masks
        merged_masks = self._merge_overlapping_masks(masks)
        return merged_masks
    
    def _merge_overlapping_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge masks with high IoU."""
        if not masks:
            return []
        
        # Sort by size (largest first)
        masks_sorted = sorted(masks, key=lambda m: m.sum(), reverse=True)
        
        merged = []
        for mask in masks_sorted:
            # Check overlap with existing masks
            overlap = False
            for existing in merged:
                iou = self._calculate_iou(mask, existing)
                if iou > 0.5:
                    overlap = True
                    break
            
            if not overlap:
                merged.append(mask)
        
        return merged
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0


# Import cv2 for color conversion
try:
    import cv2
except ImportError:
    cv2 = None


def main():
    """Test SAM2 wrapper."""
    import cv2
    
    logging.basicConfig(level=logging.INFO)
    
    # Load test image
    test_image_path = Path(__file__).parent.parent.parent / "images" / "test_frame.png"
    if not test_image_path.exists():
        print("No test image found")
        return
    
    image = cv2.imread(str(test_image_path))
    
    # Initialize SAM2
    sam = SAM2Wrapper(model_type="sam2_hiera_small")
    
    # Test box segmentation
    h, w = image.shape[:2]
    box = (w//4, h//4, 3*w//4, 3*h//4)
    mask = sam.segment_from_box(image, box)
    
    print(f"Generated mask with {mask.sum()} pixels")
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentation Mask")
    plt.show()


if __name__ == "__main__":
    main()
