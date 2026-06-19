"""
Feature Extractors for specific game state features.

This module provides specialized extractors for HUD elements and other
game features that are not detected by the main YOLO vision pipeline.

Extractors:
- GadgetExtractor: Extracts gadget_ready from HUD
- HyperchargeExtractor: Extracts hypercharge_ready from HUD
- CooldownExtractor: Extracts cooldowns of attack/super from HUD
- LineOfSightExtractor: Calculates line_of_sight_free using OccupancyGrid + raycast

Usage:
    from vision.feature_extractors import GadgetExtractor, CooldownExtractor

    gadget_ext = GadgetExtractor()
    gadget_ready = gadget_ext.extract(screenshot, player_bbox)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HUDFeatures:
    """Container for extracted HUD features."""
    gadget_ready: bool = False
    hypercharge_ready: bool = False
    cooldown_attack: float = 0.0  # 0.0 to 1.0 progress
    cooldown_super: float = 0.0  # 0.0 to 1.0 progress
    ammo_current: int = 3
    ammo_max: int = 3
    super_charge: float = 0.0  # 0.0 to 1.0


class GadgetExtractor:
    """
    Extracts gadget_ready status from the HUD.

    Uses template matching or color thresholding on the gadget icon
    in the bottom-right corner of the screen.
    """

    def __init__(self):
        """Initialize gadget extractor."""
        # Default gadget icon position (bottom-right of screen)
        self.default_region = (1700, 900, 1920, 1080)  # x1, y1, x2, y2

    def extract(self, screenshot, player_bbox=None) -> bool:
        """
        Extract gadget_ready status from HUD.

        Args:
            screenshot: Full game screenshot (numpy array)
            player_bbox: Optional player bounding box for reference

        Returns:
            True if gadget is ready, False otherwise
        """
        if screenshot is None:
            return False

        try:
            # Extract HUD region (bottom-right where gadget icon appears)
            h, w = screenshot.shape[:2]
            gadget_region = screenshot[int(h*0.85):h, int(w*0.85):w]

            # Simple heuristic: check for bright color indicating ready gadget
            # In production, use template matching with gadget icon templates
            mean_brightness = np.mean(gadget_region)

            # Threshold indicating gadget is ready (bright icon)
            gadget_ready = mean_brightness > 100

            return gadget_ready

        except Exception as e:
            logger.warning(f"[GADGET] Failed to extract gadget status: {e}")
            return False


class HyperchargeExtractor:
    """
    Extracts hypercharge_ready status from the HUD.

    Similar to gadget extraction, looks for the hypercharge icon
    in the HUD area.
    """

    def __init__(self):
        """Initialize hypercharge extractor."""
        self.default_region = (1700, 900, 1920, 1080)

    def extract(self, screenshot, player_bbox=None) -> bool:
        """
        Extract hypercharge_ready status from HUD.

        Args:
            screenshot: Full game screenshot (numpy array)
            player_bbox: Optional player bounding box for reference

        Returns:
            True if hypercharge is ready, False otherwise
        """
        if screenshot is None:
            return False

        try:
            h, w = screenshot.shape[:2]
            hyper_region = screenshot[int(h*0.85):h, int(w*0.85):w]

            # Check for hypercharge indicator (glowing effect)
            # In production, use specific color detection for hypercharge glow
            mean_brightness = np.mean(hyper_region)
            hypercharge_ready = mean_brightness > 120

            return hypercharge_ready

        except Exception as e:
            logger.warning(f"[HYPERCHARGE] Failed to extract hypercharge status: {e}")
            return False


class CooldownExtractor:
    """
    Extracts cooldown progress for attack and super abilities.

    Analyzes the cooldown indicators in the HUD (usually circular
    progress bars or filled bars).
    """

    def __init__(self):
        """Initialize cooldown extractor."""
        pass

    def extract(self, screenshot, player_bbox=None) -> tuple[float, float]:
        """
        Extract attack and super cooldown progress.

        Args:
            screenshot: Full game screenshot (numpy array)
            player_bbox: Optional player bounding box for reference

        Returns:
            (cooldown_attack, cooldown_super) as 0.0-1.0 progress values
            1.0 = ready (no cooldown), 0.0 = full cooldown
        """
        if screenshot is None:
            return 1.0, 1.0  # Default to ready

        try:
            h, w = screenshot.shape[:2]

            # Attack button region (bottom-right joystick area)
            attack_region = screenshot[int(h*0.75):int(h*0.90), int(w*0.75):int(w*0.90)]

            # Super button region (bottom-left or near attack button)
            super_region = screenshot[int(h*0.75):int(h*0.90), int(w*0.60):int(w*0.75)]

            # Simple heuristic: check darkness of cooldown overlay
            # Darker = more cooldown (less progress)
            attack_brightness = np.mean(attack_region)
            super_brightness = np.mean(super_region)

            # Normalize to 0-1 range (1.0 = ready, 0.0 = full cooldown)
            cooldown_attack = min(1.0, attack_brightness / 150.0)
            cooldown_super = min(1.0, super_brightness / 150.0)

            return cooldown_attack, cooldown_super

        except Exception as e:
            logger.warning(f"[COOLDOWN] Failed to extract cooldown status: {e}")
            return 1.0, 1.0  # Default to ready


class LineOfSightExtractor:
    """
    Calculates line_of_sight_free using OccupancyGrid + raycast.

    Determines if there's a clear line of sight from player to target
    by checking for obstacles (walls, bushes) along the ray.
    """

    def __init__(self, grid_resolution=50):
        """
        Initialize line of sight extractor.

        Args:
            grid_resolution: Size of grid cells in pixels
        """
        self.grid_resolution = grid_resolution

    def extract(
        self,
        player_pos: tuple[float, float],
        target_pos: tuple[float, float],
        walls: list,
        bushes: list
    ) -> bool:
        """
        Calculate if line of sight is free.

        Args:
            player_pos: Player (x, y) position
            target_pos: Target (x, y) position
            walls: List of wall bounding boxes
            bushes: List of bush bounding boxes

        Returns:
            True if line of sight is clear, False if blocked
        """
        if player_pos is None or target_pos is None:
            return True  # Default to clear if positions unknown

        try:
            # Simple raycast: check if any obstacle intersects the line
            # In production, use proper raycast algorithm with OccupancyGrid

            # Calculate line parameters
            dx = target_pos[0] - player_pos[0]
            dy = target_pos[1] - player_pos[1]
            distance = (dx**2 + dy**2)**0.5

            if distance == 0:
                return True

            # Sample points along the line
            num_samples = int(distance / self.grid_resolution) + 1
            for i in range(1, num_samples):
                t = i / num_samples
                sample_x = player_pos[0] + dx * t
                sample_y = player_pos[1] + dy * t

                # Check if sample point is inside any obstacle
                for wall in walls:
                    if self._point_in_bbox((sample_x, sample_y), wall):
                        return False  # Blocked by wall

                # Bushes also block line of sight (for shooting)
                for bush in bushes:
                    if self._point_in_bbox((sample_x, sample_y), bush):
                        return False  # Blocked by bush

            return True  # Clear line of sight

        except Exception as e:
            logger.warning(f"[LOS] Failed to calculate line of sight: {e}")
            return True  # Default to clear

    def _point_in_bbox(self, point: tuple[float, float], bbox: tuple) -> bool:
        """Check if point is inside bounding box."""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class VelocityCalculator:
    """
    Calculates player velocity from position history.
    """

    def __init__(self, history_length=5):
        """
        Initialize velocity calculator.

        Args:
            history_length: Number of positions to keep for velocity calculation
        """
        self.history_length = history_length
        self.position_history = []

    def update(self, position: tuple[float, float], timestamp: float) -> tuple[float, float]:
        """
        Update position history and calculate velocity.

        Args:
            position: Current (x, y) position
            timestamp: Current timestamp

        Returns:
            Velocity as (vx, vy) in pixels/second
        """
        self.position_history.append((position, timestamp))

        # Keep only recent history
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)

        # Calculate velocity from last two positions
        if len(self.position_history) >= 2:
            (x1, y1), t1 = self.position_history[-2]
            (x2, y2), t2 = self.position_history[-1]

            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                return (vx, vy)

        return (0.0, 0.0)


class HUDFeatureExtractor:
    """
    Combined extractor for all HUD features.

    Provides a single interface to extract all HUD-related features
    in one call for efficiency.
    """

    def __init__(self):
        """Initialize all extractors."""
        self.gadget_extractor = GadgetExtractor()
        self.hypercharge_extractor = HyperchargeExtractor()
        self.cooldown_extractor = CooldownExtractor()
        self.velocity_calculator = VelocityCalculator()

    def extract_all(
        self,
        screenshot,
        player_bbox=None,
        player_position=None,
        timestamp=None
    ) -> HUDFeatures:
        """
        Extract all HUD features.

        Args:
            screenshot: Full game screenshot
            player_bbox: Player bounding box
            player_position: Player position for velocity calculation
            timestamp: Current timestamp for velocity calculation

        Returns:
            HUDFeatures dataclass with all extracted features
        """
        features = HUDFeatures()

        # Extract gadget and hypercharge status
        features.gadget_ready = self.gadget_extractor.extract(screenshot, player_bbox)
        features.hypercharge_ready = self.hypercharge_extractor.extract(screenshot, player_bbox)

        # Extract cooldowns
        features.cooldown_attack, features.cooldown_super = self.cooldown_extractor.extract(
            screenshot, player_bbox
        )

        # Derive a continuous super-charge proxy from the HUD.
        # The HUD only exposes readiness reliably here, so we keep the scalar
        # feature continuous by treating a ready super as fully charged.
        features.super_charge = 1.0 if features.cooldown_super >= 0.95 else features.cooldown_super

        # Calculate velocity
        if player_position and timestamp:
            features.velocity = self.velocity_calculator.update(player_position, timestamp)

        return features
