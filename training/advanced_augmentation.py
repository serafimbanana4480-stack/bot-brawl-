"""
advanced_augmentation.py

Practical augmentation toolkit for the current repo.

This module intentionally avoids hard dependencies on Albumentations so it
works in the current environment, but it can transparently use it if present.
The fallback path still provides strong, deterministic-friendly transforms:
- random brightness/contrast
- Gaussian blur
- Gaussian noise
- hue/saturation jitter
- horizontal flip
- mild perspective warp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AugmentationPolicy:
    """Config for image augmentation strength."""

    flip_prob: float = 0.5
    blur_prob: float = 0.15
    noise_prob: float = 0.20
    color_prob: float = 0.65
    perspective_prob: float = 0.10
    brightness_limit: Tuple[int, int] = (-20, 20)
    contrast_limit: Tuple[float, float] = (0.85, 1.15)
    saturation_limit: Tuple[float, float] = (0.85, 1.20)
    hue_limit: Tuple[int, int] = (-8, 8)


class AdvancedAugmenter:
    """Augmentation helper with optional Albumentations support."""

    def __init__(self, policy: Optional[AugmentationPolicy] = None, seed: Optional[int] = None):
        self.policy = policy or AugmentationPolicy()
        self.rng = np.random.default_rng(seed)
        self._albumentations = None
        self._build_albumentations()

    def _build_albumentations(self) -> None:
        try:
            import albumentations as A  # type: ignore
            self._albumentations = A.Compose([
                A.HorizontalFlip(p=self.policy.flip_prob),
                A.RandomBrightnessContrast(
                    brightness_limit=self.policy.brightness_limit,
                    contrast_limit=(self.policy.contrast_limit[0] - 1.0, self.policy.contrast_limit[1] - 1.0),
                    p=self.policy.color_prob,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=self.policy.hue_limit,
                    sat_shift_limit=(int((self.policy.saturation_limit[0] - 1.0) * 100), int((self.policy.saturation_limit[1] - 1.0) * 100)),
                    val_shift_limit=10,
                    p=self.policy.color_prob,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=self.policy.blur_prob),
                A.GaussNoise(std_range=(0.01, 0.05), p=self.policy.noise_prob),
                A.Perspective(scale=(0.02, 0.06), p=self.policy.perspective_prob),
            ])
            logger.info("Albumentations detected - using full augmentation stack")
        except Exception:
            self._albumentations = None

    def _ensure_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        alpha = float(self.rng.uniform(*self.policy.contrast_limit))
        beta = int(self.rng.integers(self.policy.brightness_limit[0], self.policy.brightness_limit[1] + 1))
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        sigma = float(self.rng.uniform(5.0, 20.0))
        noise = self.rng.normal(0.0, sigma, image.shape).astype(np.float32)
        out = image.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = float(self.rng.uniform(*self.policy.saturation_limit))
        hue_shift = int(self.rng.integers(self.policy.hue_limit[0], self.policy.hue_limit[1] + 1))
        hsv[:, :, 1] *= sat_scale
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _perspective(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        dx = int(w * float(self.rng.uniform(0.01, 0.04)))
        dy = int(h * float(self.rng.uniform(0.01, 0.04)))
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst = np.float32([
            [dx, dy],
            [w - 1 - dx, dy],
            [dx, h - 1 - dy],
            [w - 1 - dx, h - 1 - dy],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply one augmentation pipeline to an image."""
        image = self._ensure_uint8(image)

        if self._albumentations is not None:
            result = self._albumentations(image=image)
            return self._ensure_uint8(result["image"])

        out = image.copy()
        if self.rng.random() < self.policy.flip_prob:
            out = cv2.flip(out, 1)
        if self.rng.random() < self.policy.color_prob:
            out = self._brightness_contrast(out)
        if self.rng.random() < self.policy.color_prob:
            out = self._color_jitter(out)
        if self.rng.random() < self.policy.blur_prob:
            out = cv2.GaussianBlur(out, (3, 3), 0)
        if self.rng.random() < self.policy.noise_prob:
            out = self._gaussian_noise(out)
        if self.rng.random() < self.policy.perspective_prob:
            out = self._perspective(out)
        return self._ensure_uint8(out)
