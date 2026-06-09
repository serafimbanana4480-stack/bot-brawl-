"""
Test script for vision fixes.
Loads the UnifiedStateDetector against real screenshots and prints detected states.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on path

from pylaai_real.unified_state_detector import UnifiedStateDetector


def main():
    images_path = Path("images")
    detector = UnifiedStateDetector(images_path=images_path, window_w=1920, window_h=1080)

    # Use screenshots with known/expected states
    test_cases = [
        ("lobby", "auto_test_093747_c1.png"),
        ("matchmaking", "auto_play_100328_matchmaking.png"),
        ("matchmaking", "auto_play_100338_matchmaking.png"),
    ]

    print("=" * 70)
    print("VISION FIX TEST RESULTS")
    print("=" * 70)

    all_pass = True
    for expected, filename in test_cases:
        filepath = Path(filename)
        if not filepath.exists():
            print(f"[SKIP] {filename} not found")
            continue

        img_bgr = cv2.imread(str(filepath))
        if img_bgr is None:
            print(f"[FAIL] Could not load {filename}")
            all_pass = False
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = detector.detect(img_rgb)

        status = "PASS" if result.state == expected else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(
            f"[{status}] {filename:45s} -> "
            f"state={result.state:12s} conf={result.confidence:.2f} method={result.method}"
        )
        print(f"       details={result.details}")
        print()

    # Also run one extra screenshot without a strict expectation just to verify no crash
    extra = "screenshot_now.png"
    if Path(extra).exists():
        img_bgr = cv2.imread(extra)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = detector.detect(img_rgb)
        print(
            f"[INFO] {extra:45s} -> "
            f"state={result.state:12s} conf={result.confidence:.2f} method={result.method}"
        )
        print(f"       details={result.details}")
        print()

    print("=" * 70)
    if all_pass:
        print("All targeted tests PASSED")
    else:
        print("Some targeted tests FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
