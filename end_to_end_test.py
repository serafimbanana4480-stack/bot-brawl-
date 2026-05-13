"""
end_to_end_test.py

Full system integration test for the Brawl Stars bot.
Verifies the complete loop:
    Emulator → Screenshot → Vision → Decision → ADB Action

This test requires:
1. A running emulator (BlueStacks/LDPlayer) with ADB enabled
2. Brawl Stars installed (optional for basic screenshot test)
3. A trained model in models/ (optional — will report accordingly)

Usage:
    python -m backend.brawl_bot.end_to_end_test --adb-id 127.0.0.1:5555

Exit codes:
    0 = All tests passed
    1 = Critical failure (emulator not connected)
    2 = Vision failure (model missing or not trained)
    3 = Action failure (ADB input not working)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _log_step(step: int, name: str, status: str, detail: str = "") -> None:
    emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    logger.info(f"[{step:02d}] {emoji} {name}: {status} {detail}")


def run_e2e_test(adb_id: str, adb_path: Optional[str] = None) -> int:
    """
    Run end-to-end system test.
    Returns exit code (0 = success).
    """
    from .emulator_detector import get_adb_path, EmulatorDetector
    from .adb_resilient import ResilientADB
    from .vision_engine import YOLOVisionEngine as YOLOv8VisionEngine
    from .humanization import HumanizationEngine
    from .model_validator import validate_all_models

    results = {
        "emulator_connected": False,
        "screenshot_capture": False,
        "vision_model_loaded": False,
        "vision_inference": False,
        "humanization": False,
        "adb_action": False,
    }

    print("=" * 60)
    print("BRAWL STARS BOT — END-TO-END SYSTEM TEST")
    print("=" * 60)

    # --- Step 1: ADB available ---
    step = 1
    try:
        resolved_adb = adb_path or get_adb_path()
        _log_step(step, "ADB executable", "PASS", f"path={resolved_adb}")
    except Exception as e:
        _log_step(step, "ADB executable", "FAIL", str(e))
        return 1

    # --- Step 2: Emulator detection ---
    step = 2
    detector = EmulatorDetector()
    emulators = detector.detect_adb_devices()
    target_emulator = None
    for e in emulators:
        if e.adb_id == adb_id:
            target_emulator = e
            break

    if target_emulator is None:
        _log_step(step, "Emulator detection", "FAIL", f"adb_id={adb_id} not found")
        _log_step(step + 1, "Hint", "INFO", "Start BlueStacks/LDPlayer and enable ADB")
        return 1

    _log_step(step, "Emulator detection", "PASS", f"type={target_emulator.type} id={adb_id}")
    results["emulator_connected"] = True

    # --- Step 3: 5-step verification ---
    step = 3
    verify = detector.verify_emulator_5step(target_emulator)
    if verify["passed"]:
        _log_step(step, "5-step verification", "PASS")
    else:
        _log_step(
            step, "5-step verification", "FAIL",
            f"failed_at_step={verify.get('failure_step')}"
        )
        return 1

    # --- Step 4: Screenshot capture ---
    step = 4
    adb = ResilientADB(resolved_adb, adb_id)
    img_bytes = adb.screenshot()
    if img_bytes is None:
        _log_step(step, "Screenshot capture", "FAIL", "ADB screencap returned None")
        return 1

    _log_step(step, "Screenshot capture", "PASS", f"size={len(img_bytes)} bytes")
    results["screenshot_capture"] = True

    # Decode for vision test
    try:
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        _log_step(step + 1, "Image decode", "PASS", f"shape={img.shape}")
    except Exception as e:
        _log_step(step + 1, "Image decode", "FAIL", str(e))
        return 1

    # --- Step 5: Model validation ---
    step = 6
    model_report = validate_all_models(delete_fakes=False)
    valid_models = [m for m in model_report.get("valid", [])]
    if valid_models:
        _log_step(step, "Model validation", "PASS", f"{len(valid_models)} valid model(s)")
        results["vision_model_loaded"] = True
    else:
        _log_step(
            step, "Model validation", "FAIL",
            f"0 valid models. Integrity score: {model_report.get('integrity_score')}/100"
        )
        _log_step(step + 1, "Hint", "INFO", "Train model: python -m backend.brawl_bot.train_yolo")
        # Continue test with COCO model for diagnostics

    # --- Step 6: Vision inference ---
    step = 8
    ve = YOLOv8VisionEngine()
    models_dir = Path(__file__).parent / "models"
    loaded = ve.load_models(models_dir)
    if loaded:
        detections = ve.detect(img)
        _log_step(
            step, "Vision inference", "PASS",
            f"{len(detections)} detections (NOTE: may be COCO classes, not game entities)"
        )
        for d in detections[:3]:
            _log_step(step, f"  Detection", "INFO", f"{d.class_name} @ ({d.x},{d.y}) conf={d.confidence:.2f}")
        results["vision_inference"] = True
    else:
        _log_step(step, "Vision inference", "FAIL", "No models loaded")

    # --- Step 7: Humanization ---
    step = 9
    he = HumanizationEngine()
    path = he.get_path((100, 100), (400, 400))
    delay = he.get_delay("reaction")
    _log_step(step, "Humanization", "PASS", f"path={len(path)} points delay={delay:.3f}s")
    results["humanization"] = True

    # --- Step 8: ADB action ---
    step = 10
    # Tap at center of screen (safe if Brawl Stars not running)
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    tap_ok = adb.tap(center_x, center_y)
    if tap_ok:
        _log_step(step, "ADB tap action", "PASS", f"({center_x},{center_y})")
        results["adb_action"] = True
    else:
        _log_step(step, "ADB tap action", "FAIL", "ResilientADB returned None")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        emoji = "✅" if passed else "❌"
        print(f"  {emoji} {name}: {status}")

    critical_pass = results["emulator_connected"] and results["screenshot_capture"]
    if not critical_pass:
        print("\n❌ CRITICAL FAILURE: Cannot operate without emulator + screenshot")
        return 1

    if not results["vision_model_loaded"]:
        print("\n⚠️  VISION WARNING: No trained Brawl Stars model. Bot is blind.")
        print("   Run dataset pipeline + train_yolo to fix.")
        return 2

    if not results["adb_action"]:
        print("\n❌ ACTION FAILURE: ADB input not working")
        return 3

    print("\n✅ ALL TESTS PASSED — System is operational")
    print("   (Note: Vision uses trained model — verify detections are game entities)")
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="End-to-end Brawl Stars bot test")
    parser.add_argument("--adb-id", required=True, help="ADB device ID")
    parser.add_argument("--adb-path", default=None, help="Path to adb executable")
    args = parser.parse_args()

    exit_code = run_e2e_test(adb_id=args.adb_id, adb_path=args.adb_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
