"""
test_detection.py

Script de teste rápido para verificar screenshot + deteção YOLO.
Abre o Brawl Stars no BlueStacks, captura um screenshot, corre inference,
e guarda a imagem com bounding boxes em test_detection.jpg.

Uso:
    py test_detection.py --model models/brawlstars_yolov8.pt --output test_detection.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("test_detection")


def main():
    parser = argparse.ArgumentParser(description="Test screenshot + YOLO detection")
    parser.add_argument("--model", type=str, default="models/brawlstars_yolov8.pt", help="Path to YOLO model")
    parser.add_argument("--output", type=str, default="test_detection.jpg", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    args = parser.parse_args()

    from ultralytics import YOLO
    from pylaai_real.screenshot_taker import ScreenshotTaker
    import cv2
    import numpy as np

    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)
    logger.info("Model classes: %s", model.names)

    logger.info("Finding emulator window...")
    st = ScreenshotTaker("auto")
    if not st.find_window():
        logger.error("Emulator window not found! Open BlueStacks first.")
        return 1

    logger.info("Taking screenshot...")
    img = st.take()
    if img is None:
        logger.error("Screenshot failed!")
        return 1
    logger.info("Screenshot captured: shape=%s", img.shape)

    logger.info("Running inference (conf=%.2f)...", args.conf)
    results = model(img, conf=args.conf, verbose=False)
    result = results[0]

    # Draw boxes
    colors = {
        0: (0, 255, 0),    # player - green
        1: (0, 0, 255),    # enemy - red
        2: (255, 255, 0),  # cubebox - cyan
        3: (255, 0, 255),  # powerup - magenta
    }
    drawn = img.copy()
    detections = 0
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        color = colors.get(cls_id, (255, 255, 255))
        label = f"{model.names.get(cls_id, 'unknown')} {conf:.2f}"
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        cv2.putText(drawn, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detections += 1
        logger.info("  Detected: %s (conf=%.2f) at [%d,%d,%d,%d]", model.names.get(cls_id), conf, x1, y1, x2, y2)

    cv2.imwrite(args.output, drawn)
    logger.info("Saved result with %d detections to %s", detections, args.output)
    logger.info("Open %s to inspect detections.", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
