"""
live_collector.py

Script de captura de dados em tempo real do Brawl Stars.
Captura screenshots do BlueStacks, faz inferência com YOLO,
e guarda imagens + pseudo-labels no formato YOLO para treino futuro.

Uso:
    py live_collector.py --model models/brawlstars_yolov8.pt --output dataset/real_capture --interval 2.0
"""

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger("live_collector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def collect_loop(model_path: str, output_dir: str, interval: float, conf_threshold: float, max_images: int):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    from ultralytics import YOLO
    from pylaai_real.screenshot_taker import ScreenshotTaker

    logger.info("Loading model: %s", model_path)
    model = YOLO(model_path)
    logger.info("Model loaded. Classes: %s", model.names)

    st = ScreenshotTaker("auto")
    if not st.find_window():
        logger.error("Emulator window not found!")
        return

    logger.info("Starting collection: interval=%.1fs, conf=%.2f, max=%d", interval, conf_threshold, max_images)
    collected = 0
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    while collected < max_images:
        start = time.time()
        img = st.take()
        if img is None:
            logger.warning("Screenshot failed, retrying...")
            time.sleep(1)
            continue

        # Inference
        results = model(img, conf=conf_threshold, verbose=False)
        result = results[0]

        # Save image
        img_name = f"{session_id}_f{collected:05d}.jpg"
        img_path = images_dir / img_name
        import cv2
        cv2.imwrite(str(img_path), img)

        # Save YOLO labels (pseudo-labeling)
        label_path = labels_dir / f"{session_id}_f{collected:05d}.txt"
        lines = []
        h, w = img.shape[:2]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Convert to YOLO format: cls_id x_center y_center width height (normalized)
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            label_path.write_text("\n".join(lines) + "\n")
            logger.info("Saved %s with %d detections", img_name, len(lines))
        else:
            logger.info("Saved %s (no detections)", img_name)

        collected += 1
        elapsed = time.time() - start
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)

    logger.info("Collection complete: %d images saved to %s", collected, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Live dataset collector for Brawl Stars")
    parser.add_argument("--model", type=str, default="models/brawlstars_yolov8.pt", help="Path to YOLO model")
    parser.add_argument("--output", type=str, default="dataset/real_capture", help="Output directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between captures")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--max", type=int, default=1000, help="Max images to collect")
    args = parser.parse_args()

    collect_loop(args.model, args.output, args.interval, args.conf, args.max)


if __name__ == "__main__":
    main()
