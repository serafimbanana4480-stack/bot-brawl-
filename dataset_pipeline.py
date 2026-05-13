"""
dataset_pipeline.py

Complete dataset collection and preprocessing pipeline for Brawl Stars YOLO training.

Features:
- Automated screenshot capture via ADB
- Frame deduplication (skip near-duplicate frames)
- Metadata tagging (game state, timestamp, resolution)
- Action logging (movement, attack, super)
- Export to YOLO format (images + labels directory structure)
- Auto-annotation helper (saves empty label files for manual annotation)
- Integration with gameplay_recorder for rich data

Usage:
    python -m backend.brawl_bot.dataset_pipeline --adb-id 127.0.0.1:5555 --duration 300

Output structure:
    dataset/
    ├── raw/              ← All captured screenshots
    ├── metadata/         ← Rich metadata for each frame
    ├── actions/          ← Logged actions (for behavior cloning)
    ├── train/images/     ← Training set images
    ├── train/labels/     ← Training set labels (empty .txt ready for annotation)
    ├── val/images/       ← Validation set images
    ├── val/labels/       ← Validation set labels
    └── data.yaml         ← YOLO dataset config
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BRAWL_STARS_CLASSES = [
    "enemy",
    "teammate",
    "player",
    "wall",
    "bush",
    "powerup",
    "box",
    "bullet",
]


def _adb_screencap(adb_path: str, adb_id: str) -> Optional[bytes]:
    """Capture screenshot from emulator. Returns PNG bytes or None."""
    try:
        result = subprocess.run(
            [adb_path, "-s", adb_id, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.error(f"screencap failed: rc={result.returncode}")
            return None
        if len(result.stdout) < 1024:
            logger.error(f"screencap returned tiny output ({len(result.stdout)} bytes)")
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.error("screencap timeout")
        return None
    except Exception as e:
        logger.error(f"screencap exception: {e}")
        return None


def _image_hash(img_bytes: bytes) -> str:
    """Fast perceptual hash using average pixel value of downscaled image."""
    try:
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return hashlib.md5(img_bytes).hexdigest()
        small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        avg = small.mean()
        bits = "".join("1" if p > avg else "0" for p in small.flatten())
        return bits
    except Exception:
        return hashlib.md5(img_bytes).hexdigest()


def _is_duplicate(last_hash: Optional[str], current_hash: str, threshold: int = 10) -> bool:
    """Check if two perceptual hashes are similar (Hamming distance)."""
    if last_hash is None:
        return False
    if len(last_hash) != len(current_hash):
        return False
    distance = sum(a != b for a, b in zip(last_hash, current_hash))
    return distance < threshold


class DatasetPipeline:
    def __init__(
        self,
        adb_id: str,
        adb_path: Optional[str] = None,
        output_dir: Path = Path("dataset"),
        capture_interval: float = 1.0,
        dedup_threshold: int = 10,
        train_split: float = 0.8,
        record_actions: bool = True,
        extract_metadata: bool = True,
    ):
        self.adb_id = adb_id
        self.adb_path = adb_path or self._find_adb()
        self.output_dir = output_dir
        self.capture_interval = capture_interval
        self.dedup_threshold = dedup_threshold
        self.train_split = train_split
        self.record_actions = record_actions
        self.extract_metadata = extract_metadata

        self.raw_dir = output_dir / "raw"
        self.metadata_dir = output_dir / "metadata"
        self.actions_dir = output_dir / "actions"
        self.train_img_dir = output_dir / "train" / "images"
        self.train_lbl_dir = output_dir / "train" / "labels"
        self.val_img_dir = output_dir / "val" / "images"
        self.val_lbl_dir = output_dir / "val" / "labels"

        self.frame_count = 0
        self.duplicate_count = 0
        self.last_hash: Optional[str] = None
        
        # Metadata extractor
        self.metadata_extractor = None
        if self.extract_metadata:
            try:
                from .automation.metadata_extractor import MetadataExtractor
                self.metadata_extractor = MetadataExtractor()
                logger.info("Metadata extractor initialized")
            except ImportError:
                logger.warning("Metadata extractor not available")

    def _find_adb(self) -> str:
        from .emulator_detector import get_adb_path
        return get_adb_path()

    def _ensure_dirs(self) -> None:
        for d in [
            self.raw_dir,
            self.metadata_dir,
            self.actions_dir,
            self.train_img_dir,
            self.train_lbl_dir,
            self.val_img_dir,
            self.val_lbl_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _write_data_yaml(self) -> None:
        yaml_path = self.output_dir / "data.yaml"
        content = f"""path: {self.output_dir.absolute()}
train: train/images
val: val/images

nc: {len(BRAWL_STARS_CLASSES)}
names: {BRAWL_STARS_CLASSES}
"""
        yaml_path.write_text(content, encoding="utf-8")
        logger.info(f"data.yaml written to {yaml_path}")

    def capture_batch(
        self,
        duration_seconds: float,
        max_frames: Optional[int] = None,
    ) -> dict:
        """
        Capture screenshots for a given duration.
        Deduplicates frames and splits into train/val.
        """
        self._ensure_dirs()
        start_time = time.time()
        consecutive_failures = 0
        max_failures = 5

        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                logger.info(f"Duration reached ({duration_seconds}s)")
                break
            if max_frames and self.frame_count >= max_frames:
                logger.info(f"Max frames reached ({max_frames})")
                break

            img_bytes = _adb_screencap(self.adb_path, self.adb_id)
            if img_bytes is None:
                consecutive_failures += 1
                logger.warning(f"Capture failed ({consecutive_failures}/{max_failures})")
                if consecutive_failures >= max_failures:
                    logger.error("Too many failures, stopping capture")
                    break
                time.sleep(self.capture_interval)
                continue

            consecutive_failures = 0
            phash = _image_hash(img_bytes)

            if _is_duplicate(self.last_hash, phash, self.dedup_threshold):
                self.duplicate_count += 1
                time.sleep(self.capture_interval)
                continue

            self.last_hash = phash
            self.frame_count += 1

            # Save raw
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            raw_name = f"frame_{timestamp}_{self.frame_count:05d}.png"
            raw_path = self.raw_dir / raw_name
            raw_path.write_bytes(img_bytes)
            
            # Extract and save metadata
            if self.extract_metadata and self.metadata_extractor:
                try:
                    import cv2
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        metadata = self.metadata_extractor.extract(img)
                        metadata_path = self.metadata_dir / f"{raw_name.replace('.png', '.json')}"
                        import json
                        with open(metadata_path, 'w') as f:
                            json.dump({
                                'frame_id': self.frame_count,
                                'timestamp': timestamp,
                                'metadata': metadata.__dict__
                            }, f, indent=2)
                except Exception as e:
                    logger.debug(f"Metadata extraction failed: {e}")

            # Split train/val
            is_train = np.random.random() < self.train_split
            if is_train:
                img_dir = self.train_img_dir
                lbl_dir = self.train_lbl_dir
            else:
                img_dir = self.val_img_dir
                lbl_dir = self.val_lbl_dir

            img_path = img_dir / raw_name
            shutil.copy2(raw_path, img_path)

            # Write empty label file (ready for annotation)
            lbl_name = raw_name.replace(".png", ".txt")
            lbl_path = lbl_dir / lbl_name
            if not lbl_path.exists():
                lbl_path.write_text("")

            if self.frame_count % 10 == 0:
                logger.info(
                    f"Captured {self.frame_count} unique frames "
                    f"({self.duplicate_count} duplicates skipped)"
                )

            time.sleep(self.capture_interval)

        self._write_data_yaml()

        return {
            "total_frames": self.frame_count,
            "duplicates_skipped": self.duplicate_count,
            "train_dir": str(self.train_img_dir),
            "val_dir": str(self.val_img_dir),
            "data_yaml": str(self.output_dir / "data.yaml"),
        }

    def validate_dataset(self) -> dict:
        """Verify dataset structure and report statistics."""
        stats = {
            "train_images": len(list(self.train_img_dir.glob("*.png"))),
            "train_labels": len(list(self.train_lbl_dir.glob("*.txt"))),
            "val_images": len(list(self.val_img_dir.glob("*.png"))),
            "val_labels": len(list(self.val_lbl_dir.glob("*.txt"))),
            "data_yaml_exists": (self.output_dir / "data.yaml").exists(),
        }

        # Check for orphaned images (no label file)
        orphaned = []
        for img in self.train_img_dir.glob("*.png"):
            if not (self.train_lbl_dir / img.name.replace(".png", ".txt")).exists():
                orphaned.append(img.name)
        for img in self.val_img_dir.glob("*.png"):
            if not (self.val_lbl_dir / img.name.replace(".png", ".txt")).exists():
                orphaned.append(img.name)

        stats["orphaned_images"] = orphaned
        stats["is_valid"] = (
            stats["train_images"] > 0
            and stats["train_labels"] == stats["train_images"]
            and stats["val_images"] > 0
            and stats["val_labels"] == stats["val_images"]
            and stats["data_yaml_exists"]
            and len(orphaned) == 0
        )

        return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Brawl Stars Dataset Pipeline")
    parser.add_argument("--adb-id", required=True, help="ADB device ID")
    parser.add_argument("--adb-path", default=None, help="Path to adb executable")
    parser.add_argument("--duration", type=float, default=300, help="Capture duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between captures")
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to capture")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset")
    args = parser.parse_args()

    pipeline = DatasetPipeline(
        adb_id=args.adb_id,
        adb_path=args.adb_path,
        output_dir=args.output,
        capture_interval=args.interval,
    )

    if args.validate:
        stats = pipeline.validate_dataset()
        print(json.dumps(stats, indent=2))
        sys.exit(0 if stats["is_valid"] else 1)

    result = pipeline.capture_batch(
        duration_seconds=args.duration,
        max_frames=args.max_frames,
    )

    print("\n" + "=" * 50)
    print("DATASET CAPTURE COMPLETE")
    print("=" * 50)
    print(f"Total unique frames: {result['total_frames']}")
    print(f"Train dir: {result['train_dir']}")
    print(f"Val dir: {result['val_dir']}")
    print(f"data.yaml: {result['data_yaml']}")
    print("\nNEXT STEP: Label frames with Label Studio or Roboflow")
    print("  Classes:", BRAWL_STARS_CLASSES)
    print("=" * 50)


if __name__ == "__main__":
    main()
