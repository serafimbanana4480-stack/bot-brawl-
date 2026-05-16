"""
download_roboflow_dataset.py

Download e preprocess dataset do Roboflow Universe para usar como baseline.

Dataset: https://universe.roboflow.com/bloxxy/brawl-stars-dataset
- 2551 imagens
- 10 classes: Ball, Enemy, Friendly, Gem, Hot_Zone, Me, PP, PP_Box, Safe_Enemy, Safe_Friendly
- YOLO11m pré-treinado com 80.5% mAP@50

Uso:
    python training/download_roboflow_dataset.py
"""

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.enhanced_training_pipeline import STANDARD_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("roboflow_download")

# Roboflow dataset URL (formato download API)
ROBOFLOW_URLS = {
    "bloxxy": "https://universe.roboflow.com/ds/bloxxy/brawl-stars-dataset/download?format=yolov8",
}

# Classe mapping: Roboflow -> nosso formato
# Roboflow: Ball, Enemy, Friendly, Gem, Hot_Zone, Me, PP, PP_Box, Safe_Enemy, Safe_Friendly
# Nosso:     Player, Bush, Enemy, Cubebox, Wall, Powerup, Bullet, Super
CLASS_MAP = {
    "Enemy": 2,        # Enemy -> Enemy (index 2)
    "Safe_Enemy": 2,   # Safe_Enemy -> Enemy
    "Friendly": 0,     # Friendly -> Player
    "Me": 0,           # Me -> Player
    "Safe_Friendly": 0,# Safe_Friendly -> Player
    "Ball": 3,         # Ball -> Cubebox (aproximado)
    "Gem": 3,          # Gem -> Cubebox (power cube)
    "PP": 5,           # PP -> Powerup
    "PP_Box": 5,       # PP_Box -> Powerup
    "Hot_Zone": -1,    # Hot_Zone -> skip (nao existe no nosso)
}

# Roboflow class names in order (from dataset)
ROBOFLOW_CLASS_NAMES = ["Ball", "Enemy", "Friendly", "Gem", "Hot_Zone", "Me", "PP", "PP_Box", "Safe_Enemy", "Safe_Friendly"]

# Classes que queremos manter
KEEP_CLASSES = {0, 2, 3, 5}  # Player, Enemy, Cubebox, Powerup


def download_dataset(url: str, output_dir: Path, api_key: str = None) -> bool:
    """Download dataset do Roboflow."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if api_key:
        download_url = f"{url}?api_key={api_key}"
    else:
        logger.warning("No Roboflow API key - using public URL")
        download_url = url

    zip_path = output_dir / "dataset.zip"

    logger.info(f"Downloading from {download_url}")
    logger.info("This may take a few minutes for large datasets...")

    try:
        urlretrieve(download_url, zip_path)
        logger.info(f"Downloaded to {zip_path}")
    except URLError as e:
        logger.error(f"Download failed: {e}")
        logger.error("Make sure you have a valid Roboflow API key")
        return False

    logger.info("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        logger.info(f"Extracted to {output_dir}")
    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid ZIP")
        return False

    zip_path.unlink()
    return True


def remap_classes(dataset_dir: Path) -> int:
    """
    Remapeia classes do dataset Roboflow para nosso formato.
    Remove classes que não nos interessam.
    """
    labels_dir = dataset_dir / "labels"
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return 0

    remapped = 0
    removed = 0

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            cls_id = int(parts[0])
            bbox = parts[1:]

            # Map Roboflow class to our class
            if cls_id < len(ROBOFLOW_CLASS_NAMES):
                roboflow_class_name = ROBOFLOW_CLASS_NAMES[cls_id]
                new_cls_id = CLASS_MAP.get(roboflow_class_name, -1)
            else:
                new_cls_id = -1

            # Skip unwanted classes
            if new_cls_id == -1 or new_cls_id not in KEEP_CLASSES:
                removed += 1
                continue

            new_lines.append(f"{new_cls_id} {' '.join(bbox)}\n")
            remapped += 1

        with open(label_file, 'w') as f:
            f.writelines(new_lines)

    logger.info(f"Remapped {remapped} annotations, removed {removed}")
    return remapped


def get_classes(dataset_dir: Path) -> set:
    """Get all class IDs from dataset."""
    labels_dir = dataset_dir / "labels"
    classes = set()
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            classes.add(int(parts[0]))
    return classes


def verify_compatibility(local_dir: Path, roboflow_dir: Path) -> bool:
    """Verify datasets have compatible classes."""
    local_classes = get_classes(local_dir)
    robo_classes = get_classes(roboflow_dir)
    required = {0, 1, 3, 5}  # Player, Enemy, Cubebox, Powerup
    
    if not required.issubset(robo_classes):
        logger.error(f"Roboflow dataset missing required classes: {required - robo_classes}")
        return False
    
    logger.info(f"Compatibility check passed. Roboflow classes: {robo_classes}")
    return True


def verify_dataset(dataset_dir: Path) -> dict:
    """Verifica integridade do dataset (suporta estrutura YOLO com splits)."""
    stats = {"images": 0, "labels": 0, "classes": set(), "empty_labels": 0}

    # Check for YOLO split structure (train/val/test/images + train/val/test/labels)
    has_splits = False
    for split in ["train", "val", "test"]:
        if (dataset_dir / split / "images").exists():
            has_splits = True
            break

    if has_splits:
        image_dirs = [(dataset_dir / s / "images", dataset_dir / s / "labels")
                      for s in ["train", "val", "test"]
                      if (dataset_dir / s / "images").exists()]
    else:
        # Flat structure: images/ and labels/ directly
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if not images_dir.exists():
            logger.error(f"Images dir not found: {images_dir}")
            return stats
        image_dirs = [(images_dir, labels_dir)]

    for images_dir, labels_dir in image_dirs:
        for img_file in images_dir.glob("*"):
            if img_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                stats["images"] += 1

                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    stats["labels"] += 1

                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        if not lines or all(l.strip() == "" for l in lines):
                            stats["empty_labels"] += 1

                        for line in lines:
                            if line.strip():
                                parts = line.split()
                                if parts:
                                    stats["classes"].add(int(parts[0]))

    return stats


def merge_with_local(local_dir: Path, roboflow_dir: Path, output_dir: Path) -> dict:
    """
    Faz merge do dataset local com Roboflow.
    Prioriza dataset local para classes que já temos.
    """
    logger.info("=" * 60)
    logger.info("MERGING DATASETS")
    logger.info("=" * 60)

    # Verify compatibility
    if not verify_compatibility(local_dir, roboflow_dir):
        logger.error("Datasets are not compatible, aborting merge")
        return {}

    # Verify minimum dataset size
    local_images = len(list(local_dir.glob("**/*.png"))) + len(list(local_dir.glob("**/*.jpg")))
    robo_images = len(list(roboflow_dir.glob("**/*.png"))) + len(list(roboflow_dir.glob("**/*.jpg")))
    
    if local_images < 100:
        logger.error(f"Local dataset too small: {local_images} images (minimum: 100)")
        return {}
    
    if robo_images < 100:
        logger.error(f"Roboflow dataset too small: {robo_images} images (minimum: 100)")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy local dataset
    logger.info(f"Copying local dataset from {local_dir}")
    local_stats = _copy_split(local_dir, output_dir)

    # Append roboflow dataset
    logger.info(f"Appending Roboflow dataset from {roboflow_dir}")
    robo_stats = _copy_split(roboflow_dir, output_dir, append=True)

    return {"local": local_stats, "roboflow": robo_stats}


def _copy_split(source_dir: Path, dest_dir: Path, append: bool = False) -> dict:
    """Copy a dataset split to destination."""
    stats = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        src_images = source_dir / split / "images"
        src_labels = source_dir / split / "labels"

        if not src_images.exists():
            continue

        dst_images = dest_dir / split / "images"
        dst_labels = dest_dir / split / "labels"

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        start_idx = 0
        if append:
            start_idx = len(list(dst_images.glob("*")))

        for img_file in src_images.glob("*"):
            if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            i = list(src_images.glob("*")).index(img_file)
            new_name = f"robo_{start_idx + i:05d}{img_file.suffix}"
            shutil.copy2(img_file, dst_images / new_name)

            label_file = src_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, dst_labels / f"robo_{start_idx + i:05d}.txt")

            stats[split] += 1

    return stats


def create_merged_yaml(dataset_dir: Path) -> Path:
    """Cria data.yaml para dataset mesclado."""
    import yaml

    # Detect actual classes in dataset
    actual_classes = get_classes(dataset_dir)
    
    # Only include classes that actually exist
    classes = {i: name for i, name in STANDARD_CLASSES.items() if i in actual_classes}
    
    config = {
        "path": str(dataset_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": classes,
        "nc": len(classes),
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Created {yaml_path} with {len(classes)} classes: {classes}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow Brawl Stars dataset")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Roboflow API key (get from https://app.roboflow.com/settings/api)")
    parser.add_argument("--url", type=str, default=None,
                       help="Custom dataset URL")
    parser.add_argument("--output", type=str, default="dataset/roboflow_raw",
                       help="Output directory")
    parser.add_argument("--merge", action="store_true",
                       help="Merge with local dataset")
    parser.add_argument("--local-dataset", type=str, default="dataset/yolo_v2",
                       help="Local dataset to merge with")
    parser.add_argument("--remap-only", action="store_true",
                       help="Only remap classes in existing dataset")
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Remap only mode
    if args.remap_only:
        logger.info(f"Remapping classes in {output_dir}")
        remap_classes(output_dir)
        verify_dataset(output_dir)
        return

    # Download dataset
    url = args.url or ROBOFLOW_URLS.get("bloxxy", "https://universe.roboflow.com/ds/bloxxy/brawl-stars-dataset/download?format=yolov8")

    logger.info("=" * 60)
    logger.info("DOWNLOADING ROBOFLOW DATASET")
    logger.info("=" * 60)
    logger.info("Note: You need a valid Roboflow API key to download")
    logger.info("Get one free at: https://app.roboflow.com/settings/api")

    if not args.api_key:
        logger.error("Roboflow API key is REQUIRED")
        logger.error("Get one free at: https://app.roboflow.com/settings/api")
        return

    success = download_dataset(url, output_dir, args.api_key)

    if not success:
        logger.error("Download failed. Try with --api-key YOUR_KEY")
        logger.error("Or download manually from: https://universe.roboflow.com/bloxxy/brawl-stars-dataset")
        return

    logger.info("Remapping classes...")
    remap_classes(output_dir)

    logger.info("Verifying dataset...")
    stats = verify_dataset(output_dir)
    logger.info(f"Dataset stats: {stats['images']} images, {len(stats['classes'])} classes")

    if args.merge:
        local_dir = Path(args.local_dataset)
        if local_dir.exists():
            merge_with_local(local_dir, output_dir, Path("dataset/merged"))
            create_merged_yaml(Path("dataset/merged"))
        else:
            logger.warning(f"Local dataset not found: {local_dir}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
