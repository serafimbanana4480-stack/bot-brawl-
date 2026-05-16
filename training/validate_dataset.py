"""
validate_dataset.py

Valida o dataset YOLO: verifica estrutura, conta classes, gera estatísticas.
Executar antes do treino para garantir que o dataset está pronto.

Usage:
    python training/validate_dataset.py
    python training/validate_dataset.py --dataset dataset/yolo_final
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("validate_dataset")

CLASS_NAMES = {
    0: "Player",
    1: "Enemy",
    2: "Bush",
    3: "Cubebox",
    4: "Wall",
    5: "Powerup",
    6: "Bullet",
    7: "Super",
}


def validate_structure(dataset_dir: Path) -> dict:
    """Valida estrutura de diretórios do dataset."""
    errors = []
    warnings = []

    required_dirs = {
        "train": ["images", "labels"],
        "val": ["images", "labels"],
        "test": ["images", "labels"],
    }

    for split, subdirs in required_dirs.items():
        for subdir in subdirs:
            path = dataset_dir / split / subdir
            if not path.exists():
                errors.append(f"Missing directory: {path}")
            elif not any(path.iterdir()):
                warnings.append(f"Empty directory: {path}")

    return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}


def count_classes(dataset_dir: Path) -> dict:
    """Conta distribuição de classes no dataset."""
    class_counts = defaultdict(int)
    image_counts = {"train": 0, "val": 0, "test": 0}
    total_boxes = 0
    empty_images = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / split / "labels"
        images_dir = dataset_dir / split / "images"

        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            # Count images
            img_name = label_file.stem
            img_extensions = [".png", ".jpg", ".jpeg"]
            has_image = any((images_dir / f"{img_name}{ext}").exists() for ext in img_extensions)

            if has_image:
                image_counts[split] += 1

            # Parse labels
            boxes_in_image = 0
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_counts[cls_id] += 1
                        boxes_in_image += 1
                        total_boxes += 1

            if boxes_in_image == 0:
                empty_images[split] += 1

    return {
        "class_counts": dict(class_counts),
        "image_counts": dict(image_counts),
        "empty_images": dict(empty_images),
        "total_boxes": total_boxes,
    }


def check_image_quality(dataset_dir: Path, sample_size: int = 20) -> dict:
    """Verifica qualidade de sample de imagens."""
    blurry = 0
    dark = 0
    small = 0
    checked = 0

    for split in ["train", "val", "test"]:
        images_dir = dataset_dir / split / "images"
        if not images_dir.exists():
            continue

        images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
        images = images[:sample_size]

        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                checked += 1
                h, w = img.shape[:2]

                if h < 100 or w < 100:
                    small += 1

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < 50:
                    dark += 1

                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                if laplacian.var() < 50:
                    blurry += 1

            except Exception as e:
                logger.warning(f"Error checking {img_path.name}: {e}")

    return {
        "checked": checked,
        "blurry": blurry,
        "dark": dark,
        "small": small,
        "blurry_pct": blurry / checked * 100 if checked > 0 else 0,
        "dark_pct": dark / checked * 100 if checked > 0 else 0,
    }


def check_bbox_sizes(dataset_dir: Path) -> dict:
    """Analisa tamanhos de bboxes para detectar anomalias."""
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []

    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        _, _, _, bw, bh = map(float, parts[:5])
                        bbox_widths.append(bw)
                        bbox_heights.append(bh)
                        bbox_areas.append(bw * bh)

    if not bbox_widths:
        return {}

    return {
        "avg_width": np.mean(bbox_widths),
        "avg_height": np.mean(bbox_heights),
        "avg_area": np.mean(bbox_areas),
        "min_width": np.min(bbox_widths),
        "max_width": np.max(bbox_widths),
        "min_height": np.min(bbox_heights),
        "max_height": np.max(bbox_heights),
    }


def generate_report(dataset_dir: Path, output_path: Path = None) -> dict:
    """Gera relatório completo do dataset."""
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION REPORT")
    logger.info("=" * 60)

    # Structure validation
    logger.info("\n[1] STRUCTURE VALIDATION")
    structure = validate_structure(dataset_dir)
    if structure["valid"]:
        logger.info("  ✓ Dataset structure is valid")
    else:
        logger.error("  ✗ Dataset structure has errors:")
        for err in structure["errors"]:
            logger.error(f"    - {err}")

    for warn in structure["warnings"]:
        logger.warning(f"  ⚠ {warn}")

    # Class distribution
    logger.info("\n[2] CLASS DISTRIBUTION")
    stats = count_classes(dataset_dir)

    logger.info(f"  Total images:")
    for split, count in stats["image_counts"].items():
        logger.info(f"    {split}: {count}")

    logger.info(f"\n  Total bounding boxes: {stats['total_boxes']}")
    logger.info(f"\n  Per-class counts:")
    for cls_id, count in sorted(stats["class_counts"].items()):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        pct = count / stats['total_boxes'] * 100 if stats['total_boxes'] > 0 else 0
        logger.info(f"    {cls_id:2d} ({name:10s}): {count:6d} ({pct:5.1f}%)")

    empty_total = sum(stats["empty_images"].values())
    logger.info(f"\n  Empty label files: {empty_total}")

    # Image quality
    logger.info("\n[3] IMAGE QUALITY (sample)")
    quality = check_image_quality(dataset_dir)
    if quality["checked"] > 0:
        logger.info(f"  Checked {quality['checked']} images")
        logger.info(f"  Blurry: {quality['blurry']} ({quality['blurry_pct']:.1f}%)")
        logger.info(f"  Dark: {quality['dark']} ({quality['dark_pct']:.1f}%)")
        logger.info(f"  Too small: {quality['small']}")
    else:
        logger.warning("  No images checked (dataset may be empty)")

    # Bbox sizes
    logger.info("\n[4] BOUNDING BOX SIZES")
    bboxes = check_bbox_sizes(dataset_dir)
    if bboxes:
        logger.info(f"  Avg width: {bboxes['avg_width']:.3f} ({bboxes['avg_width']*640:.1f}px)")
        logger.info(f"  Avg height: {bboxes['avg_height']:.3f} ({bboxes['avg_height']*640:.1f}px)")
        logger.info(f"  Width range: {bboxes['min_width']:.3f} - {bboxes['max_width']:.3f}")
        logger.info(f"  Height range: {bboxes['min_height']:.3f} - {bboxes['max_height']:.3f}")

    # Summary
    total_images = sum(stats["image_counts"].values())
    logger.info("\n[5] SUMMARY")
    logger.info(f"  Dataset: {dataset_dir}")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Total boxes: {stats['total_boxes']}")
    logger.info(f"  Classes: {len(stats['class_counts'])}")

    report = {
        "dataset": str(dataset_dir),
        "structure": structure,
        "stats": stats,
        "quality": quality,
        "bboxes": bboxes,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n  Report saved to: {output_path}")

    logger.info("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset")
    parser.add_argument("--dataset", type=str, default="dataset/yolo_final",
                       help="Dataset directory to validate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON report path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output) if args.output else dataset_dir / "validation_report.json"

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    generate_report(dataset_dir, output_path)


if __name__ == "__main__":
    main()
