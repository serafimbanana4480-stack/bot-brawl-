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
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.class_schema import CORE_CLASSES, EXTENDED_CLASSES
from core.class_registry import ROBOFLOW_TO_CANONICAL, get_class_id

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("roboflow_download")

# Roboflow dataset URL (formato download API)
ROBOFLOW_URLS = {
    "bloxxy": "https://universe.roboflow.com/ds/bloxxy/brawl-stars-dataset/download?format=yolov8",
}

# Classe mapping: Roboflow -> nosso formato (using unified registry)
# Maps Roboflow class names to canonical class IDs in core schema
CLASS_MAP = {}
for roboflow_name, canonical_name in ROBOFLOW_TO_CANONICAL.items():
    if canonical_name is not None:
        class_id = get_class_id(canonical_name, schema="core")
        if class_id is not None:
            CLASS_MAP[roboflow_name] = class_id
        else:
            CLASS_MAP[roboflow_name] = -1  # Skip if not in core schema
    else:
        CLASS_MAP[roboflow_name] = -1  # Skip explicitly marked classes

# Roboflow class names in order (from dataset)
ROBOFLOW_CLASS_NAMES = ["Ball", "Enemy", "Friendly", "Gem", "Hot_Zone", "Me", "PP", "PP_Box", "Safe_Enemy", "Safe_Friendly"]

# Classes que queremos manter
KEEP_CLASSES = set(CORE_CLASSES.keys())  # Player, Enemy, Cubebox, Powerup
STANDARD_CLASSES = EXTENDED_CLASSES


def _coerce_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def download_dataset(url: str, output_dir: Path, api_key: str) -> bool:
    """Download dataset do Roboflow (api_key is required)."""
    if not api_key:
        logger.error("Roboflow API key is REQUIRED for download")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    download_url = f"{url}?api_key={api_key}"

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


def remap_classes(dataset_dir: Path, roboflow_class_names: list = None) -> int:
    """
    Remapeia classes do dataset Roboflow para nosso formato.
    Remove classes que nao nos interessam.
    
    Args:
        dataset_dir: Path to the dataset split directory (contains labels/).
        roboflow_class_names: Optional list of Roboflow class names in index order.
                              If None, reads from data.yaml in the parent directory.
    """
    dataset_dir = _coerce_path(dataset_dir)
    labels_dir = dataset_dir / "labels"
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return 0

    # Auto-detect Roboflow class names from data.yaml if not provided
    if roboflow_class_names is None:
        parent = dataset_dir.parent
        yaml_path = parent / "data.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    cfg = yaml.safe_load(f)
                roboflow_class_names = cfg.get("names", [])
                logger.info(f"Read {len(roboflow_class_names)} class names from {yaml_path}")
            except Exception:
                roboflow_class_names = ROBOFLOW_CLASS_NAMES
                logger.warning(f"Could not read {yaml_path}, using hardcoded class names")
        else:
            roboflow_class_names = ROBOFLOW_CLASS_NAMES

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
            if cls_id < len(roboflow_class_names):
                roboflow_class_name = roboflow_class_names[cls_id]
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
    dataset_dir = _coerce_path(dataset_dir)
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
    local_dir = _coerce_path(local_dir)
    roboflow_dir = _coerce_path(roboflow_dir)

    _local_classes = get_classes(local_dir)  # reserved for future cross-validation
    robo_classes = get_classes(roboflow_dir)
    required = set(CORE_CLASSES.keys())  # Player, Enemy, Cubebox, Powerup
    
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
                        if not lines or all(line.strip() == "" for line in lines):
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
    extensions = {".png", ".jpg", ".jpeg"}
    local_images = sum(1 for f in local_dir.glob("**/*") if f.suffix.lower() in extensions)
    robo_images = sum(1 for f in roboflow_dir.glob("**/*") if f.suffix.lower() in extensions)
    
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

        # Collect image files once (avoid O(n^2) re-globbing)
        image_files = [f for f in sorted(src_images.glob("*"))
                      if f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        for i, img_file in enumerate(image_files):
            new_name = f"robo_{start_idx + i:05d}{img_file.suffix}"
            shutil.copy2(img_file, dst_images / new_name)

            label_file = src_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, dst_labels / f"robo_{start_idx + i:05d}.txt")

            stats[split] += 1

    return stats


def create_merged_yaml(dataset_dir: Path, schema: str = "core") -> Path:
    """Cria data.yaml para dataset mesclado."""
    import yaml
    from training.class_schema import get_schema

    # Detect actual classes in dataset
    actual_classes = get_classes(dataset_dir)
    
    expected_schema = get_schema(schema)
    # Only include classes that actually exist and are expected in the target schema.
    classes = {i: name for i, name in expected_schema.items() if i in actual_classes}
    
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

    logger.info(f"Created {yaml_path} with {len(classes)} classes for schema={schema}: {classes}")
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
    parser.add_argument(
        "--schema",
        type=str,
        default="core",
        choices=["core", "extended", "full"],
        help="Target class schema for remap/merge (default: core)",
    )
    parser.add_argument("--dataset", type=str, action="append", default=None,
                       help="Additional dataset(s) to download in format workspace/project (can be used multiple times). Delegates to multi-downloader.")
    parser.add_argument("--discover", action="store_true",
                       help="Auto-discover compatible datasets from Roboflow Universe and download them. Delegates to multi-downloader.")
    parser.add_argument("--no-merge", action="store_true",
                       help="Skip merging datasets after multi-download")
    parser.add_argument("--merge-output-dir", type=str, default="dataset/merged",
                       help="Output directory for merged dataset in multi-download mode")
    args = parser.parse_args()

    # Multi-dataset mode: delegate to roboflow_multi_downloader
    if args.dataset or args.discover:
        try:
            from training.roboflow_multi_downloader import run_multi_download
            from training.roboflow_dataset_discoverer import (
                DatasetInfo,
                filter_compatible,
                score_dataset,
                search_universe,
                KNOWN_DATASETS,
            )
        except ImportError as e:
            logger.error(f"Multi-downloader not available: {e}")
            return 1

        datasets = []
        if args.discover:
            all_datasets = list(KNOWN_DATASETS)
            try:
                web_results = search_universe(query="brawl stars", max_results=50)
                all_datasets.extend(web_results)
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
            seen = {}
            for ds in all_datasets:
                key = f"{ds.workspace}/{ds.project}"
                if key not in seen:
                    seen[key] = ds
            for ds in seen.values():
                score_dataset(ds, schema=args.schema)
            datasets = filter_compatible(list(seen.values()), min_score=0.0)
            logger.info(f"Discovered {len(datasets)} compatible datasets")

        if args.dataset:
            for ds_str in args.dataset:
                if "/" not in ds_str:
                    logger.error(f"Invalid dataset format: {ds_str}")
                    continue
                workspace, project = ds_str.split("/", 1)
                datasets.append(DatasetInfo(workspace=workspace, project=project, source="cli"))

        if not datasets:
            logger.error("No datasets to download.")
            return 1

        if not args.api_key:
            logger.error("Roboflow API key is REQUIRED for multi-download")
            return 1

        results = run_multi_download(
            datasets=datasets,
            output_dir=Path(args.output),
            api_key=args.api_key,
            schema=args.schema,
            merge=not args.no_merge,
            merge_output_dir=Path(args.merge_output_dir) if args.merge_output_dir else None,
        )
        logger.info(f"Multi-download complete. Prepared: {len(results.get('prepared_dirs', []))} datasets")
        return

    # Legacy single-dataset mode
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
            create_merged_yaml(Path("dataset/merged"), schema=args.schema)
        else:
            logger.warning(f"Local dataset not found: {local_dir}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
