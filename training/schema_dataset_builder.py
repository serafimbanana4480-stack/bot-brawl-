"""
schema_dataset_builder.py

Create schema-specific training datasets from legacy label sets.

The current core dataset stores labels as 0,2,3,5. Ultralytics requires
contiguous indices starting at 0, so this builder remaps labels into:
- core:      0=Player, 1=Enemy, 2=Cubebox, 3=Powerup
- extended:  0..7 contiguous (identity in the current extended dataset)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

from training.class_schema import EXTENDED_CLASSES, CORE_CLASSES, remap_label_id

logger = logging.getLogger(__name__)


def _iter_images(split_dir: Path) -> Iterable[Path]:
    for candidate in sorted(split_dir.glob("*")):
        if candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield candidate


def _remap_label_file(source: Path, dest: Path, schema: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    lines = []
    if source.exists():
        for raw in source.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) < 5:
                continue
            try:
                old_id = int(parts[0])
            except ValueError:
                continue
            new_id = remap_label_id(old_id, schema=schema)
            if new_id is None:
                continue
            lines.append(f"{new_id} {' '.join(parts[1:5])}\n")
            count += 1
    dest.write_text("".join(lines), encoding="utf-8")
    return count


def build_schema_dataset(source_dir: Path, target_dir: Path, schema: str = "core") -> Dict[str, int]:
    """Copy a YOLO dataset and remap labels into the requested schema."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "labels": 0, "skipped": 0}
    for split in ("train", "val", "test"):
        src_images = source_dir / split / "images"
        src_labels = source_dir / split / "labels"
        if not src_images.exists():
            continue

        dst_images = target_dir / split / "images"
        dst_labels = target_dir / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img_path in _iter_images(src_images):
            shutil.copy2(img_path, dst_images / img_path.name)
            label_path = src_labels / f"{img_path.stem}.txt"
            label_count = _remap_label_file(label_path, dst_labels / f"{img_path.stem}.txt", schema=schema)
            if label_count == 0:
                stats["skipped"] += 1
            else:
                stats["labels"] += label_count
            stats["images"] += 1

    return stats
