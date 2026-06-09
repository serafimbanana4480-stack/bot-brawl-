"""
model_validator.py

Validates YOLO model files: SHA-256 integrity, architecture check,
class label verification, and detection of fake/COCO-reused models.

RULES:
- A model is FAKE if it shares the same SHA-256 as yolov8n.pt (the base download).
- A model is FAKE if its class labels are COCO-80 classes (not Brawl Stars entities).
- A model is INVALID if it cannot be loaded or has no Ultralytics metadata.
- Only models with Brawl Stars-specific classes are marked VALID.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

BRAWL_STARS_CORE_CLASSES = {
    "player", "enemy", "bush", "cubebox", "wall",
    "powerup", "bullet", "super",
    # Legacy/alternative names
    "teammate", "box", "super_indicator",
    "health_bar", "joystick", "attack_button"
}

# Backwards-compatible alias used by older tests and tooling.
BRAWL_STARS_CLASSES = (
    "player", "enemy", "cubebox", "powerup",
    "bush", "wall", "bullet", "super",
    "teammate", "box", "super_indicator",
    "health_bar", "joystick", "attack_button",
)

# Core production schema used by the currently deployed dataset/model.
# The validator accepts the 4-class core set as valid and logs the optional
# 8-class extensions when present.
BRAWL_STARS_CORE_4_CLASSES = {"player", "enemy", "cubebox", "powerup"}

COCO_CLASSES_SAMPLE = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "suitcase", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "laptop", "mouse", "remote", "keyboard", "cell phone", "chair"
}

MODELS_DIR = Path(__file__).parent / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


PathLike = Union[str, Path]


def _coerce_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def sha256_file(path: PathLike) -> str:
    """Compute SHA-256 hash of a file."""
    path = _coerce_path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_names_from_yaml(model_path: Path) -> Optional[List[str]]:
    """Fallback class extraction from sidecar YAML files."""
    sidecar_paths = [
        model_path.with_suffix(".yaml"),
        model_path.with_suffix(".yml"),
        model_path.parent / "data.yaml",
        model_path.parent / "data.yml",
    ]
    for candidate in sidecar_paths:
        if not candidate.exists():
            continue
        try:
            import yaml
            with open(candidate, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            names = data.get("names")
            if isinstance(names, dict):
                try:
                    return [names[k] for k in sorted(names, key=lambda value: int(value))]
                except Exception:
                    return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return list(names)
        except Exception as exc:
            logger.debug(f"Could not read sidecar metadata {candidate.name}: {exc}")
    return None


def get_model_classes(model_path: PathLike) -> Optional[List[str]]:
    """
    Extract class names from a YOLO .pt file without running inference.
    Returns None if model cannot be loaded.
    """
    try:
        import torch
        model_path = _coerce_path(model_path)
        try:
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(f"weights_only=True failed for {model_path.name}, falling back to weights_only=False")
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)

        # Ultralytics stores names in model.names or ckpt['model'].names
        names = None

        if isinstance(checkpoint, dict):
            model_obj = checkpoint.get("model", None)
            if model_obj is not None and hasattr(model_obj, "names"):
                names = model_obj.names
            elif "names" in checkpoint:
                names = checkpoint["names"]
        elif hasattr(checkpoint, "names"):
            names = checkpoint.names

        if names is None:
            names = _extract_names_from_yaml(model_path)
        if names is None:
            return None

        if isinstance(names, dict):
            return list(names.values())
        if isinstance(names, (list, tuple)):
            return list(names)
        return None

    except Exception as e:
        logger.debug(f"Could not extract classes from {model_path.name}: {e}")
        return None


def is_coco_model(classes: List[str]) -> bool:
    """Returns True if the class set overlaps significantly with COCO-80."""
    if not classes:
        return False
    class_set = {c.lower() for c in classes}
    overlap = class_set & {c.lower() for c in COCO_CLASSES_SAMPLE}
    return len(overlap) >= 10


def is_brawl_stars_model(classes: List[str]) -> bool:
    """Returns True if the core 4-class Brawl Stars schema is present."""
    if not classes:
        return False
    class_set = {c.lower() for c in classes}
    required = {c.lower() for c in BRAWL_STARS_CORE_4_CLASSES}
    overlap = class_set & required
    return overlap == required


class ModelValidationResult:
    def __init__(
        self,
        path: Path,
        sha256: str,
        classes: Optional[List[str]],
        status: str,
        reason: str
    ):
        self.path = path
        self.sha256 = sha256
        self.classes = classes or []
        self.status = status  # "valid" | "fake" | "invalid"
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "name": self.path.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "classes": self.classes,
            "status": self.status,
            "reason": self.reason,
        }


def validate_all_models(delete_fakes: bool = False, models_dir: Optional[PathLike] = None) -> Dict:
    """
    Validate all .pt files in the models directory.

    Args:
        delete_fakes: If True, physically removes fake models.

    Returns:
        {
          "valid": [...],
          "fake": [...],
          "invalid": [...],
          "integrity_score": int (0-100),
          "registry_written": bool,
        }
    """
    models_dir = _coerce_path(models_dir or MODELS_DIR)
    results: List[ModelValidationResult] = []
    pt_files = list(models_dir.glob("*.pt"))

    if not pt_files:
        logger.warning("No .pt model files found in models directory.")
        return {
            "valid": [],
            "fake": [],
            "invalid": [],
            "integrity_score": 0,
            "registry_written": False,
        }

    # First pass: compute all hashes to detect duplicates
    hash_map: Dict[str, List[Path]] = {}
    for pt in pt_files:
        try:
            digest = sha256_file(pt)
            hash_map.setdefault(digest, []).append(pt)
        except Exception as e:
            logger.error(f"Failed to hash {pt.name}: {e}")

    # Identify the yolov8n baseline hash (the model everyone copies from)
    baseline_hash: Optional[str] = None
    yolov8n_path = models_dir / "yolov8n.pt"
    if yolov8n_path.exists():
        try:
            baseline_hash = sha256_file(yolov8n_path)
            logger.info(f"Baseline yolov8n hash: {baseline_hash[:16]}...")
        except Exception:
            pass

    # Second pass: classify each model
    for pt in pt_files:
        try:
            digest = sha256_file(pt)
        except Exception as e:
            results.append(ModelValidationResult(
                path=pt, sha256="ERROR", classes=None,
                status="invalid", reason=f"Cannot hash file: {e}"
            ))
            continue

        # Check for duplicate hash with other models (clone detection)
        duplicates = [p for p in hash_map.get(digest, []) if p != pt]
        classes = get_model_classes(pt)

        if classes is None:
            results.append(ModelValidationResult(
                path=pt, sha256=digest, classes=None,
                status="invalid", reason="Cannot load model / no metadata"
            ))
            continue

        # FAKE: same hash as yolov8n base (copy/rename)
        if baseline_hash and digest == baseline_hash and pt.name != "yolov8n.pt":
            results.append(ModelValidationResult(
                path=pt, sha256=digest, classes=classes,
                status="fake",
                reason=f"Identical SHA-256 to yolov8n.pt (copy/rename detected)"
            ))
            continue

        # FAKE: duplicate hash across different named models
        if duplicates and pt.name != "yolov8n.pt":
            dup_names = [p.name for p in duplicates]
            results.append(ModelValidationResult(
                path=pt, sha256=digest, classes=classes,
                status="fake",
                reason=f"Duplicate SHA-256 with: {dup_names}"
            ))
            continue

        # FAKE: COCO class labels
        if is_coco_model(classes):
            results.append(ModelValidationResult(
                path=pt, sha256=digest, classes=classes,
                status="fake",
                reason="COCO-80 class labels detected — not trained for Brawl Stars"
            ))
            continue

        # VALID: Brawl Stars classes present
        if is_brawl_stars_model(classes):
            results.append(ModelValidationResult(
                path=pt, sha256=digest, classes=classes,
                status="valid",
                reason="Brawl Stars-specific class labels verified"
            ))
            continue

        # DEFAULT: unknown classes — mark invalid but don't delete
        results.append(ModelValidationResult(
            path=pt, sha256=digest, classes=classes,
            status="invalid",
            reason=f"Unknown classes (not COCO, not Brawl Stars): {classes[:5]}"
        ))

    valid = [r for r in results if r.status == "valid"]
    fake = [r for r in results if r.status == "fake"]
    invalid = [r for r in results if r.status == "invalid"]

    logger.info(f"Model validation complete: {len(valid)} valid, {len(fake)} fake, {len(invalid)} invalid")

    # Log fake models
    for r in fake:
        logger.error(f"FAKE MODEL: {r.path.name} — {r.reason}")

    for r in invalid:
        logger.warning(f"INVALID MODEL: {r.path.name} — {r.reason}")

    # Delete fakes if requested
    deleted = []
    if delete_fakes:
        quarantine_dir = models_dir / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        for r in fake:
            if r.path.name == "yolov8n.pt":
                logger.warning("Keeping yolov8n.pt as baseline reference — not deleting.")
                continue
            try:
                dest = quarantine_dir / r.path.name
                shutil.move(str(r.path), str(dest))
                deleted.append(r.path.name)
                logger.warning(f"Quarantined fake model: {r.path.name} → {dest}")
            except Exception as e:
                logger.error(f"Failed to quarantine {r.path.name}: {e}")

    # Write registry
    registry_written = False
    registry_path = models_dir / "model_registry.json"
    registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {r.path.name: r.to_dict() for r in results},
        "quarantined": deleted,
    }
    try:
        registry_path.write_text(json.dumps(registry, indent=2))
        registry_written = True
        logger.info(f"Model registry written to {registry_path}")
    except Exception as e:
        logger.error(f"Failed to write model registry: {e}")

    # Integrity score: % of non-fake non-invalid models
    total = len(results)
    score = int((len(valid) / total) * 100) if total > 0 else 0

    return {
        "valid": [r.to_dict() for r in valid],
        "fake": [r.to_dict() for r in fake],
        "invalid": [r.to_dict() for r in invalid],
        "quarantined": deleted,
        "integrity_score": score,
        "registry_written": registry_written,
    }


def run_audit() -> None:
    """CLI entry point: print model audit report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = validate_all_models(delete_fakes=False)

    print("\n" + "=" * 60)
    print("BRAWL STARS BOT — MODEL INTEGRITY AUDIT")
    print("=" * 60)

    print(f"\n✅ VALID ({len(report['valid'])}):")
    for m in report["valid"]:
        print(f"  {m['name']}  [{m['sha256'][:12]}...] — {m['reason']}")

    print(f"\n❌ FAKE ({len(report['fake'])}):")
    for m in report["fake"]:
        print(f"  {m['name']}  [{m['sha256'][:12]}...] — {m['reason']}")

    print(f"\n⚠️  INVALID ({len(report['invalid'])}):")
    for m in report["invalid"]:
        print(f"  {m['name']}  [{m['sha256'][:12]}...] — {m['reason']}")

    print(f"\nSystem Integrity Score: {report['integrity_score']}/100")

    # Check registry for valid models before reporting CRITICAL
    registry_valid_count = 0
    if REGISTRY_PATH.exists():
        try:
            reg_data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            registry_valid_count = sum(
                1 for m in reg_data.get("models", {}).values()
                if m.get("status") == "valid"
            )
        except Exception:
            pass  # Registry unreadable — fall through to normal logic

    if len(report["valid"]) == 0 and registry_valid_count == 0:
        print("STATUS: ❌ CRITICAL — No real Brawl Stars models present.")
        print("        Bot cannot operate. Train YOLO on real game data first.")
    elif len(report["valid"]) == 0 and registry_valid_count > 0:
        print(f"STATUS: ⚠️  WARNING — No valid models in current scan, but {registry_valid_count} valid model(s) in registry.")
        print("        Re-run validation after placing trained models in the models/ directory.")
    elif report["integrity_score"] < 80:
        print("STATUS: ⚠️  PARTIAL — Valid models present, but models directory contains fake or invalid files.")
    else:
        print("STATUS: ✅ PRODUCTION READY")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_audit()
