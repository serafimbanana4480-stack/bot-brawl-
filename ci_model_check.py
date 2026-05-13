"""
ci_model_check.py

CI/CD validation script for model integrity.
Exits 0 if all registered models pass SHA-256 + class checks.
Exits 1 on any failure (use in pre-commit / GitHub Actions).

Usage:
    python ci_model_check.py
    python ci_model_check.py --strict   # fail if no valid models
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "models" / "model_registry.json"
MODELS_DIR = Path(__file__).parent / "models"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_ci_check(strict: bool = False) -> int:
    """
    Returns 0 on pass, 1 on failure.
    strict=True: also fail if registry has zero valid models.
    """
    failures = []

    if not REGISTRY_PATH.exists():
        logger.error(f"Model registry not found at {REGISTRY_PATH}")
        logger.error("Run: python -m backend.brawl_bot.model_validator to generate it.")
        return 1

    try:
        registry = json.loads(REGISTRY_PATH.read_text())
    except Exception as e:
        logger.error(f"Cannot parse model registry: {e}")
        return 1

    models = registry.get("models", {})
    if not models:
        logger.error("Registry is empty — no models recorded.")
        return 1

    valid_count = 0
    fake_count = 0

    for name, meta in models.items():
        status = meta.get("status", "unknown")
        recorded_hash = meta.get("sha256", "")
        model_path = MODELS_DIR / name

        if status == "fake":
            fake_count += 1
            logger.error(f"FAKE: {name} — {meta.get('reason', '')}")
            failures.append(f"FAKE model in registry: {name}")
            continue

        if status == "invalid":
            logger.warning(f"INVALID: {name} — {meta.get('reason', '')}")
            continue

        if status == "valid":
            if not model_path.exists():
                logger.error(f"MISSING: {name} — registered as valid but file not found")
                failures.append(f"Missing file: {name}")
                continue

            actual_hash = _sha256(model_path)
            if actual_hash != recorded_hash:
                logger.error(
                    f"HASH MISMATCH: {name}\n"
                    f"  registered: {recorded_hash[:16]}...\n"
                    f"  on disk:    {actual_hash[:16]}..."
                )
                failures.append(f"Hash mismatch: {name}")
                continue

            valid_count += 1
            logger.info(f"OK: {name} [{actual_hash[:12]}...] classes={meta.get('classes', [])[:3]}")

    if strict and valid_count == 0:
        failures.append("No valid Brawl Stars models — cannot run bot")
        logger.error("STRICT MODE: Zero valid models in registry.")

    if failures:
        logger.error(f"\n{'=' * 50}")
        logger.error(f"CI CHECK FAILED — {len(failures)} issue(s):")
        for f in failures:
            logger.error(f"  • {f}")
        logger.error(f"{'=' * 50}")
        return 1

    logger.info(f"\n✅ CI CHECK PASSED — {valid_count} valid model(s), {fake_count} fake(s) quarantined")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI model integrity check")
    parser.add_argument("--strict", action="store_true", help="Fail if no valid Brawl Stars models")
    args = parser.parse_args()
    sys.exit(run_ci_check(strict=args.strict))
