"""
class_schema.py

Canonical class schemas used across training and validation.

Now delegates to core.class_registry for unified schema management.
Maintains backward compatibility while providing access to expanded schemas.

Schemas:
- core: 4 classes (minimum viable)
- extended: 8 classes (adds environment)
- full: 35 classes (comprehensive tactical coverage)

DEPRECATED: Legacy hardcoded mappings. Use core.class_registry instead.
"""

from __future__ import annotations

from typing import Dict, Optional

# Import unified registry
from core.class_registry import (
    VISUAL_CLASSES,
    get_schema as registry_get_schema,
    get_class_id as registry_get_class_id,
    remap_label_id as registry_remap_label_id,
    ALIASES,
    ROBOFLOW_TO_CANONICAL,
)

# ============================================================================
# Backward compatibility: re-export legacy names
# ============================================================================

DEFAULT_SCHEMA = "core"  # Safe default - use "full" explicitly for advanced use

# Re-export from registry (lowercase canonical names)
CORE_CLASSES: Dict[int, str] = VISUAL_CLASSES["core"]
EXTENDED_CLASSES: Dict[int, str] = VISUAL_CLASSES["extended"]
FULL_CLASSES: Dict[int, str] = VISUAL_CLASSES["full"]

# Legacy uppercase names (deprecated, use canonical lowercase)
_CORE_LEGACY: Dict[int, str] = {
    0: "Player",
    1: "Enemy",
    2: "Cubebox",
    3: "Powerup",
}

_EXTENDED_LEGACY: Dict[int, str] = {
    0: "Player",
    1: "Bush",
    2: "Enemy",
    3: "Cubebox",
    4: "Wall",
    5: "Powerup",
    6: "Bullet",
    7: "Super",
}

# Legacy label map (deprecated - use registry.remap_label_id)
LEGACY_CORE_LABEL_MAP: Dict[int, int] = {
    0: 0,
    2: 1,
    3: 2,
    5: 3,
}

# Roboflow mappings (deprecated - use registry.ROBOFLOW_TO_CANONICAL)
ROBOFLOW_CLASS_NAMES = list(ROBOFLOW_TO_CANONICAL.keys())
CLASS_MAP = {k: registry_get_class_id(v, "core") if v else -1
             for k, v in ROBOFLOW_TO_CANONICAL.items()}
KEEP_CLASSES = {0, 1, 2, 3}  # Core class IDs


# ============================================================================
# Public API (delegates to registry)
# ============================================================================

def get_schema(name: str = DEFAULT_SCHEMA) -> Dict[int, str]:
    """
    Return a class schema by name.

    Args:
        name: Schema name ("core", "extended", "full")

    Returns:
        Dictionary mapping class IDs to canonical names
    """
    return registry_get_schema(name)


def get_full_schema() -> Dict[int, str]:
    """Get full 15-class schema."""
    return VISUAL_CLASSES["full"].copy()


def schema_name(schema: Dict[int, str]) -> str:
    """Return canonical schema name for a mapping."""
    # Check against all available schemas
    for name, ref_schema in VISUAL_CLASSES.items():
        if schema == ref_schema:
            return name
    return "custom"


def remap_label_id(label_id: int, schema: str = "core") -> Optional[int]:
    """
    Map a source label id into the contiguous target schema.

    Uses the unified registry for consistent remapping.
    """
    return registry_remap_label_id(label_id, schema)


def get_canonical_name(name: str) -> str:
    """
    Normalize a class name to its canonical form.

    Args:
        name: Raw class name (e.g., "Player", "enemy", "teammate")

    Returns:
        Canonical lowercase name (e.g., "player", "enemy")
    """
    return ALIASES.get(name, name.lower())


def get_class_id(name: str, schema: str = "full") -> Optional[int]:
    """
    Get class ID for a canonical name in a given schema.

    Args:
        name: Class name (will be normalized to canonical)
        schema: Target schema

    Returns:
        Class ID or None if not in schema
    """
    return registry_get_class_id(name, schema)


# ============================================================================
# Schema validation utilities
# ============================================================================

def validate_schema_completeness(
    schema: Dict[int, str],
    required: set = None
) -> tuple[bool, list[str]]:
    """
    Validate that a schema contains required classes.

    Returns:
        (is_valid, list_of_missing_classes)
    """
    if required is None:
        required = {"player", "enemy"}  # Minimum required

    schema_names = {name.lower() for name in schema.values()}
    missing = required - schema_names

    return len(missing) == 0, list(missing)


def get_schema_coverage(schema: Dict[int, str]) -> dict:
    """
    Get coverage report for a schema vs ideal full schema.

    Returns dict with coverage metrics.
    """
    full_names = set(VISUAL_CLASSES["full"].values())
    schema_names = {name.lower() for name in schema.values()}

    covered = schema_names & full_names
    missing = full_names - schema_names
    extra = schema_names - full_names

    return {
        "coverage_pct": len(covered) / len(full_names) * 100,
        "covered_classes": sorted(covered),
        "missing_classes": sorted(missing),
        "extra_classes": sorted(extra),
    }


# ============================================================================
# Migration helpers
# ============================================================================

def migrate_old_detection(
    old_class_name: str,
    old_schema: str = "core",
    new_schema: str = "full"
) -> tuple[Optional[str], Optional[int]]:
    """
    Migrate a detection from old schema to new schema.

    Args:
        old_class_name: Original class name
        old_schema: Source schema
        new_schema: Target schema

    Returns:
        (new_canonical_name, new_class_id) or (None, None) if incompatible
    """
    # Normalize to canonical
    canonical = get_canonical_name(old_class_name)

    # Get new ID
    new_id = registry_get_class_id(canonical, new_schema)

    return (canonical, new_id) if new_id is not None else (None, None)
