"""
core/class_registry.py

Canonical class registry for Soberana Omega.

Single source of truth for:
- Visual class mappings (YOLO detection schema)
- Name aliases (normalize variant names to canonical)
- State feature definitions
- Unified action space

This eliminates naming inconsistencies across play.py, wrapper.py, vision/state.py, etc.

Usage:
    from core.class_registry import get_canonical, VISUAL_CLASSES, UnifiedAction

    # Normalize variant name to canonical
    canonical = get_canonical("teammate")  # returns "player"
    canonical = get_canonical("Enemy")     # returns "enemy"

    # Get class ID for schema
    class_id = get_class_id("enemy", schema="full")  # returns 2

Breaking Change Notice:
    This replaces legacy hardcoded mappings in wrapper.py, play.py, etc.
    Old models may need retraining with the new unified schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union


# ============================================================================
# VISUAL CLASSES: id → nome canonical (YOLO detection classes)
# ============================================================================

# Core schema (4 classes) — minimum viable for gameplay
CORE_CLASSES: Dict[int, str] = {
    0: "player",
    1: "enemy",
    2: "cubebox",
    3: "powerup",
}

# Extended schema (8 classes) — adds environment objects
EXTENDED_CLASSES: Dict[int, str] = {
    0: "player",
    1: "bush",
    2: "enemy",
    3: "cubebox",
    4: "wall",
    5: "powerup",
    6: "bullet_neutral",
    7: "super_area",
}

# Full schema (35 classes) — tactical coverage for spatial reasoning, hazards and game-mode objectives
FULL_CLASSES: Dict[int, str] = {
    0: "player",
    1: "bush",
    2: "enemy",
    3: "cubebox",
    4: "wall",
    5: "powerup",
    6: "bullet_neutral",
    7: "super_area",
    8: "bullet_friendly",
    9: "bullet_enemy",
    10: "poison_gas",
    11: "safe_zone",
    12: "gem",
    13: "ball",
    14: "hot_zone",
    15: "destroyable_wall",
    16: "open_path",
    17: "spawn_point",
    18: "water",
    19: "jump_pad",
    20: "teleporter",
    21: "heist_safe",
    22: "siege_bolt",
    23: "goal",
    24: "enemy_low_hp",
    25: "enemy_super_ready",
    26: "enemy_attacking",
    27: "enemy_hidden",
    28: "enemy_near",
    29: "incoming_threat",
    30: "collect_power_cube",
    31: "dodge_zone",
    32: "heal_zone",
    33: "respawn_zone",
    34: "objective_zone",
}

# Schema registry
VISUAL_CLASSES: Dict[str, Dict[int, str]] = {
    "core": CORE_CLASSES,
    "extended": EXTENDED_CLASSES,
    "full": FULL_CLASSES,
}

# Reverse lookup: name → id for each schema
_CLASS_TO_ID: Dict[str, Dict[str, int]] = {
    schema: {name: id for id, name in classes.items()}
    for schema, classes in VISUAL_CLASSES.items()
}


# ============================================================================
# ALIASES: nome alternativo → canonical
# Elimina as variantes encontradas em play.py, wrapper.py, etc.
# ============================================================================

ALIASES: Dict[str, str] = {
    # Player variants (found in play.py, wrapper.py, async_pipeline.py, dataset_pipeline.py)
    "Player": "player",
    "player": "player",  # self-alias for consistency
    "self": "player",
    "teammate": "player",
    "ally": "player",
    "person": "player",  # COCO generic fallback
    "Friendly": "player",  # Roboflow
    "Me": "player",  # Roboflow
    "Safe_Friendly": "player",  # Roboflow
    "character": "player",
    "hero": "player",
    "npc": "player",
    "brawler_self": "player",

    # Enemy variants
    "Enemy": "enemy",
    "enemy": "enemy",  # self-alias
    "brawler": "enemy",
    "opponent": "enemy",
    "Safe_Enemy": "enemy",  # Roboflow
    "brawler_enemy": "enemy",
    "foe": "enemy",
    "mob": "enemy",

    # Cubebox/Power Cube variants
    "Cubebox": "cubebox",
    "cubebox": "cubebox",  # self-alias
    "power_cube": "cubebox",
    "powercube": "cubebox",
    "box": "cubebox",
    "PP_Box": "cubebox",  # Roboflow
    "crate": "cubebox",
    "chest": "cubebox",
    "destroyable_box": "cubebox",

    # Powerup variants
    "Powerup": "powerup",
    "powerup": "powerup",  # self-alias
    "PP": "powerup",  # Roboflow shorthand
    "item": "powerup",
    "buff": "powerup",
    "ammo": "powerup",
    "power_up": "powerup",

    # Bush variants
    "Bush": "bush",
    "bush": "bush",  # self-alias

    # Wall variants
    "Wall": "wall",
    "wall": "wall",  # self-alias
    "obstacle": "wall",
    "rock": "wall",
    "block": "wall",
    "barrier": "wall",

    # Bullet variants
    "Bullet": "bullet_neutral",
    "bullet": "bullet_neutral",
    "bullet_neutral": "bullet_neutral",  # self-alias
    "projectile": "bullet_neutral",
    "shot": "bullet_neutral",
    "projectile_neutral": "bullet_neutral",
    "FriendlyBullet": "bullet_friendly",
    "bullet_friendly": "bullet_friendly",  # self-alias
    "EnemyBullet": "bullet_enemy",
    "bullet_enemy": "bullet_enemy",  # self-alias

    # Super variants
    "Super": "super_area",
    "super": "super_area",
    "super_area": "super_area",  # self-alias
    "super_attack": "super_area",

    # Game mode specific
    "Ball": "ball",
    "ball": "ball",  # self-alias (Brawl Ball)
    "Gem": "gem",
    "gem": "gem",  # self-alias (Gem Grab)
    "Hot_Zone": "hot_zone",
    "hot_zone": "hot_zone",  # self-alias
    "HotZone": "hot_zone",
    "hotzone": "hot_zone",
    "area": "hot_zone",
    "zone": "hot_zone",
    "DestroyableWall": "destroyable_wall",
    "destroyable_wall": "destroyable_wall",
    "OpenPath": "open_path",
    "open_path": "open_path",
    "SpawnPoint": "spawn_point",
    "spawn_point": "spawn_point",
    "Water": "water",
    "water": "water",
    "lake": "water",
    "river": "water",
    "puddle": "water",
    "JumpPad": "jump_pad",
    "jump_pad": "jump_pad",
    "jump": "jump_pad",
    "pad": "jump_pad",
    "Teleporter": "teleporter",
    "teleporter": "teleporter",
    "teleport": "teleporter",
    "portal": "teleporter",
    "HeistSafe": "heist_safe",
    "heist_safe": "heist_safe",
    "SiegeBolt": "siege_bolt",
    "siege_bolt": "siege_bolt",
    "Goal": "goal",
    "goal": "goal",
    "target": "goal",
    "basket": "goal",
    "net": "goal",
    "EnemyLowHP": "enemy_low_hp",
    "enemy_low_hp": "enemy_low_hp",
    "EnemySuperReady": "enemy_super_ready",
    "enemy_super_ready": "enemy_super_ready",
    "EnemyAttacking": "enemy_attacking",
    "enemy_attacking": "enemy_attacking",
    "EnemyHidden": "enemy_hidden",
    "enemy_hidden": "enemy_hidden",
    "EnemyNear": "enemy_near",
    "enemy_near": "enemy_near",
    "IncomingThreat": "incoming_threat",
    "incoming_threat": "incoming_threat",
    "CollectPowerCube": "collect_power_cube",
    "collect_power_cube": "collect_power_cube",
    "DodgeZone": "dodge_zone",
    "dodge_zone": "dodge_zone",
    "HealZone": "heal_zone",
    "heal_zone": "heal_zone",
    "RespawnZone": "respawn_zone",
    "respawn_zone": "respawn_zone",
    "ObjectiveZone": "objective_zone",
    "objective_zone": "objective_zone",

    # Environmental hazards
    "PoisonGas": "poison_gas",
    "poison_gas": "poison_gas",  # self-alias
    "gas": "poison_gas",
    "fog": "poison_gas",
    "storm": "poison_gas",
    "hazard": "poison_gas",
    "danger_zone": "poison_gas",
    "SafeZone": "safe_zone",
    "safe_zone": "safe_zone",  # self-alias
    "safe": "safe_zone",
    "shield": "safe_zone",
    "protection": "safe_zone",
    "heal_zone": "heal_zone",
    "heal": "heal_zone",
    "health": "heal_zone",
    "regen": "heal_zone",
    "dodge_zone": "dodge_zone",
    "danger": "dodge_zone",
    "warning": "dodge_zone",
    "avoid": "dodge_zone",
    "objective_zone": "objective_zone",
    "objective": "objective_zone",
    "mission_zone": "objective_zone",
    "respawn_zone": "respawn_zone",
    "respawn": "respawn_zone",
    "spawn": "spawn_point",
    "spawn_point": "spawn_point",
    "heist_safe": "heist_safe",
    "turret": "heist_safe",
    "safe_box": "heist_safe",
    "siege_bolt": "siege_bolt",
    "bolt": "siege_bolt",
    "gear": "siege_bolt",
}


# ============================================================================
# ROBOFLOW MAPPINGS: Roboflow dataset names → canonical
# ============================================================================

ROBOFLOW_TO_CANONICAL: Dict[str, Optional[str]] = {
    # Direct mappings
    "Enemy": "enemy",
    "Safe_Enemy": "enemy",
    "Friendly": "player",
    "Me": "player",
    "Safe_Friendly": "player",
    "Ball": "ball",
    "Gem": "gem",
    "Hot_Zone": "hot_zone",
    "PP_Box": "cubebox",
    "PP": "powerup",

    # Classes to skip (not useful for gameplay)
    "Hot_Zone_Area": None,
    "Robo": None,
    "Background": None,
    "UI": None,
    "button": None,
    "text": None,
    "logo": None,
    "menu": None,
    "loading": None,
    "screen": None,
}

# Keep classes filter for dataset pipeline
ROBOFLOW_KEEP_CLASSES: Set[str] = {
    "Enemy", "Safe_Enemy",
    "Friendly", "Me", "Safe_Friendly",
    "Ball", "Gem", "Hot_Zone",
    "PP_Box", "PP",
}


# ============================================================================
# STATE FEATURES: Canonical list of scalar state features
# ============================================================================

# Self-state features (11 features)
SELF_STATE_FEATURES: List[str] = [
    "hp_ratio",              # 0.0 to 1.0
    "ammo_ratio",            # 0.0 to 1.0 (current/max)
    "super_charge",          # 0.0 to 1.0 (continuous, not just bool)
    "gadget_ready",          # bool as 0/1
    "hypercharge_ready",     # bool as 0/1
    "cooldown_attack",       # 0.0 to 1.0 (progress)
    "cooldown_super",        # 0.0 to 1.0 (progress)
    "is_moving",             # bool as 0/1
    "is_in_bush",            # bool as 0/1
    "velocity_norm",         # 0.0 to 1.0 normalized movement speed
    "current_tactic",        # encoded current high-level tactic
]

# Spatial features (17 features)
SPATIAL_FEATURES: List[str] = [
    "danger_score",          # 0.0 to 1.0
    "enemies_in_range",      # count (capped)
    "escape_routes",         # count (capped)
    "dist_nearest_enemy",    # pixels or normalized
    "dist_nearest_cube",     # pixels or normalized
    "dist_nearest_cover",    # pixels or normalized
    "dist_nearest_safezone", # pixels or normalized (Showdown)
    "line_of_sight_free",    # bool as 0/1
    "safe_direction_x",      # -1 to 1 (normalized vector)
    "safe_direction_y",      # -1 to 1
    "wall_left",             # bool as 0/1
    "wall_right",            # bool as 0/1
    "wall_up",               # bool as 0/1
    "wall_down",             # bool as 0/1
    "bush_nearby",           # bool as 0/1
    "projectile_threat",     # 0.0 to 1.0
    "objective_pressure",    # 0.0 to 1.0
]

# Temporal features (10 features)
TEMPORAL_FEATURES: List[str] = [
    "time_since_enemy_seen", # seconds
    "previous_action",       # encoded as int
    "velocity_x",            # pixels/second
    "velocity_y",            # pixels/second
    "enemy_last_seen_x",     # relative x of last seen nearest enemy
    "enemy_last_seen_y",     # relative y of last seen nearest enemy
    "enemy_last_hp",         # hp ratio of last seen nearest enemy
    "enemy_last_super",      # bool as 0/1
    "enemy_last_angle",      # normalized angle
    "enemy_last_attack",     # bool as 0/1
]

# Per-enemy features (6 features for nearest enemy)
ENEMY_FEATURES: List[str] = [
    "nearest_enemy_hp_ratio",
    "nearest_enemy_has_super",
    "nearest_enemy_angle",
    "nearest_enemy_is_attacking",
    "nearest_enemy_last_seen_x",
    "nearest_enemy_last_seen_y",
]

# All state features combined
STATE_FEATURES: List[str] = (
    SELF_STATE_FEATURES +
    SPATIAL_FEATURES +
    TEMPORAL_FEATURES +
    ENEMY_FEATURES
)

# Feature dimensions for neural network
STATE_FEATURE_DIM: int = len(STATE_FEATURES)  # 44 features


# ============================================================================
# UNIFIED ACTION SPACE: Fusão RL + UtilityAI
# ============================================================================

class UnifiedAction(Enum):
    """
    Unified action space combining RL and UtilityAI actions.

    Mappings from legacy systems:
        RL "attack" → ATTACK
        RL "move_to_enemy" → MOVE_TO_ENEMY
        RL "retreat" → RETREAT
        RL "use_super" → USE_SUPER
        RL "collect_cube" → COLLECT_CUBE
        RL "idle" → IDLE

        UtilityAI ATTACK → ATTACK
        UtilityAI RETREAT → RETREAT
        UtilityAI COLLECT_CUBE → COLLECT_CUBE
        UtilityAI TAKE_COVER → TAKE_COVER
        UtilityAI HOLD_POSITION → HOLD_POSITION
        UtilityAI HEAL_UP → HEAL_UP
        UtilityAI AMBUSH → AMBUSH
        UtilityAI CHASE → CHASE
        UtilityAI KITE → KITE (attack_while_moving)
        UtilityAI USE_SUPER → USE_SUPER
    """
    IDLE = 0
    ATTACK = 1
    MOVE_TO_ENEMY = 2
    RETREAT = 3
    KITE = 4           # attack while moving (merges RL + UtilityAI)
    USE_SUPER = 5
    COLLECT_CUBE = 6
    TAKE_COVER = 7
    HOLD_POSITION = 8
    CHASE = 9          # pursue low-HP enemy
    HEAL_UP = 10       # wait in safe spot
    AMBUSH = 11        # wait in bush for enemy

    @classmethod
    def count(cls) -> int:
        """Return number of actions."""
        return len(cls)

    @classmethod
    def from_string(cls, name: str) -> Optional["UnifiedAction"]:
        """Parse action from string name (case-insensitive)."""
        name_map = {
            "idle": cls.IDLE,
            "attack": cls.ATTACK,
            "move_to_enemy": cls.MOVE_TO_ENEMY,
            "retreat": cls.RETREAT,
            "kite": cls.KITE,
            "use_super": cls.USE_SUPER,
            "collect_cube": cls.COLLECT_CUBE,
            "take_cover": cls.TAKE_COVER,
            "hold_position": cls.HOLD_POSITION,
            "chase": cls.CHASE,
            "heal_up": cls.HEAL_UP,
            "ambush": cls.AMBUSH,
        }
        return name_map.get(name.lower())


# Legacy action mappings for backward compatibility during transition
RL_ACTION_MAP: Dict[str, UnifiedAction] = {
    "attack": UnifiedAction.ATTACK,
    "move_to_enemy": UnifiedAction.MOVE_TO_ENEMY,
    "retreat": UnifiedAction.RETREAT,
    "use_super": UnifiedAction.USE_SUPER,
    "collect_cube": UnifiedAction.COLLECT_CUBE,
    "idle": UnifiedAction.IDLE,
}

UTILITY_ACTION_MAP: Dict[str, UnifiedAction] = {
    "ATTACK": UnifiedAction.ATTACK,
    "RETREAT": UnifiedAction.RETREAT,
    "COLLECT_CUBE": UnifiedAction.COLLECT_CUBE,
    "TAKE_COVER": UnifiedAction.TAKE_COVER,
    "HOLD_POSITION": UnifiedAction.HOLD_POSITION,
    "HEAL_UP": UnifiedAction.HEAL_UP,
    "AMBUSH": UnifiedAction.AMBUSH,
    "CHASE": UnifiedAction.CHASE,
    "KITE": UnifiedAction.KITE,
    "USE_SUPER": UnifiedAction.USE_SUPER,
}


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def get_canonical(name: str) -> str:
    """
    Normalize a class name to its canonical form.

    Args:
        name: Raw class name (e.g., "teammate", "Enemy", "Cubebox")

    Returns:
        Canonical name (e.g., "player", "enemy", "cubebox")

    Examples:
        >>> get_canonical("teammate")
        'player'
        >>> get_canonical("Enemy")
        'enemy'
        >>> get_canonical("unknown_name")
        'unknown_name'  # passes through if no alias
    """
    return ALIASES.get(name, name.lower())


def get_class_id(name: str, schema: str = "full") -> Optional[int]:
    """
    Get class ID for a canonical name in a given schema.

    Args:
        name: Class name (will be normalized to canonical)
        schema: One of "core", "extended", "full"

    Returns:
        Class ID or None if not in schema

    Examples:
        >>> get_class_id("enemy", "core")
        1
        >>> get_class_id("bush", "core")
        None  # bush not in core schema
        >>> get_class_id("bush", "extended")
        1
    """
    canonical = get_canonical(name)
    schema_lower = schema.lower()

    if schema_lower not in _CLASS_TO_ID:
        raise ValueError(f"Unknown schema: {schema}. Use: {list(VISUAL_CLASSES.keys())}")

    return _CLASS_TO_ID[schema_lower].get(canonical)


def get_class_name(class_id: int, schema: str = "full") -> Optional[str]:
    """
    Get canonical class name from ID.

    Args:
        class_id: Numeric class ID
        schema: One of "core", "extended", "full"

    Returns:
        Canonical name or None if ID not in schema
    """
    schema_lower = schema.lower()

    if schema_lower not in VISUAL_CLASSES:
        raise ValueError(f"Unknown schema: {schema}. Use: {list(VISUAL_CLASSES.keys())}")

    return VISUAL_CLASSES[schema_lower].get(class_id)


# Legacy schema aliases (backward compatibility)
_SCHEMA_ALIASES: Dict[str, str] = {
    "4": "core",
    "4classes": "core",
    "8": "extended",
    "8classes": "extended",
    "35": "full",
    "35classes": "full",
    "all": "full",
}


def get_schema(schema: str = "full") -> Dict[int, str]:
    """
    Get full schema mapping (id → name).

    Args:
        schema: One of "core", "extended", "full" (or legacy aliases)

    Returns:
        Dictionary mapping class IDs to canonical names
    """
    schema_lower = schema.lower()

    # Resolve legacy aliases
    if schema_lower in _SCHEMA_ALIASES:
        schema_lower = _SCHEMA_ALIASES[schema_lower]

    if schema_lower not in VISUAL_CLASSES:
        raise ValueError(f"Unknown schema: {schema}. Use: {list(VISUAL_CLASSES.keys())}")

    return VISUAL_CLASSES[schema_lower].copy()


def get_schema_for_model(model_classes: Union[Dict, Set, List]) -> str:
    """
    Detect which schema best matches a model's classes.

    Args:
        model_classes: Set of class names or id→name dict from a YOLO model

    Returns:
        Schema name ("core", "extended", or "full")

    Examples:
        >>> get_schema_for_model({"player", "enemy", "cubebox", "powerup"})
        'core'
        >>> get_schema_for_model({0: "player", 1: "bush", 2: "enemy"})
        'extended'
    """
    # Normalize to set of canonical names
    if isinstance(model_classes, dict):
        names = {get_canonical(name) for name in model_classes.values()}
    else:
        names = {get_canonical(name) for name in model_classes}

    # Check which schema is the best superset match
    core_names = set(CORE_CLASSES.values())
    extended_names = set(EXTENDED_CLASSES.values())
    full_names = set(FULL_CLASSES.values())

    if names.issubset(core_names):
        return "core"
    elif names.issubset(extended_names):
        return "extended"
    elif names.issubset(full_names):
        return "full"
    else:
        # Return closest match
        core_overlap = len(names & core_names)
        extended_overlap = len(names & extended_names)
        full_overlap = len(names & full_names)

        overlaps = [(core_overlap, "core"), (extended_overlap, "extended"), (full_overlap, "full")]
        overlaps.sort(reverse=True)
        return overlaps[0][1]


def get_classes_for_yolo(schema: str = "full") -> Dict[int, str]:
    """
    Get class mapping formatted for YOLO/Ultralytics Detect wrapper.

    Args:
        schema: One of "core", "extended", "full"

    Returns:
        Dict formatted for pylaai_real/detect.py classes parameter
    """
    return get_schema(schema)


def normalize_detections(
    detections: Dict[str, List],
    target_schema: str = "full"
) -> Dict[str, List]:
    """
    Normalize detection dict keys to canonical names for target schema.

    Args:
        detections: Dict with potentially variant keys (e.g., {"teammate": [...]})
        target_schema: Target schema for filtering

    Returns:
        Dict with canonical keys, filtered to schema classes

    Example:
        >>> normalize_detections({"teammate": [[1,2,3,4]], "Enemy": [[5,6,7,8]]})
        {'player': [[1,2,3,4]], 'enemy': [[5,6,7,8]]}
    """
    schema_classes = set(get_schema(target_schema).values())
    result: Dict[str, List] = {}

    for raw_name, boxes in detections.items():
        canonical = get_canonical(raw_name)
        if canonical in schema_classes:
            if canonical not in result:
                result[canonical] = []
            result[canonical].extend(boxes)

    return result


def get_by_type(
    detections: Dict[str, List],
    class_type: str,
    first_only: bool = False
) -> Union[List, Optional[Tuple]]:
    """
    Get detections of a specific type, handling aliases.

    Args:
        detections: Detection dict with potentially variant keys
        class_type: Canonical class type (e.g., "player", "enemy")
        first_only: If True, return first box only (or None)

    Returns:
        List of boxes, or single box if first_only=True

    Examples:
        >>> get_by_type({"teammate": [[1,2,3,4]]}, "player")
        [[1,2,3,4]]
        >>> get_by_type({"Player": [[1,2,3,4]]}, "player", first_only=True)
        (1,2,3,4)
    """
    normalized = normalize_detections(detections)
    boxes = normalized.get(class_type, [])

    if first_only:
        if not boxes:
            return None
        first_box = boxes[0]
        return tuple(first_box) if isinstance(first_box, list) else first_box
    return boxes


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def get_core_classes() -> Dict[int, str]:
    """Return core 4-class schema (for backward compatibility)."""
    return CORE_CLASSES.copy()


def get_extended_classes() -> Dict[int, str]:
    """Return extended 8-class schema (for backward compatibility)."""
    return EXTENDED_CLASSES.copy()


def remap_label_id(label_id: int, target_schema: str = "core") -> Optional[int]:
    """
    Map a label ID from one schema to another.

    This is useful for converting between model outputs with different schemas.

    Args:
        label_id: Source label ID
        target_schema: Target schema name

    Returns:
        Mapped label ID or None if class doesn't exist in target
    """
    # First, get canonical name from source (assume source is "full" if unknown)
    canonical = get_class_name(label_id, schema="full")
    if canonical is None:
        return None

    # Then get ID in target schema
    return get_class_id(canonical, schema=target_schema)


# ============================================================================
# Module initialization validation
# ============================================================================

def _validate_registry():
    """Validate registry consistency on import."""
    # Check all schemas have contiguous IDs starting from 0
    for schema_name, classes in VISUAL_CLASSES.items():
        expected_ids = list(range(len(classes)))
        actual_ids = sorted(classes.keys())
        if actual_ids != expected_ids:
            raise ValueError(
                f"Schema '{schema_name}' has non-contiguous IDs: {actual_ids}"
            )

    # Check all aliases map to valid canonical names
    full_names = set(FULL_CLASSES.values())
    for alias, canonical in ALIASES.items():
        if alias != canonical and canonical not in full_names:
            raise ValueError(
                f"Alias '{alias}' maps to unknown canonical '{canonical}'"
            )

    # Check state features list is consistent
    if len(STATE_FEATURES) != STATE_FEATURE_DIM:
        raise ValueError(
            f"STATE_FEATURES length ({len(STATE_FEATURES)}) != STATE_FEATURE_DIM ({STATE_FEATURE_DIM})"
        )


# Run validation on import
_validate_registry()
