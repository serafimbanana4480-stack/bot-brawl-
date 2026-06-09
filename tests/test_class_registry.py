"""
tests/test_class_registry.py

Tests for the unified class registry system.

Validates:
- Class name normalization (aliases → canonical)
- Schema consistency (core/extended/full)
- Detection normalization
- Action space unification
"""

import pytest
import sys
from pathlib import Path

# Add project root to path


class TestClassNormalization:
    """Test class name normalization via aliases."""

    def test_get_canonical_basic(self):
        """Test basic canonical name resolution."""
        from core.class_registry import get_canonical

        assert get_canonical("Player") == "player"
        assert get_canonical("player") == "player"  # self-alias
        assert get_canonical("teammate") == "player"
        assert get_canonical("self") == "player"
        assert get_canonical("person") == "player"

    def test_get_canonical_enemy_variants(self):
        """Test enemy name variants."""
        from core.class_registry import get_canonical

        assert get_canonical("Enemy") == "enemy"
        assert get_canonical("enemy") == "enemy"
        assert get_canonical("brawler") == "enemy"
        assert get_canonical("opponent") == "enemy"

    def test_get_canonical_cubebox_variants(self):
        """Test cubebox/power cube variants."""
        from core.class_registry import get_canonical

        assert get_canonical("Cubebox") == "cubebox"
        assert get_canonical("power_cube") == "cubebox"
        assert get_canonical("powercube") == "cubebox"
        assert get_canonical("box") == "cubebox"

    def test_unknown_name_passes_through(self):
        """Unknown names should pass through unchanged."""
        from core.class_registry import get_canonical

        assert get_canonical("unknown_class") == "unknown_class"
        assert get_canonical("XYZ") == "xyz"  # lowercase


class TestSchemaDefinitions:
    """Test schema definitions are valid."""

    def test_core_schema_completeness(self):
        """Core schema has exactly 4 classes with contiguous IDs."""
        from core.class_registry import CORE_CLASSES, VISUAL_CLASSES

        assert len(CORE_CLASSES) == 4
        assert set(CORE_CLASSES.keys()) == {0, 1, 2, 3}
        assert set(CORE_CLASSES.values()) == {"player", "enemy", "cubebox", "powerup"}

    def test_extended_schema_completeness(self):
        """Extended schema has exactly 8 classes."""
        from core.class_registry import EXTENDED_CLASSES

        assert len(EXTENDED_CLASSES) == 8
        assert set(EXTENDED_CLASSES.keys()) == {0, 1, 2, 3, 4, 5, 6, 7}

    def test_full_schema_completeness(self):
        """Full schema has exactly 35 classes."""
        from core.class_registry import FULL_CLASSES

        assert len(FULL_CLASSES) == 35
        assert set(FULL_CLASSES.keys()) == set(range(35))

    def test_schema_containment(self):
        """Core ⊆ Extended ⊆ Full."""
        from core.class_registry import CORE_CLASSES, EXTENDED_CLASSES, FULL_CLASSES

        core_names = set(CORE_CLASSES.values())
        extended_names = set(EXTENDED_CLASSES.values())
        full_names = set(FULL_CLASSES.values())

        assert core_names.issubset(extended_names)
        assert extended_names.issubset(full_names)


class TestClassIDLookup:
    """Test class ID lookup in different schemas."""

    def test_get_class_id_core(self):
        """Test ID lookup in core schema."""
        from core.class_registry import get_class_id

        assert get_class_id("player", "core") == 0
        assert get_class_id("enemy", "core") == 1
        assert get_class_id("cubebox", "core") == 2
        assert get_class_id("powerup", "core") == 3

    def test_get_class_id_extended(self):
        """Test ID lookup in extended schema."""
        from core.class_registry import get_class_id

        assert get_class_id("player", "extended") == 0
        assert get_class_id("bush", "extended") == 1
        assert get_class_id("enemy", "extended") == 2
        assert get_class_id("bullet_neutral", "extended") == 6

    def test_get_class_id_full(self):
        """Test ID lookup in full schema."""
        from core.class_registry import get_class_id

        assert get_class_id("player", "full") == 0
        assert get_class_id("gem", "full") == 12
        assert get_class_id("ball", "full") == 13
        assert get_class_id("hot_zone", "full") == 14
        assert get_class_id("enemy_low_hp", "full") == 24
        assert get_class_id("objective_zone", "full") == 34

    def test_get_class_id_with_aliases(self):
        """Test ID lookup using aliases."""
        from core.class_registry import get_class_id

        # Should normalize alias to canonical first
        assert get_class_id("teammate", "core") == 0  # teammate → player
        assert get_class_id("Enemy", "core") == 1  # Enemy → enemy

    def test_class_not_in_schema_returns_none(self):
        """Classes not in schema should return None."""
        from core.class_registry import get_class_id

        assert get_class_id("bush", "core") is None  # bush not in core
        assert get_class_id("gem", "extended") is None  # gem not in extended


class TestDetectionNormalization:
    """Test detection dict normalization."""

    def test_normalize_detections_basic(self):
        """Test basic detection normalization."""
        from core.class_registry import normalize_detections

        detections = {
            "teammate": [[1, 2, 3, 4]],
            "Enemy": [[5, 6, 7, 8]],
        }

        normalized = normalize_detections(detections, "core")

        assert "player" in normalized
        assert "enemy" in normalized
        assert "teammate" not in normalized  # alias should be gone
        assert normalized["player"] == [[1, 2, 3, 4]]

    def test_normalize_detections_filters_by_schema(self):
        """Normalization should filter classes not in target schema."""
        from core.class_registry import normalize_detections

        detections = {
            "player": [[1, 2, 3, 4]],
            "enemy": [[5, 6, 7, 8]],
            "bush": [[9, 10, 11, 12]],  # not in core schema
        }

        normalized = normalize_detections(detections, "core")

        assert "player" in normalized
        assert "enemy" in normalized
        assert "bush" not in normalized  # filtered out

    def test_get_by_type_first_only(self):
        """Test get_by_type with first_only=True."""
        from core.class_registry import get_by_type

        detections = {
            "Player": [[1, 2, 3, 4], [5, 6, 7, 8]],  # 2 players detected
        }

        first = get_by_type(detections, "player", first_only=True)
        assert first == (1, 2, 3, 4)

    def test_get_by_type_all(self):
        """Test get_by_type with first_only=False."""
        from core.class_registry import get_by_type

        detections = {
            "teammate": [[1, 2, 3, 4], [5, 6, 7, 8]],
        }

        all_boxes = get_by_type(detections, "player", first_only=False)
        assert len(all_boxes) == 2


class TestUnifiedAction:
    """Test unified action space."""

    def test_action_count(self):
        """Test we have exactly 12 actions."""
        from core.class_registry import UnifiedAction

        assert UnifiedAction.count() == 12

    def test_action_from_string(self):
        """Test parsing actions from strings."""
        from core.class_registry import UnifiedAction

        assert UnifiedAction.from_string("attack") == UnifiedAction.ATTACK
        assert UnifiedAction.from_string("KITE") == UnifiedAction.KITE
        assert UnifiedAction.from_string("idle") == UnifiedAction.IDLE

    def test_action_values_unique(self):
        """All action values should be unique."""
        from core.class_registry import UnifiedAction

        values = [a.value for a in UnifiedAction]
        assert len(values) == len(set(values))


class TestActionMapper:
    """Test action mapper functionality."""

    def test_rl_to_unified_mapping(self):
        """Test RL action → UnifiedAction mapping."""
        from decision.action_mapper import rl_to_unified
        from core.class_registry import UnifiedAction

        assert rl_to_unified("attack") == UnifiedAction.ATTACK
        assert rl_to_unified("retreat") == UnifiedAction.RETREAT
        assert rl_to_unified("idle") == UnifiedAction.IDLE

    def test_unified_to_rl_mapping(self):
        """Test UnifiedAction → RL action mapping."""
        from decision.action_mapper import unified_to_rl
        from core.class_registry import UnifiedAction

        assert unified_to_rl(UnifiedAction.ATTACK) == "attack"
        assert unified_to_rl(UnifiedAction.RETREAT) == "retreat"

    def test_action_metadata(self):
        """Test action metadata retrieval."""
        from decision.action_mapper import get_action_metadata
        from core.class_registry import UnifiedAction

        meta = get_action_metadata(UnifiedAction.ATTACK)
        assert meta.name == "attack"
        assert meta.requires_target is True
        assert meta.requires_ammo is True


class TestRoboflowMappings:
    """Test Roboflow dataset mappings."""

    def test_roboflow_to_canonical(self):
        """Test Roboflow class names map to canonical."""
        from core.class_registry import ROBOFLOW_TO_CANONICAL

        assert ROBOFLOW_TO_CANONICAL["Enemy"] == "enemy"
        assert ROBOFLOW_TO_CANONICAL["Me"] == "player"
        assert ROBOFLOW_TO_CANONICAL["Ball"] == "ball"

    def test_roboflow_skip_classes(self):
        """Some Roboflow classes should be skipped (None)."""
        from core.class_registry import ROBOFLOW_TO_CANONICAL

        assert ROBOFLOW_TO_CANONICAL.get("Hot_Zone_Area") is None


class TestSchemaDetection:
    """Test automatic schema detection."""

    def test_detect_core_schema(self):
        """Detect core schema from model classes."""
        from core.class_registry import get_schema_for_model

        model_classes = {"player", "enemy", "cubebox", "powerup"}
        assert get_schema_for_model(model_classes) == "core"

    def test_detect_extended_schema(self):
        """Detect extended schema from model classes."""
        from core.class_registry import get_schema_for_model

        model_classes = {"player", "bush", "enemy", "cubebox", "wall"}
        assert get_schema_for_model(model_classes) == "extended"

    def test_detect_full_schema(self):
        """Detect full schema from model classes."""
        from core.class_registry import get_schema_for_model

        model_classes = {"player", "enemy", "cubebox", "gem", "ball", "hot_zone"}
        assert get_schema_for_model(model_classes) == "full"


class TestStateFeatures:
    """Test state feature definitions."""

    def test_state_features_count(self):
        """Test we have expected number of state features."""
        from core.class_registry import STATE_FEATURES, STATE_FEATURE_DIM

        assert len(STATE_FEATURES) == STATE_FEATURE_DIM
        assert STATE_FEATURE_DIM == 44  # 11 self + 17 spatial + 10 temporal + 6 per-enemy

    def test_self_state_features(self):
        """Test self-state features are defined."""
        from core.class_registry import SELF_STATE_FEATURES

        expected = [
            "hp_ratio", "ammo_ratio", "super_charge",
            "gadget_ready", "hypercharge_ready",
            "cooldown_attack", "cooldown_super",
            "is_moving", "is_in_bush",
            "velocity_norm", "current_tactic"
        ]
        for feat in expected:
            assert feat in SELF_STATE_FEATURES


class TestIntegration:
    """Integration tests with actual project files."""

    def test_class_schema_imports_registry(self):
        """Test that class_schema.py properly imports from registry."""
        from training import class_schema

        # Should have access to canonical functions
        assert hasattr(class_schema, "get_canonical_name")
        assert class_schema.get_canonical_name("Player") == "player"

    def test_play_py_can_use_registry(self):
        """Test that play.py detection methods work with registry."""
        # This is more of a smoke test - actual integration tested at runtime
        from core.class_registry import get_by_type

        # Simulate detections dict that play.py would receive
        detections = {
            "teammate": [[100, 200, 150, 250]],  # player alias
            "Enemy": [[300, 400, 350, 450]],  # enemy
            "Cubebox": [[500, 600, 550, 650]],  # cubebox alias
        }

        player = get_by_type(detections, "player", first_only=True)
        assert player == (100, 200, 150, 250)

        enemies = get_by_type(detections, "enemy", first_only=False)
        assert len(enemies) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
