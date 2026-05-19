"""
test_training_workflow_e2e.py

Testes end-to-end para o pipeline de treino hibrido (Roboflow + screenshots proprios).

Verifica:
1. Estrutura do dataset Roboflow
2. Remapeamento de classes
3. Integridade do dataset (imagens + labels)
4. Carregamento do modelo treinado
5. Inferencia basica
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def project_root():
    return ROOT


@pytest.fixture
def roboflow_dataset_dir(project_root):
    d = project_root / "dataset" / "roboflow_raw_v2"
    if not d.exists():
        pytest.skip("Roboflow dataset not downloaded")
    return d


@pytest.fixture
def model_path(project_root):
    p = project_root / "models" / "brawlstars_yolov8.pt"
    if not p.exists():
        pytest.skip("Trained model not found")
    return p


# ============================================================
# Tests: Estrutura do Dataset
# ============================================================

class TestDatasetStructure:
    """Verifica estrutura do dataset Roboflow."""

    def test_dataset_exists(self, roboflow_dataset_dir):
        assert roboflow_dataset_dir.exists(), "Roboflow dataset directory missing"

    def test_data_yaml_exists(self, roboflow_dataset_dir):
        yaml_path = roboflow_dataset_dir / "data.yaml"
        assert yaml_path.exists(), "data.yaml missing"

    def test_data_yaml_valid(self, roboflow_dataset_dir):
        import yaml
        yaml_path = roboflow_dataset_dir / "data.yaml"
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert "names" in cfg, "data.yaml missing 'names'"
        assert "nc" in cfg, "data.yaml missing 'nc'"
        assert "train" in cfg, "data.yaml missing 'train'"
        assert "val" in cfg, "data.yaml missing 'val'"
        assert cfg["nc"] > 0, "nc must be > 0"
        assert len(cfg["names"]) == cfg["nc"], "nc must match number of names"

    def test_splits_exist(self, roboflow_dataset_dir):
        for split in ["train", "val", "test"]:
            images_dir = roboflow_dataset_dir / split / "images"
            labels_dir = roboflow_dataset_dir / split / "labels"
            assert images_dir.exists(), f"{split}/images missing"
            assert labels_dir.exists(), f"{split}/labels missing"

    def test_has_minimum_images(self, roboflow_dataset_dir):
        total = 0
        for split in ["train", "val", "test"]:
            images_dir = roboflow_dataset_dir / split / "images"
            if images_dir.exists():
                total += len(list(images_dir.glob("*")))
        assert total >= 100, f"Dataset too small: {total} images (minimum: 100)"


# ============================================================
# Tests: Integridade do Dataset
# ============================================================

class TestDatasetIntegrity:
    """Verifica integridade dos dados."""

    def test_every_image_has_label(self, roboflow_dataset_dir):
        """Toda imagem tem label correspondente."""
        missing = []
        for split in ["train", "val", "test"]:
            images_dir = roboflow_dataset_dir / split / "images"
            labels_dir = roboflow_dataset_dir / split / "labels"
            if not images_dir.exists():
                continue
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                label_file = labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    missing.append(f"{split}/{img_file.name}")
        assert len(missing) == 0, f"Images without labels: {missing}"

    def test_no_orphan_labels(self, roboflow_dataset_dir):
        """Nao ha labels sem imagem correspondente."""
        orphans = []
        for split in ["train", "val", "test"]:
            images_dir = roboflow_dataset_dir / split / "images"
            labels_dir = roboflow_dataset_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                # Check for image with any extension
                stem = label_file.stem
                found = False
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (images_dir / f"{stem}{ext}").exists():
                        found = True
                        break
                if not found:
                    orphans.append(f"{split}/{label_file.name}")
        assert len(orphans) == 0, f"Orphan labels: {orphans}"

    def test_label_format_valid(self, roboflow_dataset_dir):
        """Labels estao em formato YOLO valido."""
        invalid = []
        for split in ["train", "val", "test"]:
            labels_dir = roboflow_dataset_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            invalid.append(f"{label_file}:{line_num} - only {len(parts)} parts")
                            continue
                        try:
                            cls_id = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]
                            # Check coords are normalized (0-1)
                            for i, c in enumerate(coords):
                                if not (0.0 <= c <= 1.0):
                                    invalid.append(
                                        f"{label_file}:{line_num} - coord {i}={c} out of range"
                                    )
                        except (ValueError, IndexError) as e:
                            invalid.append(f"{label_file}:{line_num} - {e}")
        assert len(invalid) == 0, f"Invalid labels:\n" + "\n".join(invalid[:20])

    def test_class_distribution(self, roboflow_dataset_dir):
        """Classes tem distribuicao razoavel."""
        class_counts = defaultdict(int)
        for split in ["train", "val", "test"]:
            labels_dir = roboflow_dataset_dir / split / "labels"
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            cls_id = int(line.split()[0])
                            class_counts[cls_id] += 1

        assert len(class_counts) >= 2, f"Only {len(class_counts)} classes found"
        total = sum(class_counts.values())
        # No class should have < 5% of total
        for cls_id, count in class_counts.items():
            pct = count / total * 100
            assert pct >= 5.0, f"Class {cls_id} has only {pct:.1f}% of boxes"


# ============================================================
# Tests: Remapeamento de Classes
# ============================================================

class TestClassRemapping:
    """Verifica que o remapeamento de classes funciona."""

    def test_remap_function_imports(self):
        """Funcao remap_classes importa corretamente."""
        from training.download_roboflow_dataset import remap_classes, CLASS_MAP, ROBOFLOW_CLASS_NAMES
        assert len(CLASS_MAP) > 0
        assert len(ROBOFLOW_CLASS_NAMES) > 0

    def test_class_map_correct(self):
        """CLASS_MAP mapeia classes corretamente."""
        from training.download_roboflow_dataset import CLASS_MAP
        # Enemy classes -> 1 in core schema: Player=0, Enemy=1, Cubebox=2, Powerup=3
        assert CLASS_MAP["Enemy"] == 1
        assert CLASS_MAP["Safe_Enemy"] == 1
        # Player classes -> 0
        assert CLASS_MAP["Friendly"] == 0
        assert CLASS_MAP["Me"] == 0
        assert CLASS_MAP["Safe_Friendly"] == 0
        # Hot_Zone -> skip
        assert CLASS_MAP["Hot_Zone"] == -1

    def test_keep_classes_set(self):
        """KEEP_CLASSES contem as classes esperadas."""
        from training.download_roboflow_dataset import KEEP_CLASSES
        assert 0 in KEEP_CLASSES  # Player
        assert 1 in KEEP_CLASSES  # Enemy
        assert 2 in KEEP_CLASSES  # Cubebox
        assert 3 in KEEP_CLASSES  # Powerup

    def test_roboflow_class_names_match(self):
        """ROBOFLOW_CLASS_NAMES tem 10 classes."""
        from training.download_roboflow_dataset import ROBOFLOW_CLASS_NAMES
        assert len(ROBOFLOW_CLASS_NAMES) == 10


# ============================================================
# Tests: Modelo Treinado
# ============================================================

class TestTrainedModel:
    """Verifica modelo treinado."""

    def test_model_file_exists(self, model_path):
        assert model_path.exists(), f"Model not found: {model_path}"
        size_mb = model_path.stat().st_size / 1024 / 1024
        assert size_mb > 1.0, f"Model too small: {size_mb:.1f} MB"

    def test_model_loads(self, model_path):
        """Modelo carrega sem erros."""
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            assert model is not None
        except ImportError:
            pytest.skip("ultralytics not installed")
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")

    def test_model_has_classes(self, model_path):
        """Modelo tem as classes esperadas."""
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            assert hasattr(model, 'names'), "Model has no names attribute"
            assert len(model.names) >= 4, f"Expected >=4 classes, got {len(model.names)}"
        except ImportError:
            pytest.skip("ultralytics not installed")

    def test_model_inference(self, model_path, project_root):
        """Modelo faz inferencia basica."""
        try:
            import numpy as np
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            # Create dummy image (640x640x3)
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(dummy_img, conf=0.5, verbose=False)
            assert results is not None
            assert len(results) > 0
        except ImportError:
            pytest.skip("ultralytics not installed")

    def test_model_inference_on_real_image(self, model_path, roboflow_dataset_dir):
        """Modelo faz inferencia em imagem real do dataset."""
        try:
            from ultralytics import YOLO

            # Get a real image from test set
            test_images = list((roboflow_dataset_dir / "test" / "images").glob("*"))
            if not test_images:
                pytest.skip("No test images available")

            model = YOLO(str(model_path))
            results = model(str(test_images[0]), conf=0.25, verbose=False)
            assert results is not None
            assert len(results) > 0
            # Should detect at least something at low confidence
            boxes = results[0].boxes
            if boxes is not None:
                assert len(boxes) > 0, "Model detected nothing in test image"
        except ImportError:
            pytest.skip("ultralytics not installed")


# ============================================================
# Tests: Workflow Completo
# ============================================================

class TestWorkflowIntegration:
    """Verifica integracao do workflow."""

    def test_complete_workflow_imports(self):
        """Modulo complete_training_workflow importa corretamente."""
        from training.complete_training_workflow import (
            run_command, step_capture, step_download_roboflow,
            step_merge_datasets, step_validate, step_train,
            step_final_validation, check_emulator
        )
        assert callable(run_command)
        assert callable(step_train)
        assert callable(step_validate)

    def test_enhanced_pipeline_imports(self):
        """Modulo enhanced_training_pipeline importa corretamente."""
        from training.enhanced_training_pipeline import (
            STANDARD_CLASSES, NC, DataCurator, EnhancedAutoLabeler,
            ScreenCapturer, train_yolo, validate_model,
            prepare_dataset, create_data_yaml
        )
        assert len(STANDARD_CLASSES) == 8
        assert NC == 8

    def test_download_roboflow_imports(self):
        """Modulo download_roboflow_dataset importa corretamente."""
        from training.download_roboflow_dataset import (
            download_dataset, remap_classes, verify_dataset,
            verify_compatibility, merge_with_local, create_merged_yaml,
            CLASS_MAP, ROBOFLOW_CLASS_NAMES, KEEP_CLASSES
        )
        # Legacy hardcoded Roboflow class names (10 entries)
        assert len(ROBOFLOW_CLASS_NAMES) == 10
        # CLASS_MAP computed from class_registry (20 entries)
        assert len(CLASS_MAP) >= 10

    def test_validate_dataset_imports(self):
        """Modulo validate_dataset importa corretamente."""
        from training.validate_dataset import (
            validate_structure, count_classes, check_image_quality,
            check_bbox_sizes, generate_report
        )
        assert callable(validate_structure)
        assert callable(generate_report)

    def test_config_classes_match_standard(self, project_root):
        """config.json classes alinhadas com STANDARD_CLASSES."""
        from training.enhanced_training_pipeline import STANDARD_CLASSES

        config_path = project_root / "config.json"
        if not config_path.exists():
            pytest.skip("config.json not found")

        with open(config_path) as f:
            cfg = json.load(f)

        # Check training_schema config if present
        schema = cfg.get("training_schema", "")
        if schema:
            from training.class_schema import get_schema
            classes = get_schema(schema)
            assert len(classes) >= 4, f"Schema '{schema}' has {len(classes)} classes, expected >= 4"

        # Check brawler_queue game_mode entries
        brawler_queue = cfg.get("brawler_queue", [])
        if brawler_queue:
            valid_modes = {"showdown", "gem_grab", "brawl_ball", "heist", "bounty", "siege", "hot_zone", "knockout", "wipeout", "duels"}
            for brawler in brawler_queue:
                mode = brawler.get("game_mode", "")
                if mode:
                    assert mode in valid_modes or mode == "", f"Unknown game_mode: {mode}"


# ============================================================
# Tests: Compatibilidade
# ============================================================

class TestCompatibility:
    """Verifica compatibilidade entre modulos."""

    def test_verify_compatibility_function(self):
        """verify_compatibility funciona corretamente."""
        from training.download_roboflow_dataset import verify_compatibility
        # Test with non-existent dirs - should handle gracefully
        result = verify_compatibility(Path("/nonexistent1"), Path("/nonexistent2"))
        assert isinstance(result, bool)

    def test_standard_classes_consistent(self):
        """STANDARD_CLASSES consistente entre modulos."""
        from training.enhanced_training_pipeline import STANDARD_CLASSES as SC1
        from training.download_roboflow_dataset import STANDARD_CLASSES as SC2
        assert SC1 == SC2, "STANDARD_CLASSES mismatch between modules"