from pathlib import Path

import yaml

from training.class_schema import CORE_CLASSES, EXTENDED_CLASSES, get_schema
from training.enhanced_training_pipeline import create_data_yaml
from training.validate_dataset import generate_report


def test_get_schema_defaults_to_core():
    assert get_schema() == CORE_CLASSES


def test_create_data_yaml_uses_requested_schema(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    yaml_path = create_data_yaml(dataset_dir, schema="core")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    assert data["nc"] == 4
    assert data["names"] == CORE_CLASSES


def test_generate_report_tracks_expected_schema(tmp_path):
    dataset_dir = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    label = dataset_dir / "train" / "labels" / "sample.txt"
    label.write_text("0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.2 0.2\n2 0.3 0.3 0.2 0.2\n3 0.2 0.2 0.2 0.2\n")

    report = generate_report(dataset_dir, schema="core")
    schema_info = report["class_schema"]

    assert schema_info["expected"] == sorted(CORE_CLASSES.keys())
    assert schema_info["missing_expected"] == []
    assert schema_info["extra_classes"] == []
