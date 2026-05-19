"""
test_roboflow_multi_downloader.py

Testes para training/roboflow_multi_downloader.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from training.roboflow_multi_downloader import (
    DatasetInfo,
    build_dataset_class_map,
    detect_dataset_structure,
    get_class_map_for_dataset,
    get_classes_from_labels,
    merge_datasets,
    prefix_filenames,
    read_yaml_classes,
    remap_dataset_classes,
)


class TestBuildDatasetClassMap:
    def test_returns_map(self):
        result = build_dataset_class_map(schema="core")
        assert "__default__" in result
        assert isinstance(result["__default__"], dict)

    def test_core_classes_present(self):
        result = build_dataset_class_map(schema="core")
        default = result["__default__"]
        assert "Enemy" in default
        assert default["Enemy"] >= 0


class TestGetClassMapForDataset:
    def test_auto_detect_known_classes(self):
        ds = DatasetInfo(workspace="test", project="test")
        class_names = ["Enemy", "Friendly", "PP_Box", "PP"]
        result = get_class_map_for_dataset(ds, schema="core", auto_detect_classes=class_names)
        assert len(result) == 4
        assert "Enemy" in result
        assert "Friendly" in result

    def test_auto_detect_heuristic(self):
        ds = DatasetInfo(workspace="test", project="test")
        class_names = ["brawler", "crate", "item", "gem"]
        result = get_class_map_for_dataset(ds, schema="full", auto_detect_classes=class_names)
        assert "brawler" in result
        assert "crate" in result
        assert "item" in result
        assert "gem" in result

    def test_unknown_classes_skipped(self):
        ds = DatasetInfo(workspace="test", project="test")
        class_names = ["UnknownClass123", "Background"]
        result = get_class_map_for_dataset(ds, schema="core", auto_detect_classes=class_names)
        assert len(result) == 0


class TestDetectDatasetStructure:
    def test_detects_splits(self, tmp_path):
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "train" / "labels").mkdir(parents=True)
        (tmp_path / "val" / "images").mkdir(parents=True)
        (tmp_path / "val" / "labels").mkdir(parents=True)
        splits, yaml = detect_dataset_structure(tmp_path)
        assert len(splits) == 2
        assert splits[0][0].name == "images"

    def test_detects_flat(self, tmp_path):
        (tmp_path / "images").mkdir(parents=True)
        (tmp_path / "labels").mkdir(parents=True)
        splits, yaml = detect_dataset_structure(tmp_path)
        assert len(splits) == 1
        assert splits[0][0].name == "images"

    def test_no_structure(self, tmp_path):
        splits, yaml = detect_dataset_structure(tmp_path)
        assert len(splits) == 0


class TestReadYamlClasses:
    def test_reads_yaml(self, tmp_path):
        import yaml
        data = {"names": {0: "player", 1: "enemy", 2: "cubebox"}}
        yaml_path = tmp_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)
        result = read_yaml_classes(yaml_path)
        assert result == ["player", "enemy", "cubebox"]

    def test_reads_list(self, tmp_path):
        import yaml
        data = {"names": ["player", "enemy"]}
        yaml_path = tmp_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)
        result = read_yaml_classes(yaml_path)
        assert result == ["player", "enemy"]


class TestRemapDatasetClasses:
    def test_remaps_labels(self, tmp_path):
        (tmp_path / "images").mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label_file = labels_dir / "test.txt"
        with open(label_file, "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
            f.write("1 0.3 0.3 0.2 0.2\n")

        class_map = {"old_cls_0": 3, "old_cls_1": 1}
        class_names = ["old_cls_0", "old_cls_1"]
        remapped, removed = remap_dataset_classes(tmp_path, class_map, class_names)
        assert remapped == 2
        assert removed == 0

        with open(label_file, "r") as f:
            lines = f.readlines()
        assert lines[0].startswith("3")
        assert lines[1].startswith("1")

    def test_removes_unmapped(self, tmp_path):
        (tmp_path / "images").mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label_file = labels_dir / "test.txt"
        with open(label_file, "w") as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
            f.write("99 0.3 0.3 0.2 0.2\n")

        class_map = {"old_cls_0": 3}
        class_names = ["old_cls_0"]  # ID 99 não existe na lista
        remapped, removed = remap_dataset_classes(tmp_path, class_map, class_names)
        assert remapped == 1
        assert removed == 1


class TestPrefixFilenames:
    def test_prefixes_files(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        img = images_dir / "test.jpg"
        lbl = labels_dir / "test.txt"
        img.write_text("")
        lbl.write_text("0 0.5 0.5 0.1 0.1")

        prefix_filenames(tmp_path, "ds1")

        assert (images_dir / "ds1_test.jpg").exists()
        assert (labels_dir / "ds1_test.txt").exists()
        assert not img.exists()
        assert not lbl.exists()


class TestGetClassesFromLabels:
    def test_reads_classes(self, tmp_path):
        (tmp_path / "images").mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        with open(labels_dir / "a.txt", "w") as f:
            f.write("0 0.1 0.2 0.3 0.4\n")
            f.write("3 0.1 0.2 0.3 0.4\n")
        with open(labels_dir / "b.txt", "w") as f:
            f.write("1 0.1 0.2 0.3 0.4\n")

        result = get_classes_from_labels(tmp_path)
        assert result == {0, 1, 3}


class TestMergeDatasets:
    def test_merge_two_datasets(self, tmp_path):
        # Criar dataset 1
        ds1 = tmp_path / "ds1"
        (ds1 / "train" / "images").mkdir(parents=True)
        (ds1 / "train" / "labels").mkdir(parents=True)
        (ds1 / "train" / "images" / "a.jpg").write_text("")
        with open(ds1 / "train" / "labels" / "a.txt", "w") as f:
            f.write("0 0.1 0.2 0.3 0.4\n")

        # Criar dataset 2
        ds2 = tmp_path / "ds2"
        (ds2 / "train" / "images").mkdir(parents=True)
        (ds2 / "train" / "labels").mkdir(parents=True)
        (ds2 / "train" / "images" / "b.jpg").write_text("")
        with open(ds2 / "train" / "labels" / "b.txt", "w") as f:
            f.write("1 0.1 0.2 0.3 0.4\n")

        output = tmp_path / "merged"
        stats = merge_datasets([ds1, ds2], output, schema="core")
        assert stats["total_images"] == 2
        assert stats["total_labels"] == 2
        assert (output / "data.yaml").exists()

    def test_empty_input(self, tmp_path):
        output = tmp_path / "merged"
        stats = merge_datasets([], output, schema="core")
        assert stats["total_images"] == 0
