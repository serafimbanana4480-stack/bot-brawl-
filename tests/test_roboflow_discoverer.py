"""
test_roboflow_discoverer.py

Testes para training/roboflow_dataset_discoverer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from training.roboflow_dataset_discoverer import (
    DatasetInfo,
    KNOWN_DATASETS,
    compute_compatibility_score,
    filter_compatible,
    score_dataset,
    search_universe,
)


class TestDatasetInfo:
    def test_download_url(self):
        ds = DatasetInfo(workspace="bloxxy", project="brawl-stars-dataset")
        assert "universe.roboflow.com/ds/bloxxy/brawl-stars-dataset/download?format=yolov8" in ds.download_url

    def test_universe_url(self):
        ds = DatasetInfo(workspace="bloxxy", project="brawl-stars-dataset")
        assert ds.universe_url == "https://universe.roboflow.com/bloxxy/brawl-stars-dataset"

    def test_to_dict(self):
        ds = DatasetInfo(workspace="w", project="p", name="test")
        d = ds.to_dict()
        assert d["workspace"] == "w"
        assert d["project"] == "p"
        assert d["name"] == "test"


class TestCompatibilityScoring:
    def test_perfect_match_core(self):
        classes = ["Player", "Enemy", "Cubebox", "Powerup"]
        score = compute_compatibility_score(classes, schema="core")
        assert score == 1.0

    def test_partial_match(self):
        classes = ["Player", "Enemy"]
        score = compute_compatibility_score(classes, schema="core")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        classes = ["Background", "UI", "button"]
        score = compute_compatibility_score(classes, schema="core")
        assert score == 0.0

    def test_empty_classes(self):
        score = compute_compatibility_score([], schema="core")
        assert score == 0.0

    def test_extended_schema(self):
        classes = ["player", "enemy", "bush", "wall"]
        score = compute_compatibility_score(classes, schema="extended")
        assert 0.0 < score <= 1.0


class TestScoreDataset:
    def test_score_known_dataset(self):
        ds = KNOWN_DATASETS[0]  # bloxxy dataset
        scored = score_dataset(ds, schema="core")
        assert scored.compatibility_score > 0.0

    def test_unknown_classes_zero_score(self):
        ds = DatasetInfo(workspace="test", project="test", class_names=["UnknownClass"])
        scored = score_dataset(ds, schema="core")
        assert scored.compatibility_score == 0.0


class TestFilterCompatible:
    def test_filters_by_score(self):
        datasets = [
            DatasetInfo(workspace="a", project="a", compatibility_score=0.9),
            DatasetInfo(workspace="b", project="b", compatibility_score=0.1),
            DatasetInfo(workspace="c", project="c", compatibility_score=0.5),
        ]
        result = filter_compatible(datasets, min_score=0.5)
        assert len(result) == 2
        assert result[0].workspace == "a"

    def test_filters_by_type(self):
        datasets = [
            DatasetInfo(workspace="a", project="a", type="Object Detection", compatibility_score=0.9),
            DatasetInfo(workspace="b", project="b", type="Instance Segmentation", compatibility_score=0.9),
        ]
        result = filter_compatible(datasets, target_types={"Object Detection"})
        assert len(result) == 1
        assert result[0].workspace == "a"


class TestSearchUniverse:
    def test_returns_empty_without_dependencies(self, monkeypatch):
        # Simular que requests/bs4 não estão instalados
        monkeypatch.setitem(sys.modules, "requests", None)
        result = search_universe(max_results=10)
        assert result == []

    def test_basic_structure(self):
        # Sem fazer scraping real, verificar que a função existe e retorna lista
        result = search_universe(max_results=1)
        assert isinstance(result, list)
