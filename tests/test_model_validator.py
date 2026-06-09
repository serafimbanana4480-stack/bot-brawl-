"""
Testes para model_validator.py
"""
import sys
from pathlib import Path
import types


import pytest
from model_validator import (
    sha256_file, is_coco_model, is_brawl_stars_model,
    validate_all_models, get_model_classes, BRAWL_STARS_CLASSES, COCO_CLASSES_SAMPLE
)


class TestModelValidatorUtilities:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = sha256_file(f)
        assert len(h) == 64
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_sha256_file_accepts_string_path(self, tmp_path):
        f = tmp_path / "string.txt"
        f.write_text("hello world")
        h = sha256_file(str(f))
        assert len(h) == 64

    def test_is_coco_model(self):
        assert is_coco_model(list(COCO_CLASSES_SAMPLE)[:15])
        assert not is_coco_model(["enemy", "player"])
        assert not is_coco_model([])

    def test_is_brawl_stars_model(self):
        assert is_brawl_stars_model(list(BRAWL_STARS_CLASSES)[:4])
        assert not is_brawl_stars_model(["person", "car"])
        assert not is_brawl_stars_model([])

    def test_validate_all_models_empty_dir(self, tmp_path):
        import model_validator as mv
        orig_dir = mv.MODELS_DIR
        mv.MODELS_DIR = tmp_path
        try:
            result = validate_all_models()
            assert result["integrity_score"] == 0
            assert result["registry_written"] is False
        finally:
            mv.MODELS_DIR = orig_dir

    def test_get_model_classes_accepts_string_path(self, tmp_path, monkeypatch):
        model_path = tmp_path / "model.pt"
        model_path.write_bytes(b"fake checkpoint")

        fake_torch = types.SimpleNamespace(
            load=lambda *args, **kwargs: {"names": {0: "player", 1: "enemy", 2: "cubebox", 3: "powerup"}}
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        classes = get_model_classes(str(model_path))
        assert classes == ["player", "enemy", "cubebox", "powerup"]

    def test_validate_all_models_accepts_custom_models_dir(self, tmp_path, monkeypatch):
        model_path = tmp_path / "alpha.pt"
        model_path.write_bytes(b"alpha")

        import model_validator as mv
        monkeypatch.setattr(mv, "sha256_file", lambda path: "digest-alpha")
        monkeypatch.setattr(mv, "get_model_classes", lambda path: ["player", "enemy", "cubebox", "powerup"])

        result = validate_all_models(models_dir=str(tmp_path))
        assert result["integrity_score"] == 100
        assert result["valid"][0]["name"] == "alpha.pt"
        assert result["registry_written"] is True
