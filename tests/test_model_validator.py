"""
Testes para model_validator.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from model_validator import (
    sha256_file, is_coco_model, is_brawl_stars_model,
    validate_all_models, BRAWL_STARS_CLASSES, COCO_CLASSES_SAMPLE
)


class TestModelValidatorUtilities:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = sha256_file(f)
        assert len(h) == 64
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_is_coco_model(self):
        assert is_coco_model(list(COCO_CLASSES_SAMPLE)[:15])
        assert not is_coco_model(["enemy", "player"])
        assert not is_coco_model([])

    def test_is_brawl_stars_model(self):
        assert is_brawl_stars_model(list(BRAWL_STARS_CLASSES)[:5])
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
