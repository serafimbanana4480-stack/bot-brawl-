from pathlib import Path

import cv2
import numpy as np

from training.advanced_augmentation import AdvancedAugmenter
from training.hyperparameter_tuner import HyperparameterTuner
from training.semi_supervised_trainer import SemiSupervisedTrainer, PseudoLabelConfig
from training.sota_training_pipeline import run_training


def test_advanced_augmenter_preserves_image_shape():
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    augmenter = AdvancedAugmenter(seed=123)
    out = augmenter.augment(image)

    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_hyperparameter_tuner_records_history(tmp_path):
    tuner = HyperparameterTuner(seed=7)
    history_path = tmp_path / "history.json"

    def train_fn(candidate):
        return {"mAP50": candidate.lr0 * 1000 + (10 if candidate.cos_lr else 0)}

    best_candidate, results = tuner.run(train_fn, n_trials=4, history_path=history_path)

    assert best_candidate is not None
    assert len(results) == 4
    assert history_path.exists()


def test_semi_supervised_trainer_generates_pseudo_labels(tmp_path):
    image_dir = tmp_path / "unlabeled"
    output_dir = tmp_path / "labels"
    image_dir.mkdir()

    image_path = image_dir / "sample.png"
    cv2.imwrite(str(image_path), np.full((100, 120, 3), 255, dtype=np.uint8))

    class DummyTensor:
        def __init__(self, value):
            self._value = value

        def __getitem__(self, item):
            return self._value

        def tolist(self):
            return self._value

    class DummyBox:
        def __init__(self):
            self.conf = DummyTensor(0.95)
            self.cls = DummyTensor(2)
            self.xyxy = [DummyTensor([10.0, 15.0, 50.0, 60.0])]

    class DummyResult:
        boxes = [DummyBox()]

    class DummyTeacher:
        def predict(self, **kwargs):
            return [DummyResult()]

    trainer = SemiSupervisedTrainer(DummyTeacher(), PseudoLabelConfig(confidence_threshold=0.5))
    stats = trainer.generate_pseudo_labels(image_dir, output_dir)

    label_file = output_dir / "sample.txt"
    assert label_file.exists()
    assert stats["images"] == 1
    assert "2 " in label_file.read_text(encoding="utf-8")


def test_sota_pipeline_routes_schema_and_training(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def fake_create_data_yaml(dataset_dir, schema="core"):
        yaml_path = dataset_dir / f"data_{schema}.yaml"
        yaml_path.write_text("nc: 4\n", encoding="utf-8")
        return yaml_path

    monkeypatch.setattr("training.sota_training_pipeline.create_data_yaml", fake_create_data_yaml)
    monkeypatch.setattr("training.sota_training_pipeline.train_yolo", lambda **kwargs: str(tmp_path / "model.pt"))
    monkeypatch.setattr("training.sota_training_pipeline.validate_model", lambda *args, **kwargs: {"mAP50": 0.42})

    result = run_training(data_dir=data_dir, schema="core", tune_trials=0)

    assert result["model_path"].endswith("model.pt")
    assert result["validation"]["mAP50"] == 0.42
