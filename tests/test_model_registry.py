from pathlib import Path

from training.model_registry import ModelRegistry


def test_model_registry_accepts_string_paths(tmp_path):
    registry_dir = tmp_path / "registry"
    model_file = tmp_path / "demo.pt"
    model_file.write_bytes(b"demo-model")

    registry = ModelRegistry(str(registry_dir))
    model_id = registry.register_model(
        model_path=str(model_file),
        model_type="yolo",
        version="1.0.0",
        training_data="synthetic",
        training_metrics={"loss": 0.1},
        validation_metrics={"map50": 0.2},
    )

    assert model_id
    model = registry.get_model(model_id)
    assert model is not None
    assert model.class_schema == "core"
    assert Path(registry.models_file).exists()
    assert Path(registry.performance_file).exists() is False
