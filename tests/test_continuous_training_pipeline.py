import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from training.continuous_training_pipeline import ContinuousTrainingPipeline, PipelineMetrics
import tempfile


def test_pipeline_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "output"
        pipeline = ContinuousTrainingPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            min_samples=10,
            train_interval_minutes=1,
        )
        assert pipeline.data_dir == data_dir
        assert pipeline.output_dir == output_dir
        assert pipeline.min_samples == 10


def test_count_samples_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        output_dir = Path(tmpdir) / "output"
        pipeline = ContinuousTrainingPipeline(data_dir=data_dir, output_dir=output_dir)
        assert pipeline._count_samples() == 0
        assert not pipeline._has_enough_data()


def test_metrics_serialization():
    m = PipelineMetrics(
        run_id="run_001",
        timestamp="2024-01-01T00:00:00",
        data_samples=5,
        training_duration_sec=12.3,
        models_trained=["bc"],
        validation_results={"bc": {"score": 0.8}},
        deployed=False,
        notes="test",
    )
    d = m.to_dict()
    assert d["run_id"] == "run_001"
    assert d["models_trained"] == ["bc"]
    assert not d["deployed"]
