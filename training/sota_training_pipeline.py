"""
sota_training_pipeline.py

High-level orchestration for the project training stack:
- core/extended schema selection
- optional pseudo-label expansion
- optional hyperparameter search
- transfer-learning friendly training knobs

This is the practical "SOTA-ready" entry point for the current repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

from training.class_schema import get_schema, schema_name
from training.hyperparameter_tuner import HyperparameterTuner, TuningCandidate
from training.semi_supervised_trainer import PseudoLabelConfig, SemiSupervisedTrainer
from training.schema_dataset_builder import build_schema_dataset
from training.enhanced_training_pipeline import create_data_yaml, train_yolo, validate_model

logger = logging.getLogger(__name__)


def _build_train_kwargs(candidate: TuningCandidate | None, device: str, resume: bool) -> dict:
    kwargs = {"device": device, "resume": resume}
    if candidate is None:
        return kwargs
    kwargs.update(
        freeze=candidate.freeze,
        cos_lr=candidate.cos_lr,
        hyperparams=candidate.to_ultralytics_kwargs(),
    )
    return kwargs


def run_training(
    data_dir: Path,
    schema: str = "core",
    base_model: str = "models/brawlstars_yolov8.pt",
    device: str = "cpu",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    tune_trials: int = 0,
    unlabeled_dir: Path | None = None,
    pseudo_label_conf: float = 0.65,
    resume: bool = False,
) -> dict:
    """Run the end-to-end SOTA-ready training flow."""
    schema_map = get_schema(schema)
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    artifacts_dir = data_dir.parent / f"{data_dir.name}_{schema_name(schema_map)}_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    schema_dataset_dir = artifacts_dir / "dataset"
    schema_dataset_dir.mkdir(parents=True, exist_ok=True)

    if unlabeled_dir is not None and Path(unlabeled_dir).exists():
        from ultralytics import YOLO

        pseudo_dir = artifacts_dir / "pseudo_labels"
        teacher = YOLO(str(base_model))
        trainer = SemiSupervisedTrainer(teacher, PseudoLabelConfig(confidence_threshold=pseudo_label_conf))
        pseudo_stats = trainer.generate_pseudo_labels(Path(unlabeled_dir), pseudo_dir)
    else:
        pseudo_stats = {"images": 0, "labels": 0, "filtered": 0}

    build_stats = build_schema_dataset(data_dir, schema_dataset_dir, schema=schema)
    data_yaml = create_data_yaml(schema_dataset_dir, schema=schema)

    best_candidate = None
    tuning_results = []
    if tune_trials > 0:
        tuner = HyperparameterTuner()

        def objective(candidate: TuningCandidate) -> dict:
            train_path = train_yolo(
                data_yaml=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                pretrained=base_model,
                output_name=f"{schema_name(schema_map)}_tuned",
                **_build_train_kwargs(candidate, device=device, resume=resume),
            )
            if train_path is None:
                return {"mAP50": 0.0}
            metrics = validate_model(Path(train_path), data_yaml)
            return metrics

        best_candidate, tuning_results = tuner.run(
            objective,
            n_trials=tune_trials,
            history_path=artifacts_dir / "tuning_history.json",
        )

    train_kwargs = _build_train_kwargs(best_candidate, device=device, resume=resume)
    model_path = train_yolo(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        pretrained=base_model,
        output_name=f"{schema_name(schema_map)}_production",
        **train_kwargs,
    )

    validation = validate_model(Path(model_path), data_yaml) if model_path else {}
    return {
        "data_yaml": str(data_yaml),
        "model_path": model_path,
        "validation": validation,
        "pseudo_label_stats": pseudo_stats,
        "build_stats": build_stats,
        "best_candidate": best_candidate,
        "tuning_results": tuning_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA-ready training pipeline")
    parser.add_argument("--data-dir", type=str, default="dataset/yolo_final")
    parser.add_argument("--schema", type=str, default="core", choices=["core", "extended"])
    parser.add_argument("--base-model", type=str, default="models/brawlstars_yolov8.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--tune-trials", type=int, default=0)
    parser.add_argument("--unlabeled-dir", type=str, default=None)
    parser.add_argument("--pseudo-label-conf", type=float, default=0.65)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = run_training(
        data_dir=Path(args.data_dir),
        schema=args.schema,
        base_model=args.base_model,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        tune_trials=args.tune_trials,
        unlabeled_dir=Path(args.unlabeled_dir) if args.unlabeled_dir else None,
        pseudo_label_conf=args.pseudo_label_conf,
        resume=args.resume,
    )

    logger.info("Training result: %s", result)


if __name__ == "__main__":
    main()
