"""
train.py

Ponto de entrada unificado e otimizado para treinamento de modelos YOLO do Brawl Stars.

Suporta schemas: core (4 classes), extended (8 classes), full (35 classes).
Integra HyperparameterTuner, SemiSupervisedTrainer, ModelRegistry, data augmentation,
mixed-precision (AMP), multi-GPU, TensorBoard, e validação de dataset.

Todos os modelos treinados são guardados em models/ com nome padronizado e
registrados no ModelRegistry para versionamento e rollback.

Usage:
    python train.py --schema core --epochs 50 --batch 8 --device auto
    python train.py --schema extended --epochs 100 --batch 16 --device cuda
    python train.py --schema full --epochs 150 --batch 8 --device 0 --model-size m
    python train.py --schema core --tune 8 --pseudo-label --epochs 100
    python train.py --schema extended --amp --lr-find --validate-dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YOLO_SIZES = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}

DEFAULT_TRAIN_ARGS = {
    "patience": 15,
    "save": True,
    "save_period": 5,
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,
    "plots": True,
    "save_json": True,
    "single_cls": False,
    "rect": False,
    "nbs": 64,
    "overlap_mask": True,
    "mask_ratio": 4,
    "workers": 4,
    "seed": 42,
    "deterministic": True,
    "cache": False,
    "val": True,
}


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(device: str) -> str:
    """Resolve device string. Supports 'auto', 'cpu', 'cuda', '0', '0,1'."""
    if device == "cpu":
        return "cpu"
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        return "cpu"
    return device  # "0", "0,1", etc.


def is_gpu(device: str) -> bool:
    resolved = resolve_device(device)
    return resolved != "cpu"


# ---------------------------------------------------------------------------
# Dataset validation (health check)
# ---------------------------------------------------------------------------
def validate_dataset_health(dataset_dir: Path, schema: str) -> Tuple[bool, Dict]:
    """
    Pre-training dataset health check.
    Returns (ok, report).
    """
    from training.class_schema import get_schema

    schema_map = get_schema(schema)
    expected_ids = set(schema_map.keys())

    report = {
        "schema": schema,
        "expected_classes": len(schema_map),
        "checks": {},
    }

    # Structure check
    structure_ok = True
    for split in ("train", "val", "test"):
        imgs = dataset_dir / split / "images"
        lbls = dataset_dir / split / "labels"
        imgs_ok = imgs.exists() and any(imgs.iterdir())
        lbls_ok = lbls.exists() and any(lbls.iterdir())
        report["checks"][f"{split}_images"] = imgs_ok
        report["checks"][f"{split}_labels"] = lbls_ok
        if not imgs_ok or not lbls_ok:
            structure_ok = False

    if not structure_ok:
        return False, report

    # Count images and boxes
    total_images = 0
    total_boxes = 0
    present_classes = set()
    class_counts: Dict[int, int] = {}
    empty_labels = 0

    for split in ("train", "val"):
        lbls_dir = dataset_dir / split / "labels"
        if not lbls_dir.exists():
            continue
        for lbl in lbls_dir.glob("*.txt"):
            total_images += 1
            boxes = 0
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        present_classes.add(cls_id)
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                        boxes += 1
                        total_boxes += 1
                    except ValueError:
                        pass
            if boxes == 0:
                empty_labels += 1

    report["total_images"] = total_images
    report["total_boxes"] = total_boxes
    report["empty_labels"] = empty_labels
    report["present_classes"] = sorted(present_classes)
    report["class_distribution"] = {str(k): v for k, v in sorted(class_counts.items())}

    missing = sorted(expected_ids - present_classes)
    extra = sorted(present_classes - expected_ids)

    report["missing_classes"] = missing
    report["extra_classes"] = extra

    # Validate
    if total_images < 10:
        logger.warning("Dataset muito pequeno (<10 imagens). Treino pode falhar.")
    if empty_labels / max(total_images, 1) > 0.5:
        logger.warning("Mais de 50%% das imagens sem labels. Verifique o dataset.")
    if missing:
        logger.warning(f"Schema {schema}: classes esperadas ausentes: {missing}")

    ok = total_images >= 10 and total_boxes >= 10 and len(present_classes) >= 2
    return ok, report


# ---------------------------------------------------------------------------
# Pseudo-label pipeline
# ---------------------------------------------------------------------------
def run_pseudo_labeling(
    unlabeled_dir: Path,
    output_dir: Path,
    teacher_model_path: Path,
    confidence: float = 0.65,
) -> Dict[str, int]:
    """Generate pseudo-labels using SemiSupervisedTrainer."""
    from ultralytics import YOLO
    from training.semi_supervised_trainer import SemiSupervisedTrainer, PseudoLabelConfig

    logger.info(f"Carregando teacher model: {teacher_model_path}")
    teacher = YOLO(str(teacher_model_path))
    config = PseudoLabelConfig(confidence_threshold=confidence, max_images=None)
    trainer = SemiSupervisedTrainer(teacher, config)
    stats = trainer.generate_pseudo_labels(unlabeled_dir, output_dir)
    logger.info(f"Pseudo-labels gerados: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Model registry integration
# ---------------------------------------------------------------------------
def register_in_registry(
    model_path: Path,
    model_type: str,
    schema: str,
    metrics: Dict[str, float],
    training_data: str = "",
) -> Optional[str]:
    """Register trained model in the ModelRegistry."""
    try:
        from core.model_registry import ModelRegistry

        registry = ModelRegistry()
        ver = registry.register(
            name=f"yolo_{schema}",
            path=model_path,
            metrics=metrics,
            metadata={"training_data": training_data, "timestamp": datetime.now().isoformat()},
        )
        registry.set_active(f"yolo_{schema}", ver.version)
        logger.info(f"Modelo registrado: yolo_{schema}@{ver.version}")
        return ver.version
    except Exception as e:
        logger.warning(f"Nao foi possivel registar no ModelRegistry: {e}")
        return None


# ---------------------------------------------------------------------------
# TensorBoard / experiment tracking setup
# ---------------------------------------------------------------------------
def setup_tracking(run_name: str) -> Optional[str]:
    """Setup experiment tracking. Returns run ID or None."""
    run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        import tensorboard  # noqa: F401
        logger.info(f"TensorBoard disponivel. Run ID: {run_id}")
        return run_id
    except ImportError:
        pass

    try:
        import mlflow
        mlflow.set_experiment("brawlstars_yolo")
        mlflow.start_run(run_name=run_id)
        logger.info(f"MLflow tracking iniciado: {run_id}")
        return run_id
    except ImportError:
        pass

    return run_id  # fallback: use run_id for directory naming


def log_metrics_mlflow(metrics: Dict[str, float]):
    """Log metrics to MLflow if available."""
    try:
        import mlflow
        mlflow.log_metrics(metrics)
    except Exception:
        pass


def end_tracking():
    """End experiment tracking."""
    try:
        import mlflow
        mlflow.end_run()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Training report generation
# ---------------------------------------------------------------------------
def generate_training_report(
    run_id: str,
    args: argparse.Namespace,
    schema: str,
    classes: Dict[int, str],
    dataset_report: Dict,
    pseudo_stats: Optional[Dict],
    tune_result: Optional[Dict],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    model_path: Path,
    duration_sec: float,
    registry_version: Optional[str],
) -> Path:
    """Generate a comprehensive training report."""
    report_dir = Path("runs") / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration_sec, 1),
        "config": {
            "schema": schema,
            "num_classes": len(classes),
            "model_size": args.model_size,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "amp": args.amp,
            "lr_find": args.lr_find,
            "multi_scale": args.multi_scale,
            "close_mosaic": args.close_mosaic,
            "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing,
            "dropout": args.dropout,
        },
        "classes": {str(k): v for k, v in classes.items()},
        "dataset_report": dataset_report,
        "pseudo_label_stats": pseudo_stats,
        "hyperparameter_tuning": tune_result,
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "model_path": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 2) if model_path.exists() else 0,
        "registry_version": registry_version,
        "status": "success",
    }

    report_path = report_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Also save a human-readable summary
    summary_path = report_dir / "training_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Run ID:     {run_id}\n")
        f.write(f"Timestamp:  {report['timestamp']}\n")
        f.write(f"Duration:   {duration_sec:.1f}s\n")
        f.write(f"Schema:     {schema} ({len(classes)} classes)\n")
        f.write(f"Model:      {args.model_size}\n")
        f.write(f"Device:     {args.device} (resolved: {resolve_device(args.device)})\n\n")
        f.write("Validation:\n")
        for k, v in val_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nModel: {model_path}\n")

    logger.info(f"Relatorio de treino salvo: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Treinar YOLO para Brawl Stars (unificado + otimizado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Core training args ---
    parser.add_argument("--epochs", type=int, default=50, help="Numero de epocas (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, 0, 0,1 (default: auto)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem (default: 640)")
    parser.add_argument("--data", type=str, default="dataset/roboflow_raw_v2/data.yaml", help="Dataset config YAML")

    # --- Schema ---
    parser.add_argument(
        "--schema", type=str, default="core",
        choices=["core", "extended", "full"],
        help="Schema de classes (default: core)",
    )

    # --- Model architecture ---
    parser.add_argument(
        "--model-size", type=str, default="n",
        choices=list(YOLO_SIZES.keys()),
        help="Tamanho do modelo YOLOv8: n, s, m, l, x (default: n)",
    )
    parser.add_argument("--freeze", type=int, default=0, help="Freeze backbone layers (default: 0)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing (default: 0.0)")

    # --- Learning rate ---
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR factor (default: 0.01)")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR schedule")
    parser.add_argument("--lr-find", action="store_true", help="Auto-find optimal LR before training")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs (default: 3.0)")
    parser.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup initial momentum (default: 0.8)")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1, help="Warmup initial bias LR (default: 0.1)")

    # --- Mixed precision & performance ---
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision (AMP) on GPU")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk for faster training")
    parser.add_argument("--multi-scale", action="store_true", help="Enable multi-scale training (0.5-1.5x)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Close mosaic aug for last N epochs (default: 10)")

    # --- Reproducibility ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic mode")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="Disable deterministic")

    # --- Resume ---
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--warm-start", action="store_true", help="Warm-start from registry checkpoint")

    # --- Export ---
    parser.add_argument("--export-trt", action="store_true", help="Export to TensorRT after training")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--export-openvino", action="store_true", help="Export to OpenVINO after training")

    # --- Hyperparameter tuning ---
    parser.add_argument("--tune", type=int, default=0, metavar="N",
                       help="Run N hyperparameter tuning trials before final training")

    # --- Semi-supervised ---
    parser.add_argument("--pseudo-label", action="store_true", help="Generate pseudo-labels from unlabeled images")
    parser.add_argument("--unlabeled-dir", type=str, default="dataset/unlabeled",
                       help="Directory with unlabeled images (default: dataset/unlabeled)")
    parser.add_argument("--pseudo-conf", type=float, default=0.65,
                       help="Confidence threshold for pseudo-labels (default: 0.65)")

    # --- Dataset validation ---
    parser.add_argument("--validate-dataset", action="store_true", help="Run pre-training dataset health check")
    parser.add_argument("--validate-dataset-only", action="store_true",
                       help="Only validate dataset, skip training")

    # --- Augmentation ---
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation prob (default: 1.0)")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation prob (default: 0.0)")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation prob (default: 0.0)")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation (default: 0.015)")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation (default: 0.7)")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation (default: 0.4)")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees (default: 0.0)")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation (default: 0.1)")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale (default: 0.5)")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear (default: 0.0)")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective (default: 0.0)")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip prob (default: 0.0)")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip prob (default: 0.5)")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing prob (default: 0.4)")
    parser.add_argument("--crop-fraction", type=float, default=1.0, help="Crop fraction (default: 1.0)")
    parser.add_argument("--auto-augment", type=str, default=None, choices=["randaugment", "autoaugment", "augmix"],
                       help="Auto-augmentation policy")

    # --- Optimizer ---
    parser.add_argument("--optimizer", type=str, default="auto",
                       choices=["auto", "SGD", "Adam", "AdamW", "RMSProp"],
                       help="Optimizer (default: auto)")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum / Adam beta1 (default: 0.937)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay (default: 0.0005)")

    # --- NMS ---
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image (default: 300)")

    # --- Fraction ---
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (default: 1.0)")

    # --- Other ---
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights (default: True)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Train from scratch")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--save-period", type=int, default=5, help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--no-registry", action="store_true", help="Skip ModelRegistry registration")
    parser.add_argument("--profile", action="store_true", help="Enable ONNX/TensorRT profiling after export")

    args = parser.parse_args()

    # ========================================================================
    # PHASE 0: Setup tracking
    # ========================================================================
    run_id = setup_tracking(f"brawlstars_{args.schema}_{args.model_size}")

    start_time = time.time()

    # ========================================================================
    # PHASE 1: Dataset discovery & validation
    # ========================================================================
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"ERRO: Dataset nao encontrado em {data_path}")
        logger.error("Execute primeiro: python dataset_pipeline.py --adb-id <id> --duration 300")
        sys.exit(1)

    from training.class_schema import get_schema
    from training.schema_dataset_builder import build_schema_dataset

    classes = get_schema(args.schema)
    source_dataset = data_path.parent
    derived_dataset = source_dataset.parent / f"{source_dataset.name}_{args.schema}"
    build_schema_dataset(source_dataset, derived_dataset, schema=args.schema)
    derived_data = derived_dataset / "data.yaml"
    derived_config = {
        "path": str(derived_dataset.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": classes,
        "nc": len(classes),
    }
    with open(derived_data, "w", encoding="utf-8") as f:
        yaml.safe_dump(derived_config, f, sort_keys=True)

    # Pre-training dataset health check
    dataset_report: Dict = {"status": "skipped"}
    if args.validate_dataset or args.validate_dataset_only:
        logger.info("Executando validacao do dataset...")
        ds_ok, dataset_report = validate_dataset_health(derived_dataset, args.schema)
        if not ds_ok:
            logger.error("Dataset validation FAILED. Corrija os problemas antes de treinar.")
            logger.error(json.dumps(dataset_report, indent=2, ensure_ascii=False))
            if args.validate_dataset_only:
                sys.exit(1)
        else:
            logger.info("Dataset validation PASSED.")

    if args.validate_dataset_only:
        logger.info("--validate-dataset-only: a sair.")
        return

    # ========================================================================
    # PHASE 2: Pseudo-labeling (optional)
    # ========================================================================
    pseudo_stats: Optional[Dict] = None
    if args.pseudo_label:
        unlabeled = Path(args.unlabeled_dir)
        teacher = Path("models/brawlstars_yolov8.pt")
        if not teacher.exists():
            teacher = Path(YOLO_SIZES[args.model_size])
        if unlabeled.exists() and any(unlabeled.iterdir()):
            pseudo_output = derived_dataset.parent / f"{derived_dataset.name}_pseudo"
            pseudo_stats = run_pseudo_labeling(unlabeled, pseudo_output, teacher, args.pseudo_conf)
        else:
            logger.warning(f"Diretorio unlabeled nao encontrado ou vazio: {unlabeled}")

    # ========================================================================
    # PHASE 3: Hyperparameter tuning (optional)
    # ========================================================================
    tune_result: Optional[Dict] = None
    best_candidate = None

    if args.tune > 0:
        logger.info(f"Iniciando hyperparameter tuning com {args.tune} trials...")
        from training.hyperparameter_tuner import HyperparameterTuner

        tuner = HyperparameterTuner(seed=args.seed)
        resolved_dev = resolve_device(args.device)

        def tuning_objective(candidate):
            from ultralytics import YOLO
            tmp_dir = derived_dataset.parent / f"{derived_dataset.name}_tune_tmp"
            build_schema_dataset(source_dataset, tmp_dir, schema=args.schema)
            tmp_yaml = tmp_dir / "data.yaml"
            tmp_cfg = {
                "path": str(tmp_dir.resolve()),
                "train": "train/images",
                "val": "val/images",
                "names": classes,
                "nc": len(classes),
            }
            with open(tmp_yaml, "w", encoding="utf-8") as f:
                yaml.safe_dump(tmp_cfg, f, sort_keys=True)

            model = YOLO(YOLO_SIZES[args.model_size])
            kwargs = candidate.to_ultralytics_kwargs()
            kwargs.update(
                data=str(tmp_yaml),
                epochs=min(args.epochs, 20),
                batch=args.batch,
                imgsz=args.imgsz,
                device=resolved_dev,
                project="runs",
                name=f"brawlstars_{args.schema}_tune",
                exist_ok=True,
                verbose=False,
                patience=5,
                save=False,
                plots=False,
            )
            model.train(**kwargs)
            metrics = model.val(data=str(tmp_yaml), device=resolved_dev, verbose=False)
            return {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}

        best_candidate, tuning_results = tuner.run(
            tuning_objective,
            metric_name="mAP50",
            n_trials=args.tune,
            history_path=Path("runs") / run_id / "tuning_history.json",
        )

        if best_candidate:
            tune_result = {
                "best_trial_mAP50": max(r.metric_value for r in tuning_results),
                "best_config": {
                    "lr0": best_candidate.lr0,
                    "lrf": best_candidate.lrf,
                    "momentum": best_candidate.momentum,
                    "weight_decay": best_candidate.weight_decay,
                    "warmup_epochs": best_candidate.warmup_epochs,
                    "freeze": best_candidate.freeze,
                    "cos_lr": best_candidate.cos_lr,
                },
            }
            # Apply best candidate to args
            args.lr0 = args.lr0 or best_candidate.lr0
            args.lrf = args.lrf or best_candidate.lrf
            args.momentum = best_candidate.momentum
            args.weight_decay = best_candidate.weight_decay
            args.warmup_epochs = best_candidate.warmup_epochs
            args.freeze = best_candidate.freeze
            args.cos_lr = best_candidate.cos_lr

            logger.info(f"Tuning concluido. Melhor mAP50: {tune_result['best_trial_mAP50']:.4f}")

    # ========================================================================
    # PHASE 4: Training
    # ========================================================================
    print()
    print("=" * 60)
    print("TREINO YOLO UNIFICADO (OTIMIZADO)")
    print("=" * 60)
    print(f"Dataset:      {args.data}")
    print(f"Derived YAML: {derived_data}")
    print(f"Schema:       {args.schema} ({len(classes)} classes)")
    print(f"Model:        YOLOv8{args.model_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch:        {args.batch}")
    print(f"Image size:   {args.imgsz}")
    print(f"Device:       {args.device}")
    print(f"AMP:          {args.amp}")
    print(f"LR find:      {args.lr_find}")
    print(f"Multi-scale:  {args.multi_scale}")
    print(f"Close mosaic: {args.close_mosaic}")
    print(f"Warmup:       {args.warmup_epochs}")
    print(f"Patience:     {args.patience}")
    print(f"Freeze:       {args.freeze}")
    print(f"Seed:         {args.seed}")
    print(f"Deterministic:{args.deterministic}")
    print(f"Pretrained:   {args.pretrained}")
    if tune_result:
        print(f"Tuned:        {tune_result['best_trial_mAP50']:.4f} mAP50 (best)")
    print("=" * 60)

    resolved_device = resolve_device(args.device)
    print(f"Resolved device: {resolved_device}")

    # GPU checks
    if is_gpu(args.device):
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            print(f"CUDA GPUs disponiveis: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // (1024**2)} MB)")
        except Exception:
            pass

    # Set deterministic if requested
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            import torch
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # Warm-start from registry
    warm_start_path = None
    if args.warm_start:
        try:
            from core.model_registry import ModelRegistry
            registry = ModelRegistry()
            warm_start_path = registry.get_warm_start_path(f"yolo_{args.schema}")
            if warm_start_path:
                logger.info(f"Warm-start do registry: {warm_start_path}")
        except Exception:
            pass

    # Load model
    model_path = YOLO_SIZES[args.model_size]
    if warm_start_path and warm_start_path.exists():
        model_path = str(warm_start_path)
    print(f"\nCarregando modelo base: {model_path}...")
    from ultralytics import YOLO
    model = YOLO(model_path)

    # Auto LR find
    if args.lr_find:
        print("\nExecutando auto LR finder...")
        try:
            lr_results = model.tune(
                data=str(derived_data),
                method="lr_find",
                iterations=100,
                device=resolved_device,
            )
            suggested_lr = lr_results.get("lr", 0.001)
            if args.lr0 is None:
                args.lr0 = suggested_lr
            print(f"Suggested LR: {suggested_lr}")
        except Exception as e:
            logger.warning(f"LR finder falhou: {e}")

    # Build training kwargs
    train_kwargs = dict(DEFAULT_TRAIN_ARGS)
    train_kwargs.update({
        "data": str(derived_data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": resolved_device,
        "project": "runs",
        "name": f"brawlstars_{args.schema}",
        "patience": args.patience,
        "save_period": args.save_period,
        "freeze": args.freeze,
        "cos_lr": args.cos_lr,
        "resume": args.resume,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "workers": args.workers,
        "cache": args.cache,
        "amp": args.amp and is_gpu(args.device),
        "multi_scale": args.multi_scale,
        "close_mosaic": args.close_mosaic,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "iou": args.iou,
        "max_det": args.max_det,
        "fraction": args.fraction,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "erasing": args.erasing,
        "crop_fraction": args.crop_fraction,
        "auto_augment": args.auto_augment,
        "rect": False,
        "single_cls": False,
        "nbs": 64,
        "overlap_mask": True,
        "mask_ratio": 4,
    })

    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf

    # Remove None values
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    # ========================================================================
    # TRAIN
    # ========================================================================
    print("\nIniciando treino...")
    model.train(**train_kwargs)

    # ========================================================================
    # PHASE 5: Validation & metrics
    # ========================================================================
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(f"runs/brawlstars_{args.schema}")))
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    print()
    print("=" * 60)
    print("TREINO CONCLUIDO")
    print("=" * 60)
    print(f"Modelo guardado em: {best_path}")
    print(f"Ultimo modelo: {last_path}")

    # Validation
    print("\nValidando modelo...")
    val_metrics_raw = model.val(data=str(derived_data), device=resolved_device)
    val_metrics = {
        "mAP50": round(float(val_metrics_raw.box.map50), 4),
        "mAP50-95": round(float(val_metrics_raw.box.map), 4),
        "precision": round(float(getattr(val_metrics_raw.box, "mp", 0.0)), 4),
        "recall": round(float(getattr(val_metrics_raw.box, "mr", 0.0)), 4),
    }

    print(f"mAP50:     {val_metrics['mAP50']:.4f}")
    print(f"mAP50-95:  {val_metrics['mAP50-95']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall:    {val_metrics['recall']:.4f}")

    # ========================================================================
    # PHASE 6: Copy to models/ directory
    # ========================================================================
    dest = None
    if best_path.exists():
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / f"brawlstars_yolov8_{args.model_size}_{args.schema}.pt"
        shutil.copy(best_path, dest)
        print(f"\nModelo copiado para: {dest}")

        # Legacy alias for core + nano
        if args.schema == "core" and args.model_size == "n":
            legacy_dest = models_dir / "brawlstars_yolov8.pt"
            try:
                shutil.copy(best_path, legacy_dest)
                print(f"Modelo tambem copiado para legado: {legacy_dest}")
            except Exception as e:
                logger.warning(f"Aviso: nao foi possivel atualizar o legado: {e}")

    # ========================================================================
    # PHASE 7: Model registry
    # ========================================================================
    registry_version = None
    if dest and dest.exists() and not args.no_registry:
        registry_version = register_in_registry(
            model_path=dest,
            model_type=f"yolo_{args.model_size}",
            schema=args.schema,
            metrics=val_metrics,
            training_data=str(derived_dataset),
        )

    # ========================================================================
    # PHASE 8: Export (TensorRT, ONNX, OpenVINO)
    # ========================================================================
    for fmt, flag, name in [
        ("engine", args.export_trt, "TensorRT"),
        ("onnx", args.export_onnx, "ONNX"),
        ("openvino", args.export_openvino, "OpenVINO"),
    ]:
        if flag:
            print(f"\nExportando para {name}...")
            try:
                export_kwargs = {"format": fmt}
                if fmt == "engine":
                    export_kwargs["device"] = 0
                    export_kwargs["half"] = True
                if args.profile:
                    export_kwargs["profile"] = True
                model.export(**export_kwargs)
                print(f"{name} export concluido.")
            except Exception as e:
                logger.warning(f"{name} export falhou: {e}")

    # ========================================================================
    # PHASE 9: Report
    # ========================================================================
    duration = time.time() - start_time
    train_metrics = {}
    if trainer and hasattr(trainer, "metrics"):
        tm = trainer.metrics
        train_metrics = {
            "best_fitness": round(float(getattr(tm, "fitness", 0.0)), 4),
        }

    report_path = generate_training_report(
        run_id=run_id or "unknown",
        args=args,
        schema=args.schema,
        classes=classes,
        dataset_report=dataset_report,
        pseudo_stats=pseudo_stats,
        tune_result=tune_result,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        model_path=dest or best_path,
        duration_sec=duration,
        registry_version=registry_version,
    )

    # Log to MLflow
    log_metrics_mlflow(val_metrics)
    end_tracking()

    # ========================================================================
    # DONE
    # ========================================================================
    print()
    print("=" * 60)
    print("TUDO CONCLUIDO")
    print("=" * 60)
    print(f"Modelo final:    {dest}")
    print(f"Relatorio:       {report_path}")
    print(f"Duracao total:   {duration:.1f}s")
    if registry_version:
        print(f"Registrado como: {registry_version}")
    print("\nReinicie o bot para carregar o novo modelo.")


if __name__ == "__main__":
    main()