"""
train_yolo.py

YOLOv8 training script for Brawl Stars object detection.

Prerequisites:
    pip install ultralytics
    Dataset must be prepared by dataset_pipeline.py

Usage:
    python -m backend.brawl_bot.train_yolo --epochs 100 --imgsz 640 --batch 16

After training, the best model is copied to models/ and registered.
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "backend" / "brawl_bot" / "models"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"


def train(
    data_yaml: Path = DATASET_YAML,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    device: str = "cpu",
    project: str = str(PROJECT_ROOT / "runs"),
    name: str = "brawlstars",
) -> Path:
    """
    Train YOLOv8 on Brawl Stars dataset.
    Returns path to best.pt model file.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            f"Run: python -m backend.brawl_bot.dataset_pipeline --adb-id <id> --duration 300"
        )

    logger.info(f"Starting YOLO training: model={model} epochs={epochs} imgsz={imgsz} batch={batch}")
    logger.info(f"Dataset: {data_yaml}")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(" ultralytics not installed. Run: pip install ultralytics")

    # Load base model
    base_model_path = MODELS_DIR / model
    if base_model_path.exists():
        model_path = str(base_model_path)
    else:
        model_path = model  # Will download from Ultralytics hub

    yolo = YOLO(model_path)

    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    # Find best.pt
    train_dir = Path(project) / name
    best_pt = train_dir / "weights" / "best.pt"
    last_pt = train_dir / "weights" / "last.pt"

    if not best_pt.exists():
        raise RuntimeError(f"Training completed but {best_pt} not found")

    logger.info(f"Training complete. Best model: {best_pt}")

    # Copy to models/ with descriptive name
    model_name = f"brawlstars_v1_{datetime.now().strftime('%Y%m%d')}.pt"
    dest = MODELS_DIR / model_name
    shutil.copy2(best_pt, dest)
    logger.info(f"Model copied to {dest}")

    # Update model registry
    from .model_validator import validate_all_models
    validate_all_models(delete_fakes=False)

    return dest


def export_tensorrt(model_path: Path) -> Path:
    """
    Export trained model to TensorRT engine.
    Requires: TensorRT, NVIDIA GPU, ultralytics[export]
    """
    logger.info(f"Exporting {model_path} to TensorRT...")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed")

    yolo = YOLO(str(model_path))
    engine_path = model_path.with_suffix(".engine")

    yolo.export(format="engine", device=0, half=True)
    logger.info(f"TensorRT engine: {engine_path}")
    return engine_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train YOLO for Brawl Stars")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-trt", action="store_true", help="Also export to TensorRT")
    args = parser.parse_args()

    best = train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
    )

    if args.export_trt:
        try:
            export_tensorrt(best)
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            logger.info("Install TensorRT and run manually later")

    print(f"\n✅ Model ready: {best}")
    print("Restart the bot to load the new model.")


if __name__ == "__main__":
    main()
