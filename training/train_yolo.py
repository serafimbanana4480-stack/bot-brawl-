"""
YOLOv8 Training Script for Brawl Stars Bot
Trains object detection model on sample dataset.
"""

from pathlib import Path
import yaml
from ultralytics import YOLO
import torch


def train_yolo_model(
    data_yaml_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    model_name: str = "yolov8n.pt",
    device: str = "0" if torch.cuda.is_available() else "cpu",
    project: str = "models/yolo",
    name: str = "brawlstars_detection",
):
    """
    Train YOLOv8 model on Brawl Stars dataset.

    Args:
        data_yaml_path: Path to data.yaml file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        model_name: Pre-trained model to start from
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
        project: Project directory for saving results
        name: Experiment name
    """
    print("="*60)
    print("YOLOv8 Training Configuration")
    print("="*60)
    print(f"Data: {data_yaml_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Project: {project}")
    print(f"Name: {name}")
    print("="*60)

    # Load a model
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)

    # Training configuration
    training_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'patience': 10,  # Early stopping patience
        'save': True,
        'save_period': 5,  # Save every 5 epochs
        'cache': True,  # Cache dataset for faster training
        'workers': 4,
        'pretrained': True,
        'optimizer': 'Adam',
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Cls loss gain
        'dfl': 1.5,  # DFL loss gain
        'mosaic': 1.0,  # Mosaic augmentation
        'mixup': 0.0,  # Mixup augmentation
        'copy_paste': 0.0,  # Copy-paste augmentation
        'auto_augment': 'randaugment',  # Auto augmentation
        'erasing': 0.4,  # Random erasing
        'crop_fraction': 1.0,  # Fraction of image to crop
    }

    print("\nStarting training...")
    print("="*60)

    # Train the model
    results = model.train(**training_args)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()

    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    # Export the model
    print("\nExporting model...")
    export_path = Path(project) / name / "weights" / "best.pt"
    model.export(format='onnx')  # Export to ONNX for deployment

    print(f"\nModel exported to: {export_path}")
    print(f"ONNX export: {export_path.parent / 'best.onnx'}")

    return results, metrics


if __name__ == "__main__":
    # Configuration
    data_yaml = "dataset/yolo/data.yaml"
    output_dir = "models/yolo"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Train the model
    results, metrics = train_yolo_model(
        data_yaml_path=data_yaml,
        epochs=50,  # Use 50 epochs for quick training
        batch_size=16,
        img_size=640,
        model_name="yolov8n.pt",  # Use nano model for speed
        device="0" if torch.cuda.is_available() else "cpu",
        project="models/yolo",
        name="brawlstars_detection"
    )

    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)
