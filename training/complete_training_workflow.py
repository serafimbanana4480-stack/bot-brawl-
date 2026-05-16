"""
complete_training_workflow.py

Workflow completo para treinar modelo YOLO:
1. Captura screenshots (se emulador disponível)
2. Download Roboflow dataset (opcional)
3. Merge datasets
4. Validação do dataset
5. Treino do modelo
6. Validação final

Usage:
    python training/complete_training_workflow.py --capture --roboflow --epochs 100 --device 0
    python training/complete_training_workflow.py --train-only --epochs 50
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("workflow")

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list, description: str) -> bool:
    """Executa comando e retorna sucesso."""
    import subprocess

    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        return False
    return True


def step_capture(num_frames: int = 500) -> bool:
    """Step 1: Captura screenshots."""
    logger.info("=" * 60)
    logger.info("STEP 1: CAPTURING SCREENSHOTS")
    logger.info("=" * 60)

    script = PROJECT_ROOT / "training" / "enhanced_training_pipeline.py"
    cmd = [
        sys.executable, str(script),
        "--capture-only",
        "--frames", str(num_frames),
        "--interval", "0.3"
    ]

    success = run_command(cmd, "Capture screenshots")
    if success:
        logger.info(f"Captured {num_frames} frames")
    return success


def step_download_roboflow(api_key: str = None) -> bool:
    """Step 2: Download Roboflow dataset."""
    logger.info("=" * 60)
    logger.info("STEP 2: DOWNLOADING ROBOFLOW DATASET")
    logger.info("=" * 60)

    script = PROJECT_ROOT / "training" / "download_roboflow_dataset.py"
    cmd = [sys.executable, str(script), "--output", "dataset/roboflow_raw"]

    if api_key:
        cmd.extend(["--api-key", api_key])

    return run_command(cmd, "Download Roboflow dataset")


def step_merge_datasets(local_dataset: str = "dataset/captured") -> bool:
    """Step 3: Merge datasets."""
    logger.info("=" * 60)
    logger.info("STEP 3: MERGING DATASETS")
    logger.info("=" * 60)

    script = PROJECT_ROOT / "training" / "download_roboflow_dataset.py"
    cmd = [
        sys.executable, str(script),
        "--merge",
        "--local-dataset", local_dataset,
        "--output", "dataset/roboflow_raw"
    ]

    return run_command(cmd, "Merge datasets")


def step_validate() -> bool:
    """Step 4: Validação do dataset."""
    logger.info("=" * 60)
    logger.info("STEP 4: VALIDATING DATASET")
    logger.info("=" * 60)

    merged_dir = PROJECT_ROOT / "dataset" / "merged"
    if not merged_dir.exists():
        logger.error("Merged dataset not found")
        return False

    script = PROJECT_ROOT / "training" / "validate_dataset.py"
    cmd = [sys.executable, str(script), "--dataset", "dataset/merged"]

    return run_command(cmd, "Validate dataset")


def step_train(epochs: int = 50, batch_size: int = 16,
               device: str = "cpu", img_size: int = 640) -> bool:
    """Step 5: Treino do modelo."""
    logger.info("=" * 60)
    logger.info("STEP 5: TRAINING MODEL")
    logger.info("=" * 60)

    script = PROJECT_ROOT / "training" / "enhanced_training_pipeline.py"
    cmd = [
        sys.executable, str(script),
        "--train-only",
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--device", device,
        "--img-size", str(img_size)
    ]

    return run_command(cmd, "Train model")


def step_final_validation() -> bool:
    """Step 6: Validação final do modelo."""
    logger.info("=" * 60)
    logger.info("STEP 6: FINAL VALIDATION")
    logger.info("=" * 60)

    model_path = PROJECT_ROOT / "models" / "brawlstars_yolov8.pt"
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return False

    logger.info(f"Model ready: {model_path}")
    logger.info(f"Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Try to load model and verify classes
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        logger.info(f"Model loaded successfully: {model.names}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def check_emulator() -> bool:
    """Verifica se emulador está disponível."""
    logger.info("Checking for emulator...")

    try:
        from pylaai_real.screenshot_taker import ScreenshotTaker
    except ImportError:
        logger.warning("ScreenshotTaker not available")
        return False

    titles = [
        "BlueStacks App Player",
        "LDPlayer",
        "Nox",
        "MEmu",
    ]

    for title in titles:
        st = ScreenshotTaker(title)
        if st.find_window():
            logger.info(f"Found emulator: {title}")
            return True

    logger.warning("No emulator found")
    return False


def main():
    parser = argparse.ArgumentParser(description="Complete training workflow")
    parser.add_argument("--capture", action="store_true",
                       help="Capture screenshots from emulator")
    parser.add_argument("--roboflow", action="store_true",
                       help="Download Roboflow dataset")
    parser.add_argument("--merge", action="store_true",
                       help="Merge datasets")
    parser.add_argument("--train", action="store_true",
                       help="Train model")
    parser.add_argument("--all", action="store_true",
                       help="Run full pipeline (capture + roboflow + merge + train)")

    parser.add_argument("--frames", type=int, default=500,
                       help="Number of frames to capture")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or 0 for GPU)")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Image size")

    parser.add_argument("--api-key", type=str, default=None,
                       help="Roboflow API key")

    parser.add_argument("--local-dataset", type=str, default="dataset/captured",
                       help="Local dataset directory for merge (default: dataset/captured)")

    parser.add_argument("--skip-capture", action="store_true",
                       help="Skip capture step")
    parser.add_argument("--skip-roboflow", action="store_true",
                       help="Skip Roboflow download")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip validation steps")
    parser.add_argument("--train-only", action="store_true",
                       help="Only run training (skip everything else)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("COMPLETE TRAINING WORKFLOW")
    logger.info("=" * 60)

    start_time = time.time()

    # Train-only mode
    if args.train_only or args.train:
        # GPU verification
        if args.device == "0":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    args.device = "cpu"
            except ImportError:
                logger.warning("PyTorch not available, using CPU")
                args.device = "cpu"
        
        if not step_validate():
            logger.error("Validation failed - cannot train with invalid dataset")
            return
        if not step_train(args.epochs, args.batch, args.device, args.img_size):
            logger.error("Training failed")
            return
        step_final_validation()
        return

    # Full pipeline
    if args.all:
        args.capture = True
        args.roboflow = True
        args.merge = True
        args.train = True

    # Step 1: Capture
    if args.capture and not args.skip_capture:
        if check_emulator():
            step_capture(args.frames)
        else:
            logger.warning("Skipping capture - no emulator found")
    else:
        logger.info("Skipping capture step")

    # Step 2: Roboflow
    if args.roboflow and not args.skip_roboflow:
        step_download_roboflow(args.api_key)
    else:
        logger.info("Skipping Roboflow download")

    # Step 3: Merge
    if args.merge:
        step_merge_datasets(args.local_dataset)
    else:
        logger.info("Skipping merge step")

    # Step 4: Validate
    if not args.skip_validation:
        step_validate()

    # Step 5: Train
    if args.train:
        if not step_train(args.epochs, args.batch, args.device, args.img_size):
            logger.error("Training failed")
            return

    # Step 6: Final validation
    step_final_validation()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"WORKFLOW COMPLETE in {elapsed/60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
