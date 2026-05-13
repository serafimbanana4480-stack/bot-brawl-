#!/usr/bin/env python3
"""
Enhanced main entry point for Brawl Stars Bot.
Integrates all modules: vision, decision, control, training.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import create_bot_orchestrator
from core.orchestrator import BrawlStarsOrchestrator, BotConfig
from vision.vision_engine import YOLOv8VisionEngine
from training.auto_labeler import auto_label_dataset
from end_to_end_test import run_e2e_test


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bot.log', encoding='utf-8')
        ]
    )


def cmd_check(args):
    """Run system checks and validation."""
    print("🔍 Running system validation...")
    
    # Run end-to-end test
    success = run_e2e_test()
    
    if success:
        print("✅ System validation passed")
        return 0
    else:
        print("❌ System validation failed")
        return 1


def cmd_auto_label(args):
    """Auto-label dataset."""
    print(f"🏷️  Auto-labeling dataset in {args.images_dir}...")
    
    stats = auto_label_dataset(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        templates_dir=args.templates_dir,
        format=args.format
    )
    
    print(f"\n📊 Labeling Statistics:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Labeled images: {stats['labeled_images']}")
    print(f"   Total labels: {stats['total_labels']}")
    print(f"   Labels by class:")
    for class_name, count in stats['labels_by_class'].items():
        print(f"      - {class_name}: {count}")
    
    return 0


def cmd_train(args):
    """Run training pipeline."""
    print(f"🚀 Starting training...")
    
    # Import training module
    from training import trainer
    
    # Run training
    result = trainer.train_yolo(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_size=args.model_size
    )
    
    if result:
        print(f"✅ Training complete: {result}")
        return 0
    else:
        print("❌ Training failed")
        return 1


def cmd_run(args):
    """Run the bot."""
    print("🎮 Starting Brawl Stars Bot...")
    
    # Create orchestrator
    orchestrator = create_bot_orchestrator(
        models_dir=args.models_dir,
        dataset_dir=args.dataset_dir,
        confidence_threshold=args.confidence,
        enable_auto_learning=args.auto_learn,
        max_apm=args.max_apm
    )
    
    # Initialize
    if not orchestrator.initialize():
        print("❌ Failed to initialize bot")
        return 1
    
    # Setup callbacks
    def on_state_change(old_state, new_state):
        print(f"🔄 State: {old_state.name} → {new_state.name}")
    
    def on_action(action_type, data):
        if args.verbose:
            print(f"⚡ Action: {action_type}")
    
    orchestrator.on_state_change = on_state_change
    orchestrator.on_action = on_action
    
    # Start bot
    try:
        orchestrator.start()
        
        print("✅ Bot is running. Press Ctrl+C to stop.")
        print(f"   Models: {args.models_dir}")
        print(f"   Auto-learning: {'enabled' if args.auto_learn else 'disabled'}")
        print(f"   Max APM: {args.max_apm}")
        
        # Keep running until interrupted
        import time
        while orchestrator.is_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping bot...")
    finally:
        orchestrator.stop()
    
    print("👋 Bot stopped")
    return 0


def cmd_status(args):
    """Show bot status."""
    # This would connect to a running bot instance
    print("📊 Bot Status")
    print("   Not connected to running instance")
    print("   Use 'main.py run' to start the bot")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Brawl Stars Bot - Enhanced Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check                    # Validate system
  %(prog)s run                      # Start bot with default settings
  %(prog)s auto-label ./captures    # Auto-label captured frames
  %(prog)s train -d ./dataset       # Train YOLO model
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser(
        'check',
        help='Run system validation'
    )
    check_parser.set_defaults(func=cmd_check)
    
    # Auto-label command
    label_parser = subparsers.add_parser(
        'auto-label',
        help='Auto-label dataset using heuristics'
    )
    label_parser.add_argument(
        'images_dir',
        help='Directory with images to label'
    )
    label_parser.add_argument(
        '-o', '--output-dir',
        default='./labels',
        help='Output directory for labels'
    )
    label_parser.add_argument(
        '-t', '--templates-dir',
        default=None,
        help='Directory with template images'
    )
    label_parser.add_argument(
        '-f', '--format',
        choices=['yolo', 'coco'],
        default='yolo',
        help='Output label format'
    )
    label_parser.set_defaults(func=cmd_auto_label)
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train YOLO model'
    )
    train_parser.add_argument(
        '-d', '--dataset',
        required=True,
        help='Path to dataset'
    )
    train_parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )
    train_parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    train_parser.add_argument(
        '-s', '--model-size',
        choices=['n', 's', 'm', 'l', 'x'],
        default='s',
        help='YOLO model size'
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run the bot'
    )
    run_parser.add_argument(
        '-m', '--models-dir',
        default='./models',
        help='Directory with YOLO models'
    )
    run_parser.add_argument(
        '-d', '--dataset-dir',
        default='./dataset',
        help='Directory for auto-learning data'
    )
    run_parser.add_argument(
        '-c', '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    run_parser.add_argument(
        '--auto-learn',
        action='store_true',
        default=True,
        help='Enable auto-learning'
    )
    run_parser.add_argument(
        '--max-apm',
        type=int,
        default=180,
        help='Maximum actions per minute'
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show bot status'
    )
    status_parser.set_defaults(func=cmd_status)
    
    # Parse args
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run command
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
