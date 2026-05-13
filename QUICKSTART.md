# Brawl Stars Bot - Quick Start Guide

## Prerequisites

- Windows 10/11
- Python 3.8+ (tested on 3.14)
- BlueStacks 5 with ADB enabled
- NVIDIA GPU (recommended for YOLO)

## Installation

```bash
# Install dependencies
pip install ultralytics opencv-python numpy

# Verify installation
python main_enhanced.py check
```

## Configuration

### 1. BlueStacks Setup

1. Enable ADB: Settings → Advanced → ADB Debugging → ON
2. Note the ADB port (usually 5555)
3. Keep BlueStacks at 1920x1080 resolution

### 2. Create Directory Structure

```bash
mkdir -p models dataset captures logs
```

### 3. Get YOLO Model

**Option A: Download Pre-trained (if available)**
```bash
# Place model in ./models/
# Should be named brawl_stars_yolov8s.pt
```

**Option B: Train Your Own**
```bash
# 1. Capture gameplay
python dataset_pipeline.py --capture --duration 300

# 2. Auto-label frames
python main_enhanced.py auto-label ./captures -o ./dataset/labels

# 3. Review and correct labels (manual step)
# Use labelImg or similar tool

# 4. Train model
python main_enhanced.py train -d ./dataset -e 100
```

## Running the Bot

### Basic Usage

```bash
# Start bot with auto-learning
python main_enhanced.py run -m ./models --auto-learn

# Or with custom settings
python main_enhanced.py run \
    -m ./models \
    -d ./dataset \
    -c 0.6 \
    --max-apm 150
```

### Command Options

```
-m, --models-dir      Path to YOLO models (default: ./models)
-d, --dataset-dir     Path for auto-learning data (default: ./dataset)
-c, --confidence      Detection threshold 0.0-1.0 (default: 0.5)
--auto-learn          Enable continuous learning (default: True)
--max-apm             Max actions per minute (default: 180)
-v, --verbose         Enable debug logging
```

## Monitoring

### Real-time Status

```bash
# In another terminal
python main_enhanced.py status
```

### Logs

```bash
# View live logs
tail -f bot.log

# View performance metrics
cat logs/metrics_*.json
```

## Training Pipeline

### Auto-Labeling

```bash
# Label captured frames
python main_enhanced.py auto-label \
    ./captures \
    -o ./dataset/train/labels \
    -f yolo

# Statistics will show labeling quality
```

### Manual Review

After auto-labeling, review and correct:
1. Open images in labelImg
2. Fix incorrect boxes
3. Add missing labels
4. Remove false positives

### Training

```bash
# Start training
python main_enhanced.py train \
    -d ./dataset \
    -e 100 \
    -b 16 \
    -s s

# Options:
# -e, --epochs      Number of training epochs
# -b, --batch-size  Batch size (depends on VRAM)
# -s, --model-size  n/s/m/l/x (nano to extra large)
```

### Validation

```bash
# After training, validate new model
python main_enhanced.py check
```

## Safety & Anti-Detection

### APM Limits

```bash
# Conservative (safer)
python main_enhanced.py run --max-apm 120

# Aggressive (riskier)
python main_enhanced.py run --max-apm 200
```

### Humanization Settings

Edit `core/orchestrator.py`:
```python
reaction_delay_min = 0.08  # 80ms minimum
reaction_delay_max = 0.22  # 220ms maximum
```

### Miss-Click Rate

Edit `humanization.py`:
```python
MISTAKE_RATE = 0.04  # 4% miss-clicks
```

## Troubleshooting

### Bot Not Detecting Anything

1. Check model loaded: `python main_enhanced.py check`
2. Verify BlueStacks visible: Screenshot test
3. Adjust confidence: `-c 0.3` for more detections
4. Ensure proper resolution: 1920x1080

### Bot Detected / Banned

1. Lower APM: `--max-apm 100`
2. Add more delays: Edit reaction times
3. Increase random pauses
4. Use miss-clicks: Set MISTAKE_RATE = 0.05

### Poor Performance

1. Use smaller model: `-s n` (nano)
2. Enable frame skip: Edit `frame_skip = 2`
3. Reduce inference resolution
4. Close unnecessary programs

### Training Not Starting

1. Check dataset format: Should be YOLO format
2. Verify data.yaml exists
3. Ensure enough images: Minimum 100
4. Check GPU available: `nvidia-smi`

## Advanced Configuration

### Custom State Machine

```python
from decision.state_machine import BotState, StateContext

def my_custom_handler(context: StateContext):
    # Custom behavior
    pass

orchestrator.state_machine.register_handler(
    BotState.ENGAGE,
    my_custom_handler
)
```

### Custom Scoring

```python
from decision.scorer import TargetScorer

class MyScorer(TargetScorer):
    def score_target(self, target, ...):
        # Custom scoring logic
        return my_score
```

### Performance Callbacks

```python
def on_state_change(old, new):
    print(f"Changed from {old} to {new}")

def on_action(action_type, data):
    print(f"Executed {action_type}")

orchestrator.on_state_change = on_state_change
orchestrator.on_action = on_action
```

## Best Practices

1. **Start Conservative**: Use low APM, high delays
2. **Monitor Logs**: Check performance regularly
3. **Review Labels**: Quality data = quality model
4. **Test Small**: Validate on 1-2 matches before long sessions
5. **Keep Updated**: Retrain when meta changes

## Getting Help

- Check logs: `bot.log` and `logs/`
- Run validation: `python main_enhanced.py check`
- Review ARCHITECTURE.md for detailed docs
- Check GitHub issues (if applicable)

## Next Steps

1. Capture gameplay footage
2. Build dataset (500+ images)
3. Train custom model
4. Run bot with monitoring
5. Review performance, iterate
