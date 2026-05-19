# Error Handling Policy

This document defines the comprehensive error handling policy for the Brawl Stars Bot project to ensure consistent, predictable, and maintainable error handling throughout the codebase.

## Principles

### 1. Fail Fast on Critical Errors
Critical dependencies and configuration issues should fail immediately with clear error messages rather than silently degrading functionality.

### 2. Graceful Degradation for Optional Features
Optional features (e.g., advanced AI modules, visualization) should fail gracefully and log warnings without stopping the bot.

### 3. Consistent Exception Types
Use specific exception types for different error categories to enable precise error handling.

### 4. Never Swallow Exceptions Silently
All exceptions must be logged with context before being handled or re-raised.

### 5. Provide Actionable Error Messages
Error messages must include:
- What went wrong
- Where it went wrong (file, function, line)
- Why it went wrong (context)
- How to fix it (actionable guidance)

## Exception Hierarchy

```python
# Base exception for all bot-specific errors
class BotError(Exception):
    """Base exception for all Brawl Stars Bot errors."""
    pass

# Critical errors that should stop the bot
class CriticalBotError(BotError):
    """Critical error that requires bot shutdown."""
    pass

# Configuration errors
class ConfigurationError(BotError):
    """Configuration-related error."""
    pass

# Vision/detection errors
class VisionError(BotError):
    """Vision/detection system error."""
    pass

# Emulator/ADB errors
class EmulatorError(BotError):
    """Emulator or ADB communication error."""
    pass

# Model errors
class ModelError(BotError):
    """Model loading or inference error."""
    pass

# State detection errors
class StateDetectionError(BotError):
    """Game state detection error."""
    pass
```

## Error Handling Patterns

### Pattern 1: Critical Dependency Loading

```python
# BAD - Silent failure
try:
    import torch
except ImportError:
    torch = None  # Silent failure

# GOOD - Fail fast with clear message
try:
    import torch
except ImportError as e:
    raise CriticalBotError(
        "PyTorch is required for model inference. "
        "Install with: pip install torch"
    ) from e
```

### Pattern 2: Optional Feature Loading

```python
# BAD - No logging
try:
    from core.world_model import WorldModel
    world_model = WorldModel()
except ImportError:
    world_model = None

# GOOD - Graceful degradation with logging
try:
    from core.world_model import WorldModel
    world_model = WorldModel()
    logger.info("World model initialized successfully")
except ImportError as e:
    world_model = None
    logger.warning(
        "World model not available (optional feature disabled). "
        "Install with: pip install <dependencies>. Error: %s",
        e
    )
```

### Pattern 3: Operation Retry

```python
# BAD - Infinite retry loop
while True:
    try:
        result = risky_operation()
        break
    except Exception:
        time.sleep(1)

# GOOD - Bounded retry with backoff
MAX_RETRIES = 3
RETRY_DELAY = 1.0

for attempt in range(MAX_RETRIES):
    try:
        result = risky_operation()
        break
    except (NetworkError, TimeoutError) as e:
        if attempt == MAX_RETRIES - 1:
            raise BotError(
                f"Operation failed after {MAX_RETRIES} attempts"
            ) from e
        logger.warning(
            "Attempt %d/%d failed: %s. Retrying in %.1fs...",
            attempt + 1, MAX_RETRIES, e, RETRY_DELAY
        )
        time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
```

### Pattern 4: Context-Rich Logging

```python
# BAD - Generic error message
except Exception as e:
    logger.error(f"Error: {e}")

# GOOD - Context-rich error with traceback
except Exception as e:
    logger.error(
        "Failed to process detection frame",
        extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "component": "vision_pipeline",
            "frame_number": frame_id,
            "detection_count": len(detections)
        },
        exc_info=True  # Include full traceback
    )
```

## Component-Specific Guidelines

### Vision System

**Critical Errors (Fail Fast):**
- YOLO model file not found
- Screenshot capture failure (after retries)
- OpenCV not installed

**Non-Critical Errors (Graceful Degradation):**
- Tracking initialization failure → Use detection-only mode
- Feature extraction failure → Use basic detection
- Async pipeline unavailable → Use synchronous mode

### Emulator Control

**Critical Errors (Fail Fast):**
- ADB not found or not executable
- No compatible emulator window found
- ADB connection timeout (after retries)

**Non-Critical Errors (Graceful Degradation):**
- Input method unavailable → Log warning, continue
- Screenshot method unavailable → Try fallback method

### State Detection

**Critical Errors (Fail Fast):**
- All state detection methods failed
- Unknown state timeout exceeded

**Non-Critical Errors (Graceful Degradation):**
- OCR unavailable → Use pixel heuristics
- Template matching failed → Try alternative method

### Training Pipeline

**Critical Errors (Fail Fast):**
- Dataset directory not found
- Insufficient training data
- GPU memory error (if GPU required)

**Non-Critical Errors (Graceful Degradation):**
- Data augmentation unavailable → Train without augmentation
- Hyperparameter tuning unavailable → Use default hyperparameters

## Error Recovery Levels

The project already implements a Circuit Breaker pattern in `core/error_recovery.py`. This policy defines when to use each recovery level:

### Level 1: Retry (3 attempts, 100ms delay)
- Transient network errors
- Temporary ADB timeouts
- Occasional screenshot failures

### Level 2: Fallback to Alternative Method
- Primary detection method failed → Try alternative
- Primary screenshot method failed → Try fallback
- OCR failed → Use pixel heuristics

### Level 3: Degrade to Basic Mode
- Advanced features unavailable → Use basic mode
- Async pipeline failed → Use synchronous
- Tracking failed → Use detection-only

### Level 4: Skip Frame with Warning
- Single frame processing error
- Non-critical feature failure
- Temporary resource exhaustion

### Level 5: Emergency Stop
- Critical dependency missing
- Repeated failures (circuit breaker open)
- Safety system triggered

## Logging Standards

### Log Levels

- **DEBUG**: Detailed diagnostic information (development only)
- **INFO**: Normal operational events
- **WARNING**: Non-critical issues that don't stop operation
- **ERROR**: Errors that require attention but don't stop the bot
- **CRITICAL**: Errors that require immediate bot shutdown

### Log Format

All log messages should follow this pattern:
```
[COMPONENT] Action/Event: Details (Context)
```

Examples:
```python
logger.info("[VISION] Model loaded: yolov8n.pt (inference ready)")
logger.warning("[ADB] Connection timeout, retrying... (attempt 2/3)")
logger.error("[STATE] Unknown state timeout exceeded (60s), forcing reset")
logger.critical("[EMULATOR] No compatible window found, cannot start")
```

### Structured Logging

For critical operations, use structured logging with extra context:

```python
logger.error(
    "[DETECTION] Frame processing failed",
    extra={
        "frame_id": frame_id,
        "error_type": type(e).__name__,
        "detection_count": len(detections),
        "processing_time_ms": processing_time
    },
    exc_info=True
)
```

## Implementation Checklist

### Immediate Actions

1. **Add custom exception hierarchy** to `core/exceptions.py`
2. **Update wrapper.py** to use custom exceptions
3. **Add context-rich logging** to all critical operations
4. **Update try/except blocks** in wrapper.py to follow policy
5. **Add startup dependency check** with fail-fast for critical deps

### Short-term Actions

6. **Add error metrics** to dashboard (error rate by type)
7. **Add error recovery dashboard** showing circuit breaker state
8. **Implement error alerting** (webhook on critical errors)
9. **Add error log aggregation** for analysis

### Long-term Actions

10. **Add distributed tracing** for error root cause analysis
11. **Implement automated error categorization** using ML
12. **Add error trend analysis** for proactive detection
13. **Create error runbooks** for common error scenarios

## Migration Guide

### Step 1: Add Exception Hierarchy

Create `core/exceptions.py` with the custom exception hierarchy defined above.

### Step 2: Update Imports

Replace generic exception imports:
```python
# Before
from core.exceptions import BotError, CriticalBotError, ConfigurationError, VisionError, EmulatorError, ModelError, StateDetectionError
```

### Step 3: Update Error Handling

Replace silent failures with policy-compliant handling:
```python
# Before
try:
    from optional_module import Feature
except ImportError:
    feature = None

# After
try:
    from optional_module import Feature
    feature = Feature()
    logger.info("[OPTIONAL] Feature initialized successfully")
except ImportError as e:
    feature = None
    logger.warning(
        "[OPTIONAL] Feature not available (graceful degradation). "
        "Install with: pip install <package>. Error: %s",
        e
    )
```

### Step 4: Add Context to Logs

Replace generic error messages with context-rich ones:
```python
# Before
except Exception as e:
    logger.error(f"Error: {e}")

# After
except Exception as e:
    logger.error(
        "[COMPONENT] Operation failed",
        extra={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "context": "relevant_context_here"
        },
        exc_info=True
    )
```

## Testing Error Handling

Add tests to verify error handling policy compliance:

```python
def test_critical_dependency_fail_fast():
    """Test that critical dependencies fail fast with clear message."""
    with pytest.raises(CriticalBotError) as exc_info:
        initialize_bot_without_critical_dep()
    assert "PyTorch is required" in str(exc_info.value)

def test_optional_feature_graceful_degradation():
    """Test that optional features degrade gracefully."""
    bot = initialize_bot_without_optional_feature()
    assert bot.optional_feature is None
    assert "not available" in caplog.text

def test_error_logging_context():
    """Test that errors are logged with full context."""
    with caplog.at_level(logging.ERROR):
        trigger_error()
    assert "error_type" in caplog.records[0].extra
    assert caplog.records[0].exc_info is not None
```

## References

- Existing error recovery system: `core/error_recovery.py`
- Circuit breaker pattern implementation
- Safety system: `safety_system.py`
