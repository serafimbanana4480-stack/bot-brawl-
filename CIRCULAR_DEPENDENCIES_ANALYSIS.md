# Circular Dependencies Analysis

This document analyzes circular dependencies in the Brawl Stars Bot codebase and provides recommendations for resolution.

## Current Import Structure

### wrapper.py Imports

wrapper.py imports from the following modules:
- `realtime_logs` (optional)
- `core.world_model` (optional)
- `core.pressure_map` (optional)
- `core.cover_system` (optional)
- `core.occupancy_grid` (optional)
- `core.behavioral_profile` (optional)
- `core.central_coordinator` (optional)
- `core.async_pipeline` (optional)
- `core.class_registry` (optional)
- `decision.utility_ai` (optional)
- `decision.sticky_target` (optional)
- `decision.intent_system` (optional)
- `decision.enemy_intention` (optional)
- `decision.meta_awareness` (optional)
- `pylaai_real.rl_engine` (optional)
- `pylaai_real.elo_tracker` (optional)
- `pylaai_real.state_manager` (critical)
- `pylaai_real.play` (critical)
- `pylaai_real.detect` (critical)
- `pylaai_real.movement` (critical)
- `pylaai_real.lobby_automator` (critical)
- `pylaai_real.screenshot_taker` (critical)
- `pylaai_real.unified_state_detector` (critical)
- `emulator_controller` (critical)
- `safety_system` (critical)
- `meta_learning` (optional)
- `world_model_integration` (optional)
- `auto_tuner` (optional)
- `behavioral_profile_system` (optional)
- `auto_calibrator` (optional)
- `pylaai_real.ocr_state_detector` (optional)
- `pylaai_real.debug_visualizer` (optional)

### Modules that Import wrapper.py

- `api.py` (imports `from .wrapper import PylaAIEnhanced`)
- `api/brawl_stars_routes.py` (imports `from wrapper import PylaAIEnhanced`)
- `tests/test_wrapper_diagnostics_status.py` (imports `from backend.brawl_bot.wrapper import PylaAIEnhanced`)

## Potential Circular Dependencies

### 1. wrapper.py ↔ core modules

**Issue**: wrapper.py imports from core modules, but core modules may need to access wrapper state.

**Analysis**: Most core modules are independent and don't import wrapper.py. However, some may have indirect dependencies.

**Resolution**: 
- Ensure core modules never import wrapper.py
- Use dependency injection pattern if core modules need bot state
- Pass required state as parameters instead of importing

### 2. wrapper.py ↔ decision modules

**Issue**: wrapper.py imports decision modules, but decision modules may need to access bot configuration.

**Analysis**: Decision modules should be stateless and accept configuration as parameters.

**Resolution**:
- Make decision modules stateless
- Pass configuration/objects as parameters to decision methods
- Avoid decision modules importing wrapper

### 3. wrapper.py ↔ pylaai_real modules

**Issue**: wrapper.py imports pylaai_real modules, but these modules may have cross-dependencies.

**Analysis**: 
- `pylaai_real/play.py` may import from decision modules
- `pylaai_real/state_manager.py` may import from other pylaai_real modules

**Resolution**:
- Audit pylaai_real modules for cross-imports
- Use lazy imports where possible
- Refactor shared code into separate utility modules

### 4. api.py ↔ wrapper.py

**Issue**: api.py imports wrapper.py, but wrapper.py doesn't import api.py (no circular dependency here).

**Analysis**: This is a proper dependency hierarchy (API depends on bot, not vice versa).

**Resolution**: No action needed - this is correct.

## Recommended Refactoring

### Phase 1: Eliminate Direct wrapper.py Imports in Core/Decision Modules

**Action**: Search for any imports of wrapper.py in core/ or decision/ modules and remove them.

**Example**:
```python
# BAD - core module importing wrapper
from wrapper import PylaAIEnhanced

# GOOD - dependency injection
def some_function(bot_state: BotState):
    # Use bot_state parameter instead of importing wrapper
    pass
```

### Phase 2: Extract Shared Configuration

**Action**: Move configuration access to the new ConfigManager instead of accessing through wrapper.

**Example**:
```python
# BAD - accessing config through wrapper
from wrapper import PylaAIEnhanced
config = PylaAIEnhanced.config

# GOOD - using ConfigManager
from core.config_manager import get_config
config = get_config()
```

### Phase 3: Use Lazy Imports for Optional Modules

**Action**: Move optional module imports inside functions where they're actually used.

**Example**:
```python
# BAD - top-level optional import
try:
    from core.world_model import WorldModel
except ImportError:
    WorldModel = None

# GOOD - lazy import inside function
def get_world_model():
    try:
        from core.world_model import WorldModel
        return WorldModel()
    except ImportError:
        logger.warning("World model not available")
        return None
```

### Phase 4: Create Dependency Injection Container

**Action**: Create a DI container to manage component lifecycles and dependencies.

```python
class BotContainer:
    """Dependency injection container for bot components."""
    
    def __init__(self):
        self._instances = {}
    
    def get(self, component_type: type):
        """Get or create component instance."""
        if component_type not in self._instances:
            self._instances[component_type] = self._create(component_type)
        return self._instances[component_type]
    
    def _create(self, component_type: type):
        """Create component with its dependencies."""
        # Implement factory pattern for component creation
        pass
```

## Implementation Priority

### High Priority (Critical)

1. **Audit all core/ and decision/ modules for wrapper.py imports**
   - Search: `grep -r "from wrapper import" core/ decision/`
   - Remove any found imports
   - Refactor to use dependency injection

2. **Audit pylaai_real modules for cross-imports**
   - Search: `grep -r "from pylaai_real" pylaai_real/`
   - Identify circular imports
   - Extract shared code to utility modules

### Medium Priority (Stability)

3. **Implement lazy imports for optional modules**
   - Move optional imports inside functions
   - Reduce startup time
   - Improve error handling

4. **Extract configuration access to ConfigManager**
   - Replace wrapper.config with get_config()
   - Decouple modules from wrapper

### Low Priority (Enhancement)

5. **Implement dependency injection container**
   - Improve testability
   - Better component lifecycle management
   - Clearer dependency graph

## Verification

After refactoring, verify no circular dependencies exist:

```bash
# Use pydeps to visualize dependencies
pip install pydeps
pydeps wrapper --max-bacon=3 --show-cycles

# Or use circular-dependencies detector
pip install circular-dependency-detector
circular-dependency-detector .
```

## Testing

Add tests to verify no circular dependencies:

```python
def test_no_wrapper_imports_in_core():
    """Verify core modules don't import wrapper."""
    for module_file in Path('core').rglob('*.py'):
        content = module_file.read_text()
        assert 'from wrapper import' not in content
        assert 'import wrapper' not in content

def test_no_wrapper_imports_in_decision():
    """Verify decision modules don't import wrapper."""
    for module_file in Path('decision').rglob('*.py'):
        content = module_file.read_text()
        assert 'from wrapper import' not in content
        assert 'import wrapper' not in content
```

## Success Criteria

- No core/ or decision/ modules import wrapper.py
- No circular dependencies detected by pydeps
- All optional imports are lazy-loaded
- Configuration accessed via ConfigManager, not wrapper
- Dependency graph is acyclic and clear
