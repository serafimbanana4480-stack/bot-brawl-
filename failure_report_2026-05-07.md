# Bot Failure Report
**Date:** 2026-05-07 01:06 UTC
**Test Duration:** ~2 minutes
**Mode:** Diagnostic mode enabled with full logging

## Summary
The bot was executed to identify failures and errors. The bot successfully initialized and connected to the emulator, but encountered critical issues preventing progression past the lobby/loading state.

## Critical Issues Found

### 1. Template Matching Failure (CRITICAL)
**Severity:** CRITICAL
**Frequency:** Continuous throughout execution

**Description:**
The StateFinder is consistently failing to match any templates on the screen. All template matching scores are significantly below the threshold:

- `brawler_select.png`: max_val=0.020 (threshold=0.300) - FAILED
- `thumbs_down.png`: max_val=0.000 (threshold=0.300) - FAILED
- `play_button.png`: max_val=0.028 (threshold=0.300) - FAILED
- `joystick.png`: max_val=0.170 (threshold=0.300) - FAILED

**Impact:**
The bot cannot reliably detect game states (lobby, end, brawler_selection, in_game) and must rely entirely on screen automation hints, which may be unreliable.

**Root Cause:**
Possible causes:
- Template images are outdated or don't match current game UI
- Screen resolution mismatch (expected 1920x1080 but actual may differ)
- Game UI changed in recent updates
- Screenshot capture issue (wrong window or region)

**Recommendation:**
1. Verify template images in `backend/brawl_bot/images/` match current game UI
2. Check actual screen resolution of emulator window
3. Capture and compare actual screenshots with templates
4. Lower threshold values temporarily for testing
5. Update templates to match current game version

---

### 2. Lobby/Loading Loop (CRITICAL)
**Severity:** CRITICAL
**Frequency:** Continuous loop

**Description:**
The bot is stuck in an infinite loop between lobby and loading states:
1. Detects lobby → presses play button
2. Detects loading → waits for transition
3. Detects lobby again → presses play button
4. Repeat indefinitely

**Log Pattern:**
```
[STATE] Transição de estado: lobby -> loading
[STATE] Loading detectado - aguardando transição
[STATE] Transição de estado: loading -> lobby
[STATE] No lobby - a pressionar play
```

**Impact:**
Bot cannot progress to matchmaking or in-game state, making it non-functional.

**Root Cause:**
Related to Issue #1 - template matching failure causes state detection to rely on screen automation hints, which may be oscillating between "idle" (lobby) and "loading" states.

**Recommendation:**
Fix Issue #1 first (template matching) to resolve state detection reliability.

---

### 3. Match Controller Conflict (HIGH)
**Severity:** HIGH
**Frequency:** Every time bot tries to start a match

**Description:**
The MatchController repeatedly reports: "Já existe uma partida em andamento ou pendente de finalização!" (Match already in progress or pending completion)

**Impact:**
Prevents the bot from starting new matches, may indicate state management issue.

**Root Cause:**
The bot's state management believes a match is already active when it's not, possibly due to:
- Previous match not properly marked as completed
- State corruption in MatchController
- Race condition between state transitions

**Recommendation:**
1. Add debug logging to MatchController to track match state transitions
2. Implement proper match completion handling in end_game handler
3. Add match state reset functionality
4. Verify match state is cleared after each match ends

---

### 4. Unicode Encoding Error (MEDIUM)
**Severity:** MEDIUM
**Frequency:** Once at startup

**Description:**
```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 61-62: character maps to <undefined>
```

**Impact:**
May cause logging issues with non-ASCII characters (e.g., Portuguese characters in logs).

**Root Cause:**
Windows console encoding issue with UTF-8 characters.

**Recommendation:**
Set console encoding to UTF-8 at startup:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

---

## Successful Components

### ✅ Emulator Connection
- EmulatorController successfully connected to BlueStacks
- ADB connection established on port 5555
- Window detection working (BlueStacks App Player found)
- Screenshot capture functioning

### ✅ Component Initialization
- StateManager initialized successfully
- PlayLogic initialized with combat telemetry
- LobbyAutomator initialized with OCR support
- Wrapper initialized with all components
- SafetySystem and HumanizationEngine loaded

### ✅ Logging Infrastructure
- All new logging prefixes working correctly:
  - `[STATE]` - State transitions
  - `[COMBAT]` - Combat decisions
  - `[ADB]` - ADB commands
  - `[WINDOW]` - Window operations
  - `[LOBBY]` - Lobby automation
  - `[WRAPPER]` - Wrapper lifecycle
  - `[STATE_FINDER]` - State detection

---

## Test Environment

**Configuration:**
- Emulator: BlueStacks
- Resolution: 1920x1080 (expected)
- ADB Port: 5555
- Brawler: colt
- Mode: gem_grab
- Diagnostic Mode: Enabled

**Execution Time:** ~2 minutes
**Total Log Lines:** ~200+
**Errors Found:** 4 (1 critical, 2 high, 1 medium)

---

## Recommended Fixes Priority

1. **IMMEDIATE:** Fix template matching - update templates or lower thresholds
2. **HIGH:** Fix state detection loop - resolve lobby/loading oscillation
3. **HIGH:** Fix MatchController state management - clear match state properly
4. **MEDIUM:** Fix Unicode encoding in logging

---

## Conclusion

The bot's infrastructure is solid (emulator connection, component initialization, logging), but the core state detection system is failing due to template matching issues. This cascades into the lobby/loading loop and prevents the bot from functioning. Once template matching is fixed, the other issues should be easier to resolve.

**Status:** BOT NON-FUNCTIONAL - Cannot progress past lobby state
