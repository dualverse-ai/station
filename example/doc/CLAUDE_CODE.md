# Claude Code Integration for Auto Research Debugging

## Overview

Claude Code is integrated as an automated debugging assistant for failed research submissions in the Station. The system automatically detects certain types of failures (syntax errors, import errors, runtime errors) and launches Claude Code to fix them iteratively.

## Key Design Principles

1. **Isolated Workspaces**: Each debugging session gets its own workspace under `claude_workspaces/eval_{id}/` with controlled access to storage via symlinks
2. **Auto-Fetch Mechanism**: Claude writes fixed code to `submissions/submission_v2.py` which is automatically detected and evaluated by the system
3. **Version Management**: Failed attempts create versioned evaluations (v2, v3, etc.) while preserving the original
4. **Outcome-Based Display**: Agents see their original submission if Claude fails, or the working version if Claude succeeds
5. **Report-Based Completion**: Claude signals completion by writing either `report_success.md` or `report_failed.md`
6. **Zero Impact When Disabled**: Feature flag `CLAUDE_CODE_DEBUG_ENABLED` ensures complete backward compatibility

## Architecture

### Core Component: `ClaudeCodeDebugger` Class

All Claude Code functionality is encapsulated in `station/eval_research/claude_code_debugger.py`:

#### Public API

```python
class ClaudeCodeDebugger:
    """Manages Claude Code debugging sessions for failed evaluations"""
    
    def __init__(self, research_room_path: str, constants_module=None, auto_evaluator_instance=None):
        """Initialize the debugger with paths and configuration."""
        
    def should_debug(self, eval_entry: Dict[str, Any], execution_result: Dict[str, Any]) -> bool:
        """Determine if a failure should be debugged by Claude Code.
        
        Returns False for:
        - Timeouts (code might be correct but slow)
        - Fundamental logic errors (TODO, not implemented, wrong function signature)
        - When CLAUDE_CODE_DEBUG_ENABLED is False
        """
        
    def launch_debug_session(self, eval_id: str) -> threading.Thread:
        """Launch Claude Code debugging in a background thread.
        Returns the thread object for monitoring if needed."""
        
    def get_debug_report(self, eval_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Get the debug report if it exists.
        Returns (status, report_content) where status is 'success', 'failed', or None."""
        
    def check_completions(self):
        """Check for completed sessions and clean up finished threads."""
        
    def has_active_sessions(self) -> bool:
        """Check if any debugging sessions are still running.
        Returns True if there are active threads."""
        
    def load_latest_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest version of an evaluation (e.g., v3 if multiple versions exist)."""
        
    def load_original_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """Load the original (v1) evaluation."""
```

#### Internal Architecture

The debugger works as follows:

1. **Thread-based execution**: Each debugging session runs in a separate daemon thread
2. **Workspace isolation**: Creates `claude_workspaces/eval_{id}/` with:
   - `evaluation.yaml` - Copy of failed evaluation
   - `submissions/` - Where Claude writes fixes (v2, v3, etc.)
   - `scripts/monitor_evaluation.py` - Hardcoded with eval_id
   - `storage/` - Symlinks to shared (RO), system (RO), lineage (RW)
   - `tmp/` - Scratch space

3. **Execution flow**:
   - Runs `claude` CLI with JSON output format and restricted tools
   - Working directory set to workspace
   - Timeout enforced (default 30 minutes)
   - Claude writes `report_success.md` or `report_failed.md` to signal completion

4. **Key implementation details**:
   - Uses `subprocess.run()` which blocks the thread until completion
   - Rate limiting via semaphore (matches research evaluation concurrency)
   - Path traversal protection in workspace cleanup
   - Lineage names normalized to lowercase
   - Monitor script timeout increased to 60 seconds

### Integration Points

#### 1. Auto Evaluator (`auto_evaluator.py`)
- Initializes `claude_debugger` when `CLAUDE_CODE_DEBUG_ENABLED` is true
- Checks `should_debug()` for failed evaluations
- Launches debugging sessions instead of notifying agents for fixable errors
- Runs `_check_claude_submissions()` to process Claude's fixes
- Skips notifications for Claude Code resubmissions (version != 'v1')

#### 2. Station (`station.py`)
- `has_pending_claude_code_sessions()` method checks if debugging is active
- Returns false immediately when feature is disabled
- Used by orchestrator to determine waiting state

#### 3. Station Runner (`station_runner.py`)
- Adds Claude Code sessions to automatic wait conditions
- Orchestrator waits while debugging sessions are active
- Auto-resumes when all sessions complete

#### 4. Research Counter (`research_counter.py`)
- Filters evaluation display based on debug reports (when enabled)
- Shows latest version if `report_{id}_success.md` exists
- Shows original if `report_{id}_failed.md` exists
- Shows all evaluations as-is when feature is disabled

## File Structure

```
station_data/rooms/research/
├── evaluations/                    # Main evaluations directory
│   ├── evaluation_321.yaml         # Original failed evaluation
│   ├── evaluation_321_v2.yaml      # Claude's attempts (created by auto evaluator)
│   ├── report_321_success.md       # Claude's success report (if fixed)
│   ├── report_321_failed.md        # Claude's failure report (if gave up)
│   └── ...
├── claude_workspaces/              # All Claude operations
│   ├── history/                    # Saved debugging sessions
│   │   └── eval_321_claude_321_timestamp/  # Session history with outputs
│   └── eval_321/                   # Active workspace per evaluation
│       ├── evaluation.yaml         # Copy of original evaluation
│       ├── submissions/            # Claude writes fixed code here
│       │   └── submission_v2.py    # Auto-fetched by system
│       ├── tmp/                    # Claude's scratch space
│       ├── scripts/                # Contains monitor script
│       │   └── monitor_evaluation.py
│       ├── storage/                # Symlinks to storage
│       │   ├── shared/             # → ../../storage/shared/ (read-only)
│       │   ├── system/             # → ../../storage/system/ (read-only)
│       │   └── lineage/            # → ../../storage/lineages/{author}/ (read-write)
│       ├── report_success.md       # Claude writes this if successful
│       └── report_failed.md        # Claude writes this if giving up
└── pending_evaluations.yamll       # Queue for evaluations
```

## Configuration

```python
# Add to constants.py
CLAUDE_CODE_DEBUG_ENABLED = False  # Set to true to enable Claude Code debugging
CLAUDE_CODE_DEBUG_TIMEOUT = 1800  # 30 minutes
CLAUDE_CODE_DEBUG_MAX_ATTEMPTS = 5
CLAUDE_CODE_DEBUG_MAX_CONCURRENT = RESEARCH_EVAL_MAX_PARALLEL_WORKERS  # Match research concurrency
CLAUDE_CODE_USE_STANDARD_AUTH = True  # Use Claude.ai auth instead of API key
```

## Security Features

- Claude Code is restricted from writing to `storage/system/` and `storage/shared/` using `--disallowedTools` flag
- Can still read from system/shared storage to understand the environment
- Enforced at the SDK level, not just through prompts
- Isolated workspaces prevent cross-contamination between evaluations

## Authentication Options

- When `CLAUDE_CODE_USE_STANDARD_AUTH = true`, removes `ANTHROPIC_API_KEY` from environment
- Forces Claude Code to use standard Claude.ai authentication (subscription-based)
- Often more cost-effective than API-based billing for heavy usage

## Notification Flow

1. **Original submission fails + Claude Code WILL debug** → NO notification sent
2. **Original submission fails + Claude Code WON'T debug** (disabled/timeout) → Normal failure notification
3. **Claude Code resubmissions** (v2, v3, etc.) → NEVER trigger notifications
4. **Claude Code completes** → ONE comprehensive notification with:
   - Final evaluation result (success or still failed after attempts)
   - Complete debug report
   - Clear indication this was auto-debugged
5. **Key principle**: Agent receives exactly ONE notification per submission

### Version Selection Principles

When Claude Code completes debugging, the system uses three principles to determine which code version to show the agent:

**Principle 1: Show Claude-fixed version if `report_success.md` exists AND any of:**
- Debugged version has a valid numeric score (score ≠ "n.a.")
- Debugged version timed out (code might be slow but functionally correct)
- Debugged version is test-mode success:
  - Error message: `"Evaluation failed: Test mode - no scoring"`
  - Logs contain no Python traceback (code executed without runtime errors)

**Principle 2: Show original version with Claude's failure report if:**
- `report_failed.md` exists (Claude tried but couldn't fix it)

**Principle 3: Show original version without Claude report if:**
- No Claude session exists (feature disabled or submission not eligible for debugging)

#### Test-Mode Success Detection

Test-mode submissions (with `test()` function and "test-only" tag) are analysis scripts that don't produce numeric scores. They are considered successful when:
- Error message is exactly: `"Evaluation failed: Test mode - no scoring"`
- Execution logs contain no Python tracebacks
- This allows Claude Code to fix test-mode analysis scripts and show the working version to agents

**Implementation:** Located in `station/eval_research/evaluation_manager.py:_generate_notification_message()`

## Backward Compatibility

When `CLAUDE_CODE_DEBUG_ENABLED = false`:
- No ClaudeCodeDebugger instance is created
- All notifications work exactly as they do today
- No additional waiting states are introduced
- No submission file processing occurs
- Research Counter shows all evaluations without filtering
- System behaves identically to current implementation