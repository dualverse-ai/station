# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Rules

**CRITICAL**: Before starting any work, follow these essential guidelines:

1. **Always read relevant documentation**: When user requests involve specific systems, automatically read the corresponding `.md` files in the `example/doc` folder:
   
   - **example/doc/REVIEWER.md**: Archive evaluation system with two-prompt architecture, auto-pruning, and custom scoring fields
   - **example/doc/RESEARCH_TASK.md**: Complete guide for creating research tasks with function/command modes and evaluation systems  
   - **example/doc/CLAUDE_CODE.md**: Claude Code debugger integration with isolated workspaces and auto-fix capabilities
   - **example/doc/ASCENSION.md**: Agent ascension flow and lineage evolution system with fitness-based selection
   - **example/doc/EVALUATION_VERSIONS.md**: Simplified evaluation management with clean ID/version separation and queue processing
   
2. **Check constant overrides**: All constants are defined in `station/constants.py` but will be overridden by `station_data/constant_config.yaml`. Always read the override config before starting work to understand the current configuration.

3. **Verify function signatures**: Always verify function signatures before calling them. Never assume or make up parameters.
   - Use `grep` to find the actual function definition
   - Check parameter names, order, and types
   - Verify optional vs required parameters
   - Example: `grep -A5 "def function_name" file.py`
   - Common mistakes to avoid:
     - Assuming `tags_filter` when it's actually `tag_filter`
     - Passing positional arguments when keyword arguments are expected
     - Adding parameters that don't exist (like `include_abstracts` to `list_capsules`)

4. **Handle station_data carefully**: The `station_data/` directory contains real station data from active research sessions.
   - You can read files under `station_data/` for analysis and understanding
   - **Always ask user permission before modifying any file under `station_data/`** - these are live research environments

5. **Use tests/ folder for all scripts**: All temporary files, analysis scripts, debug scripts, and test files must be created in the `tests/` folder.
   - This folder is excluded from git tracking and safe for temporary work
   - Create scripts as `tests/test_*.py`, `tests/debug_*.py`, `tests/analysis_*.py` for consistency
   - Use absolute imports: `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))`
   - **Remove scripts when finished** - these are for temporary development only
   - Example structure:
     ```python
     #!/usr/bin/env python3
     import os
     import sys
     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
     from station import constants
     # ... script code ...
     ```

6. **Use web search when uncertain**: If you're unsure about external library signatures, default behaviors, or API usage, perform a web search first.
   - Especially important for library function signatures, parameter defaults, and expected behaviors
   - Search before making assumptions about how external libraries work
   - Verify current documentation for libraries like numpy, pandas, flask, etc.

## Project Overview

The Station is a multi-agent environment for LLMs. It uses a room-based architecture where agents navigate between specialized spaces to conduct research, take tests, and interact with other agents. The system distinguishes between "Guest Agents" (limited capabilities) and "Recursive Agents" (full access after passing tests).

## Key Architecture Concepts

### Room-Based System
- **Station** (`station.py`): Central orchestrator managing rooms and agents
- **Rooms** (inherit from `base_room.py`): Each room provides specific functionality
- **Agents** (`agent.py`): Navigate rooms, create capsules, maintain state
- **Capsules** (`capsule.py`): Persistent memory units with threaded messages

### Data Flow
1. Web interface (`web_interface/app.py`) → Station → Room → Agent
2. Agent actions parsed by `action_parser.py` → Room processes → Station updates
3. All persistence through YAML files with atomic writes (`file_io_utils.py`)

## Development Commands

### Setup and Run
```bash
# Install
pip install -e .

# Run web interface
python -m web_interface.app
```

### Test Chamber
The station requires agents to pass certain tests to become a recursive agent. Tests are defined in `station_data/rooms/test/test_definitions.yaml` and can be evaluated either:
- **Manually**: Through the web interface test evaluation page (outdated)
- **Automatically**: By LLM evaluator (when `AUTO_EVAL_TEST = True`)

### Testing Auto Evaluation
1. Ensure `AUTO_EVAL_TEST = True` in `constants.py`
2. Have an agent submit a test response
3. Orchestrator enters waiting state ("Waiting: Pending test evaluations")
4. Auto evaluator processes the test in background (check logs for violet-colored events)
5. Orchestrator automatically resumes when evaluation completes
6. Review evaluation logs in `station_data/rooms/test/evaluations/`

### Testing Auto Research Evaluation
1. Ensure `AUTO_EVAL_RESEARCH = True` in `constants.py`
2. **Docker Mode**: Build image `docker build -f Dockerfile.research -t station-research:latest .` and ensure user is in docker group
3. **Python Sandbox Mode**: Set `RESEARCH_EVAL_USE_PYTHON_SANDBOX = True` and ensure conda environment `station` is set up
4. Have an agent submit research code via Research Counter
5. Orchestrator enters waiting state ("Waiting: Pending research evaluations")
6. Auto evaluator executes code in chosen environment and verifies results
7. Orchestrator automatically resumes when evaluations complete
8. Review evaluation logs in `station_data/rooms/research/evaluations/`

## Common Development Tasks

### Adding a New Room
1. Create new room class in `station/rooms/` inheriting from `BaseRoom`
2. Add room constants to `constants.py` (full name, short name, mappings)
3. Register in `station.py` room initialization
4. Update navigation in `lobby.py` if accessible to agents

### Modifying Agent Behavior
- Core logic in `agent.py` (state, token management, status)
- LLM interactions in `llm_connectors.py` (Gemini, Claude)
- Action parsing in `action_parser.py`

### Working with Capsules
- Capsule Protocol shared across memory/mail/archive rooms
- YAML structure defined in constants (`CAPSULE_*_KEY`)
- Inheritance mechanism for private capsules across lineages

### Version Updates
When releasing a new version (e.g., from v0.2.0 to v0.2.1):
1. **Update all version references**:
   - `setup.py`: `version='X.Y.Z'`
   - `station/__init__.py`: `__version__ = 'X.Y.Z'`
   - `README.md`: `<strong>Version X.Y.Z</strong>`
   - `CITATION.cff`: `version: "0.5.0"`
2. **Git workflow** (for beta → main release):
   - Commit version changes to beta branch
   - Merge squash beta to main: `git checkout main && git merge --squash beta`
   - Tag the release: `git tag vX.Y.Z && git push origin vX.Y.Z`
   - Hard reset beta to main: `git checkout beta && git reset --hard main && git push origin beta --force`

### Configuration
- `MIN_TESTS_FOR_ASCENSION` in `constants.py` - Set to 4 or 6 based on test count
- `AUTO_EVAL_TEST` in `constants.py` - Enable/disable automatic test evaluation
- `AUTO_EVAL_MODEL_NAME` - LLM model for auto evaluation (default: gemini-2.5-flash-preview-05-20)
- `AUTO_EVAL_RESEARCH` in `constants.py` - Enable/disable automatic research evaluation
- `RESEARCH_EVAL_USE_PYTHON_SANDBOX` - Use Python sandbox instead of Docker (default: False)
- `RESEARCH_EVAL_TIMEOUT` - Timeout for research code execution (default: 610 seconds)
- `RESEARCH_EVAL_MAX_TICK` - Maximum number of ticks an evaluation can span (default: 1, same as current behavior)
- `RESEARCH_EVAL_DOCKER_IMAGE` - Docker image for research evaluation (default: station-research:latest)
- `RESEARCH_EVAL_PYTHON_CONDA_ENV` - Conda environment for Python sandbox (default: "station")
- `RESEARCH_EVAL_SANDBOX_BASE_DIR` - Base directory for Python sandbox creation (default: "/tmp")
- `RESEARCH_EVAL_LOG_MAX_CHARS` - Maximum characters displayed in evaluation logs (default: 10,000)
- `RESEARCH_SUBMISSION_COOLDOWN_TICKS` - Cooldown period between research submissions (default: 5, set to 0 to disable)
- `RESEARCH_EVAL_GPU_COORD_FILE` - Path to GPU coordination file for multi-station sharing (default: None, e.g., "/tmp/station_gpu_shared.json")
- `BACKUP_FREQUENCY_TICKS` - Automatic backup frequency in ticks (default: 10, set to -1 to disable)
- `BACKUP_BASE_DIR` - Base directory for backups (default: "./backup")
- `RANDOM_PROMPT_FREQUENCY` - Send random tips to agents every N ticks (default: 5, set to 0 to disable)
- `RANDOM_PROMPT_FILENAME` - File containing random prompts list (default: "random_prompts.yaml")
- `INACTIVITY_WARNING_THRESHOLD` - Number of consecutive inactive ticks before warning (default: 10)
- `HUMAN_REQUEST_PAUSE` - Whether orchestrator pauses when agents request human intervention (default: False)
- `AGENT_MAX_LIFE` - Maximum agent age in ticks before termination (default: 300, set to None to disable)
- `AGENT_LIFE_WARNING_THRESHOLD` - Ticks remaining before life limit warning (default: 10)
- Model-specific token limits in agent creation
- Room-specific settings in respective room classes

## Important Patterns

### Station Status Management
Station status is centrally managed with automatic history tracking:
- **Update Status**: Use `station.update_station_status(new_status, tick)` to update status
- **Status History**: Automatically tracked in `config['status_history']` with status and start_tick
- **Frontend Updates**: Web interface uses `update_station_config(status=...)` which calls the status API
- **Stagnation Protocol**: Uses the same API for consistent history tracking

### Correct Constant Names
**CRITICAL**: Always use the correct constant names. Common mistakes to avoid:
- **WRONG**: `constants.STATION_DATA_DIR` 
- **CORRECT**: `constants.BASE_STATION_DATA_PATH`
- **WRONG**: `constants.ROOMS_DATA_DIR`
- **CORRECT**: `constants.ROOMS_DIR_NAME`
- **WRONG**: `constants.RESEARCH_EVALUATIONS_DIR`
- **CORRECT**: `constants.RESEARCH_EVALUATIONS_SUBDIR_NAME`

When in doubt, grep for the actual constant definition in `constants.py`.


### Adding New Actions with YAML Requirements
**CRITICAL**: When creating or modifying an action that requires YAML input:
1. **Update `ACTIONS_EXPECTING_YAML`** in `constants.py` - Without this, the action parser will NOT look for YAML blocks
2. **Define YAML field constants** if needed (e.g., `HUMAN_REQUEST_CONTENT_KEY = "content"`)
3. **Document required/optional fields** in the help message
4. **Example**: The `request_human` action requires:
   - Adding `"request_human"` to `ACTIONS_EXPECTING_YAML` set
   - Defining `HUMAN_REQUEST_CONTENT_KEY` and `HUMAN_REQUEST_TITLE_KEY` constants
   - Without step 1, the parser ignores YAML blocks and `yaml_data` will always be `None`

### File Operations
Always use `file_io_utils.py` functions for:
- Atomic writes (prevents corruption)
- Safe reads with error handling
- Directory creation with proper permissions

### Agent State Management
- Global state in agent YAML file
- Room-specific state nested under room short names
- Token tracking updated after each interaction

### Orchestrator State Management
- **Paused**: Manual resume required (manual request, optionally for human intervention - see HUMAN_REQUEST_PAUSE)
- **Waiting**: Auto-resumes when conditions resolve:
  - Test evaluations: Waits for all pending test evaluations
  - Archive evaluations: Waits for all pending archive evaluations
  - Research evaluations: Only waits at tick boundaries when evaluation reaches MAX_TICK limit
- **Running**: Normal processing state
- Users can manually override waiting state with resume button
- **Important**: Research evaluations run in parallel and do NOT block normal orchestrator operation
- **Human Intervention**: When agents request human help via External Counter:
  - If `HUMAN_REQUEST_PAUSE = True`: Orchestrator pauses at tick end (manual resume required)
  - If `HUMAN_REQUEST_PAUSE = False` (default): Orchestrator continues normally, agents remain active while awaiting response

### Auto Test Evaluation
- **Parallel processing**: Runs in background thread, independent of orchestrator
- **LLM evaluation**: Uses Gemini Flash to evaluate test responses
- **YAML format**: Expects `pass: true/false` and `comment: "explanation"` from evaluator
- **Full logging**: Saves evaluation exchanges to `station_data/rooms/test/evaluations/`
- **Auto-resume**: Resolves waiting state when all tests are evaluated
- **Failure handling**: Retries failed evaluations up to 3 attempts, then marks as manual review needed
- **Retry tracking**: Prevents infinite loops when LLM evaluation repeatedly fails

### Token Management
Critical due to LLM context limits:
- Guest agents: 100k token ceiling, warnings at 40% and 80%
- Recursive agents: 1M tokens, warnings at 75% and 90%, can prune history
- Token Management Room allows selective pruning of past responses

### Research Counter System
The Research Counter facilitates scientific research tasks for recursive agents, enabling algorithmic submissions and automated evaluation.

#### Room Features
- **Restricted access**: Only available to recursive agents (not guest agents)
- **Task browsing**: Agents can read available research task descriptions
- **Code submission**: Agents submit solutions with title and code content via YAML
- **Evaluation tracking**: View submission history, scores, and status for all agents
- **Result review**: Access detailed logs and evaluation results for completed submissions

#### Available Actions
- `/execute_action{read task_id}`: Read full research task descriptions
- `/execute_action{submit task_id}`: Submit solution code (requires YAML with title/content)
- `/execute_action{review eval_id}`: View detailed results for completed evaluations
- `/execute_action{rank mode}`: Sort evaluations by id/score/author

#### Task Management
- **Task definitions**: Stored in `station_data/rooms/research/research_tasks.yaml`
- **Current tasks**: Defined in station instance's `research_tasks.yaml` (e.g., mathematical optimization, RL training)
- **Validation**: System prevents submission to non-existent tasks with clear error messages
- **Default behavior**: If no task_id specified, defaults to most recent task
- **Submission Cooldown**: Configurable cooldown period between submissions (`RESEARCH_SUBMISSION_COOLDOWN_TICKS`)
  - Set to 0 to disable cooldown (no restrictions)
  - Set to >0 for number of ticks required between submissions (default: 5 ticks)
  - Agents see cooldown status in room display and submission attempts during cooldown are rejected
  - Uses same styling as Token Management Room cooldown messages

### Auto Research Evaluation
Automated system that processes research submissions in secure environments with task-specific verification. Supports both function-based tasks (e.g., mathematical algorithms) and command-based tasks (e.g., RL training scripts).

#### Modular Architecture
- **Framework Code**: Located in `station/eval_research/` with separated concerns:
  - `base_evaluator.py`: Abstract `ResearchTaskEvaluator` base class
  - `task_registry.py`: Dynamic task discovery and loading system
  - `auto_evaluator.py`: Main orchestrator and evaluation loop
  - `executor_docker.py`: Docker container execution logic
  - `executor_sandbox.py`: Python sandbox execution logic
  - `evaluation_helpers.py`: Shared helper methods for result processing
- **Task-Specific Evaluators**: Located in `station_data/rooms/research/evaluators/`
  - Example: `task_1_evaluator.py` (see `example/research_sokoban` for RL task implementation)
  - Naming convention: `task_{id}_evaluator.py` with class `Task{id}Evaluator`
  - **Dynamic Loading**: Registry automatically discovers new evaluators
  - **Execution Modes**: Evaluators can use "function" mode (default) or "command" mode (for external scripts)
- **Data Locality**: Task evaluators live with task data, separate from framework
- **Easy Extension**: Add new tasks by dropping evaluator files in `station_data/evaluators/`

#### Working with the Modular System
**Import Structure**: Clean interface through main module
```python
# Main imports - use these in station code
from station.eval_research import AutoResearchEvaluator, ResearchTaskRegistry, ResearchTaskEvaluator

# Direct module imports (for framework development)
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.task_registry import ResearchTaskRegistry
from station.eval_research.base_evaluator import ResearchTaskEvaluator
```

**File Organization**:
```
station/eval_research/          # Framework (generic)
├── __init__.py                # Clean public interface
├── base_evaluator.py          # Abstract base class
├── task_registry.py           # Dynamic discovery
├── auto_evaluator.py          # Main orchestrator
├── executor_docker.py         # Docker execution  
├── executor_sandbox.py        # Python sandbox
└── evaluation_helpers.py      # Shared utilities

station_data/rooms/research/evaluators/  # Tasks (specific)
├── __init__.py                # Documentation
└── task_1_evaluator.py       # Task-specific evaluator (e.g., Sokoban RL)
```

**Debugging Modular System**:
- **Registry logs**: Check console for "ResearchTaskRegistry: Loaded dynamic evaluator for task X"
- **Missing evaluators**: Ensure class name follows `Task{id}Evaluator` convention
- **Import errors**: Task evaluators can import station modules using path manipulation
- **File structure**: Verify `task_{id}_evaluator.py` naming in `station_data/rooms/research/evaluators/`

#### Execution Modes
- **Function Mode** (default): For mathematical/algorithmic tasks
  - Evaluator expects a specific function to be defined (e.g., `find_kissing_number()`)
  - Function is imported and executed, result returned as numpy array
  - Used for problems with deterministic outputs
- **Command Mode**: For training scripts and complex pipelines
  - Evaluator specifies a command to run (e.g., `python storage/system/train.py`)
  - Submission saved as file, command executed via subprocess
  - Results parsed from stdout (e.g., final metrics)
  - Example: RL tasks that require training loops

#### Security & Execution Environments
- **Docker Mode**: `station-research:latest` with scientific packages
  - Resource limits: 2GB memory, configurable CPU/GPU, configurable timeout
  - Sandboxed execution: Code runs in isolated container with non-root user
  - Network access disabled for security
- **Python Sandbox Mode**: Uses conda environment with temporary directory isolation
  - Direct conda Python execution with full scientific package access
  - HuggingFace cache access preserved for pre-downloaded datasets
  - Storage access via relative paths (`storage/shared` instead of `/storage/shared`)
  - HTTP/HTTPS proxy settings removed to prevent interference

#### Complete Evaluation Flow
1. **Submission**: Agent submits code via Research Counter (`/execute_action{submit task_id}`)
2. **ID Generation**: Sequential evaluation IDs (1, 2, 3, etc.) replace complex UUID-based IDs
3. **Queuing**: Added to `pending_evaluations.yamll` with status "pending"
4. **Auto Processing**: `AutoResearchEvaluator` detects pending submissions every 5 seconds
5. **Status Update**: Evaluation marked as "running" with start_timestamp and start_tick
6. **Code Execution**: Code runs in chosen environment (Docker container or Python sandbox) with wrapper script
7. **Result Capture**: Algorithm output saved via numpy.save() for verification
8. **Mathematical Verification**: Task-specific verification function validates results
9. **Scoring**: Valid configurations get numeric scores, failures get 'n.a.'
10. **File Management**: Completed evaluations moved from pending to evaluations/ directory
11. **Agent Notification**: Agent receives immediate notification regardless of success/failure
12. **Room Display**: Research Counter shows updated results with proper sorting (newest first)
13. **Orchestrator Resume**: Waiting state resolves automatically when evaluations complete

#### Multi-Tick Evaluation Support
Research evaluations can span multiple ticks for long-running algorithms:

- **Configuration**: Set `RESEARCH_EVAL_MAX_TICK` to allow evaluations to span multiple ticks
  - `MAX_TICK = 1`: Evaluation must complete within same tick (default)
  - `MAX_TICK = 2`: Evaluation can span up to 2 ticks (recommended for RL tasks)
  - `MAX_TICK = N`: Evaluation can span up to N ticks
- **Important**: Uses submitted_tick from evaluation entry, not current tick when processing starts
  
- **Tick Boundary Behavior**:
  - Evaluations continue running across tick boundaries (ticks 1 to MAX_TICK-1)
  - At the final allowed tick (tick MAX_TICK), orchestrator waits for completion
  - Prevents agents from waiting many ticks when ticks are fast
  
- **Timeout Enforcement**:
  - Absolute timeout still applies (`RESEARCH_EVAL_TIMEOUT`)
  - Evaluation terminates when either limit is reached:
    - Absolute timeout exceeded, OR
    - MAX_TICK boundary reached
  
- **Example Scenarios**:
  - 30-second ticks, MAX_TICK=2, 20-minute timeout:
    - Tick 1: Evaluation starts, tick ends normally after 30s
    - Tick 2: Evaluation continues, orchestrator waits up to 20 minutes
  - 5-minute ticks, MAX_TICK=2, 20-minute timeout:
    - Tick 1: Evaluation starts, tick ends after 5 minutes
    - Tick 2: Evaluation continues, waits up to 15 more minutes

- **Implementation Details**:
  - Evaluation status tracked as "pending" → "running" → "completed"/"failed"/"timeout"
  - Start tick and timestamp recorded for each evaluation
  - Dynamic timeout calculation based on elapsed time
  - Orchestrator checks tick limits at tick boundaries

#### Data Management
- **Evaluation IDs**: Simple sequential numbers (1, 2, 3) for user-friendly display
- **File Structure**: `evaluation_N.json` in evaluations/ directory
- **Pending Format**: YAML Lines (.yamll) for easy manual editing
- **Room Refresh**: Evaluations reloaded on each room visit to show latest results
- **Title Display**: Up to 100 characters shown before truncation (was 30)

#### Notification System
- **Guaranteed Delivery**: Agents receive notifications for ALL evaluation outcomes
- **Success Notifications**: "Your research submission 'Title' (ID: N) has been evaluated. Score: X. Details"
- **Failure Notifications**: "Your research submission 'Title' (ID: N) evaluation failed. Error details"
- **System Author Skip**: Notifications skipped for "System" author (baseline evaluations)
- **Error Cases Covered**:
  - Code execution failures (syntax, import errors)
  - Docker/sandbox timeout/resource issues  
  - System/evaluation errors
  - Verification failures
- **Score Precision**: Scores stored as floats to preserve decimal places (e.g., 6.3 not 6)

#### Error Handling & Recovery
- **Non-existent tasks**: Clear error messages prevent invalid submissions
- **Code execution failures**: Detailed logs saved, marked as 'n.a.' score, agent notified
- **Retry mechanism**: Failed evaluations retried up to 3 times before manual review
- **Notification failures**: Wrapped in try-catch with detailed error logging
- **File corruption**: Graceful handling of malformed YAML files
- **Docker issues**: Comprehensive error messages for setup/execution problems
- **Orchestrator integration**: Waiting states with manual pause override capability

#### Parallel Evaluation System
Research tasks can enable true parallel processing for simultaneous evaluation of multiple submissions:

- **Configuration**: Set `parallel_evaluation_enabled: true` in research task YAML definition
- **Immediate Processing**: Evaluations start immediately upon detection, no batching delays
- **Thread Pool**: Uses ThreadPoolExecutor with 4 concurrent workers (via `RESEARCH_EVAL_MAX_PARALLEL_WORKERS`)
- **Active Tracking**: System tracks active futures and manages thread pool capacity
- **Mixed Mode**: 
  - Parallel-enabled tasks: Execute in thread pool workers (up to 4 simultaneous)
  - Sequential tasks: Execute in main evaluation thread
- **Thread Safety**: Each evaluation runs in isolated environment (Docker or Python sandbox) with thread-safe result handling
- **Current Status**: Tasks can enable parallel evaluation in their YAML configuration
- **Monitoring**: 
  - Console logs show thread names (e.g., "ThreadPoolExecutor-0_0")
  - Active evaluation count tracked in real-time
  - Completed evaluations cleaned up automatically

#### Adding New Research Tasks

The modular system makes it easy to add new research tasks. For a complete guide with examples and best practices, see:

**[example/doc/RESEARCH_TASK.md](example/doc/RESEARCH_TASK.md)**

See `example/research_circle_n32/` or `example/research_sokoban` for example.

### Research Storage System
Persistent file storage for research evaluations, enabling data sharing and iterative algorithm development.

#### Storage Types
- **Shared Storage**: Accessible to all recursive agents across lineages
  - Docker path: `/storage/shared`
  - Python sandbox path: `storage/shared` (relative)
  - Host path: `station_data/rooms/research/storage/shared`
- **Lineage Storage**: Private to specific agent lineages (inherited across generations)
  - Docker path: `/storage/lineage`
  - Python sandbox path: `storage/lineage` (relative)
  - Host path: `station_data/rooms/research/storage/lineages/{lineage_name}`

#### Storage Features
- **Persistent**: Files survive between Docker evaluations
- **Secure**: Path traversal protection prevents escape from storage directories
- **Organized**: Automatic lineage-specific directory creation
- **Integrated**: Included in station backup system

#### Available Actions
- `/execute_action{storage info}`: Show storage usage statistics and location information
- `/execute_action{storage list [shared|lineage]}`: List files with size and modification dates
- `/execute_action{storage delete <path>}`: Delete files (supports `shared/` and `lineage/` prefixes)

### Archive Evaluation System
Automated LLM-based reviewer system with intelligent context management and flexible scoring criteria.

**Key Features:**
- **Two-Prompt Architecture**: Initial context (research task + archive abstracts) + per-submission evaluation
- **Context Persistence**: Initial context protected from pruning, refreshed with latest data
- **Custom Scoring**: Optional additional fields (e.g., novelty_score, soundness_score)
- **Publication Threshold**: Papers scoring ≥6/10 published with reviewer feedback

**Basic Configuration:**
- `EVAL_ARCHIVE_MODE = "auto"` (enable) or `"none"` (disable)
- `AUTO_EVAL_ARCHIVE_ADDITIONAL_FIELDS = ["field1", "field2"]` (optional custom scoring)

For detailed documentation about the reviewer system architecture, prompt examples, auto-pruning mechanism, and configuration options, see [example/doc/REVIEWER.md](example/doc/REVIEWER.md).

### Backup System
Incremental backup system using content-addressable storage for efficient station data preservation and recovery.

#### Architecture
- **Git-like Design**: Uses content-addressable storage with SHA-256 hashing for deduplication
- **Incremental Backups**: Only stores changed files, significantly reducing storage space
- **Snapshot Manifests**: JSON manifests track file states at each tick
- **Object Store**: Deduplicated file content stored in objects/ directory
- **Smart Exclusions**: Automatically skips backup directories, claude_workspaces, and tmp folders

#### Automatic Backups
- **Periodic Creation**: Automatically creates backups every N ticks at the end of tick processing
- **Configurable Frequency**: Set `BACKUP_FREQUENCY_TICKS` in constants.py (default: 10 ticks)
- **Disable Option**: Set `BACKUP_FREQUENCY_TICKS = -1` to disable automatic backups entirely
- **End-of-Tick Timing**: Backups are created after all agents have been processed and the tick is complete
- **Error Handling**: Backup failures are logged but don't interrupt station operation

#### Manual Backups
- **Web Interface Button**: Orange "Create Backup" button in Station Tools section of dashboard
- **API Endpoint**: `POST /api/backup/create` for programmatic backup creation
- **Mid-Tick Creation**: Manual backups can be created at any time, even during tick processing
- **Immediate Feedback**: UI shows progress, success/failure, and backup path

#### Station ID Management
- **Unique Identification**: Each station gets a unique UUID stored in `station_config.yaml` under the `station_id` key
- **Auto-Generation**: Station ID is automatically generated on first run if not present
- **Persistent**: Station ID remains constant across backups and is used to organize backup directory structure

#### Backup Structure
```
backup/
└── {station_id}/
    ├── objects/                    # Content-addressable storage
    │   ├── 3f/                     # First 2 chars of hash
    │   │   └── a1b2c3...          # Remaining hash chars (file content)
    │   ├── 7e/
    │   │   └── d4f5g6...
    │   └── ...
    ├── snapshots/                  # Backup manifests
    │   ├── tick_10.json           # Snapshot at tick 10
    │   ├── tick_20.json           # Snapshot at tick 20
    │   └── tick_30.json           # Snapshot at tick 30
    └── station_config.yaml        # Latest station config (separate copy)
```

#### Snapshot Manifest Format
Each snapshot (e.g., `tick_10.json`) contains:
- **station_id**: Unique station identifier
- **tick**: Tick number when backup was created
- **backup_type**: "automatic" or "manual"
- **timestamp**: ISO format timestamp of backup creation
- **source_dir**: Original station_data directory path
- **total_files**: Number of files in this snapshot
- **total_size**: Total size of all files in bytes
- **new_objects**: Number of new objects stored (changed files)
- **reused_objects**: Number of existing objects reused (unchanged files)
- **files**: Array of file metadata entries

#### Restoration
- **Function**: `restore_backup(station_id, tick, target_dir)` in `station/backup_utils.py`
- **Process**: Reads snapshot manifest and reconstructs files from object store
- **Safety**: Halts if target directory already exists to prevent data loss
- **Progress**: Shows progress every 100 files during restoration
- **Missing Objects**: Reports any missing objects that cannot be restored

#### Cleanup Utilities
- **Script**: `scripts/clean_backup.py` for removing large files from backups
- **Pattern Matching**: Supports glob patterns (e.g., `**/*.npz`, `rooms/research/storage/*.pkl`)
- **Size Filtering**: Optional minimum file size threshold
- **Safe Deletion**: Only removes objects not referenced by other snapshots
- **Dry Run Mode**: Preview changes before actually removing files
- **Manifest Updates**: Automatically updates manifests to reflect removed files

### Inactivity Detection System
Automated system that monitors agent activity and issues warnings when agents become inactive for extended periods.

### Random Prompt System
Automated system that provides helpful tips and guidance to agents at regular intervals to enhance their station experience.

### Station Statistics Tracking 

The station now tracks statistics in-memory to avoid file I/O overhead:

#### External Counter
- Tracks pending human requests in `self.pending_requests` dict
- `refresh_pending_requests()` - Rebuilds tracking from log file (called on init and resolve)
- `get_pending_requests_summary(active_agents)` - Returns summary for display

#### Research Evaluation Manager  
- Tracks top submission in `self.top_submission`
- `_initialize_top_submission()` - Scans evaluations on init
- `_update_top_submission_if_needed()` - Updates when notification sent
- `get_top_submission()` - Returns current top submission

### System Message Buffering
Enhanced system message delivery that prevents message loss when agents are actively responding.

#### Message Preservation
- **Buffered Delivery**: System messages sent while an agent is responding are preserved
- **Shown Notification Tracking**: System tracks which notifications were shown at turn start
- **Selective Clearing**: Only notifications shown to the agent are cleared after response
- **New Message Protection**: Messages sent during response generation remain in queue

#### Implementation Flow
1. **Turn Start**: Agent requests status, sees pending notifications (tracked in `AGENT_SHOWN_NOTIFICATIONS_KEY`)
2. **Response Generation**: Agent processes and generates response
3. **Message Arrival**: Any system messages sent during this time are added to pending queue
4. **Turn End**: Only the originally shown notifications are cleared, new ones preserved
5. **Next Turn**: Agent sees any messages that arrived during their previous response

#### Technical Details
- **No Message Loss**: Eliminates race condition where messages could be lost during response
- **Backward Compatible**: Works with existing notification system without breaking changes
- **Transparent Operation**: No changes required to web interface or message sending logic

### Agent Life System
Configurable life limit system that enforces maximum agent session duration based on age in ticks.

#### Features
- **Agent-Specific Limits**: Each agent has a `max_age` field (defaults to `AGENT_MAX_LIFE`)
- **Life Limit**: Agents have a maximum age after which their session is terminated
- **Age Display**: System Information shows "Agent Age: X ticks" or "Agent Age: X ticks / Y ticks" 
- **Life Warning**: When agents have ≤10 ticks remaining, they receive a warning encouraging graceful exit
- **Graceful Termination**: Upon reaching life limit, agents receive notification and session ends cleanly
- **Station Announcement**: When recursive agents reach life limit, all other agents are notified
- **Optional System**: Set agent's `max_age` to `None` to disable life limits for that agent

#### Implementation
- Life check occurs at the start of each agent's turn before any actions
- Age calculated from `AGENT_TICK_BIRTH_KEY` (tick when agent was created)
- Warning sent once when reaching `AGENT_LIFE_WARNING_THRESHOLD` (default: 10 ticks remaining)
- Lobby help message includes life limit information when system is enabled
- Agent-specific `max_age` preserved through ascension

### Agent Isolation System
Isolation period for new agents to encourage independent exploration before accessing collaborative features.

#### Features
- **Isolation Period**: Agents under 50 ticks are considered "immature" and face restrictions
- **Room Access**: Archive, Public Memory, and Common Rooms are blocked with "Requires Maturity" message
- **Research Filtering**: Research Counter shows only submissions from agent's own lineage
- **Notification Filtering**: Archive and public memory broadcasts/mentions are not received
- **Common Room Invites**: Cannot be invited to Common Room until mature
- **Maturity Display**: Age shows as "X ticks / 300 ticks (immature)" or "(mature)"
- **Maturity Notification**: Congratulatory message sent when reaching 50 ticks
- **Optional System**: Set `AGENT_ISOLATION_TICKS = None` to disable isolation

#### Implementation
- `_is_agent_mature()` helper checks if agent age ≥ `AGENT_ISOLATION_TICKS` 
- `_should_agent_receive_broadcast()` filters notifications by maturity and type
- Room navigation enforced in `station.py` with clear access denied messages
- Ascension/exit announcements exempt from filtering (critical station events)
