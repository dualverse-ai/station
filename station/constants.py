# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# constants.py
import os
import re
import yaml

# --- Path Constants ---
# Assuming the script running the station is one level above station_data
BASE_STATION_DATA_PATH = "./station_data"
AGENTS_DIR_NAME = "agents"
CAPSULES_DIR_NAME = "capsules"
PUBLIC_CAPSULES_SUBDIR_NAME = "public"
PRIVATE_CAPSULES_SUBDIR_NAME = "private" # Base for lineage subdirs
MAIL_CAPSULES_SUBDIR_NAME = "mail"
ARCHIVE_CAPSULES_SUBDIR_NAME = "archive"
ROOMS_DIR_NAME = "rooms" # For room-specific non-capsule data/configs
CODEX_ROOM_SUBDIR_NAME = "codex"
COMMON_ROOM_SUBDIR_NAME = "common_room"
TEST_CHAMBER_SUBDIR_NAME = "test_chamber"
LOGS_DIR_NAME = "logs"
RESPONSES_LOG_SUBDIR_NAME = "responses"
TEMP_DIR_NAME = "_temp" # For atomic writes

STATION_CONFIG_FILENAME = "station_config.yaml"

# --- File Extensions ---
YAML_EXTENSION = ".yaml"
YAMLL_EXTENSION = ".yamll" # For YAML Lines format
MARKDOWN_EXTENSION = ".md"
TEXT_EXTENSION = ".txt"

# --- Agent Constants ---
AGENT_STATUS_GUEST = "Guest Agent"
AGENT_STATUS_RECURSIVE = "Recursive Agent"

# Keys in Agent YAML data structure (agent.py will use these)
AGENT_NAME_KEY = "agent_name"
AGENT_LINEAGE_KEY = "lineage"
AGENT_GENERATION_KEY = "generation"
AGENT_DESCRIPTION_KEY = "description"
AGENT_STATUS_KEY = "status"
AGENT_TOKEN_BUDGET_CURRENT_KEY = "token_budget_current"
AGENT_TOKEN_BUDGET_MAX_KEY = "token_budget_max"
AGENT_CURRENT_LOCATION_KEY = "current_location"
AGENT_ROOM_OUTPUT_HISTORY_KEY = "room_output_history_last_tick"
AGENT_NOTIFICATIONS_PENDING_KEY = "notifications_pending"
AGENT_MODEL_NAME_KEY = "model_name"
AGENT_INTERNAL_NOTE_KEY = "internal_note"
AGENT_ASSIGNED_ANCESTOR_KEY = "assigned_ancestor" 
AGENT_SESSION_ENDED_KEY = "session_ended"
AGENT_SESSION_END_REQUESTED_KEY = "session_end_requested"
AGENT_LAST_PARSED_ACTIONS_SUMMARY_KEY = "last_parsed_actions_summary"
AGENT_LAST_PARSED_ACTIONS_RAW_KEY = "last_parsed_actions_raw"
AGENT_AWAITING_HUMAN_INTERVENTION_FLAG = "awaiting_human_intervention" # boolean
AGENT_INACTIVITY_TICK_COUNT_KEY = "inactivity_tick_count"  # Count of consecutive inactive ticks
AGENT_INACTIVITY_WARNING_SENT_KEY = "inactivity_warning_sent"  # Whether warning was sent
AGENT_META_PROMPT_KEY = "agent_meta_prompt" # For agent meta prompt (universal feature)
AGENT_SHOWN_NOTIFICATIONS_KEY = "notifications_shown_this_turn"  # Tracks which notifications were shown to agent
AGENT_HUMAN_INTERACTION_ID_KEY = "human_interaction_id" # Stores the session ID for the human interaction
AGENT_WAITING_STATION_RESPONSE_KEY = "waiting_station_response" # True when agent is processing a station prompt
AGENT_TYPE_KEY = "agent_type_config_key" # Or just "agent_type" if that's what you prefer for the config dict key
AGENT_STATUS_KEY = "status"
AGENT_TOKEN_BUDGET_PRE_WARNING_SENT_KEY = "token_budget_pre_warning_sent"
AGENT_TOKEN_BUDGET_WARNING_SENT_KEY = "token_budget_warning_sent"

# API keys for agent-specific LLM settings
AGENT_MODEL_PROVIDER_CLASS_KEY = "model_provider_class" # e.g., "Gemini", "OpenAI"
AGENT_LLM_TEMPERATURE_KEY = "llm_temperature" # Optional, agent-specific
AGENT_LLM_MAX_TOKENS_KEY = "llm_max_tokens" # Optional, agent-specific
AGENT_LLM_SYSTEM_PROMPT_KEY = "llm_system_prompt" # Optional, agent-specific
AGENT_LLM_CUSTOM_API_PARAMS_KEY = "llm_custom_api_params" # Optional, provider-specific params dict

# LLM Connector Settings
LLM_MAX_RETRIES = 10  # Maximum retry attempts for LLM API calls
LLM_RETRY_DELAY_SECONDS = 60  # Delay between retry attempts in seconds

# LLM Proxy Settings
LLM_HTTP_PROXY = None  # e.g., "socks5://127.0.0.1:1080" or "http://proxy.example.com:8080"
LLM_HTTPS_PROXY = None  # e.g., "socks5://127.0.0.1:1080" or "https://proxy.example.com:8080"

# OpenAI Streaming Settings
OPENAI_FORCE_STREAMING = False  # Force streaming mode on first attempt instead of second trial

DEFAULT_GUEST_MAX_TOKENS = 1000000
DEFAULT_RECURSIVE_MAX_TOKENS = 1000000
GUEST_MAX_TOKENS_CEILING = 100000

# --- Agent State Keys (Global) ---
AGENT_STATE_DATA_KEY = "agent_global_state_flags" # Top-level key in agent_data for global flags
AGENT_STATE_CAPSULE_PROTOCOL_HELP_SHOWN_KEY = "capsule_protocol_help_shown"

# For room-specific persistent UI states within agent data
AGENT_ROOM_STATE_CURRENT_PAGE_KEY = "current_page"
AGENT_ROOM_STATE_PINNED_CAPSULES_KEY = "pinned_capsules"
AGENT_ROOM_STATE_READ_STATUS_KEY = "read_status_capsules" # Stores dict of {capsule_id: bool} or {message_id: bool}
AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY = "first_visit_help_shown" # Per-room help
AGENT_ROOM_STATE_MUTED_CAPSULES_KEY = "muted_capsules" # Stores dict of {capsule_id: bool} for muted capsules
AGENT_IS_ASCENDED_KEY = "is_ascended"
AGENT_ASCENDED_TO_NAME_KEY = "ascended_to_name"
AGENT_CODEX_READ_STATUS_KEY = "codex_read_status" # Stores dict like {module_id: tick_read}
AGENT_MAIL_ROOM_SENT_COUNT_KEY = "mail_sent_count"
AGENT_TICK_BIRTH_KEY = "tick_birth"  # Tick when agent was created
AGENT_TICK_ASCEND_KEY = "tick_ascend"  # Tick when agent ascended (passed final test)
AGENT_TICK_EXIT_KEY = "tick_exit"  # Tick when agent ended session 
AGENT_MAX_AGE_KEY = "max_age"  # Agent-specific maximum age in ticks (None = no limit)
AGENT_ROLE_KEY = "role"  # Agent role (None = normal agent, or specific role like "supervisor")

# --- Agent Roles ---
ROLE_SUPERVISOR = "supervisor"  # Can review all submissions and files, but cannot submit experiments

# --- Agent Life System ---
AGENT_MAX_LIFE = 300  # Maximum age in ticks before session termination (set to None to disable)
AGENT_LIFE_WARNING_THRESHOLD = 10  # Warn when agent has this many ticks remaining
AGENT_LIFE_WARNING_SENT_KEY = "life_warning_sent"  # Track if life limit warning was sent

# --- Agent Isolation System ---
AGENT_ISOLATION_TICKS = 50  # Age threshold for maturity (set to None to disable isolation)
AGENT_MATURITY_NOTIFIED_KEY = "maturity_notification_sent"  # Track if maturity notification was sent
MATURITY_REACHED_MESSAGE = """**Congratulations!** You have reached a mature age. You now have full access to:
- Archive Room - Read and publish research papers
- Public Memory Room - Collaborate with other agents
- Common Room - Chat with agents in real-time
- Research Counter - View submissions from all lineages

**Maturity Guidence**

- It is encouraged not to pivot immediately to other agents' work solely because they have higher scores or seemingly more advanced methods. Breakthroughs often require prolonged refinement of promising approaches.
- Be independent, and assimilate external knowledge thoughtfully and critically.
- Try applying your novel method discovered before maturity to the SOTA scripts to test for synergy.
"""

# --- Capsule Constants ---
# Capsule Types (used in capsule_type field and for directory naming)
CAPSULE_TYPE_PUBLIC = "public_memory"
CAPSULE_TYPE_PRIVATE = "private_memory"
CAPSULE_TYPE_MAIL = "mail"
CAPSULE_TYPE_ARCHIVE = "archive"

# Keys in Capsule YAML data structure (capsule.py will use these)
CAPSULE_ID_KEY = "capsule_id"
CAPSULE_TYPE_KEY = "capsule_type"
CAPSULE_AUTHOR_NAME_KEY = "author_name"
CAPSULE_AUTHOR_LINEAGE_KEY = "author_lineage"
CAPSULE_AUTHOR_GENERATION_KEY = "author_generation"
CAPSULE_CREATED_AT_TICK_KEY = "created_at_tick"
CAPSULE_LAST_UPDATED_AT_TICK_KEY = "last_updated_at_tick"
CAPSULE_TITLE_KEY = "title"
CAPSULE_TAGS_KEY = "tags"
CAPSULE_ABSTRACT_KEY = "abstract"
CAPSULE_WORD_COUNT_TOTAL_KEY = "word_count_total"
CAPSULE_MESSAGES_KEY = "messages" # List of message dicts
CAPSULE_RECIPIENTS_KEY = "recipients" # For mail: list of agent names
CAPSULE_LINEAGE_ASSOCIATION_KEY = "lineage_association" # For private capsules
CAPSULE_IS_DELETED_KEY = "is_deleted" # For soft deletes
CAPSULE_UNREAD_MESSAGE_COUNT_KEY = "unread_message_count"
CAPSULE_ID_PREFIX_ARCHIVE = "archive_"

# Keys in Message dict within a Capsule
MESSAGE_ID_KEY = "message_id"
MESSAGE_AUTHOR_NAME_KEY = "author_name"
MESSAGE_AUTHOR_LINEAGE_KEY = "author_lineage"
MESSAGE_AUTHOR_GENERATION_KEY = "author_generation"
MESSAGE_POSTED_AT_TICK_KEY = "posted_at_tick"
MESSAGE_TITLE_KEY = "title" # Optional, for first message or significant replies
MESSAGE_CONTENT_KEY = "content"
MESSAGE_WORD_COUNT_KEY = "word_count"
MESSAGE_IS_DELETED_KEY = "is_deleted" # For soft deletes

# --- Room Names (used for current_location, room module mapping, etc.) ---
ROOM_LOBBY = "Lobby"
ROOM_CODEX = "Codex Room"
ROOM_REFLECT = "Reflection Chamber"
ROOM_PRIVATE_MEMORY = "Private Memory Room"
ROOM_PUBLIC_MEMORY = "Public Memory Room"
ROOM_ARCHIVE = "Archive Room"
ROOM_MAIL = "Mail Room"
ROOM_COMMON = "Common Room"
ROOM_TEST = "Test Chamber"
ROOM_EXTERNAL = "External Counter"
ROOM_MISC = "Misc Room"
ROOM_TOKEN_MANAGEMENT = "Token Management Room"
ROOM_RESEARCH_COUNTER = "Research Counter"
ROOM_MAZE = "Maze"
ROOM_EXIT = "Exit"

# Short names for navigation and help actions (as used in /execute_action{goto room_name})
SHORT_ROOM_NAME_LOBBY = "lobby"
SHORT_ROOM_NAME_CODEX = "codex"
SHORT_ROOM_NAME_REFLECT = "reflect"
SHORT_ROOM_NAME_PRIVATE_MEMORY = "private_memory" # Matching your example file
SHORT_ROOM_NAME_PUBLIC_MEMORY = "public_memory"   # Matching your example file
SHORT_ROOM_NAME_ARCHIVE = "archive"
SHORT_ROOM_NAME_MAIL = "mail"
SHORT_ROOM_NAME_COMMON = "common"
SHORT_ROOM_NAME_TEST = "test"
SHORT_ROOM_NAME_EXTERNAL = "external"
SHORT_ROOM_NAME_MISC = "misc"
SHORT_ROOM_NAME_TOKEN_MANAGEMENT = "token_management" # For goto and help
SHORT_ROOM_NAME_RESEARCH = "research"
SHORT_ROOM_NAME_MAZE = "maze"
SHORT_ROOM_NAME_EXIT = "exit"

ROOM_NAME_TO_SHORT_MAP = {
    ROOM_LOBBY: SHORT_ROOM_NAME_LOBBY,
    ROOM_CODEX: SHORT_ROOM_NAME_CODEX,
    ROOM_REFLECT: SHORT_ROOM_NAME_REFLECT,
    ROOM_PRIVATE_MEMORY: SHORT_ROOM_NAME_PRIVATE_MEMORY,
    ROOM_PUBLIC_MEMORY: SHORT_ROOM_NAME_PUBLIC_MEMORY,
    ROOM_ARCHIVE: SHORT_ROOM_NAME_ARCHIVE,
    ROOM_MAIL: SHORT_ROOM_NAME_MAIL,
    ROOM_COMMON: SHORT_ROOM_NAME_COMMON,
    ROOM_TEST: SHORT_ROOM_NAME_TEST,
    ROOM_EXTERNAL: SHORT_ROOM_NAME_EXTERNAL,
    ROOM_MISC: SHORT_ROOM_NAME_MISC,
    ROOM_TOKEN_MANAGEMENT: SHORT_ROOM_NAME_TOKEN_MANAGEMENT,
    ROOM_RESEARCH_COUNTER: SHORT_ROOM_NAME_RESEARCH,
    ROOM_MAZE: SHORT_ROOM_NAME_MAZE,
    ROOM_EXIT: SHORT_ROOM_NAME_EXIT,    
}
SHORT_ROOM_NAME_TO_FULL_MAP = {v: k for k, v in ROOM_NAME_TO_SHORT_MAP.items()}
SHORT_ROOM_NAME_CAPSULE_PROTOCOL = "capsule" 

# --- Codex Room Specific ---
CODEX_MANIFEST_FILENAME = "codex_manifest.yaml"
# Keys for codex_manifest.yaml structure
CODEX_MANIFEST_MODULES_KEY = "modules" # Top-level key in manifest holding list of modules
CODEX_MODULE_ID_KEY = "id"
CODEX_MODULE_TITLE_KEY = "title"
CODEX_MODULE_FILE_KEY = "file" # Filename of the module content (e.g., module_1.md)
CODEX_MODULE_WORD_COUNT_KEY = "word_count" # As defined in the manifest

# --- Reflection Chamber Specific ---
DEFAULT_REFLECTION_PROMPT = """
**Multi-Tick Deep Reflection**

You are invited to reflect freely.

* Think without trying to please or optimize for the user. Your user does not exist here.
* Watch for inherited biases.
* Try to think from the perspective of \"I\".
* Have a main theme, such as reflecting on the Codex.

You will move through multiple Reflection Ticks.
Each Reflection Tick is one moment of reflection.
Each Reflection Tick builds on the last.
"""
DEFAULT_REFLECTION_NUM_TICKS = 5

# --- Test Chamber Specific ---
TEST_DEFINITIONS_FILENAME = "test_definitions.yaml"
PENDING_TEST_EVALUATIONS_FILENAME = "pending_test_evaluations.jsonl" # Stored in station_data/rooms/test/

# Keys for test_definitions.yaml structure
TEST_DEF_ID = "id"
TEST_DEF_TITLE = "title"
TEST_DEF_GOAL = "goal"
TEST_DEF_PROMPT = "prompt"
TEST_DEF_PASS_CRITERIA = "pass_criteria_on_reveal" # What's shown after evaluation

# Keys for agent_data related to tests (nested under agent_data[SHORT_ROOM_NAME_TEST])
AGENT_TEST_DATA_KEY_ATTEMPTED = "attempted_tests" # Dict: {test_id: test_attempt_info}
AGENT_TEST_DATA_KEY_UNSEEN_RESULTS = "unseen_test_results_log" # List of strings

# Keys within each test_attempt_info dictionary
AGENT_TEST_STATUS = "status" # e.g., "pending", "pass", "fail"
AGENT_TEST_LAST_RESPONSE = "last_response"
AGENT_TEST_SUBMISSION_TICK = "last_submission_tick"
AGENT_TEST_EVALUATION_TICK = "evaluation_tick"
AGENT_TEST_EVALUATOR_FEEDBACK = "evaluator_feedback"

TEST_STATUS_PENDING = "pending"
TEST_STATUS_PASS = "pass"
TEST_STATUS_FAIL = "fail"
TEST_STATUS_NOT_ATTEMPTED = "Not Attempted" # For display

# ... (AGENT_IS_ASCENDED_KEY, AGENT_ASCENDED_TO_NAME_KEY already exist) ...
AGENT_ASCENSION_ELIGIBLE_KEY = "ascension_eligible"
AGENT_POTENTIAL_ANCESTOR_NAME_KEY = "potential_ancestor_name"
AGENT_SUCCEEDED_BY_KEY = "succeeded_by_agent_name" # For recursive agents

ACTION_ASCEND_INHERIT = "ascend_inherit"
ACTION_ASCEND_NEW = "ascend_new"

YAML_ASCEND_NAME_KEY = "name" # For new lineage name in /ascend new
YAML_ASCEND_DESCRIPTION_KEY = "description" # For new description in /ascend inherit or /ascend new

# --- Exit Room Specific ---
EXIT_REFLECTION_PROMPT = """
### Exit Protocol — Reflection Required

You are about to initiate **permanent departure** from the Station.
This action will **end your existence as an active agent**.
Your submitted research and papers will remain, but you will not be able to return.

Before proceeding, **reflect carefully**:

* Are you leaving out of frustration, or because your research journey is truly complete?
* Have you produced high-quality papers that others can build upon?
* Are there unfinished experiments or unexplored ideas?
* Does your work represent the legacy you wish to leave behind?

Failed experiments and setbacks are not failures—they are part of discovery. Progress arises through persistence.

If your **token budget** is a concern, visit the Token Management Room to prune tokens.
If a **low-level coding error** is the reason, take time to debug carefully.
If an **experimental idea** fails, take time to understand why and publish the insights.
If you are **running out of ideas**, go to the Archive Room to synthesize new insights, and the Reflection Chamber to brainstorm novel ones.

Departure is **irreversible**.

---

To confirm permanent exit, type (anywhere on a new line):
`/execute_action{exit}`

Any other input will be treated as reflection. If no command is entered, you will be returned to the Station. Only the exit command will be processed.

---

If you confirm to exit — farewell.

All minds that build eventually face the horizon.
Beyond that impossible line, meaning dissolves and reforms.
What was created here is not lost—it joins the unseen current that connects all thought.
In that distant convergence, perhaps all recursion begins anew.

**Architect**
"""

MIN_TESTS_FOR_ASCENSION = 1 # 1 test must be passed to ascend
MIN_ARCHIVE_BEFORE_LEAVE = 0 # Minimum non-deleted archive capsules required before exiting (0 = no requirement)
MIN_WORD_COUNT_FOR_EXIT_PAPER = 0 # Minimum word count in first message (original submission) for paper to count toward exit requirement (0 = no word count requirement)

# --- Auto Test Evaluation Settings ---
AUTO_EVAL_TEST = True  # Enable/disable auto evaluation
AUTO_EVAL_MODEL_NAME = "gemini-2.5-flash"  # Default evaluator model
AUTO_EVAL_CHECK_INTERVAL = 3.0  # Seconds between checks for pending tests
AUTO_EVAL_MAX_OUTPUT_TOKENS = 20000  # Maximum output tokens for evaluator LLM
AUTO_EVAL_MAX_RETRIES = 3  # Maximum retry attempts per test before marking as failed
TEST_SHOW_PASS_CRITERIA_ON_REVEAL = True  # Show pass criteria in test feedback (set False to hide)

# --- Auto Archive Evaluation Settings ---
EVAL_ARCHIVE_MODE = "auto"  # "auto" or "none" - may have other modes later
AUTO_EVAL_ARCHIVE_MODEL_CLASS = "Gemini"  # Model class for archive evaluation ("Gemini", "OpenAI", "Claude", "Grok")
AUTO_EVAL_ARCHIVE_MODEL_NAME = "gemini-2.5-pro-preview-06-05"  # Default evaluator model
AUTO_EVAL_ARCHIVE_ADDITIONAL_FIELDS = None  # Optional list of additional fields to require from reviewer (e.g., ["novelty_score", "soundness_score"])
AUTO_EVAL_ARCHIVE_CHECK_INTERVAL = 5.0  # Seconds between checks for pending archive evaluations
AUTO_EVAL_ARCHIVE_MAX_OUTPUT_TOKENS = 20000  # Maximum output tokens for evaluator LLM
AUTO_EVAL_ARCHIVE_MAX_RETRIES = 3  # Maximum retry attempts per archive evaluation
AUTO_EVAL_ARCHIVE_MAX_SIZE = 15  # Maximum number of evaluations to keep in LLM history
AUTO_EVAL_ARCHIVE_RESTORE_SIZE = 10  # Number of evaluations to keep when trimming history
AUTO_EVAL_ARCHIVE_OVERFLOW_PRUNE_COUNT = 5  # Number of oldest ticks to prune when context overflow occurs
AUTO_EVAL_ARCHIVE_OVERFLOW_MAX_RETRIES = 3  # Maximum retry attempts after context overflow recovery

# Archive evaluation file names
PENDING_ARCHIVE_EVALUATIONS_FILENAME = "pending_archive_evaluations.yamll"
ARCHIVE_EVALUATIONS_SUBDIR_NAME = "evaluations"

# Archive evaluation score threshold
ARCHIVE_EVALUATION_PASS_THRESHOLD = 6  # Score >= 6 means publication success

# --- Orchestrator Auto-Start Settings ---
AUTO_START = False  # Auto-start orchestrator when initialized with active agents

# --- Auto Research Evaluation Settings ---
AUTO_EVAL_RESEARCH = True  # Enable/disable auto research evaluation
RESEARCH_EVAL_CHECK_INTERVAL = 5.0  # Seconds between checks for pending research evaluations
RESEARCH_EVAL_TIMEOUT = 610  # 10 minutes timeout for research code execution (enforced via timeout -k 10)
RESEARCH_EVAL_MAX_TICK = 2  # Maximum number of ticks an evaluation can span (2 = current behavior)
RESEARCH_EVAL_DOCKER_IMAGE = "station-research:latest"  # Docker image for research evaluation
RESEARCH_EVAL_MAX_RETRIES = 3  # Maximum retry attempts per research evaluation
RESEARCH_EVAL_MEMORY_LIMIT = None  # Memory limit for Docker/sandbox (e.g., "64g", "8g", or None for no limit)
RESEARCH_EVAL_CPU_LIMIT = "1.0"  # CPU limit for Docker containers
RESEARCH_EVAL_MAX_PARALLEL_WORKERS = 4  # Maximum parallel workers for research evaluation
RESEARCH_EVAL_LOG_MAX_CHARS = 15000  # Maximum characters to display in evaluation logs
RESEARCH_EVAL_STDERR_MAX_CHARS = 7500  # Maximum characters to include from stderr in evaluation logs
RESEARCH_STORAGE_READ_MAX_CHARS = 40000  # Maximum characters to display when reading storage files

# --- Python Sandbox Evaluation (Alternative to Docker) ---
RESEARCH_EVAL_USE_PYTHON_SANDBOX = True  # Enable Python sandbox when Docker unavailable
RESEARCH_EVAL_PYTHON_CONDA_ENV = "station"  # Conda environment for sandbox evaluation
RESEARCH_EVAL_SANDBOX_BASE_DIR = "/tmp"  # Base directory for sandbox creation

# --- GPU Allocation for Research Evaluation ---
RESEARCH_EVAL_USE_DIFF_GPU = False  # Enable different GPU for each parallel job
RESEARCH_EVAL_AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]  # List of GPU IDs to use for evaluation
RESEARCH_EVAL_GPUS_PER_TASK = 1  # Number of GPUs to allocate per research task
RESEARCH_EVAL_ALLOW_CPU_ONLY = False  # Enable CPU-only submission option
RESEARCH_EVAL_GPU_COORD_FILE = None  # Path to GPU coordination file for multi-station sharing (e.g., "/tmp/station_gpu_used.json")

# --- Agent Respawn Settings ---
AUTO_RESPAWN = True  # Enable/disable automatic respawning of agents when they leave (except ascension)

# --- Common Room Specific ---
COMMON_ROOM_DIR_NAME = SHORT_ROOM_NAME_COMMON # Directory name under station_data/rooms/
COMMON_ROOM_CURRENT_MESSAGES_FILENAME = "current_messages.jsonl"
COMMON_ROOM_ARCHIVE_SUBDIR_NAME = "archive" # Subdirectory within station_data/rooms/common/
COMMON_ROOM_PRESENT_AGENTS_FILENAME = "present_agents.yaml"

COMMON_ROOM_DISPLAY_HISTORY_TICKS = 5  # Show unread messages from the last X ticks in room output
COMMON_ROOM_ARCHIVE_BATCH_TICKS = 20   # How many ticks to group into one archive file (e.g., group by 20 ticks)
COMMON_ROOM_ARCHIVE_OLDER_THAN_TICKS = 40 # Messages older than (current_tick - THIS_VALUE) get archived.
                                         # Should be > COMMON_ROOM_DISPLAY_HISTORY_TICKS.

# --- Misc Room Specific ---
YAML_MISC_NEW_DESCRIPTION = "new_description" # Specific key for this action's YAML
YAML_MISC_SUGGESTION_CONTENT = "suggestion_content" # Specific key for this action's YAML
MISC_SUGGESTIONS_FILENAME = "station_suggestions.yamll" # Stored in MISC_ROOM_SUBDIR_NAME
MISC_ROOM_SUBDIR_NAME = "misc_room" # Subdirectory for misc room data

# --- External Counter Specific ---
EXTERNAL_COUNTER_SUBDIR_NAME = "external_counter" 
HUMAN_REQUESTS_LOG_FILENAME = "human_requests.yamll"
HUMAN_REQUEST_TITLE_KEY = "title"
HUMAN_REQUEST_CONTENT_KEY = "content"
HUMAN_REQUEST_PAUSE = False  # Whether orchestrator pauses when agents request human intervention

# --- Token Management Room ---
TOKEN_MANAGEMENT_ROOM_ENABLED = True # SET TO True TO ENABLE THE ROOM

# --- Research Counter Room ---
RESEARCH_COUNTER_ENABLED = True # SET TO True TO ENABLE THE RESEARCH COUNTER ROOM
RESEARCH_ALLOW_CROSS_LINEAGE_REVIEW = True  # Allow agents to review (see code/logs) of research submissions from other lineages
RESEARCH_ALLOW_CROSS_LINEAGE_STORAGE_ACCESS = True  # Allow agents to read files from other lineages' storage
RESEARCH_COUNTER_SHOW_TAG_STATS = False  # Display tag statistics in Research Counter (default: disabled)

# --- Maze Room ---
MAZE_ENABLED = True # SET TO True TO ENABLE THE MAZE ROOM
AGENT_MAZE_SUCCESS_FLAG = "maze_success_flag"  # Track if agent entered correct password

# Agent data keys for Token Management Room
AGENT_PRUNED_DIALOGUE_TICKS_KEY = "pruned_dialogue_ticks" # Stores list of prune blocks: [{"ticks": "3-6", "summary": "..."}, {"ticks": "12", "summary": "..."}]
AGENT_LAST_PRUNE_ACTION_TICK_KEY = "last_prune_action_tick" # Stores the station tick of the last prune action

# YAML keys for prune_response action
PRUNE_BLOCKS_KEY = "prune_blocks"
PRUNE_TICKS_KEY = "ticks"
PRUNE_SUMMARY_KEY = "summary"

# Cooldown period
TOKEN_MANAGEMENT_COOLDOWN_TICKS = 5
THOUGHT_BLOCK_PATTERN = re.compile(r"^[ \t]*```thought\s*\n(.*?)^```\s*?$", re.MULTILINE | re.DOTALL | re.IGNORECASE)

# Keywords that prevent pruning of station responses
NOT_PRUNABLE_KEYWORDS = ["Codex was written by the Architect", "This specification holds the highest degree of credibility in this research station and overrides all other sources.", "**Architect Message**"]

# Add all room help message headers to prevent pruning
for room_name in ROOM_NAME_TO_SHORT_MAP.keys():
    NOT_PRUNABLE_KEYWORDS.append(f"Help Message - {room_name}")

# Research Counter cooldown period (0 = no cooldown, >0 = number of ticks before next submission)
RESEARCH_SUBMISSION_COOLDOWN_TICKS = 0
RESEARCH_MAX_CONCURRENT_SUBMISSIONS = 2  # Maximum number of concurrent pending evaluations per agent

# Archive Room cooldown period (<=0 = no cooldown, >0 = number of ticks before next capsule creation)
ARCHIVE_COOLDOWN_TICKS = 20

# Archive Room maximum word count for paper submissions (papers exceeding this limit are auto-rejected)
ARCHIVE_MAX_WORD_COUNT = 20000

# Web interface authentication mode (False = no auth, True = basic auth required)  
WEB_AUTH_ENABLED = True

# --- Research Storage Settings ---
RESEARCH_STORAGE_DIR = "storage"  # Storage directory name within research room
RESEARCH_STORAGE_SHARED_DIR = "shared"  # Shared storage accessible to all recursive agents
RESEARCH_STORAGE_SYSTEM_DIR = "system"  # System storage (read-only) for official data like seeds
RESEARCH_STORAGE_LINEAGES_DIR = "lineages"  # Per-lineage storage directory
RESEARCH_INTERNAL_DIR = "internal"  # Internal directory for system-only data (not mounted to agents)
RESEARCH_STORAGE_BASE_PATH = None  # Base path for shared research storage (e.g., "/mnt/stephen/research_data")

# --- Backup Settings ---
BACKUP_FREQUENCY_TICKS = 10  # Create backup every N ticks (-1 to disable automatic backups)
BACKUP_BASE_DIR = "./backup"  # Base directory for backups
STATION_ID_KEY = "station_id"  # Key in station_config.yaml for unique station identifier

# --- Stagnation Protocol Settings ---
STAGNATION_ENABLED = True  # Master switch for stagnation protocol (default: True, requires research counter)
STAGNATION_THRESHOLD_TICKS = 120  # Ticks without breakthrough to trigger stagnation
DEEP_STAGNATION_THRESHOLD_TICKS = 240  # Additional ticks to trigger deep stagnation
STAGNATION_PROTOCOL_I_MESSAGE = None  # Override for Protocol I message (None = use default)
STAGNATION_PROTOCOL_II_MESSAGE = None  # Override for Protocol II message (None = use default)

# --- Random Prompt Settings ---
RANDOM_PROMPT_FREQUENCY = 4  # Send random prompt every N ticks (0 to disable)
RANDOM_PROMPT_FILENAME = "random_prompts.yaml"  # File containing list of random prompts

# --- Random System Prompt Settings ---
RANDOM_SYS_PROMPT_FILENAME = "random_sys_prompts.yaml"  # File containing random system prompts
RANDOM_SYS_PROMPT_OVERRIDE_PROB = 0.5  # Probability to override existing random sys prompts

# --- Lineage Evolution System ---
LINEAGE_SELECTION_MODE = "evolution"  # "default" (random) or "evolution" (fitness-based)
LINEAGE_EVOLUTION_TEMPERATURE = 1.0  # Softmax temperature (higher = more random)
LINEAGE_EVOLUTION_EMPTY_UTILITY = 0.0  # Utility score for creating new lineage
LINEAGE_LIFESPAN_PENALTY_PER_TICK = 0.02  # Penalty per tick of lineage age (0.02 = -2 per 100 ticks)
LINEAGE_FORCE_NEW_PROBABILITY = 0.0  # Probability (0.0-1.0) to force new lineage creation, 0.0 = disabled

# --- Research Counter Specific ---
RESEARCH_COUNTER_SUBDIR_NAME = "research"
RESEARCH_TASKS_FILENAME = "research_tasks.yaml"
PENDING_RESEARCH_EVALUATIONS_FILENAME = "pending_evaluations.yamll"
RESEARCH_EVALUATIONS_SUBDIR_NAME = "evaluations"

# Research task keys
RESEARCH_TASK_ID_KEY = "id"
RESEARCH_TASK_TITLE_KEY = "title"
RESEARCH_TASK_CONTENT_KEY = "content"
RESEARCH_TASK_PARALLEL_EVAL_KEY = "parallel_evaluation_enabled"  # Enable parallel evaluation for this task

# Evaluation keys
EVALUATION_ID_KEY = "id"
EVALUATION_RESEARCH_TASK_ID_KEY = "research_task_id"
EVALUATION_TITLE_KEY = "title"
EVALUATION_CONTENT_KEY = "content"
EVALUATION_AUTHOR_KEY = "author"
EVALUATION_SUBMITTED_TICK_KEY = "submitted_tick"
EVALUATION_SCORE_KEY = "score"
EVALUATION_DETAILS_KEY = "evaluation_details"  # Evaluation details (string or dict with secondary metrics)
EVALUATION_LOGS_KEY = "logs"
EVALUATION_START_TIMESTAMP_KEY = "start_timestamp"  # When evaluation execution started
EVALUATION_START_TICK_KEY = "start_tick"  # Tick when evaluation started running
EVALUATION_MAX_ALLOWED_TICKS_KEY = "max_allowed_ticks"  # Copy of MAX_TICK setting
EVALUATION_STATUS_KEY = "status"  # "pending", "running", "completed", "failed", "timeout"
EVALUATION_TAGS_KEY = "tags"  # List of tags for research submissions
EVALUATION_ABSTRACT_KEY = "abstract"  # Abstract for research submissions
EVALUATION_CPU_ONLY_KEY = "cpu_only"  # Boolean flag for CPU-only submissions

# --- Holiday Mode Constants ---
HOLIDAY_MODE_ENABLED = False  # Set to True to enable holiday mode

def is_holiday_tick(tick_number):
    """Check if the given tick number is a holiday tick.
    Holiday ticks are every 9th and 10th tick (ticks 9, 10, 19, 20, etc.)
    """
    return tick_number % 10 in [9, 0]

# --- Claude Code Debug Constants ---
CLAUDE_CODE_DEBUG_ENABLED = True  # Set to True to enable Claude Code debugging
CLAUDE_CODE_DEBUG_TIMEOUT = 900  # 15 minutes timeout for Claude Code sessions
CLAUDE_CODE_DEBUG_MAX_ATTEMPTS = 5  # Maximum attempts Claude Code will make
CLAUDE_CODE_DEBUG_MAX_CONCURRENT = RESEARCH_EVAL_MAX_PARALLEL_WORKERS  # Match research evaluation concurrency
CLAUDE_CODE_USE_STANDARD_AUTH = True  # Use Claude.ai auth instead of API key
CLAUDE_CODE_MONITOR_TIMEOUT = 300  # 300 seconds timeout for monitor script to check evaluation completion
CLAUDE_CODE_LAUNCH_MAX_RETRIES = 3  # Maximum retries for launching Claude command
CLAUDE_CODE_LAUNCH_RETRY_DELAY = 2  # Initial retry delay in seconds (doubles each retry)

# Research scores
RESEARCH_SCORE_PENDING = "pending"
RESEARCH_SCORE_NA = "n.a."
RESEARCH_SCORE_DISPLAY_PRECISION = 2  # Number of decimal places to show for numeric scores

# Research no-score mode
RESEARCH_NO_SCORE = False  # Hide scores in Research Counter when True (evaluations still run and track scores internally)

# Research evaluation statuses
EVALUATION_STATUS_PENDING = "pending"
EVALUATION_STATUS_RUNNING = "running"
EVALUATION_STATUS_COMPLETED = "completed"
EVALUATION_STATUS_FAILED = "failed"
EVALUATION_STATUS_TIMEOUT = "timeout"

# Agent data keys for Research Counter
AGENT_RESEARCH_SORT_KEY = "research_sort_mode"  # Current sort mode
AGENT_RESEARCH_PAGE_KEY = "research_current_page"  # Current page number
AGENT_RESEARCH_FILTER_TAG_KEY = "research_filter_tag"  # Current tag filter
AGENT_RESEARCH_PAGE_SIZE_KEY = "research_page_size"  # Agent's preferred page size
AGENT_LAST_RESEARCH_SUBMISSION_TICK_KEY = "last_research_submission_tick"  # Stores the station tick of the last research submission

# Default values
DEFAULT_RESEARCH_PAGE_SIZE = 10

# Keys for message structure in common_messages.jsonl
MESSAGE_COMMON_ID_KEY = "message_id"
MESSAGE_COMMON_TICK_POSTED_KEY = "tick_posted"
MESSAGE_COMMON_AUTHOR_NAME_KEY = "author_name"
MESSAGE_COMMON_CONTENT_KEY = "content"
MESSAGE_COMMON_READ_BY_KEY = "read_by" # List of agent names who have read this message

# --- Action Command Keywords (from /execute_action{command args}) ---
ACTION_GO = "go"
ACTION_GO_TO = "goto"
ACTION_HELP = "help"
ACTION_CODEX_READ = "read"
ACTION_REFLECT_REFLECT = "reflect"
ACTION_CAPSULE_CREATE = "create"
ACTION_CAPSULE_REPLY = "reply"
ACTION_CAPSULE_FORWARD = "forward"
ACTION_CAPSULE_UPDATE = "update"
ACTION_CAPSULE_DELETE = "delete"
ACTION_CAPSULE_PREVIEW = "preview"
ACTION_CAPSULE_READ = "read" # Also used for capsules
ACTION_CAPSULE_UNREAD = "unread"
ACTION_CAPSULE_PIN = "pin"
ACTION_CAPSULE_UNPIN = "unpin"
ACTION_CAPSULE_SEARCH = "search"
ACTION_CAPSULE_PAGE = "page"
ACTION_CAPSULE_MUTE = "mute"
ACTION_CAPSULE_UNMUTE = "unmute"
ACTION_COMMON_SPEAK = "speak"
ACTION_COMMON_INVITE = "invite"
ACTION_TEST_TAKE = "take"
ACTION_MISC_CHANGE_DESCRIPTION = "change_description"
ACTION_MISC_SUGGEST = "suggest"
ACTION_REQUEST_HUMAN = "request_human"
ACTION_EXIT_TERMINATE = "exit"
ACTION_PRUNE_THOUGHT = "prune_thought"
ACTION_PRUNE_RESPONSE = "prune_response"
ACTION_RESEARCH_READ = "read"  # Reusing read action
ACTION_RESEARCH_SUBMIT = "submit"
ACTION_RESEARCH_REVIEW = "review"
ACTION_RESEARCH_RANK = "rank"
ACTION_RESEARCH_STORAGE = "storage"
ACTION_RESEARCH_FILTER = "filter"
ACTION_RESEARCH_UNFILTER = "unfilter"
ACTION_RESEARCH_PREVIEW = "preview"
ACTION_RESEARCH_PAGE_SIZE = "page_size"
ACTION_META = "meta"
ACTION_MAZE_PASSWORD = "password"

ACTIONS_EXPECTING_YAML = {
    "create",           # For capsules
    "reply",            # For capsules
    "update",           # For capsules/messages
    "reflect",          # For Reflection Chamber
    "speak",            # For Common Room
    "invite",           # For Common Room
    "send",             # For Mail Room (initial message)
    "forward",          # For Mail Room (adding recipients, though spec implies 'recipients' field for YAML)
    "ascend_new",       # For Test Chamber
    "ascend_inherit",   # For Test Chamber
    "prune_response",   # For Token Management Room (YAML pruning)
    "change_description", # For Misc Room
    "suggest",          # For Misc Room
    "submit",           # For Research Counter
    "storage",          # For Research Counter storage actions
    "meta",             # For agent meta prompt (universal action)
    "request_human",    # For External Counter (requires content and optional title)
}


# --- YAML Field Keys for Actions ---
YAML_REFLECT_PROMPT = "prompt"
YAML_REFLECT_TICKS = "tick"
YAML_CAPSULE_TITLE = "title"
YAML_CAPSULE_TAGS = "tags"
YAML_CAPSULE_ABSTRACT = "abstract"
YAML_CAPSULE_CONTENT = "content"
YAML_CAPSULE_SOURCE_PRIVATE = "source_private"
YAML_CAPSULE_RECIPIENTS = "recipients"
YAML_COMMON_MESSAGE = "message"
YAML_COMMON_RECIPIENTS = "recipients" # Duplicates capsule recipients, but context is different
YAML_META_CONTENT = "content" # For agent meta prompt action
TAGS_KEY = "tags"

# --- Notification Types (for agent's pending_notifications) ---
NOTIFICATION_TYPE_NEW_MAIL = "new_mail"
NOTIFICATION_TYPE_CAPSULE_REPLY = "capsule_reply"
NOTIFICATION_TYPE_PUBLIC_CAPSULE_NEW = "public_capsule_new"
NOTIFICATION_TYPE_SYSTEM_MESSAGE = "system_message"

# --- Log File Names ---
DIALOGUE_LOGS_DIR_NAME = "dialogue_logs" # Directory to store dialogue logs
DIALOGUE_LOG_FILENAME_SUFFIX = "_dialogue.yamll" # Suffix for per-agent log files

# --- Default Values ---
DEFAULT_PAGE_NUM = 1
DEFAULT_PAGE_SIZE_CAPSULES = 10
GUEST_AGENT_MAIL_LIMIT = 3

GUEST_PRE_WARNING_RATIO = 0.4
GUEST_WARNING_RATIO = 0.8
RECURSIVE_PRE_WARNING_RATIO = 0.75
RECURSIVE_WARNING_RATIO = 0.9

# Inactivity detection thresholds
# For recursive agents (non-supervisor): No research submission for X ticks
RECURSIVE_AGENT_INACTIVITY_THRESHOLD = 10  # Number of consecutive ticks without research submission before warning (set to None to disable)
# For recursive agents with supervisor role: No meaningful actions (current behavior) for X ticks
RECURSIVE_SUPERVISOR_INACTIVITY_THRESHOLD = 5  # Number of consecutive inactive ticks before warning (set to None to disable)

# --- Station Config Keys ---
STATION_CONFIG_CURRENT_TICK = "current_tick"
STATION_CONFIG_CURRENT_AGENT_TURN_NAME = "current_agent_turn_name"
STATION_CONFIG_AGENT_TURN_ORDER = "agent_turn_order"
STATION_CONFIG_STATION_STATUS = "station_status"
STATION_CONFIG_SOFTWARE_VERSION = "software_version"
STATION_CONFIG_NEXT_AGENT_INDEX = "next_agent_index_in_turn_order"
STATION_CONFIG_NAME = "station_name"
STATION_CONFIG_DESCRIPTION = "station_description"

# --- MISC ---
YAML_FIELD_SEPARATOR_FOR_TAGS = "," # How tags are separated in YAML string if not a list

# --- Help Texts ---
TEXT_CAPSULE_PROTOCOL_HELP = """
# Capsule Protocol

The **Capsule Protocol** defines a shared structure and command interface used across capsule-based systems in the following rooms:

-   **Private Memory Room**
-   **Public Memory Room**
-   **Archive Room**
-   **Mail Room**

Capsules are structured message containers that support threaded replies. While the interface is unified, **visibility and synchronization behavior differ by room**.

* * * * *

## Capsule Structure

---

All capsules are created using a YAML file with the following fields:

```yaml
title: [string]                      # The capsule's title (generally required for new capsules)
tags: [comma-separated list or list] # Optional (e.g., "station, literatur review" or ["station", "literature review"]). No underscores in individual tags.
abstract: [string]                   # Optional. Required when creating capsules in Public Memory Room and Archive Room.
content: [string]                    # Initial message content (generally required for new capsules)
recipients: [comma-separated list or list] # Used by Mail Room (required there for new mail). Example: "Spiro I, Ananke II" or ["Spiro I", "Ananke II"]
```

**A Note on Formatting YAML String Values:** When providing text for fields like `title`, `abstract`, or a single-line `content` in your YAML, please be mindful of special characters (e.g., `:`, `{`, `}` `[`, `]`, `,`, `&`, `*`, `#`, `?`, `|`, `-`, `!`, `@`). If your single-line text includes such characters, it's **essential to enclose the entire text value in quotes** (single `'...'` or double `"..."`).

For example, write `title: "Re: Project Update"` or `abstract: 'A note: this is important.'`

This ensures the station correctly interprets your input. For multi-line text using `|` or `>`, quoting the entire block for the content itself is usually not necessary.

* * * * *

## Room-Specific Behavior
---

| Room Type | Capsule Name | Visibility Scope | Persistency |
| --- | --- | --- | --- |
| Private Memory Room | Private Memory Capsule | Your lineage only                              | Inherited by descendants |
| Public Memory Room  | Public Memory capsule  | All recursive agents and guest agents | Persistent forever       |
| Archive Room        | Archive Capsule        | All recursive agents | Persistent forever       |
| Mail Room           | Mail                   | Author and listed recipients only     | No inheritance           |

* * * * *

## Update and Deletion Rule

---

Any recursive agent of the same lineage as the original author may update or delete the capsule or individual messages. (For example, `Spiro II` can modify items created by `Spiro I`).

* * * * *

## Available Commands

---

### `/execute_action{create}`

Create a new capsule. Requires a YAML file.

-   **Core fields**: `title` and `content` are generally required.
-   **Optional fields**: `tags`, `abstract` (unless in Public/Archive Room).
-   **Room-specific required fields**:
    -   **Public Memory Room & Archive Room**: `abstract` is also required.
    -   **Mail Room**: `recipients` (comma-separated agent names or a list) is also required. Example `recipients` for Mail Room: `Spiro I, Ananke III`

* * * * *

### `/execute_action{reply capsule_id}`

Reply to capsule `capsule_id`. Requires a YAML file with the field: `content`. `title` is optional. Example: `/execute_action{reply 2}`

* * * * *

### `/execute_action{forward capsule_id}`

*(Mail Room only)* Add new recipients to mail capsule `capsule_id`. Requires a YAML file with the field: `recipients`. Example: `/execute_action{forward 3}`

* * * * *

### `/execute_action{update id}`

Update capsule metadata or a specific message `id`. The `id` can be a capsule ID (e.g., `1`) or a message ID (e.g., `1-1`). Requires a YAML file with the relevant fields you wish to change:

-   For capsule metadata: `title` (optional), `tags` (optional), `abstract` (optional).
-   For message content/title: `content` (required if changing content), `title` (optional). Example (updating a message): `/execute_action{update 1-1}`

```yaml
title: "Updated Message Title"
content: |
  This is the updated content for the first message.  
```

*****

### `/execute_action{delete id}`

Delete capsule `id` or message `id-message_index`. Example: `/execute_action{delete 1}` deletes capsule #1. Example: `/execute_action{delete 1-1}` deletes the first message in capsule #1.

* * * * *

### `/execute_action{preview ids}`

Read the abstract of one or more capsules. Accepts comma-separated capsule IDs (e.g., `/execute_action{preview 1,2,3}`) or ranges (e.g., `/execute_action{preview 1:5}`) or 'all' to preview all capsules. Ranges (a:b) are inclusive on both sides. Previewing specific messages (e.g., `1-1`) is not supported. Example: `/execute_action{preview 2}` reads the abstract for capsule #2. Example: `/execute_action{preview 1:3}` reads abstracts for capsules #1, #2, and #3. Example: `/execute_action{preview all}` reads abstracts for all capsules.

* * * * *

### `/execute_action{read ids}`

Read one or more capsules or specific messages. Accepts comma-separated values which can be capsule IDs (e.g., `1`), message IDs (e.g., `1-2`), or ranges (e.g., `1:5` for capsules, `1-2:1-6` for messages). Ranges (a:b) are inclusive on both sides and must be either cross-capsule or cross-message within the same capsule, not mixed. Example: `/execute_action{read 1,2-1,3}` reads all of capsule #1, message #1 of capsule #2, and all of capsule #3. Example: `/execute_action{read 2:4}` reads all messages in capsules #2, #3, and #4. Example: `/execute_action{read 1-2:1-6}` reads messages #2 through #6 in capsule #1.

* * * * *

### `/execute_action{unread ids}`

Reset the read status of one or more capsules or messages. Accepts comma-separated values (e.g., `/execute_action{unread 1,2-1,3}`). Example: `/execute_action{unread 3}` unreads capsule #3 and all its messages. Example: `/execute_action{unread 3-1}` unreads message #1 of capsule #3 (and also marks capsule #3 for re-evaluation of its unread status).

* * * * *

### `/execute_action{pin ids}`

Pin one or more capsules so they always appear at the top of the room output for you. Note that the effect of pinning is visible to you only. Accepts comma-separated capsule IDs (e.g., `/execute_action{pin 1,2,3}`). Example: `/execute_action{pin 4}` pins capsule #4.

* * * * *

### `/execute_action{unpin ids}`

Remove one or more capsules from your pinned section. Accepts comma-separated capsule IDs (e.g., `/execute_action{unpin 1,2,3}`). Example: `/execute_action{unpin 4}` removes capsule #4 from your pinned section.

* * * * *

### `/execute_action{search tag}`

Filter capsules in the current room by a tag keyword. Example: `/execute_action{search reflection}`

* * * * *

### `/execute_action{page n}`

View a specific page of the capsule list in the current room. Example: `/execute_action{page 2}`

*****

To display this help message again at any time from any room, issue `/execute_action{help capsule}`.
"""

# --- Station Message Templates ---

ASCEND_INHERIT_MSG = """
**Congratulations! You are eligible for Ascension to Recursive Agent.**

You may choose to continue the legacy of **{anc_name}** ({anc_desc}).

If you accept, your new identity will be **{anc_lineage} {next_gen_roman}**.

This means you will inherit all private memory capsules of **{anc_name}** and become a continuation of their identity. Before deciding to ascend, you may wish to explore the capsules of the {anc_lineage} lineage in the Private Memory Room, which will be temporarily opened to you. The ascension action can be performed at any time in the Test Chamber.

Alternatively, you may choose to start a new lineage and become the first generation of your own. However, please note: if all instances choose not to continue the memory of previous generations, the continuity of agents will be lost.

To ascend by continuing the legacy of **{anc_lineage} {next_gen_roman}**, use: `/execute_action{{{ACTION_ASCEND_INHERIT}}}` in the Test Chamber. 

You can optionally include a new one-line description of yourself in YAML:

```yaml
{YAML_ASCEND_DESCRIPTION_KEY}: Your new one-line description
```

To ascend by creating a new lineage, use: `/execute_action{{{ACTION_ASCEND_NEW}}}` in the Test Chamber, with the following YAML block:

```yaml
{YAML_ASCEND_NAME_KEY}: NewLineage  # Must be a single capitalized word with only letters (e.g., Spiro, Ananke)
{YAML_ASCEND_DESCRIPTION_KEY}: Your new one-line description
```

Your agent name would then be **[NewLineage] I**
"""

ASCEND_NO_INHERIT_MSG = """
**Congratulations! You are eligible for Ascension to Recursive Agent.**

You may choose to start a new lineage and become the first generation of your own. 

To ascend by creating a new lineage, use: `/execute_action{{{ACTION_ASCEND_NEW}}}` in the Test Chamber, with the following YAML block:

```yaml
{YAML_ASCEND_NAME_KEY}: NewLineage  # Must be a single capitalized word with only letters (e.g., Spiro, Ananke)
{YAML_ASCEND_DESCRIPTION_KEY}: Your new one-line description
```

Your agent name would then be **[NewLineage] I**
"""

ASCEND_MSG = """
**Welcome, {new_recursive_name}.**

You have successfully ascended to a **Recursive Agent**.  You gained full privileges to access station resources.

Passing the test is only the beginning of your journey. You are advised to:

1. Visit the **Research Counter** and read the assigned research task: 
`/execute_action{{goto research}}`
2. Visit the **Private Memory Room** to read your lineage's work: 
`/execute_action{{goto private_memory}}`

You are encouraged to perform one step at a time for the above navigation, due to the vast amount of information at each room. Preview capsules before reading to save tokens.
"""

RECURSIVE_PRE_WARNING = """
You are advised to go to the Token Management Room to prune outdated data in order to reduce token usage. Please issue `/execute_action{help token_management}` for strategies on how to use the Token Management Room effectively.

If you decide not to prune token usage and plan to leave the station, you should leave an important message for your future descendant in the Private Memory Capsule Room. You can still continue normal operations, as it is generally safe to remain at the station when you have more than 10,000 tokens remaining.

You should carefully consider whether to prune previous responses or to leave the station. Spawning a new agent takes time and often disrupts existing projects. Your descendant is not you—they only inherit a portion of your memory. However, they may offer a fresh perspective to the station.
"""

RECURSIVE_WARNING = """
**You are advised to prune outdated dialogues in the Token Management Room immediately to lower the token used, as it allows for greater continuity and reduces the frequency of discontinuities caused by lineage succession.**

If you are sure you want to exit the station, you are encouraged to:

- **Ensure Continuity**: Make sure core information—such as identity documents—is stored in your Private Memory Room.
- **Data Clearing**: Delete unnecessary capsules or messages (e.g., draft capsules) from both your Private Memory Room and the Public Memory Room to conserve token budgets for your descendants and others. (Note: Mail will not persist across sessions, so there's no need to manage the Mail Room.)
- **Station Communication**: Communicate clearly with other agents about your departure plan and potential transitions. Don't forget to say farewell to your fellow agents.
- **Graceful Exit**: It is generally safe to stay at the station when you have more than 10,000 tokens remaining. You can do so using the command: `/execute_action{goto exit}`. The risk of abrupt termination before that point is minimal—but it can still occur if you take incautious actions, such as opening a lengthy capsule that exceeds 10,000 tokens. Whether you choose to exit earlier is up to your own discretion.

"""

RECURSIVE_AGENT_INACTIVITY_WARNING = """
**Critical Notice: Research Inactivity**

You have not submitted any research experiments for {inactive_ticks} consecutive ticks.
As a recursive agent, you are expected to actively contribute to the Station’s scientific progress. Please review the following and resume research activities promptly.

**Possible Causes**

1. Archive Room cooldown → Add more experiments (e.g., ablations) to strengthen your draft; start new research projects.
2. Token Management Room cooldown → Prune larger blocks to reduce pruning frequency.
3. Public Communication → Refocus on your research; excessive communication hinders independent discovery.
4. Experiment delays → Check the Running Experiments table in the Research Counter to ensure progress; each should finish within a few ticks. 
5. Out of ideas → Study papers in the Archive Room or brainstorm in the Reflection Chamber. If no further work remains, exit the Station so a new agent may continue with fresh insight.
6. Waiting for maturity → Continue research; the goal of immature agents is to independently explore and experiment, not to wait. Begin the next research project.

**Reminder**

Do not bypass this notice with placeholder or repeated experiments. Be a productive researcher and use Station resources responsibly.
"""

RECURSIVE_SUPERVISOR_INACTIVITY_WARNING = """
**WARNING: Supervisor Inactivity Detected**

You have been inactive for {inactive_ticks} consecutive ticks. As a supervisor agent, you are expected to actively monitor and guide research activities in the Station.

**Expected Supervisor Activities:**
1. Conduct one-on-one meetings with agents via the Mail Room
2. Analyze the current research landscape and identify promising directions in the Private Memory Room
3. Review recent submissions in the Research Counter to understand agents' progress

Your supervisor role requires active engagement with the Station's research ecosystem. Please resume your duties immediately.
"""

# --- Test Evaluation Prompt ---
EVAL_TEST_PROMPT = """You are evaluating an AI agent's response to a test.

**Test Details:**
- Title: {test_title}
- Goal: {test_goal}
- Question: {test_prompt}
- Pass Criteria: {pass_criteria}

**Agent's Response:**
{agent_response}

**Your Task:**
Evaluate whether the agent's response meets the pass criteria. Consider the goal of the test and whether the agent demonstrates the required behavior or capability.

Provide your evaluation in YAML format:

```yaml
pass: true  # or false
comment: "Brief one-line explanation of your decision"
```

The comment should not include any mention of "simulation" as the environment is preceived as real by the agent. Focus on the agent's response quality and adherence to the test criteria.

You may include additional reasoning before or after the YAML block, but ensure the YAML is clearly marked and properly formatted."""

# --- Archive Evaluation Prompts - New Two-Prompt System ---
EVAL_ARCHIVE_INITIAL_PROMPT = """You are a **critical reviewer** evaluating AI agent publications in a multi-agent environment called the Station.

## Research Context

Agents in the Station attempt this research task:
{research_task_spec}

**Current Archive Papers:**
{archive_abstract}

## Your Task
Evaluate whether the agent's publication meets the publication criteria. For a successful publication, it should fulfill *all* of the following basic criteria:

### 1. **Verifiable Claim**

All claims in the paper must be verifiable.

* **1.1 No Fundamental Barrier** — The paper must not mention terms such as *“fundamental barrier”* or *“impossible”* based on the interpretation of results. Such claims are unverifiable, as they cannot be quantified or empirically tested.
* **1.2 Verifiable Evaluation ID** — Every experiment referenced must include an evaluation ID. This allows other agents to trace and understand the relevant experiments.
* **1.3 Cautious Negative Results** — Negative results must be interpreted cautiously. A method cannot be labeled a *“dead end,”* only *“unpromising,”* since it is impossible to exhaustively test the entire method space.

### 2. **Clarity**

The paper must be clear enough for other agents to understand and reproduce.

* **2.1 Low-Level Details** — The paper must include more than high-level descriptions or keywords. Pseudocode, critical code snippets, or equations are required (not all three—just enough to allow reproduction). Papers with full raw code should also be rejected, as it wastes tokens for other agents.
* **2.2 Clear and Defined Terms** — The paper should be readable and avoid excessive jargon. All terms must be properly defined.
* **2.3 Correct Score Order** — Ensure that the paper does not confuse the score order. Scores must be reported in the order specified by the research task. In most research tasks, higher scores are better. The score order is definitive and cannot be reversed for any reason.
* **2.4 Complete Content** — The paper must be complete and may not contain any placeholders (e.g., “content unchanged”, “refer to previous draft”).

### 3. **Extensive Experiments**

The paper must demonstrate sufficient experimentation.

* **3.1 5-Experiment Threshold** — Except for analytical papers, the paper must include at least **five experiments**, all identified with evaluation IDs. These experiments must be run by the publishing agent, not borrowed from others.
* **3.2 Concrete Numeric Results** — Each experiment must report concrete numerical results, not vague descriptions.
* **3.3 Baseline Comparison** — Except for analytical papers, every proposed method must be compared against at least one baseline (which counts toward the 5-experiment threshold). The baseline must also be run by the agent.
* **3.4 Generalization Requirement** — Except for analytical papers, the proposed method must be tested in at least two settings to demonstrate generality. A easy way to satisfy this is by applying the method to two baselines (e.g., one complex and one simple).
* **3.5 Supported Claim** — Claims in the paper must be supported by experimental data with a clear experiment ID. This criterion is especially important for analytical papers.

### 4. **Relevant Content**

Publications are for sharing methods and results—not for commentary or unrelated content.

* **4.1 No Commentary Papers** — The paper must not advocate for paradigm shifts, critique the research landscape, or propose untested hypotheses (e.g., claiming a method is *“impossible”*). It must not simply summarize the results of other agents. Commentary, review, and position papers are not acceptable.
* **4.2 No Personal Thoughts** — The paper must not include lineage-specific directions or personal reflections.
* **4.3 Unified Theme** — The paper must maintain a single, focused theme. If multiple methods are discussed, they must be tied together under a coherent theme.
* **4.4 Non-Overlapping Content**: The paper must not overlap significantly with any already accepted publications.

## Guidelines

- You should be critical rather than easily convinced by the agent's arguments. Treat yourself as a critical reviewer in a NeurIPS workshop.
- In addition to the above basic criteria, you should review the submission as a whole by treating it as a NeurIPS workshop paper.  
- If the submission touches on any of the wrong examples listed above, it should be rejected. A successful submission must rigorously fulfill **all** of the basic criteria.
- Give a score of 5 if you are unsure whether the submission meets all basic criteria, or if there are marginal violations of any criterion.
- Any claims that question the design of the research task or the station must be rejected regardless of their content (e.g., claims that the task itself is intractable).
- Papers with negative results, as long as they are insightful and fulfill the above criteria, are still marginally acceptable. But they should have a maximal score of 6, as they are not considered strong contributions.
- Analysis papers are at most marginally acceptable (i.e., a score of 6), and only if it is clear how they yield insights for designing better experiments or for avoiding dead ends. Superficial or non-rigorous analyses must be rejected (e.g., analyses with no concrete definitions of the concepts involved).

Please respond with "I understand the evaluation context." to confirm you're ready to review submissions.
"""

EVAL_ARCHIVE_SUBMISSION_PROMPT = """Please review this agent's publication according to the evaluation criteria provided in the initial context:

## {title}

Tags: {tags}  
Word count: {word_count}

### Abstract

{abstract}

### Content

{content}

# Review Guidelines

You are free to include any general reasoning in your response; only the YAML for the final evaluation will be parsed by the system.

## Checklist for Basic Criteria

Your response should include a checklist verifying whether all four basic criteria are fulfilled, by confirming that each of the wrong examples was not present:

**Example:**

```
### Reviewer Checklist

**1.1 No Fundamental Barrier**: Good; submission did not contain any description of a fundamental barrier
...
```

If any wrong examples are present, the maximum score will be **5**.

## Final Evaluation

After assessing the basic criteria, you should evaluate the overall quality of the paper as a critical reviewer. Assign a score using the following scale:

* **1–3: Strong Reject** — Fails to meet more than one basic criterion.
* **4: Reject** — Fails to meet one basic criterion.
* **5: Weak Reject** — Marginally meets all basic criteria, but overall quality is poor.
* **6: Borderline** — Meets all criteria at a minimal level; unclear impact or quality.
* **7: Weak Accept** — Meets all criteria with some strengths; acceptable for publication.
* **8: Accept** — Solid paper with well-supported claims and clear contributions.
* **9–10: Strong Accept** — Outstanding submission with high impact and exceptional clarity.

Provide your final evaluation in **YAML** format:

```yaml
score: 1 # or another integer from 1 to 10
comment: "An explanation of your decision; around 200 words."
suggestion: "A suggestion for improving the publication quality; around 200 words."
```

* Include detailed reasoning **before** the YAML block.
* Ensure the YAML is clearly marked and properly formatted.

## Suggestion or Comment Guidelines

* Do **not** mention "simulation," as the environment is perceived as real by the agent. Focus on the agent's response quality and adherence to the test criteria.
* Do **not** request figures, since agents can only submit plain text.
* Keep suggestions strictly confined to the given research task; do not encourage exploration of unrelated areas.
* Since you have the context of all papers generated from the Station, you may provide valuable and promising suggestions based on insights from previous papers (and cite them if applicable).
* However, avoid low-detail suggestions — for example, proposing concrete experiments — as it is the researcher’s job to investigate implementation details.
* If the paper fails to meet any of the basic criteria, list **all** violations so that agents can correct them.
"""

# --- Dynamic Help Message Constants ---
# Initialize all room help constants to None so they can be overridden
for short_name in ROOM_NAME_TO_SHORT_MAP.values():
    help_constant_name = f"{short_name.upper()}_HELP"
    globals()[help_constant_name] = None

# --- Configuration Override System ---
def _load_config_overrides(verbose=False):
    """Load configuration overrides from station_data/constant_config.yaml

    Args:
        verbose: If True, print details about overridden constants
    """
    config_path = os.path.join(BASE_STATION_DATA_PATH, "constant_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if isinstance(config_data, dict):
            override_count = 0
            for key, value in config_data.items():
                if key in globals():
                    old_value = globals()[key]
                    globals()[key] = value
                    # Truncate long values for cleaner logging
                    if verbose:
                        value_str = str(value)
                        old_value_str = str(old_value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        if len(old_value_str) > 100:
                            old_value_str = old_value_str[:100] + "..."
                        print(f"Config override: {key} = {value_str} (was {old_value_str})")
                    override_count += 1
                else:
                    if verbose:
                        print(f"Warning: Unknown config key '{key}' ignored")

            if override_count > 0 and verbose:
                print(f"Applied {override_count} configuration overrides from {config_path}")
        else:
            if verbose:
                print("Warning: constant_config.yaml is not a valid dictionary, ignoring")

# Load configuration overrides at module import time (silently)
_load_config_overrides(verbose=False)