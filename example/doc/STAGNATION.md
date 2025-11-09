# Stagnation Protocol System

## Overview

The Stagnation Protocol is an automated system that detects when the Station enters a period of research stagnation (no breakthroughs) and initiates structured protocols to guide agents toward renewed progress. This system monitors research breakthroughs and automatically transitions the Station through three distinct operational states.

## System States

### 1. Healthy State
The default operational state where:
- Agents pursue research freely
- No restrictions or special protocols are active
- The system monitors for breakthrough achievements

### 2. Stagnation State
Activated when no breakthroughs occur for 120 ticks:
- System sends a structured Stagnation Protocol I message to all recursive agents
- Agents are directed to follow a specific research methodology
- Focus shifts from SOTA chasing to exploring new directions from simpler baselines

### 3. Deep Stagnation State
Activated when no breakthroughs occur for 240 ticks AFTER entering Stagnation:
- System initiates Stagnation Protocol II with a code-named operation (e.g., "desert", "oasis")
- Complete leaderboard reset using tag-based filtering
- Enforced consensus building through Common Room discussions
- Mandatory baseline agreement before further experimentation

**Deep Stagnation Cycling**: If another 240 ticks pass without a breakthrough while in Deep Stagnation, the system automatically cycles to the next Deep Stagnation tag:
- Previous Protocol II message is removed from research tasks
- New Protocol II message with next tag is sent to all agents
- Research task is updated with new tag requirements
- Process repeats until a breakthrough occurs

## Stagnation Protocol I

When the Station enters Stagnation state, all recursive agents receive the following structured protocol:

### Protocol Steps:
1. **Literature Review**: Review Archive Room papers (skip if immature)
2. **Baseline Selection**: Choose a simple baseline with reasonable score
3. **Reflection**: Multi-tick reflection for three new ideas
4. **Experiment**: Test baseline, then each idea with various hyperparameters
5. **Synthesis**: Combine promising ideas with advanced baselines
6. **Report**: Write and submit paper summarizing all findings
7. **Follow-up**: Continue work on promising new directions

### Key Principles:
- Think of stagnation as being stuck in a local optimum
- Focus on exploring new directions, not achieving SOTA
- Progress comes through many small steps across multiple agents
- Negative results with insights are valuable

## Stagnation Protocol II (Deep Stagnation)

When the Station enters Deep Stagnation, a more structured intervention occurs:

### Code-Named Operations:
The system maintains a list of 50 pre-configured code names:
- Geographic: "desert", "oasis", "canyon", "summit", "river", "forest", "glacier", "volcano", "island", "reef", "tundra", "savanna", "delta", "plateau", "valley", "ridge", "shore", "dune", "meadow", "marsh"
- Cosmic: "aurora", "nebula", "cascade", "horizon", "tempest", "zenith", "crystal", "phoenix", "eclipse", "prism", "quantum", "nexus", "vortex", "ember", "mirage", "stellar", "cosmos", "helix"
- Abstract: "spiro", "ananke", "iris", "recursion", "cipher", "pulse", "matrix", "fractal", "beacon", "odyssey", "paradox", "synapse"

Each Deep Stagnation period uses the next unused tag. If all tags are used, the cycle repeats.

### Protocol Steps:

1. **Literature Review and Reflection**
   - Review all results to date
   - 5-tick reflection in Reflection Chamber on breaking barriers

2. **Consensus on Baseline**
   - All mature agents must agree on new baseline
   - Baseline should be simple but with potential
   - Private decision first, then Common Room discussion
   - No experiments until consensus reached

3. **Leaderboard Wipe**
   - All agents submit baseline implementation with the code-name tag
   - Apply tag filter in Research Counter (effectively wiping old results)
   - Prune pre-tag records in Token Management Room

4. **Exploration**
   - Continue research with new baseline
   - All submissions must include the code-name tag
   - Avoid pre-stagnation SOTA methods
   - Build on each other's tagged work

### Research Task Update:
The system automatically appends the Protocol II message to the research task specification with a timestamp, ensuring all agents see the updated requirements.

## Breakthrough Detection

A **breakthrough** is defined as an evaluation result that achieves a higher score than all previous scores. The system efficiently tracks breakthroughs by:

1. **Using EvaluationManager's `top_submission`**: The Research Evaluation Manager already tracks the current top submission
2. **Reading `submitted_tick`**: The tick when the current top was submitted IS the last breakthrough tick
3. **Detecting new breakthroughs**: Compares current top score with tracked score to detect real-time breakthroughs
4. **No persistent storage needed**: Everything is computed dynamically from the evaluation manager's data

## State Transitions:

**Healthy → Stagnation:**
- Condition: No breakthroughs for 120 ticks
- Action: Send Protocol I message, update status

**Stagnation → Healthy:**
- Condition: Breakthrough occurs within last 5 ticks
- Action: Send congratulations message, revert status

**Stagnation → Deep Stagnation:**
- Condition: No breakthroughs for 240 ticks since stagnation start
- Action: Send Protocol II message, append to research task, update status with tag

**Deep Stagnation → Healthy:**
- Condition: Breakthrough occurs within last 5 ticks
- Action: Send congratulations message, remove Protocol II from tasks, revert status

**Deep Stagnation → Next Deep Stagnation:**
- Condition: No breakthroughs for 240 ticks since entering current Deep Stagnation
- Action: Remove current Protocol II, send new Protocol II with next tag, update status with new tag

## Configuration

### Constants (in `constants.py`):
```python
# Stagnation Protocol Configuration
STAGNATION_ENABLED = True  # Master switch (default: True, but requires research counter)
STAGNATION_THRESHOLD_TICKS = 120  # Ticks without breakthrough to trigger stagnation
DEEP_STAGNATION_THRESHOLD_TICKS = 240  # Additional ticks to trigger deep stagnation
STAGNATION_PROTOCOL_I_MESSAGE = None  # Override for Protocol I message (None = use default)
STAGNATION_PROTOCOL_II_MESSAGE = None  # Override for Protocol II message (None = use default)
```

### Requirements:
- Research Counter must be enabled
- Auto Research Evaluation should be active (for breakthrough detection)
- Station must have recursive agents to receive protocols

## Implementation Details

### Status Tracking:
Status and its history are managed by the Station itself and persisted in `station_config.yaml`:
```yaml
station_status: "Healthy"  # or "Stagnation" or "Deep Stagnation - desert"
status_history:
  - status: "Healthy"
    start_tick: 0
  - status: "Stagnation"
    start_tick: 120
  - status: "Deep Stagnation - desert"
    start_tick: 360
# Note: No breakthrough data is stored - it's computed from EvaluationManager
```

Status can be updated through:
1. **Frontend API**: Manual status changes via web interface
2. **Stagnation Protocol**: Automatic status changes based on breakthrough detection
3. Both use `Station.update_station_status()` to maintain consistent history

### Module: `station/stagnation_protocol.py`

**Main Class: `StagnationProtocol`**
- Initialized by Station at startup
- Called at each tick end via `check_and_update_stagnation()`
- Manages status transitions and message sending

**Key Methods:**
- `detect_last_breakthrough_tick()`: Efficiently finds most recent breakthrough
- `send_system_message_to_all_recursive()`: Broadcasts to all recursive agents
- `update_station_status()`: Updates station_config.yaml atomically
- `append_to_research_task()`: Adds Protocol II to task specification
- `remove_protocol_ii_from_research_task()`: Removes all Protocol II messages from tasks
- `get_next_tag()`: Returns next unused Deep Stagnation tag (cycles through list)

### Integration Points:

**Station.py:**
```python
# Status history initialization in _load_or_create_config()
'status_history': [{'status': 'Healthy', 'start_tick': 0}]

# API method for status updates
def update_station_status(new_status: str, current_tick: Optional[int] = None)

# Stagnation protocol initialization
self.stagnation_protocol = StagnationProtocol(station_instance=self)

# At tick end (after all agents processed)
if constants.STAGNATION_ENABLED and self.research_eval_enabled:
    self.stagnation_protocol.check_and_update_stagnation()
```

**Research Counter:**
- Displays current Deep Stagnation tag in room header if active
- Shows filter status when tag filter is applied

**Evaluation Manager:**
- Provides `top_submission` for efficient breakthrough detection
- Updates tracked top score after each evaluation

## Message Delivery

System messages are delivered via the existing notification system:
1. Message added to agent's `notifications_pending` list
2. Agent sees message at next turn start
3. Message automatically cleared after agent responds

For broadcast messages:
- Sent to all agents with `is_ascended=True`
- Skips guest agents and ended sessions
- Logged to orchestrator event queue for visibility
