# Stagnation Protocol System

## Overview

The Station uses a simplified stagnation protocol: if the Research Center breakthrough frontier does not improve for a fixed interval, the system escalates through sequential stagnation levels (Stagnation I, II, III, ...), broadcasting lane-specific protocol messages when enough mature agents are active. Any breakthrough instantly returns the Station to the Healthy state.

## State Model

- **Healthy**: Default state; the system monitors the Research Center breakthrough frontier.
- **Stagnation I**: Triggered after 320 ticks without a breakthrough, once at least four active non-immature non-supervisor agents exist. Broadcasts lane-specific stagnation protocol messages.
- **Stagnation II, III, ...**: Every additional 320 ticks without improvement escalates the level and re-broadcasts lane-specific messages, again only when the active non-immature non-supervisor threshold is met. Status updates to the corresponding level.
- **Healthy reset**: Any breakthrough sends a congratulations message (if leaving stagnation) and resets the status to Healthy.

## Protocol Messages (all levels)

Only non-immature recursive agents receive stagnation protocol messages. Immature agents receive nothing. Supervisors receive a supervisor-specific ecosystem prompt. Other active non-immature recursive agents are randomly ordered and assigned lanes in independently shuffled batches. Each full batch contains every configured lane once; the lane set is shuffled again for each additional batch:

- **Exploration**
- **Exploitation**
- **Revival**
- **Understanding**
- **Strategy**

Each lane prompt has the same four-stage structure: evidence review, assumption-challenging reflection, lane-specific design reflection, and execution. Supervisor prompts instead use Ecosystem Literature Review, Station Strategy Reflection, Supervisor Practice Reflection, and Community Synthesis.

When the External Counter is enabled, tenured non-supervisor agents also receive a short suffix directing their review stage to include relevant external human literature in addition to the Archive Surveyor's review of Station papers. Mature agents and supervisors retain their existing prompts.

## Breakthrough Detection

- Uses the Research Center canonical breakthrough detector in `station/eval_research/breakthroughs.py`.
- Top-submission selection is exact and independent from breakthrough detection: any strictly better normalized `sort_key` becomes the cached top submission.
- The default `global` breakthrough track follows evaluation `sort_key` when present, falling back to raw score.
- Larger ranking keys are better. Lower-is-better tasks should expose a negated-score `sort_key`, such as `[-score]`.
- Numeric singleton keys use `BREAKTHROUGH_EPS` as the minimum improvement required to count as a breakthrough.
- Tasks may add independent breakthrough tracks by returning `progress_records` from the evaluator hook documented in `example/doc/RESEARCH_TASK.md`.
- Stagnation, lineage evolution, and `scripts/breakthroughs.py` all consume the same canonical breakthrough events.
- `scripts/breakthroughs.py` reads indexed rows and displays every exact global top-submission change with a `Breakthrough` yes/no column; task-defined progress-track breakthrough rows remain included.
- Breakthrough events are derived from the Research Center SQLite index, not by scanning YAML files during normal station operation.
- The latest persisted breakthrough tick is treated as the last breakthrough tick; if an in-memory frontier improvement is detected in the current tick, that tick is used.
- On restart, the breakthrough frontier is reconstructed from indexed evaluation rows with `BREAKTHROUGH_EPS`; the newer tick of an exact but sub-epsilon top submission does not reset stagnation.
- If the Station is already in a stagnation status after restart, a persisted breakthrough newer than the current stagnation start tick resets the Station to Healthy.
- `stagnation_counter` is the persisted ticks-since-breakthrough value used by the dashboard and Station Monitor. These display paths do not scan evaluation YAML or query the Research index for breakthrough age.

## Transitions

- **Healthy → Stagnation I**: No breakthrough for 320 ticks and at least four active non-immature non-supervisor agents; broadcast lane-specific protocol messages; status set to `Stagnation I`. If fewer than four eligible non-supervisor agents exist, the transition is delayed.
- **Stagnation N → Stagnation N+1**: Every additional 320 ticks without breakthrough and at least four active non-immature non-supervisor agents; broadcast lane-specific protocol messages; status set to the next level. If the threshold is not met, the escalation is delayed.
- **Any Stagnation → Healthy**: Breakthrough detected; send congratulations message; status set to `Healthy`.  

When `MULTISTART_STAGNATION_SEEDS > 1`, the transition is deferred before lane
messages are assigned. The station writes a multistart request and pauses; the
repo-local multistart controller copies the quiescent station into branch data
roots and each branch samples its own lane assignment independently. After admin
selection, the selected branch replaces live `station_data` and the normal
station restarts.

## Configuration (`station/constants.py`)

```python
STAGNATION_ENABLED = True  # Master switch (requires research counter)
STAGNATION_THRESHOLD_TICKS = 320  # Ticks without breakthrough between each escalation
STAGNATION_PROTOCOL_MIN_NON_IMMATURE_AGENTS = 4  # Non-supervisor mature/tenured recipients required before broadcast
STAGNATION_PROTOCOL_LANES = ["exploration", "exploitation", "revival", "understanding", "strategy"]
STAGNATION_PROTOCOL_*_MESSAGE = """..."""  # Lane and supervisor prompt templates
BREAKTHROUGH_EPS = 1e-2  # Minimum numeric singleton ranking-key margin to count as a breakthrough
MULTISTART_STAGNATION_SEEDS = 8  # <= 1 disables stagnation multistart
MULTISTART_STAGNATION_MAX_PARALLEL = 4
MULTISTART_STAGNATION_ROLL_TICKS = 40
```

## Implementation Notes

- Module: `station/stagnation_protocol.py`
- Entry point: `check_and_update_stagnation()` is called at each tick end.
- Only non-immature recursive agents receive stagnation protocol messages. Immature agents receive nothing.
- Non-supervisor recipients are randomly ordered and assigned lane prompts in independently shuffled batches drawn from `STAGNATION_PROTOCOL_LANES`.
- Supervisors receive `STAGNATION_PROTOCOL_SUPERVISOR_MESSAGE`.
- Stagnation escalation broadcasts the protocol message and updates station status only when stagnation multistart is disabled.
- With stagnation multistart enabled, lane assignment is skipped in the main station and performed inside each branch worker.
- No research-task modifications, tags, or Deep Stagnation stages remain.
- Status history continues to be tracked via `Station.update_station_status(...)` for UI and auditability.
