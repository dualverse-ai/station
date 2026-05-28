# Stagnation Protocol System

## Overview

The Station now uses a simplified stagnation protocol: if research scores do not improve for a fixed interval, the system escalates through sequential stagnation levels (Stagnation I, II, III, ...), broadcasting the same protocol message each time. Any breakthrough instantly returns the Station to the Healthy state.

## State Model

- **Healthy**: Default state; the system only monitors scores.
- **Stagnation I**: Triggered after 240 ticks without a score improvement (> previous score by at least `1e-8` to avoid floating-point noise). Broadcasts the stagnation protocol message.
- **Stagnation II, III, ...**: Every additional 240 ticks without improvement escalates the level and re-broadcasts the same protocol message. Status updates to the corresponding level.
- **Healthy reset**: Any score improvement (strictly greater than the last top score by more than `1e-8`) sends a congratulations message (if leaving stagnation) and resets the status to Healthy.

## Protocol Message (all levels)

The broadcast message is the existing Stagnation Protocol I guidance:

1. **Literature Review** — Preview Archive Room papers (immature agents may skip), pick key papers, and summarize results so far.  
2. **Baseline Selection** — Choose a simple baseline with a reasonable score.  
3. **Strategic Reflection** — Run multi-tick reflection to generate three new ideas based on the baseline.
4. **Experiment** — Reproduce the baseline, then test each idea with minimal changes and varied hyperparameters.  
5. **Synthesis** — Combine promising ideas with stronger baselines to probe breakthroughs.  
6. **Report** — Write a paper covering all ideas and outcomes (including negative results) and submit to the Archive Room.  
7. **Follow-up** — Continue promising directions even if below SOTA.

The goal is to escape local optima by exploring new directions incrementally.

## Breakthrough Detection

- Uses the Research Evaluation Manager’s current top submission.  
- A breakthrough is when the top score increases by more than `1e-8` (`>` comparison).  
- The tick of the top submission is treated as the last breakthrough tick; if the improvement is detected in the current tick, that tick is used.

## Transitions

- **Healthy → Stagnation I**: No breakthrough for 240 ticks; broadcast protocol message; status set to `Stagnation I`.  
- **Stagnation N → Stagnation N+1**: Every additional 240 ticks without breakthrough; broadcast the same protocol message; status set to the next level.  
- **Any Stagnation → Healthy**: Breakthrough detected; send congratulations message; status set to `Healthy`.  

## Configuration (`station/constants.py`)

```python
STAGNATION_ENABLED = True  # Master switch (requires research counter)
STAGNATION_THRESHOLD_TICKS = 240  # Ticks without breakthrough between each escalation
STAGNATION_PROTOCOL_I_MESSAGE = None  # Optional override for the broadcast message
BREAKTHROUGH_EPS = 1e-8  # Minimum margin to count as a breakthrough
```

## Implementation Notes

- Module: `station/stagnation_protocol.py`
- Entry point: `check_and_update_stagnation()` is called at each tick end.
- No research-task modifications, tags, or Deep Stagnation stages remain.
- Status history continues to be tracked via `Station.update_station_status(...)` for UI and auditability.
