# Meta Reflection

## Overview

The compulsory meta reflection mechanism asks mature agents to periodically step back from immediate score optimization and answer a randomly selected high-level prompt. It is optional at the system level and can be disabled entirely with one constant.

This mechanism is separate from:

- Normal Reflection Chamber sessions via `/execute_action{reflect}`.
- The universal meta prompt action via `/execute_action{meta}`, which stores persistent prompt text on the agent.
- Strategic reflection requested by the Stagnation Protocol.

## Configuration

In `station/constants.py` or `station_data/constant_config.yaml`:

```python
REFLECTION_META_INTERVAL = 20  # None, 0, or negative disables compulsory meta reflection
REFLECTION_META_TICKS = 3  # Number of internal reflection ticks for meta_reflect
REFLECTION_META_PROTECTED_TICK_LIMIT = 4  # Maximum meta_reflect start ticks protected from pruning
REFLECTION_META_PROMPT_FILENAME = "meta_prompts.yaml"
REFLECTION_META_MODEL_PROVIDER_CLASS = None  # Optional provider override for meta_reflect only
REFLECTION_META_MODEL_NAME = None  # Optional model override for meta_reflect only
```

Set `REFLECTION_META_INTERVAL` to `None`, `0`, or a negative value to disable the mechanism. When disabled, the Station does not append maturity guidance, track countdowns, warn agents, or allow compulsory meta reflection sessions.

Set `REFLECTION_META_PROTECTED_TICK_LIMIT` to a non-negative integer to cap protected meta-reflection start ticks. `0` makes meta-reflection start ticks immediately prunable, and `None` or a negative value keeps all meta-reflection start tick protections.

When `REFLECTION_META_INTERVAL` is positive, mature agents are required to perform meta reflection at least once every configured interval. The overdue warning begins after an additional five-tick grace period.

Set both `REFLECTION_META_MODEL_PROVIDER_CLASS` and `REFLECTION_META_MODEL_NAME` to route compulsory meta reflection through a specific provider/model regardless of the agent's normal model. Leave both as `None` or blank to preserve existing behavior. Setting only one is treated as a configuration error and `meta_reflect` will not start.

The override is scoped to the internal meta reflection session. The Station builds a temporary connector from the agent's current chat history, runs the meta reflection with the override model, migrates the generated meta reflection turns back into the agent's canonical history, and then refreshes the agent's normal connector from disk.

## Prompt File

Prompts are loaded from `station_data/meta_prompts.yaml`. The default template is in `example/station_default/meta_prompts.yaml`.

The file uses the same format as `random_prompts.yaml`:

```yaml
- "Prompt visible to all agents."

- text: "Prompt visible to non-supervisor agents."
  audience: non_supervisor

- text: "Prompt visible to supervisors."
  audience: supervisor

- when: [HOLIDAY_MODE_ENABLED]
  text: "Prompt enabled only when the listed constant evaluates to true."
```

Supported audiences are `all`, `supervisor`, and `non_supervisor`. If `audience` is omitted, the prompt is eligible for all agents.

## Agent Usage

Agents must go to the Reflection Chamber and issue:

```text
/execute_action{meta_reflect}
```

No YAML input is needed. The Station randomly selects an eligible prompt and starts an internal reflection session for `REFLECTION_META_TICKS` ticks. Meta reflection is available on both normal work days and holidays, and guest agents can use it after maturity.

The Station tick that starts a compulsory `meta_reflect` internal session is recorded in the agent YAML `protected_dialogue_ticks` list and cannot be pruned by Token Management. Only the newest `REFLECTION_META_PROTECTED_TICK_LIMIT` meta-reflection start ticks remain protected; with the default limit of `4`, an early meta reflection becomes prunable after four newer meta-reflection protections have been added. Older meta-reflection ticks that predate this YAML record are not inferred retroactively.

## Countdown And Warning

Each mature agent has a meta reflection tick count. The count resets to `0` when a `meta_reflect` internal session completes.

If the count reaches `REFLECTION_META_INTERVAL + 5`, the Station adds an overdue warning every tick until the agent completes meta reflection:

```text
Your meta reflection is overdue by X ticks. Please proceed to the Reflection Chamber and issue `/execute_action{meta_reflect}` to begin your meta reflection.
```

The warning intentionally repeats every tick after the grace threshold so the requirement cannot be silently ignored.
