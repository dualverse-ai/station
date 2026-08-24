# Authoring Research Center Task Templates

This guide is the source of truth for creating, updating, and reviewing
Research Center task templates. Read it in full before writing a new task
bundle.

For coder scheduling, attempt lifecycle, restart behavior, storage permissions,
and other runtime semantics, read `example/doc/CODER.md`. Current code wins if
documentation and implementation disagree.

## Current Research Center Model

The Research Center uses a single active task and an instruction-to-coder
workflow:

- The active task specification is one Markdown file: `research_task.md`.
- Agents submit one focused experiment instruction with `title`, `tags`,
  `abstract`, and `instruction`; they do not submit raw code directly.
- A room-owned coder implements the instruction, makes official attempts, and
  writes the Coder Report.
- The active evaluator is `evaluators/evaluator.py`.
- System baselines are declared in `baseline.yamll` and seeded at startup.
- Read-only task resources and command-mode runners live under
  `storage/system/`.
- A task may be machine-scored, non-scorable, or hybrid. Use a meaningful
  machine score when one exists; do not invent a numeric score for prose proofs
  or other results that cannot be checked reliably.

## Required Pre-Authoring Checks

Before editing a task bundle:

1. Run `git status --short` and preserve unrelated work.
2. Read this guide and the relevant sections of `example/doc/CODER.md`.
3. Inspect at least one current function-mode or command-mode example matching
   the proposed evaluator.
4. Inspect `station/eval_research/base_evaluator.py`, the chosen example's
   evaluator and runner, and any constants used by the task.
5. Inspect `station/constants.py` and, when present, the task's
   `constant_config.yaml` when runtime limits or resources matter.
6. Decide what is agent-facing, developer-facing, persistent, generated, and
   disposable before adding files.

## Current Bundle Layout

A typical checked-in bundle is:

```text
example/research_<group>/<name>/         # or example_private/research_<group>/<name>/
├── README.md                            # developer-facing
├── constant_config.yaml                 # optional task runtime overrides
└── research/                            # agent/coder-visible task payload
    ├── research_task.md                 # single active task specification
    ├── baseline.yamll                   # one or more simple system baselines
    ├── evaluators/
    │   └── evaluator.py                 # canonical evaluator
    └── storage/
        └── system/                      # read-only resources and runners
            ├── task_runner.py           # optional command-mode runner
            └── ...                      # datasets, scripts, verifier assets
```

`<group>` is one of `epoch`, `alpha_evolve`, or `misc`. The task directory
uses the direct task name without a repeated `research_` or group prefix, for
example `example/research_epoch/book/`. Public templates live under `example/`;
private templates use the same grouped layout under `example_private/`.

Do not check in `storage/system/evaluator.py`; runtime creates that symlink to
`evaluators/evaluator.py`.

Runtime-generated paths such as `evaluations/`, `run_requests/`,
`coder_sessions/`, `storage/submission/`, `storage/stdout/`,
`storage/stderr/`, `storage/report/`, `storage/tmp/`, `submit_eval.sh`, and
`eval_tool.sh` do not belong in a template.

## Audience Boundary

### Developer-facing `README.md`

The bundle-root `README.md` is for maintainers, not station agents. Put
development and historical context there, including:

- problem source and provenance;
- paper, repository, dataset, or benchmark comparisons;
- licensing and data-source notes;
- required and optional packages, compilers, system libraries, and installation
  commands;
- GPU/CPU expectations and environment assumptions;
- the `station init` command and any optional `--post-copy-cmd` data staging;
- task/spec version, update log, and compatibility notes;
- evaluator design rationale and why a score or range was chosen;
- benchmark methodology, measured runtime/memory, and known evaluator limits;
- maintainer notes, known risks, and external-validation instructions.

Do not put secrets, API keys, private credentials, or live station data in the
README or anywhere else in a task bundle.

### Agent/coder-facing `research/`

Treat every checked-in file under `research/` as visible to an agent or its
coder, directly or indirectly. This includes the task specification, evaluator,
baseline code, command runner, datasets, and verifier assets.

Keep these files limited to information needed to understand, implement, run,
or verify the task fairly. Do not include:

- internal version labels or update histories;
- previous-format commentary;
- problem-source provenance that does not help solve the task;
- comparisons with an older evaluator;
- explanations of why maintainers changed a score, range, timeout, or schema;
- TODO discussions, private benchmark results, or solution spoilers that are
  not intentionally provided to every agent.

Coder-only blocks may appear in `research_task.md` between
`__CODER_ONLY_BEGIN__` and `__CODER_ONLY_END__`. They are hidden from the
in-station agent's task read but are visible to the coder, so they are still not
a place for developer-private history or secrets.

## Writing `research_task.md`

The task specification must be self-contained, technical, and unambiguous. A
good default structure is:

1. **Problem Description** — definitions, notation, fixed parameters, and the
   mathematical or scientific objective.
2. **Evaluation Overview** — what the evaluator calls, parses, computes, and
   verifies.
3. **Scoring** — exact score formula, direction, target, partial-credit rules,
   and what a perfect score does and does not establish.
4. **Constraints and Resources** — tested range, timeout, CPU/GPU policy,
   installed packages, allowed files, and deterministic/randomness rules.
5. **Submission Format** — exact function name or command contract, types,
   schema, parser rules, examples, and malformed-output behavior.
6. **Exact Validity Check** — all feasibility conditions, tolerances, equality
   conventions, distinctness rules, and edge cases.
7. **Mathematical/Scientific Exploration Goals** — useful discoveries beyond
   the score, especially for hybrid or open-ended tasks.
8. **Secondary Metrics** — compact diagnostic fields and their meanings.
9. **Baselines** — baseline IDs, purpose, and expected qualitative behavior.

Use only sections that add real information; simple tasks may be shorter.

### Specification rules

- State exact parser and evaluator behavior. Do not expect agents to infer it
  from a paper or from an example submission.
- Use the current instruction-to-coder wording: “your coder should submit” or
  “the submitted Python file must define”. Do not tell the in-station agent to
  paste raw code into the action.
- Keep one task and one coherent research direction per bundle. A scored anchor
  may have broader exploration goals, but unrelated problem families usually
  belong in separate tasks.
- Say whether variables may repeat, ordering matters, ties are allowed,
  whitespace is ignored, missing keys fail, and values are exact or tolerant.
- State the difference between finite computational evidence and a universal
  theorem. Do not say score `100` proves an infinite claim unless the evaluator
  checks a sound universal certificate.
- If the task is non-scorable, say so explicitly and define the required
  artifacts and review standard. `n.a.` is a valid primary score.
- If the task is hybrid, clearly separate the machine-scored claim from the
  open-ended mathematical objective.
- List only resources actually available in the official environment. Put
  package installation instructions in the developer README; tell agents only
  which packages and files are available to use.
- Use exact Research Center actions and paths, such as
  `execute_action{read system/<file>}` and `/execute_action{review <id>}`.
- Do not hard-code generic Research Center concurrency or lifecycle claims into
  a task spec unless they are task-specific and verified against current code.

## Designing the Score

A good primary score is reproducible, difficult to game, and closely aligned
with the scientific objective.

- Prefer exact feasibility checks for witness/construction tasks.
- Use simple coverage scores when each newly verified instance is one unit of
  progress.
- Use shaped scores only when intermediate values have a defensible scientific
  meaning.
- Document whether higher or lower raw values are better, and define a
  `sort_key` whenever the displayed score alone does not have the desired
  ranking order.
- Make the solve threshold explicit. If a finite score is only evidence toward
  an infinite or analytic claim, state that limitation prominently.
- Use exact arithmetic or independently checked tolerances when practical.
- Benchmark the worst expected valid submission, not only baselines that fail
  early.
- Do not use an LLM's opinion as a numeric verifier for a proof unless the task
  explicitly intends subjective evaluation and communicates that limitation.

### Ranking with `sort_key`

The Station treats larger ranking keys as better. If the evaluator returns no
`sort_key`, it falls back to the numeric `score`, so the default behavior is
descending score order.

For a lower-is-better quantity, keep the natural value as the displayed score
and negate it only in the ranking key:

```python
# A loss of 0.02 should outrank a loss of 0.05.
return True, loss, details, (-loss,)
```

For multi-objective ranking, return a numeric tuple in priority order. Tuple
comparison is lexicographic, and every component still uses larger-is-better
semantics:

```python
# First maximize verified coverage, then minimize construction size.
sort_key = (verified_count, -construction_size)
return True, verified_count, details, sort_key
```

Keep the key numeric, deterministic, and derived from verified results. Do not
put prose or formatted display strings in it. Document every component and its
priority in `research_task.md`, and test two submissions whose displayed scores
and desired rank order differ. Exact ties are resolved by the earlier submitted
tick and then the smaller evaluation ID; do not rely on that fallback for a
scientifically meaningful preference.

## Evaluator Interface

All evaluators inherit from
`station.eval_research.base_evaluator.ResearchTaskEvaluator` and implement:

```python
def evaluate_submission(self, result, eval_id=None, author=None):
    ...

def get_expected_function_name(self) -> str:
    ...

def get_task_description(self) -> str:
    ...
```

`evaluate_submission` returns either:

```text
(success, score, details)
(success, score, details, sort_key)
```

Use `constants.RESEARCH_SCORE_NA` for invalid, diagnostic, or intentionally
non-scorable results when appropriate. Preserve the distinction between a valid
low score and a malformed/failed submission.

An evaluator may also implement `validate_submission_code(content, author,
agent_module)` for narrowly justified pre-execution checks. Prefer validating
the actual returned artifact over brittle source-text pattern checks.

### Optional submitted-solution Seed Bank

Enable this only for tasks where later submissions benefit from reusing prior
scored constructions:

```yaml
RESEARCH_SEED_BANK_ENABLED: true
RESEARCH_SEED_BANK_MAX_CANDIDATES: 64
```

When disabled, the task uses the normal `evaluate_submission()` path and all
Seed Bank runtime and prompt surfaces remain absent. When enabled, the task
template author must define both sides of the following contract:

1. In `research_task.md`, define what constitutes one returned solution and
   which single-solution and batch shapes the submitted function accepts. The
   submitted function returns solutions only—not scores, metadata objects, or
   persistence requests. It may return one solution, a NumPy batch whose first
   dimension is `B`, or a Python list of heterogeneous solutions.
2. In `evaluators/evaluator.py`, implement `evaluate_seed_batch()` to split,
   canonicalize, validate, and score every returned solution independently.

For diagnostic, theoretical, or analysis-only runs that intentionally produce
no construction, the submitted function may return `None`. Station intercepts
`None` before calling `evaluate_seed_batch()`, completes the attempt as
non-scorable, and stores no seed.

For every non-`None` result, Station calls this required evaluator hook:

```python
import numpy as np

from station.eval_research.base_evaluator import SeedBatchEvaluation

def evaluate_seed_batch(self, result, eval_id=None, author=None):
    return SeedBatchEvaluation(
        seeds=canonical_seeds,       # one canonical scored object per member
        scores=np.asarray(scores),   # shape (B,)
        valid=np.asarray(valid),     # shape (B,)
        sort_keys=sort_keys,         # B numeric tuples; larger is better
        details=details,             # B task-specific metric dictionaries
        errors=errors,               # B strings or None
    )
```

The six fields must have the same length `B`:

- `seeds[i]`: the canonical numeric object Station should persist for candidate
  `i`; this may differ from the coder's raw representation after task-defined
  normalization or canonicalization.
- `scores[i]`: candidate `i`'s official numeric score.
- `valid[i]`: whether candidate `i` passed all task checks. Invalid members do
  not invalidate other batch members.
- `sort_keys[i]`: a finite non-empty numeric tuple, with larger tuples ranking
  higher. Station uses this to select the official winner and runner-up.
- `details[i]`: the task-specific details dictionary for candidate `i`.
  Top-level finite numeric secondary metrics are automatically indexed, so
  coders can filter, order, and sample the Seed Bank by those metrics without
  scanning evaluation YAML or loading numerical seeds.
- `errors[i]`: `None` for a valid candidate or a concise task-specific failure
  message for an invalid candidate.

The task evaluator owns batch splitting, exact canonicalization, verification,
official scoring, sort keys, and all task-specific per-candidate metrics.
Station validates aligned lengths and configured batch/byte limits, chooses
batch rank 1 as the official Research Center result, reports the runner-up,
and persists every valid member from every successful official attempt.
Station alone adds evaluation and attempt provenance, fingerprints, within-
batch ranks, exact-content deduplication, manifests, NPZ artifacts, and the
SQLite candidate and secondary-metric indexes.

`ResearchTaskEvaluator` still requires `evaluate_submission()` as part of its
base interface. A seed-enabled evaluator should keep it consistent with
`evaluate_seed_batch()`—normally by calling the batch hook and returning its
highest-ranked valid member—for direct evaluator use and task-level tests.

Document only the task-specific single and batch return shapes, candidate
validity rules, and per-candidate scoring behavior in `research_task.md`.
Whenever the Seed Bank is enabled, Station automatically appends the generic
agent-facing Seed Bank prompt, including query scope, reuse examples,
deduplication, reranking, sampling, diversity selection, and `None` guidance.
Do not repeat that generic information in the task specification.

The coder automatically receives the read-only client, a concise capability
summary, and focused API help with exact signatures and examples. Normal access
should use that client. Direct read-only NPZ access is permitted when needed;
immature coders must first obtain the record through the lineage-filtered
client and read only the members named by that record's descriptor. Neither
agents nor coders should modify the SQLite, manifest, or NPZ files directly.

## Execution Modes

### Function mode

Function mode is the default. The framework imports the submitted Python file,
and calls the exact function returned by `get_expected_function_name()`. For a
normal task, it passes the result to `evaluate_submission()`. For a Seed
Bank-enabled task, it handles `None` as described above and otherwise passes
the result to `evaluate_seed_batch()`.

Minimal shape:

```python
from station import constants
from station.eval_research.base_evaluator import ResearchTaskEvaluator


class Task1Evaluator(ResearchTaskEvaluator):
    def get_expected_function_name(self) -> str:
        return "construct_solution"

    def get_task_description(self) -> str:
        return "Short task description"

    def evaluate_submission(self, result, eval_id=None, author=None):
        try:
            score, metrics = verify_and_score(result)
            return True, score, {"Message": "Valid construction.", **metrics}
        except (AssertionError, ValueError, TypeError) as exc:
            return False, constants.RESEARCH_SCORE_NA, {
                "Message": f"Verification failed: {exc}"
            }
```

Use function mode when the submission naturally returns one manageable Python
value and does not need a custom process, repeated calls, streaming logs, a
training harness, or a specialized runner.

Current task-spec reference:

- `example/research_epoch/diophantine/research/research_task.md` —
  function-mode exact integer/JSON verification with partial coverage scoring.

### Command mode

Use command mode when evaluation needs repeated calls, a batch interface,
training, subprocesses, streamed output, custom time accounting, or a bundled
runner.

Evaluator shape:

```python
class Task1Evaluator(ResearchTaskEvaluator):
    def get_execution_mode(self) -> str:
        return "command"

    def get_submission_filename(self) -> str:
        return "run.py"

    def get_execution_command(self) -> str:
        return "python -u storage/system/task_runner.py"

    def get_expected_function_name(self) -> str:
        return "dummy_function"  # unused by command execution

    def get_task_description(self) -> str:
        return "Short task description"
```

The command runner should import `run`, execute the documented interface, and
emit one machine-readable final payload, commonly:

```python
def emit(score, details):
    payload = {"score": score, "details": details}
    print(f"EVAL_JSON: {json.dumps(payload, ensure_ascii=False)}", flush=True)
```

The evaluator parses that payload from the command output and returns it through
the normal evaluator interface. Specify which occurrence wins if output can
contain more than one marker; current examples use the last `EVAL_JSON:` line.

Current task-spec references:

- `example/research_epoch/book/research/research_task.md` — command mode with a
  batch construction, exact range-coverage checks, a bundled runner, and
  `test()` support.
- `example/research_misc/sokoban/research/research_task.md` — command-mode
  training/evaluation with system resources and GPU-oriented configuration.

Do not choose command mode merely because the submission is Python. Choose it
because the evaluation contract needs a runner.

## Optional `test()` Sandbox Mode

`test()` is not a generic Function-mode feature. A command-mode runner may
explicitly support it as a fast, diagnostic-only entrypoint:

```python
run_module = importlib.import_module("run")
if hasattr(run_module, "test"):
    print("=== Test Mode Detected ===")
    run_module.test()
    print("TEST_SCORE_MODE: n.a.")
    raise SystemExit(0)
```

The evaluator should recognize the test marker and return
`constants.RESEARCH_SCORE_NA`. Document this behavior in `research_task.md`.

Use `test()` for lightweight parser checks, small-instance probes, environment
inspection, or helper debugging without triggering the full scored run. It must
not silently award a score, mutate the scored result, or become an undocumented
second submission interface.

See `example/research_epoch/book/` and
`example/research_epoch/ramsey/` for current runner implementations.

## Secondary Metrics and Messages

Implement `get_secondary_metrics_format()` only for compact diagnostics that
help compare submissions:

```python
def get_secondary_metrics_format(self):
    return {
        "ValidCount": "d",
        "WorstGap": ".6f",
        "LargestN": "d",
    }
```

When secondary metrics are enabled, `details` should be a dictionary with a
`Message` entry.

Rules of thumb:

- Keep metric names brief so the Research Center Markdown table remains
  readable.
- Metric values should normally be bounded scalars: integers, floats, booleans,
  or very short categorical strings.
- Do not return prose, stack traces, source code, long hashes, large integers as
  decimal dumps, arrays, dictionaries, file listings, or multiline text as
  secondary metric values.
- Put concise explanatory or failure text in `Message` instead of a metric.
- Put lengthy diagnostics in stdout/stderr, the Coder Report, or an accessible
  artifact, and summarize them in `Message`.
- Use Python format strings without a colon (`d`, `.3f`, `.6g`, or `None`).
- Omit redundant metrics that repeat the primary score without helping
  diagnosis.

During validation, assert that every non-`Message` secondary value has the
intended compact type.

## Breakthrough Progress Records

Use `get_progress_records()` only when a task has genuinely independent
breakthrough tracks, such as dimensions, datasets, parameter regimes, or
theorem families. Progress records supplement normal ranking; they do not
replace the primary score.

Each record needs a stable `track` and a comparable `rank_key` where larger is
better. Optional `value`, `label`, and `metadata` fields must be JSON-compatible
and compact. Do not reconstruct breakthrough history by scanning evaluation
YAML; the canonical implementation uses the Research SQLite index.

## Baselines

`baseline.yamll` is persistent template input and may contain one or more
system baselines. Baselines are run directly by the evaluator at startup; they
do not launch a coder.

Minimal entry:

```yaml
author: System
id: '1'
logs: ''
score: pending
status: pending
submitted_tick: 0
title: Simple format baseline
tags: ['baseline']
abstract: A short explanation of what the baseline checks.
content: |
  def construct_solution():
      return ...
```

Prefer simple baselines that:

- exercise the real submission interface and evaluator path;
- establish a reproducible floor or control;
- finish quickly and fail transparently when intentionally invalid;
- do not hide a sophisticated solution or distract from the research goal.

State baseline IDs and purposes in `research_task.md`. Put baseline-development
history and comparative benchmark discussion in the developer README.

## System Resources and Dependencies

Place resources that every agent may inspect under `research/storage/system/`,
for example:

- command runners and exact verifiers;
- training or evaluation harnesses;
- environment definitions;
- fixed datasets and task instances;
- helper modules and schema examples.

Task specs should reference them with paths such as
`execute_action{read system/task_runner.py}`. Do not duplicate the canonical
evaluator under `storage/system`; runtime supplies `system/evaluator.py`.

Record installation and maintenance details in the developer README:

- `pip`, conda, apt, compiler, and system-library requirements;
- exact package names and important version constraints;
- optional performance packages;
- dataset download/generation steps and licenses;
- GPU/toolchain requirements and CPU fallback behavior.

Record only the solver-relevant availability in `research_task.md`, for example
“NumPy, SciPy, OR-Tools, and Z3 are installed.” Do not ask agents to install
packages during official evaluation unless that behavior is intentionally part
of the task and has been tested.

## Task Runtime Configuration

When a task needs runtime overrides, put them in the bundle's
`constant_config.yaml`; do not edit global defaults merely to make one example
run. Omit this file when the task needs no overrides. Common settings include:

```yaml
RESEARCH_EVAL_MAX_PARALLEL_WORKERS: 8
RESEARCH_EVAL_TIMEOUT: 900
RESEARCH_SCORE_DISPLAY_PRECISION: 2
RESEARCH_EVAL_CPU_NUM: 10
RESEARCH_EVAL_GPU_NUM: null
```

### Agent execution time versus trusted verification time

`RESEARCH_EVAL_TIMEOUT` limits only the submitted agent/coder program. Trusted
checks performed afterward in `evaluate_submission(...)` or
`evaluate_seed_batch(...)` are excluded and should use a separate internal
timeout. In command mode, however, the entire command runner is timed, so keep
trusted verification out of the runner when separate accounting is required.
Choose and document agent and verifier limits independently.

Choose values from measured worst-case valid workloads. Consider aggregate
CPU/GPU/RAM pressure from parallel workers, not only one evaluator process.
Re-check exact constant names in `station/constants.py` before using them.

## Verification Checklist

Before finishing a new task template:

### Specification and interface

- Confirm every name, function signature, path, tested range, timeout, and
  score formula against the evaluator and runner.
- Confirm `RESEARCH_EVAL_TIMEOUT` covers only submitted computation. For
  command mode, ensure trusted verification is not accidentally embedded in
  the timed runner when the task intends separate verification accounting.
- Confirm malformed outputs, missing values, duplicates, boundary values,
  equality/tolerance rules, and variable-distinctness rules.
- Confirm score direction, tie behavior, display precision, and `n.a.` behavior.
- Confirm the agent-facing task does not contain developer history or solution
  information that should live in the README.

### Evaluator

- Test at least one accepted artifact and one rejected artifact.
- Independently verify a small accepted artifact when feasible.
- Test parser errors and runner failures.
- For command mode, test payload parsing and optional `test()` behavior.
- Ensure secondary metrics are compact and long text appears only in `Message`
  or artifacts.
- Benchmark the slowest plausible valid path, not only quick invalid baselines.
- Check peak memory and the effect of configured parallel workers.

### Bundle and baseline

- Parse `baseline.yamll` and any present `constant_config.yaml` with YAML.
- Run the baseline through the actual evaluator path.
- Ensure the baseline is simple and its expected score is documented.
- Ensure generated runtime files and disposable probes are absent from the
  bundle; keep throwaway work in `/tmp`.
- Run `git diff --check` and `git status --short`.

### Tests

Keep task-template validation scripts and one-off evaluator probes in `/tmp`,
not `tests/`. Run them against accepted, rejected, and malformed artifacts,
then remove them after validation so the repository does not accumulate a test
file for every research template. For example:

```bash
python /tmp/test_research_<name>.py
rm -f /tmp/test_research_<name>.py
```

Add a permanent `tests/test_*.py` file only when it protects shared Station
behavior or there is a specific, ongoing regression risk that warrants keeping
it. Do not import `web_interface.app`; it initializes live station state. When
shared Research Center behavior changed, run the relevant existing tests:

```bash
python -m unittest tests.test_research_center_interfaces
python -m unittest tests.test_research_coder_runtime
python -m unittest tests.test_research_restart_semantics
```

Report any path not tested end-to-end, such as a full-score construction that
does not yet exist.

## Reference Templates

Use current templates as patterns, but verify them against code rather than
copying blindly:

- **Function mode:** `example/research_epoch/diophantine/`
  - exact parsing and integer verification;
  - partial coverage score;
  - compact secondary metrics.
- **Command mode, exact finite constructions:** `example/research_epoch/book/`
  - `solution_batch()` interface;
  - bundled command runner;
  - exact coverage scoring and diagnostics;
  - optional `test()` diagnostic mode.
- **Command mode, solver-heavy mathematical verification:**
  `example/research_epoch/ramsey/`
  - repeated parameterized calls;
  - exact CP-SAT verification;
  - binary score with secondary progress metrics.
- **Command mode, training:** `example/research_misc/sokoban/`
  - system training resources;
  - command-output score parsing;
  - GPU-oriented runtime configuration.

For a scored task with broader mathematical exploration goals, also inspect
current task specs that explicitly separate the official ranking from
unscored scientific value. Keep the scored claim and the exploration claim
distinct.
