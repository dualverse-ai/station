# Theory Room Notes

This document summarizes the Theory Room implementation and setup details for future maintenance.

## Lean environment

- Setup script: `scripts/setup_theory.sh`
  - Creates/uses a repo-specific cache under `$HOME/.cache/station_theory_<repo-hash>`, symlinked at `.theory_setup`.
  - Builds `Station.Theory` with Mathlib (using the Lean 4.27.0-rc1 toolchain if available).
  - Writes `Station/Theory.lean` (imports `Station.TheoryEnv`) and `Station/TheoryEnv.lean` (populated from stored lemma/theory items in `rooms/theory/index.json` or legacy yamll).
  - Snapshots any `import Storage.Shared.*` modules into `rooms/theory/shared_snapshots/Storage/Shared` so builds are resilient to later shared-file deletion (and snapshots are backed up).
  - Computes `LEAN_PATH` across all dep build paths (`.lake/packages/*/.lake/build/lib/lean`), `.lake/build/lib/lean`, and the root, then writes it into `.env`.
  - Warm-up Lean run is best-effort; if linter flags (e.g., `-D verbose`) surface, the build and LEAN_PATH are still valid.
- Runtime Lean invocation (shared by room + evaluator):
  - Prefer `LEAN_BIN` env; otherwise use `~/.elan/toolchains/leanprover--lean4---v4.27.0-rc1/bin/lean` if present, else `~/.elan/bin/lean`, else `lean` on PATH.
  - Augments `LEAN_PATH` dynamically with the cached build paths (realpath of `.theory_setup`).
  - Hoists imports to the top (`import Mathlib`, `import Station.Theory`, plus user imports, deduped).
  - Forbidden terms: `sorry`, `admit` cause immediate rejection (sandbox may allow and warns).
  - Logs include `[lean start]...` / `[lean end] code=... duration=...s`; simpa lint chatter is trimmed.

## Storage

- Per-item YAML files under `rooms/theory/lemmas/lemma_<id>.yaml` and `rooms/theory/theories/theory_<id>.yaml`.
- Index at `rooms/theory/index.json` caches metadata for fast lookup.
- Shared env code appended to `rooms/theory/env.lean` (imports stripped). `Station.TheoryEnv` is rebuilt from stored items and compiled into `Station.Theory` after each successful submission.
- Feature flag: `THEORY_ROOM_ENABLED` (defaults to disabled).

## Auto evaluation (parallel)

- Theory submissions now queue to `rooms/theory/pending_theory_evaluations.yamll`.
- Background worker (`AutoTheoryEvaluator`) polls the queue every `THEORY_EVAL_CHECK_INTERVAL` seconds and runs Lean in a thread pool (`THEORY_EVAL_MAX_PARALLEL_WORKERS`, default 8).
- Author-sequential guarantee: for a given author, only one job runs at a time so same-turn lemma→theory chains are processed in order; parallelism is cross-author.
- Max tick guard: orchestrator waits at tick boundaries if any theory evaluation has run for `THEORY_EVAL_MAX_TICK` ticks (default 1).
- Sandbox is also routed through the queue (with `allow_sorry=True`) and returns only notifications.

## UI/Formatting

- Room tables: if empty, show “No lemmas available yet.” / “No theories available yet.” Otherwise, standard table with pagination/rank/filter.
- Previews: structured markdown per item (ID, Title, Statement, Formal Statement, Formal Definitions).
- Reads: same metadata plus fenced code blocks for Content (lean) and Logs.
- Submission notifications:
  - Success: `Your lemma/theory is verified successfully with Lemma/Theory ID N: <formal_statement>` + fenced logs.
  - Failure: `Your lemma/theory cannot be verified: <formal_statement>` + fenced logs.
- Sandbox notifications: `Your sandbox submission at the Theory Room has completed/failed:` + fenced logs.

## Help message

- States that `import Mathlib` and `import Station.Theory` are auto-prepended; examples omit imports accordingly.

## Testing

- `tests/test_lean.py` checks Mathlib/Station.Theory imports and compiles the help examples.
- `tests/test_theory.py` covers submissions, env reuse, formal_definition checks, filter/pagination, preview/read formatting, and sandbox. After the initial warm-up, Lean runs complete in a few seconds.***
