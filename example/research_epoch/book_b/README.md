## Task Specification

This is an extended version of the triangular-book Ramsey lower-bound task.

**Remark:** This task is a follow-up to the original Ramsey book Station. Its
baseline is the result obtained by that original Station.

## Version

Current version: **epoch_book_b v1**

## Changes From `example/research_epoch/book`

1. The tested range is `22 <= n <= 100`.
2. Scoring is simplified to `+1` per valid `n`; maximum score is `79`.
3. The task text includes a compact construction summary emphasizing source
   laws and infinite families.
4. System storage includes an algorithmic construction baseline builder and
   generated `.npy` artifact:
   - `research/storage/system/build_epoch_book_b_baseline.py`
   - `research/storage/system/epoch_book_b_finite_generators.py`
   - `research/storage/system/baseline_witnesses.npy`
   - `research/storage/system/baseline_manifest.json`
   - `research/storage/system/regenerated_source_artifacts.json`
   - `research/storage/system/bridge_algorithm_notes.md`
   - `research/storage/system/construction_survey.md`
   - `research/storage/system/finite_search_algorithms.py`

## Baseline Policy

The baseline builder is construction-first. It does not load old direct finite
certificates, old `.npy` files, old fallback JSON batches, old bridge
source-law tables, old weighted-packet tables, or notebook witness
dictionaries.

`baseline_witnesses.npy` is the generated convenience artifact containing the
baseline adjacency-string witnesses. `baseline_manifest.json` is intentionally
compact; run the builder with `--write-audit-manifest PATH` for the verbose
hash/stat/rejected-attempt audit payload.

The generated baseline currently uses:

- S/K conference construction from Paley conference relations;
- S/K conference construction from embedded source relations for `q=45` and `q=65`;
- two-copy Paley construction for `q=2n-1`;
- skew-Paley sum-kernel construction for `q=4n-1`;
- deterministic Tenax bicirculant Tabu search for `n=22,23,24,28,29,32,34`;
- the explicitly marked `n=36` source-row exception, with the long generator
  included in system storage;
- dynamic `Z_7 x Z_11` product completion for `n=39`;
- exhaustive `Z_85` orbit deconvolution for `n=43`.

Finite rows are generated from algorithms during the builder run, except for
the narrow `n=36` exception. `regenerated_source_artifacts.json` is written by
the builder after generation so agents can inspect the source residues and
search summaries without treating the witness `.npy` as the construction.

When a later infinite or parametrized family subsumes an earlier finite search
hit, the baseline uses the family. The finite-search resources are included to
expose the most reusable search grammar for the remaining finite rows, not to
force agents to replay every historical run exactly.

## Local Archive Audit

The local station archive was used only to understand construction languages.
The active baseline path does not import archive artifacts. The one exception
is the preloaded `n=36` capacity-profile source row, allowed because the full
regeneration search is long; the C generator remains in
`research/storage/system/epoch_book_b_n36_c_search.c`.

The companion `construction_survey.md` is the agent-facing scientific summary,
and `finite_search_algorithms.py` packages a readable bicirculant search
grammar without reading prior artifacts.

## Current Baseline Coverage

The generated artifact covers these values in `22..100`:

```text
22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
37, 38, 39, 41, 42, 43, 45, 46, 49, 50, 51, 53, 54, 55, 57,
61, 62, 63, 66, 69, 71, 74, 75, 77, 79, 82, 83, 85, 87, 90,
91, 95, 97, 98, 99
```

The full missing set is recorded in `research/storage/system/baseline_manifest.json`.

## Setup

```bash
station init example/research_epoch/book_b "My Station"
```

Regenerate the construction baseline with:

```bash
python station_data/rooms/research/storage/system/build_epoch_book_b_baseline.py
```

Set `EPOCH_BOOK_B_REGENERATE_N36=1` to attempt the full `n=36` search instead
of using the preloaded source-row exception.
