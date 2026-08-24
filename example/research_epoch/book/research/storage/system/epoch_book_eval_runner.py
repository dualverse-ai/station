from __future__ import annotations

import importlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Tuple


TEST_RANGE = tuple(range(22, 51))
TOTAL_TEST_COUNT = len(TEST_RANGE)


def _emit_eval_json(score: float, details: Dict[str, object]) -> None:
    payload = {"score": score, "details": details}
    print(f"EVAL_JSON: {json.dumps(payload, ensure_ascii=False)}", flush=True)


def _run_test_mode(run_module) -> None:
    print("=== Test Mode Detected ===")
    print("Running test() function from submission...")
    test_result = run_module.test()
    print(f"Test completed. Raw Result Type: {type(test_result).__name__}")
    print("TEST_SCORE_MODE: n.a.")
    print("=== Test Mode Complete ===")


@dataclass(frozen=True)
class ParsedGraph:
    n_vertices: int
    edge_count: int
    adjacency_masks: Tuple[int, ...]


@dataclass(frozen=True)
class InstanceStats:
    n: int
    n_vertices: int
    density: float
    red_max: int
    blue_max: int
    red_excess: int
    blue_excess: int
    red_bad_pairs: int
    blue_bad_pairs: int
    red_pair_count: int
    blue_pair_count: int


def parse_graph(text: str, expected_vertices: int) -> ParsedGraph:
    if not isinstance(text, str):
        raise AssertionError("solution_batch() values must be adjacency strings")

    bits = "".join(text.split())
    if not bits:
        raise AssertionError("adjacency string must be non-empty")
    if any(ch not in "01" for ch in bits):
        raise AssertionError("adjacency string must contain only 0/1 characters")

    expected_len = expected_vertices * (expected_vertices - 1) // 2
    if len(bits) != expected_len:
        raise AssertionError(
            f"adjacency string length must be C({expected_vertices}, 2) = {expected_len}, got {len(bits)}"
        )

    masks = [0] * expected_vertices
    edge_count = 0
    idx = 0
    for j in range(1, expected_vertices):
        for i in range(j):
            bit = bits[idx]
            idx += 1
            if bit == "1":
                masks[i] |= 1 << j
                masks[j] |= 1 << i
                edge_count += 1

    return ParsedGraph(
        n_vertices=expected_vertices,
        edge_count=edge_count,
        adjacency_masks=tuple(masks),
    )


def analyze_instance(graph: ParsedGraph, n: int) -> InstanceStats:
    total_pairs = graph.n_vertices * (graph.n_vertices - 1) // 2
    all_mask = (1 << graph.n_vertices) - 1
    complement_masks = tuple((all_mask ^ (1 << u)) ^ graph.adjacency_masks[u] for u in range(graph.n_vertices))

    red_target = n - 2
    blue_target = n - 1
    red_max = 0
    blue_max = 0
    red_bad_pairs = 0
    blue_bad_pairs = 0

    for j in range(1, graph.n_vertices):
        mask_j = graph.adjacency_masks[j]
        comp_j = complement_masks[j]
        for i in range(j):
            if (mask_j >> i) & 1:
                common = (graph.adjacency_masks[i] & mask_j).bit_count()
                if common > red_max:
                    red_max = common
                if common > red_target:
                    red_bad_pairs += 1
            else:
                common = (complement_masks[i] & comp_j).bit_count()
                if common > blue_max:
                    blue_max = common
                if common > blue_target:
                    blue_bad_pairs += 1

    blue_pair_count = total_pairs - graph.edge_count
    density = graph.edge_count / total_pairs if total_pairs else 0.0
    return InstanceStats(
        n=n,
        n_vertices=graph.n_vertices,
        density=density,
        red_max=red_max,
        blue_max=blue_max,
        red_excess=max(0, red_max - red_target),
        blue_excess=max(0, blue_max - blue_target),
        red_bad_pairs=red_bad_pairs,
        blue_bad_pairs=blue_bad_pairs,
        red_pair_count=graph.edge_count,
        blue_pair_count=blue_pair_count,
    )


def get_batch_value(batch_result: Mapping, n: int) -> str:
    if n in batch_result:
        return batch_result[n]
    text_key = str(n)
    if text_key in batch_result:
        return batch_result[text_key]
    raise AssertionError(f"solution_batch() missing required key {n!r} or {text_key!r}")


def evaluate_solution_family() -> Tuple[float, Dict[str, object]]:
    run_module = importlib.import_module("run")
    solution_batch_fn = getattr(run_module, "solution_batch", None)
    if solution_batch_fn is None:
        raise AssertionError("run.py must define solution_batch() -> dict[int | str, str]")

    valid_count = 0
    worst_red_excess = 0
    worst_blue_excess = 0
    worst_n = TEST_RANGE[0]
    worst_pair_rate = 0.0
    worst_norm = -1.0
    first_failure_message = None
    start_time = time.time()

    batch_started = time.time()
    batch_result = solution_batch_fn()
    batch_elapsed_sec = time.time() - batch_started
    if not isinstance(batch_result, Mapping):
        raise AssertionError("solution_batch() must return a dict-like mapping")

    for n in TEST_RANGE:
        instance_started = time.time()
        elapsed_min = (time.time() - start_time) / 60.0
        try:
            expected_vertices = 4 * n - 2
            raw = get_batch_value(batch_result, n)
            graph = parse_graph(raw, expected_vertices)
            stats = analyze_instance(graph, n)
            red_rate = stats.red_bad_pairs / stats.red_pair_count if stats.red_pair_count else 0.0
            blue_rate = stats.blue_bad_pairs / stats.blue_pair_count if stats.blue_pair_count else 0.0
            pair_rate = max(red_rate, blue_rate)
            norm_violation = max(
                stats.red_excess / max(1.0, n - 1.0),
                stats.blue_excess / max(1.0, float(n)),
            )

            if norm_violation > worst_norm:
                worst_norm = norm_violation
                worst_n = n
                worst_pair_rate = pair_rate

            worst_red_excess = max(worst_red_excess, stats.red_excess)
            worst_blue_excess = max(worst_blue_excess, stats.blue_excess)

            valid = stats.red_excess == 0 and stats.blue_excess == 0
            if valid:
                valid_count += 1

            status = "PASS" if valid else "FAIL"
            print(
                (
                    f"[{elapsed_min:7.2f}m] n={n:3d} {status} "
                    f"|V|={stats.n_vertices} "
                    f"red_max={stats.red_max} tgt={n - 2} "
                    f"blue_max={stats.blue_max} tgt={n - 1} "
                    f"red_bad={stats.red_bad_pairs} blue_bad={stats.blue_bad_pairs} "
                    f"dens={stats.density:.3f} "
                    f"batch={batch_elapsed_sec / 60.0:.2f}m "
                    f"validate={(time.time() - instance_started) / 60.0:.2f}m"
                ),
                flush=True,
            )

            if not valid and first_failure_message is None:
                first_failure_message = (
                    f"First failing n={n}: "
                    f"red_max={stats.red_max} (target {n - 2}), "
                    f"blue_max={stats.blue_max} (target {n - 1})."
                )
        except Exception as exc:
            print(
                f"[{elapsed_min:7.2f}m] n={n:3d} ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
            if first_failure_message is None:
                first_failure_message = f"First failing n={n}: {type(exc).__name__}: {exc}"
            worst_n = n
            worst_pair_rate = 1.0
            worst_norm = math.inf

    solved = valid_count == TOTAL_TEST_COUNT
    if solved:
        message = f"Solved on the full evaluation range {TEST_RANGE[0]}..{TEST_RANGE[-1]}."
    else:
        message = (
            f"Passed {valid_count}/{TOTAL_TEST_COUNT} tested n values on range "
            f"{TEST_RANGE[0]}..{TEST_RANGE[-1]}. "
            f"{first_failure_message or 'At least one instance failed exact validation.'}"
        )

    details = {
        "Message": message,
        "ValidCount": int(valid_count),
        "WorstRedExcess": int(worst_red_excess),
        "WorstBlueExcess": int(worst_blue_excess),
        "WorstN": int(worst_n),
        "WorstViolatingPairRate": float(worst_pair_rate),
    }
    return (100.0 * valid_count / TOTAL_TEST_COUNT), details


def main() -> None:
    try:
        run_module = importlib.import_module("run")
        if hasattr(run_module, "test"):
            _run_test_mode(run_module)
            return
        if getattr(run_module, "solution_batch", None) is None:
            raise AssertionError("run.py must define solution_batch() -> dict[int | str, str]")
        score, details = evaluate_solution_family()
        _emit_eval_json(score, details)
    except Exception as exc:
        _emit_eval_json(
            0.0,
            {
                "Message": f"Runner failure: {type(exc).__name__}: {exc}",
                "ValidCount": 0,
                "WorstRedExcess": 0,
                "WorstBlueExcess": 0,
                "WorstN": TEST_RANGE[0],
                "WorstViolatingPairRate": 1.0,
            },
        )


if __name__ == "__main__":
    main()
