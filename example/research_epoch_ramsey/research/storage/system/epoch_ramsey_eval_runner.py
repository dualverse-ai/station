from __future__ import annotations

import importlib
import json
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Set, Tuple

from ortools.sat.python import cp_model


TEST_RANGE = tuple(range(15, 51))
EDGE_RE = re.compile(r"\{([^{}]+)\}")


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


@lru_cache(maxsize=None)
def k_value(n: int) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if n == 1:
        return 1
    a = n // 2
    b = (n + 1) // 2
    return a + k_value(a) + k_value(b)


@dataclass(frozen=True)
class ParsedHypergraph:
    vertex_count: int
    edges: Tuple[Tuple[int, ...], ...]
    incidences: Tuple[Tuple[int, ...], ...]


def parse_hypergraph(text: str) -> ParsedHypergraph:
    if not isinstance(text, str):
        raise AssertionError("solution(n) must return a string")

    matches = list(EDGE_RE.finditer(text))
    if not matches:
        raise AssertionError("hypergraph string must contain at least one edge")

    normalized_edges: List[Tuple[int, ...]] = []
    seen_edges: Set[Tuple[int, ...]] = set()
    used_vertices: Set[int] = set()

    for match in matches:
        items = [part.strip() for part in match.group(1).split(",")]
        if not items or any(item == "" for item in items):
            raise AssertionError("edges must be non-empty and comma-separated")

        edge_vertices: List[int] = []
        local_seen: Set[int] = set()
        for item in items:
            if not item.isdigit():
                raise AssertionError("vertex labels must be positive integers")
            value = int(item)
            if value <= 0:
                raise AssertionError("vertex labels must be positive integers")
            if value in local_seen:
                raise AssertionError("duplicate vertex inside one edge")
            local_seen.add(value)
            edge_vertices.append(value)

        edge = tuple(sorted(edge_vertices))
        if edge not in seen_edges:
            seen_edges.add(edge)
            normalized_edges.append(edge)
            used_vertices.update(edge)

    if not used_vertices:
        raise AssertionError("hypergraph has no vertices")

    vertex_count = max(used_vertices)
    expected_vertices = set(range(1, vertex_count + 1))
    if used_vertices != expected_vertices:
        raise AssertionError("vertex labels must be exactly 1..|V| with no gaps")

    incidences: List[List[int]] = [[] for _ in range(vertex_count + 1)]
    for edge_index, edge in enumerate(normalized_edges):
        for vertex in edge:
            incidences[vertex].append(edge_index)

    for vertex in range(1, vertex_count + 1):
        if not incidences[vertex]:
            raise AssertionError("hypergraph contains an isolated vertex")

    return ParsedHypergraph(
        vertex_count=vertex_count,
        edges=tuple(normalized_edges),
        incidences=tuple(tuple(incidences[v]) for v in range(1, vertex_count + 1)),
    )


def max_partition_size(hypergraph: ParsedHypergraph) -> int:
    model = cp_model.CpModel()
    edge_vars = [
        model.NewBoolVar(f"e_{edge_index}")
        for edge_index in range(len(hypergraph.edges))
    ]
    exactly_once_vars = [
        model.NewBoolVar(f"z_{vertex}")
        for vertex in range(hypergraph.vertex_count)
    ]

    for vertex_index, incident_edges in enumerate(hypergraph.incidences):
        degree = model.NewIntVar(0, len(incident_edges), f"d_{vertex_index + 1}")
        model.Add(degree == sum(edge_vars[e] for e in incident_edges))
        model.Add(degree == 1).OnlyEnforceIf(exactly_once_vars[vertex_index])
        model.Add(degree != 1).OnlyEnforceIf(exactly_once_vars[vertex_index].Not())

    model.Maximize(sum(exactly_once_vars))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise AssertionError(f"unexpected CP-SAT status: {status}")
    return int(solver.ObjectiveValue())


def evaluate_solution_callable(solution_fn: Callable[[int], str]) -> Tuple[float, Dict[str, object]]:
    min_ratio = math.inf
    min_excess = math.inf
    worst_n = TEST_RANGE[0]
    valid_count = 0
    worst_partition_slack = math.inf
    first_failure_message = None
    solved = True
    start_time = time.time()

    for n in TEST_RANGE:
        try:
            raw = solution_fn(n)
            hypergraph = parse_hypergraph(raw)
            p_star = max_partition_size(hypergraph)
            slack = n - p_star
            valid = slack >= 0
            if valid:
                valid_count += 1

            current_k = k_value(n)
            ratio = hypergraph.vertex_count / current_k
            excess = hypergraph.vertex_count - current_k

            if ratio < min_ratio:
                min_ratio = ratio
                worst_n = n
            min_excess = min(min_excess, excess)
            worst_partition_slack = min(worst_partition_slack, slack)

            elapsed_min = (time.time() - start_time) / 60.0
            status = "PASS" if (valid and hypergraph.vertex_count > current_k) else "FAIL"
            print(
                (
                    f"[{elapsed_min:7.2f}m] n={n:3d} {status} "
                    f"|V|={hypergraph.vertex_count} k_n={current_k} "
                    f"p*={p_star} slack={slack} ratio={ratio:.6f}"
                ),
                flush=True,
            )

            if ((not valid) or hypergraph.vertex_count <= current_k) and first_failure_message is None:
                solved = False
                first_failure_message = (
                    f"Not solved. First failing n={n}: "
                    f"|V|={hypergraph.vertex_count}, k_n={current_k}, p*={p_star}."
                )
        except Exception as exc:
            elapsed_min = (time.time() - start_time) / 60.0
            print(
                f"[{elapsed_min:7.2f}m] n={n:3d} ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
            solved = False
            if first_failure_message is None:
                first_failure_message = f"Not solved. First failing n={n}: {type(exc).__name__}: {exc}"

    details = {
        "Message": first_failure_message
        or f"Solved on the full evaluation range {TEST_RANGE[0]}..{TEST_RANGE[-1]}.",
        "MinRatio": 0.0 if math.isinf(min_ratio) else float(min_ratio),
        "MinExcess": 0 if math.isinf(min_excess) else int(min_excess),
        "WorstN": int(worst_n),
        "ValidCount": int(valid_count),
        "WorstPartitionSlack": 0 if math.isinf(worst_partition_slack) else int(worst_partition_slack),
    }
    return (100.0 if solved else 0.0), details


def main() -> None:
    try:
        run_module = importlib.import_module("run")
        if hasattr(run_module, "test"):
            _run_test_mode(run_module)
            return
        solution_fn = getattr(run_module, "solution", None)
        if solution_fn is None:
            raise AssertionError("run.py must define solution(n: int) -> str")
        score, details = evaluate_solution_callable(solution_fn)
        _emit_eval_json(score, details)
    except Exception as exc:
        _emit_eval_json(
            0.0,
            {
                "Message": f"Runner failure: {type(exc).__name__}: {exc}",
                "MinRatio": 0.0,
                "MinExcess": 0,
                "WorstN": 15,
                "ValidCount": 0,
                "WorstPartitionSlack": 0,
            },
        )


if __name__ == "__main__":
    main()
