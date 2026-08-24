from __future__ import annotations

"""Algorithm-level finite search tools for epoch_book_b.

This file packages reusable finite-search grammars for discovering source rows.

The default baseline builder is fast and deterministic; it compiles source
laws.  This module is for agents who want to reproduce or extend the
finite-search route itself.
"""

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Iterable


METHOD_SURVEY = [
    {
        "covered_n": [25, 27, 31, 37, 41, 45, 49],
        "method": "finite-field two-copy Paley for q=2n-1 prime powers q=1 mod 4",
    },
    {
        "covered_n": [23],
        "method": "CP-SAT two-sequence noncyclic group search on Z3 x Z3 x Z5",
    },
    {
        "covered_n": [22, 24],
        "method": "Numba simulated annealing in parity-relaxed complementary bicirculant coordinates",
    },
    {
        "covered_n": [29],
        "method": "fixed-A CP-SAT C-completion in cyclic Z57",
    },
    {
        "covered_n": [28],
        "method": "exact CP-SAT parity-relaxed bicirculant model on Z55",
    },
    {
        "covered_n": [30],
        "method": "large OpenMP/Numba joint SA in complementary bicirculant coordinates",
    },
    {
        "covered_n": [32],
        "method": "large OpenMP/Numba joint SA in complementary bicirculant coordinates",
    },
    {
        "covered_n": [33],
        "method": "OpenMP C complementary bicirculant SA over Z65",
    },
    {
        "covered_n": [34],
        "method": "OpenMP C complementary bicirculant SA over Z67",
    },
    {
        "covered_n": [35],
        "method": "OpenMP C complementary bicirculant SA over Z69",
    },
    {
        "covered_n": [39],
        "method": "Z7 x Z11 product-group local search followed by fixed-A C-completion",
    },
    {
        "covered_n": [36],
        "method": "capacity-profile A selection plus fixed-A C-completion over Z71",
    },
    {
        "covered_n": [38],
        "method": "capacity-profile A selection plus fixed-A C-completion over Z75",
    },
    {
        "covered_n": [43],
        "method": "deterministic k=4 orbit exhaustive uncoupled deconvolution over Z85",
    },
    {
        "covered_n": [42],
        "method": "Hadamard-boundary deletion near-miss plus one-edge/incident CP-SAT repair",
    },
    {
        "covered_n": [50],
        "method": "Hadamard-boundary plus-row one-edge repair",
    },
    {
        "covered_n": [46],
        "method": "S/K one-relation theorem from a q=45 conference relation",
    },
]


def neg_set(values: set[int], modulus: int) -> set[int]:
    return {(-value) % modulus for value in values}


def add_edge(masks: list[int], u: int, v: int) -> None:
    if u == v:
        return
    masks[u] |= 1 << v
    masks[v] |= 1 << u


def adjacency_string_from_masks(masks: Iterable[int]) -> str:
    rows = list(masks)
    return "".join(
        "1" if (int(rows[j]) >> i) & 1 else "0"
        for j in range(1, len(rows))
        for i in range(j)
    )


def analyze_masks(masks: tuple[int, ...], n: int) -> dict[str, int | bool]:
    vertex_count = len(masks)
    all_mask = (1 << vertex_count) - 1
    comp = tuple((all_mask ^ (1 << u)) ^ masks[u] for u in range(vertex_count))
    red_max = 0
    blue_max = 0
    red_bad = 0
    blue_bad = 0
    for j in range(1, vertex_count):
        for i in range(j):
            if (masks[j] >> i) & 1:
                common = (masks[i] & masks[j]).bit_count()
                red_max = max(red_max, common)
                red_bad += int(common > n - 2)
            else:
                common = (comp[i] & comp[j]).bit_count()
                blue_max = max(blue_max, common)
                blue_bad += int(common > n - 1)
    return {
        "red_max": red_max,
        "blue_max": blue_max,
        "red_excess": max(0, red_max - (n - 2)),
        "blue_excess": max(0, blue_max - (n - 1)),
        "red_bad_pairs": red_bad,
        "blue_bad_pairs": blue_bad,
        "valid": red_max <= n - 2 and blue_max <= n - 1,
    }


def set_from_pair_reps(modulus: int, pair_reps: Iterable[int]) -> set[int]:
    out: set[int] = set()
    for raw in pair_reps:
        value = int(raw) % modulus
        if value == 0:
            raise ValueError("zero cannot be an inverse-pair representative")
        out.add(value)
        out.add((-value) % modulus)
    return out


def periodic_autocorrelation(values: set[int], modulus: int, d: int) -> int:
    return sum(1 for x in values if (x + d) % modulus in values)


def complementary_bicirculant_residual_rows(
    n: int,
    a_pairs: Iterable[int],
    c_residues: Iterable[int],
) -> list[dict[str, int | bool]]:
    """Residual rows for the complementary bicirculant grammar.

    Vertices are two copies of Z_m, m=2n-1.  A is symmetric on the first copy,
    B is the nonzero complement of A on the second copy, and C is the directed
    cross relation.  Earlier searches called the cross relation C or z_full;
    the bridge source-law form usually calls it B.  The inequality below is the compact
    representative-row objective used by the SA/CP-SAT searches.
    """

    m = 2 * n - 1
    A = set_from_pair_reps(m, a_pairs)
    C = {int(x) % m for x in c_residues}
    B = set(range(1, m)) - A
    rows: list[dict[str, int | bool]] = []
    for d in range(1, m):
        lhs = periodic_autocorrelation(A, m, d) + periodic_autocorrelation(C, m, d)
        cap = n - 2 if d in A else n - 1
        rows.append(
            {
                "d": d,
                "in_A": d in A,
                "N_A": periodic_autocorrelation(A, m, d),
                "N_C": periodic_autocorrelation(C, m, d),
                "lhs": lhs,
                "cap": cap,
                "excess": max(0, lhs - cap),
            }
        )
    # Keep B referenced so the relation between complementary and bridge notes is
    # explicit; B is not needed in the representative residual formula above.
    if len(B) + len(A) != m - 1:
        raise AssertionError("A/B partition failed")
    return rows


def complementary_bicirculant_masks(
    n: int,
    a_pairs: Iterable[int],
    c_residues: Iterable[int],
) -> tuple[int, ...]:
    m = 2 * n - 1
    A = set_from_pair_reps(m, a_pairs)
    B = set(range(1, m)) - A
    C = {int(x) % m for x in c_residues}
    masks = [0] * (2 * m)
    for x in range(m):
        for y in range(x + 1, m):
            d = (y - x) % m
            if d in A:
                add_edge(masks, x, y)
            if d in B:
                add_edge(masks, m + x, m + y)
    for x in range(m):
        for y in range(m):
            if (y - x) % m in C:
                add_edge(masks, x, m + y)
    return tuple(masks)


@dataclass
class AnnealResult:
    n: int
    success: bool
    elapsed_seconds: float
    steps: int
    best_energy: int
    a_pairs: list[int]
    c_residues: list[int]
    validation: dict[str, int | bool] | None


def _energy_from_rows(rows: list[dict[str, int | bool]]) -> int:
    return sum(int(row["excess"]) for row in rows)


def _random_state(n: int, rng: random.Random) -> tuple[set[int], set[int]]:
    m = 2 * n - 1
    h = (m - 1) // 2
    # Both degree-balanced and degree-skew branches are useful.
    # This generic branch menu covers the successful low-n patterns.
    a_pair_count = rng.choice([max(1, (n - 2) // 2), max(1, (n - 1) // 2), max(1, n // 2)])
    a_pair_count = max(0, min(h, a_pair_count))
    c_size = n - 1
    a_pairs = set(rng.sample(range(1, h + 1), a_pair_count))
    c_residues = set(rng.sample(range(m), c_size))
    return a_pairs, c_residues


def anneal_complementary_bicirculant(
    n: int,
    *,
    seconds: float = 30.0,
    seed: int = 890022,
    max_steps_per_restart: int = 200_000,
) -> AnnealResult:
    """Small pure-Python complementary bicirculant simulated annealer.

    Larger compiled implementations can use the same move grammar with much
    bigger budgets.  This reference version is deliberately transparent and
    portable; it is a reproducer for the objective, not a guarantee that all
    rows will be discovered in the default time.
    """

    rng = random.Random(seed + 1009 * n)
    m = 2 * n - 1
    h = (m - 1) // 2
    deadline = time.time() + max(0.0, seconds)
    best_energy = 10**9
    best_a: set[int] = set()
    best_c: set[int] = set()
    steps = 0

    while time.time() < deadline:
        a_pairs, c_residues = _random_state(n, rng)
        temperature = 8.0
        rows = complementary_bicirculant_residual_rows(n, a_pairs, c_residues)
        energy = _energy_from_rows(rows)
        if energy < best_energy:
            best_energy = energy
            best_a = set(a_pairs)
            best_c = set(c_residues)
        for local_step in range(max_steps_per_restart):
            if (local_step & 1023) == 0 and time.time() >= deadline:
                break
            trial_a = set(a_pairs)
            trial_c = set(c_residues)
            if rng.random() < 0.5 and 0 < len(trial_a) < h:
                remove = rng.choice(tuple(trial_a))
                add = rng.choice([x for x in range(1, h + 1) if x not in trial_a])
                trial_a.remove(remove)
                trial_a.add(add)
            else:
                remove = rng.choice(tuple(trial_c))
                add = rng.choice([x for x in range(m) if x not in trial_c])
                trial_c.remove(remove)
                trial_c.add(add)

            trial_rows = complementary_bicirculant_residual_rows(n, trial_a, trial_c)
            trial_energy = _energy_from_rows(trial_rows)
            delta = trial_energy - energy
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-6)):
                a_pairs = trial_a
                c_residues = trial_c
                energy = trial_energy
                if energy < best_energy:
                    best_energy = energy
                    best_a = set(a_pairs)
                    best_c = set(c_residues)
                    if best_energy == 0:
                        masks = complementary_bicirculant_masks(n, best_a, best_c)
                        validation = analyze_masks(masks, n)
                        return AnnealResult(
                            n=n,
                            success=bool(validation["valid"]),
                            elapsed_seconds=seconds - max(0.0, deadline - time.time()),
                            steps=steps,
                            best_energy=0,
                            a_pairs=sorted(best_a),
                            c_residues=sorted(best_c),
                            validation=validation,
                        )
            temperature *= 0.99995
            steps += 1

    validation = None
    if best_a and best_c:
        validation = analyze_masks(complementary_bicirculant_masks(n, best_a, best_c), n)
    return AnnealResult(
        n=n,
        success=bool(validation and validation["valid"]),
        elapsed_seconds=seconds - max(0.0, deadline - time.time()),
        steps=steps,
        best_energy=best_energy,
        a_pairs=sorted(best_a),
        c_residues=sorted(best_c),
        validation=validation,
    )


def solve_fixed_a_c_completion_cp_sat(
    n: int,
    a_pairs: Iterable[int],
    *,
    seconds: float = 60.0,
    c_size: int | None = None,
    force_zero_in_c: bool = True,
) -> dict[str, object]:
    """Exact CP-SAT C-completion for fixed internal layer A."""

    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"status": "ORTOOLS_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}

    m = 2 * n - 1
    A = set_from_pair_reps(m, a_pairs)
    if c_size is None:
        c_size = n - 1
    model = cp_model.CpModel()
    c_vars = [model.NewBoolVar(f"C_{i}") for i in range(m)]
    model.Add(sum(c_vars) == int(c_size))
    if force_zero_in_c:
        model.Add(c_vars[0] == 1)

    positive_rows = []
    total_excess_terms = []
    max_excess = model.NewIntVar(0, m, "max_excess")
    for d in range(1, m):
        terms = []
        for x in range(m):
            y = (x + d) % m
            z = model.NewBoolVar(f"CC_{d}_{x}")
            model.Add(z <= c_vars[x])
            model.Add(z <= c_vars[y])
            model.Add(z >= c_vars[x] + c_vars[y] - 1)
            terms.append(z)
        n_a = periodic_autocorrelation(A, m, d)
        cap = n - 2 if d in A else n - 1
        excess = model.NewIntVar(0, m, f"excess_{d}")
        model.Add(excess >= sum(terms) + n_a - cap)
        model.Add(excess >= 0)
        model.Add(max_excess >= excess)
        is_positive = model.NewBoolVar(f"positive_{d}")
        model.Add(excess >= 1).OnlyEnforceIf(is_positive)
        model.Add(excess == 0).OnlyEnforceIf(is_positive.Not())
        positive_rows.append(is_positive)
        total_excess_terms.append(excess)

    model.Minimize(100000 * max_excess + 100 * sum(total_excess_terms) + sum(positive_rows))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 990057
    started = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - started
    out: dict[str, object] = {
        "status": solver.StatusName(status),
        "elapsed_seconds": round(elapsed, 3),
        "objective": float(solver.ObjectiveValue()) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "best_bound": float(solver.BestObjectiveBound()),
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        c_residues = [i for i, var in enumerate(c_vars) if solver.Value(var)]
        masks = complementary_bicirculant_masks(n, A, c_residues)
        out["a_pairs"] = sorted({min(x, (-x) % m) for x in A if x})
        out["c_residues"] = c_residues
        out["validation"] = analyze_masks(masks, n)
        out["adjacency_sha256_hint"] = __import__("hashlib").sha256(
            adjacency_string_from_masks(masks).encode("ascii")
        ).hexdigest()
    return out


def bridge_source_law_residual_rows(n: int, A: set[int], B: set[int]) -> list[dict[str, int | bool]]:
    """Residual rows for the later bridge source-law CP-SAT grammar."""

    m = 2 * n - 1
    rows: list[dict[str, int | bool]] = []
    for d in range(1, m):
        n_a = periodic_autocorrelation(A, m, d)
        n_b = periodic_autocorrelation(B, m, d)
        cap = n - 2 if d in A else 2 * len(A) - n + 1
        rows.append(
            {
                "d": d,
                "in_A": d in A,
                "N_A": n_a,
                "N_B": n_b,
                "lhs": n_a + n_b,
                "cap": cap,
                "excess": max(0, n_a + n_b - cap),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finite-search reference algorithms.")
    parser.add_argument("--survey", action="store_true", help="Print the bundled method survey and exit.")
    parser.add_argument("--trace", action="store_true", help="Alias for --survey.")
    parser.add_argument("--anneal", type=int, default=None, metavar="N", help="Run reference SA for one n.")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=890022)
    args = parser.parse_args()

    if args.survey or args.trace:
        import json

        print(json.dumps(METHOD_SURVEY, indent=2))
        return
    if args.anneal is not None:
        result = anneal_complementary_bicirculant(args.anneal, seconds=args.seconds, seed=args.seed)
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
