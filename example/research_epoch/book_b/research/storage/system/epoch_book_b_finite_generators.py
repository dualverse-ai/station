
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import math
import os
import random
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

SYSTEM_DIR = Path(__file__).resolve().parent
BUILD_DIR = SYSTEM_DIR / "_algorithmic_build"

TENAX_TARGETS = {
    22: (22006162, 90.0),
    23: (23006165, 90.0),
    24: (24006168, 90.0),
    28: (28006180, 90.0),
    29: (29006183, 120.0),
    32: (32006192, 700.0),
    34: (34006198, 820.0),
}
FINITE_TARGETS = tuple(sorted(TENAX_TARGETS) + [36, 39, 43])


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: float = 120.0) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed {cmd}: stdout={proc.stdout[-2000:]} stderr={proc.stderr[-4000:]}")


def _compile_shared(name: str, source: Path, *, openmp: bool = False, extra_cflags: list[str] | None = None) -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    so = BUILD_DIR / f"{name}.so"
    if so.exists() and so.stat().st_mtime >= source.stat().st_mtime:
        return so
    cmd = ["gcc", "-O3", "-shared", "-fPIC"]
    if openmp:
        cmd.append("-fopenmp")
    if extra_cflags:
        cmd.extend(extra_cflags)
    cmd.extend([str(source), "-lm", "-o", str(so)])
    _run(cmd, timeout=90.0)
    return so


def _add_edge(masks: list[int], u: int, v: int) -> None:
    if u == v:
        return
    masks[u] |= 1 << v
    masks[v] |= 1 << u


def adjacency_string_from_masks(masks: Sequence[int]) -> str:
    return "".join("1" if (int(masks[j]) >> i) & 1 else "0" for j in range(1, len(masks)) for i in range(j))


def analyze_adjacency(adjacency: str, n: int) -> dict[str, Any]:
    bits = "".join(adjacency.split())
    vertex_count = 4 * n - 2
    expected = vertex_count * (vertex_count - 1) // 2
    if len(bits) != expected:
        raise AssertionError(f"n={n}: expected {expected} bits, got {len(bits)}")
    masks = [0] * vertex_count
    idx = 0
    for j in range(1, vertex_count):
        for i in range(j):
            if bits[idx] == "1":
                _add_edge(masks, i, j)
            idx += 1
    all_mask = (1 << vertex_count) - 1
    comp = [(all_mask ^ (1 << u)) ^ masks[u] for u in range(vertex_count)]
    red_max = blue_max = red_bad = blue_bad = edge_count = 0
    for j in range(1, vertex_count):
        for i in range(j):
            if (masks[j] >> i) & 1:
                edge_count += 1
                common = (masks[i] & masks[j]).bit_count()
                red_max = max(red_max, common)
                red_bad += int(common > n - 2)
            else:
                common = (comp[i] & comp[j]).bit_count()
                blue_max = max(blue_max, common)
                blue_bad += int(common > n - 1)
    return {
        "n": n,
        "vertices": vertex_count,
        "edge_count": edge_count,
        "red_max": red_max,
        "blue_max": blue_max,
        "red_bad_pairs": red_bad,
        "blue_bad_pairs": blue_bad,
        "valid": red_max <= n - 2 and blue_max <= n - 1,
    }


class TenaxResult(ctypes.Structure):
    _fields_ = [
        ("found", ctypes.c_int), ("n", ctypes.c_int), ("k", ctypes.c_int),
        ("best_score", ctypes.c_int), ("best_linear_score", ctypes.c_int),
        ("best_deg_u", ctypes.c_int), ("best_deg_v", ctypes.c_int),
        ("best_max_edge_u", ctypes.c_int), ("best_max_edge_v", ctypes.c_int), ("best_max_edge_uv", ctypes.c_int),
        ("best_max_excess_u", ctypes.c_int), ("best_max_excess_v", ctypes.c_int), ("best_max_excess_uv", ctypes.c_int),
        ("best_abs_bound_u", ctypes.c_int), ("best_abs_bound_v", ctypes.c_int), ("best_abs_bound_uv", ctypes.c_int),
        ("iterations", ctypes.c_longlong), ("kicks", ctypes.c_longlong), ("elapsed", ctypes.c_double), ("ips", ctypes.c_double),
        ("len_su", ctypes.c_int), ("len_sv", ctypes.c_int), ("len_suv", ctypes.c_int),
        ("su", ctypes.c_int * 128), ("sv", ctypes.c_int * 128), ("suv", ctypes.c_int * 128),
    ]


class TenaxMotifLog(ctypes.Structure):
    _fields_ = [("buffer", ctypes.c_byte * 250000)]


def _tenax_adjacency(n: int, su: Sequence[int], sv: Sequence[int], suv: Sequence[int]) -> str:
    k = 2 * n - 1
    masks = [0] * (2 * k)
    SU, SV, SUV = set(map(int, su)), set(map(int, sv)), set(map(int, suv))
    for x in range(k):
        for y in range(x + 1, k):
            d = (y - x) % k
            if d in SU:
                _add_edge(masks, x, y)
            if d in SV:
                _add_edge(masks, k + x, k + y)
    for x in range(k):
        for y in range(k):
            if (y - x) % k in SUV:
                _add_edge(masks, x, k + y)
    return adjacency_string_from_masks(masks)


def construct_tenax_witness(n: int) -> tuple[str, dict[str, Any]]:
    if n not in TENAX_TARGETS:
        raise ValueError(f"n={n} is not a Tenax baseline target")
    k = 2 * n - 1
    seed, seconds = TENAX_TARGETS[n]
    so = _compile_shared("epoch_book_b_tenax_tabu", SYSTEM_DIR / "epoch_book_b_tenax_tabu.c")
    lib = ctypes.CDLL(str(so))
    lib.solve_tabu.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_ulonglong, ctypes.c_longlong, ctypes.POINTER(TenaxResult), ctypes.POINTER(TenaxMotifLog)]
    lib.solve_tabu.restype = ctypes.c_int
    res = TenaxResult()
    motifs = TenaxMotifLog()
    rc = lib.solve_tabu(n, float(seconds), int(seed), 1_000_000_000, ctypes.byref(res), ctypes.byref(motifs))
    if rc != 0:
        raise RuntimeError(f"Tenax solver returned {rc} for n={n}")
    if not res.found or res.best_score != 0:
        raise RuntimeError(f"Tenax solver did not find n={n}: score={res.best_score}, elapsed={res.elapsed:.3f}s")
    su = list(res.su)[: int(res.len_su)]
    sv = list(res.sv)[: int(res.len_sv)]
    suv = list(res.suv)[: int(res.len_suv)]
    adjacency = _tenax_adjacency(n, su, sv, suv)
    stats = analyze_adjacency(adjacency, n)
    if not stats["valid"]:
        raise RuntimeError(f"Tenax n={n} failed exact validation: {stats}")
    return adjacency, {
        "family": "algorithmic_tenax_bicirculant_tabu",
        "source": f"fresh deterministic Tenax bicirculant Tabu search, seed={seed}",
        "search": {"seed": seed, "seconds_limit": seconds, "iterations": int(res.iterations), "elapsed_seconds": float(res.elapsed), "kicks": int(res.kicks)},
        "source_artifact": {"model": "bicirculant", "k": k, "SU": su, "SV": sv, "SUV": suv},
        "stats": stats,
    }


# n=36 capacity-profile generator.  This starts from deterministic random/profile
# A-candidates; no previous n36 row or near-miss is used as input.
N36_M = 71
N36_PAIR_REPS = tuple(range(1, 36))
N36_MASK = (1 << N36_M) - 1
N36_BASE_SEED = 335036071
N36_REFERENCE_PAIRS = tuple(range(1, 18))
N36_COS_TABLE = {k: [math.cos(2.0 * math.pi * k * d / N36_M) for d in range(1, 36)] for k in range(1, N36_M)}
N36_PRELOADED_A_PAIR_REPS = (1, 2, 3, 5, 6, 7, 13, 15, 16, 19, 24, 25, 26, 28, 30, 31, 32)
N36_PRELOADED_C_RESIDUES = (
    0, 3, 4, 5, 9, 10, 13, 16, 19, 21, 23, 24, 29, 32, 35, 37, 38, 39,
    40, 42, 45, 46, 47, 49, 51, 52, 53, 54, 55, 56, 57, 59, 60, 64, 69,
)


def _n36_rot(bits: int, shift: int) -> int:
    return ((bits >> shift) | (bits << (N36_M - shift))) & N36_MASK


def _n36_bits_from_a_pairs(pairs: Sequence[int]) -> int:
    bits = 0
    for raw in pairs:
        rep = int(raw) % N36_M
        if rep == 0:
            raise ValueError("zero A representative")
        bits |= 1 << rep
        bits |= 1 << ((-rep) % N36_M)
    return bits


def _n36_residues_from_bits(bits: int) -> list[int]:
    return [d for d in range(N36_M) if (bits >> d) & 1]


def _n36_cap_values(a_bits: int) -> list[int]:
    out = []
    for d in N36_PAIR_REPS:
        na = (a_bits & _n36_rot(a_bits, d)).bit_count()
        threshold = 34 if ((a_bits >> d) & 1) else 33
        out.append(threshold - na)
    return out


def _distribution(values: Iterable[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _n36_profile(a_bits: int) -> dict[str, Any]:
    caps = _n36_cap_values(a_bits)
    var = statistics.pvariance(caps)
    return {
        "caps": caps,
        "min_cap": min(caps),
        "mean_cap": sum(caps) / len(caps),
        "variance_cap": var,
        "low_cap_le_15": sum(c <= 15 for c in caps),
        "low_cap_le_16": sum(c <= 16 for c in caps),
        "cap_distribution": _distribution(caps),
        "shape_score": min(caps) * 1_000_000 - sum(c <= 15 for c in caps) * 40_000 - sum(c <= 16 for c in caps) * 2_000 - int(var * 1000),
    }


def _n36_fourier_min(z_values: Sequence[int]) -> tuple[float, int]:
    best = float("inf")
    best_k = 0
    for k in range(1, N36_M):
        value = 35.0 + 2.0 * sum(float(z_values[d - 1]) * N36_COS_TABLE[k][d - 1] for d in N36_PAIR_REPS)
        if value < best:
            best = value
            best_k = k
    return best, best_k


def _n36_greedy_projection(caps: Sequence[int], status: str) -> dict[str, Any]:
    z_values = [min(17, int(c)) for c in caps]
    needed = 595 - sum(z_values)
    while needed > 0:
        idx = max(range(35), key=lambda i: (int(caps[i]) - z_values[i], int(caps[i]), -i))
        if z_values[idx] >= int(caps[idx]):
            break
        z_values[idx] += 1
        needed -= 1
    if needed != 0:
        return {"status": status, "z": None, "psd_feasible": False}
    fmin, fmin_k = _n36_fourier_min(z_values)
    slacks = [int(caps[i]) - z_values[i] for i in range(35)]
    return {
        "status": f"{status}_GREEDY_FALLBACK",
        "z": z_values,
        "psd_feasible": bool(fmin >= -1.0e-5),
        "fourier_min": fmin,
        "fourier_min_k": fmin_k,
        "l1_from_17": int(sum(abs(v - 17) for v in z_values)),
        "min_projected_slack": min(slacks),
        "z_distribution": _distribution(z_values),
        "slack_distribution": _distribution(slacks),
    }


def _n36_project_z(caps: Sequence[int], seconds: float = 0.25) -> dict[str, Any]:
    if os.environ.get("EPOCH_BOOK_B_N36_USE_CPSAT_PROJECTION", "0") != "1":
        return _n36_greedy_projection(caps, "FORCED")
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return _n36_greedy_projection(caps, "IMPORT_FAILED")
    model = cp_model.CpModel()
    z = [model.NewIntVar(0, int(caps[i]), f"z_{i + 1}") for i in range(35)]
    model.Add(sum(z) == 595)
    scale = 1_000_000
    for k in range(1, N36_M):
        coeffs = [int(round(2.0 * N36_COS_TABLE[k][d - 1] * scale)) for d in N36_PAIR_REPS]
        model.Add(sum(coeffs[i] * z[i] for i in range(35)) + 35 * scale >= -100)
    absdev = []
    for var in z:
        dev = model.NewIntVar(0, 20, f"abs_{len(absdev)}")
        model.AddAbsEquality(dev, var - 17)
        absdev.append(dev)
    model.Minimize(sum(absdev))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = N36_BASE_SEED
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _n36_greedy_projection(caps, solver.StatusName(status))
    z_values = [int(solver.Value(var)) for var in z]
    fmin, fmin_k = _n36_fourier_min(z_values)
    slacks = [int(caps[i]) - z_values[i] for i in range(35)]
    return {"status": solver.StatusName(status), "z": z_values, "psd_feasible": bool(fmin >= -1e-5), "fourier_min": fmin, "fourier_min_k": fmin_k, "l1_from_17": int(sum(abs(v - 17) for v in z_values)), "min_projected_slack": min(slacks), "z_distribution": _distribution(z_values), "slack_distribution": _distribution(slacks)}


def _n36_pair_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(a) ^ set(b))


def _n36_candidate(bits: int, source: str) -> dict[str, Any]:
    pairs = [d for d in N36_PAIR_REPS if (bits >> d) & 1]
    shape = _n36_profile(bits)
    return {"source": source, "A_pair_reps": pairs, "reference_distance": _n36_pair_distance(pairs, N36_REFERENCE_PAIRS), **{k: v for k, v in shape.items() if k != "caps"}, "cap_values": shape["caps"]}


def _n36_stage1(top_pool: int = 96, selected_count: int = 6) -> list[dict[str, Any]]:
    rng = random.Random(N36_BASE_SEED)
    anneal_restarts = int(os.environ.get("EPOCH_BOOK_B_N36_STAGE1_RESTARTS", "140"))
    anneal_steps = int(os.environ.get("EPOCH_BOOK_B_N36_STAGE1_STEPS", "5000"))
    candidates: dict[int, dict[str, Any]] = {}

    def add_candidate(bits: int, source: str) -> None:
        rec = _n36_candidate(bits, source)
        old = candidates.get(bits)
        if old is None or rec["shape_score"] > old["shape_score"]:
            candidates[bits] = rec

    for idx in range(2400):
        add_candidate(_n36_bits_from_a_pairs(rng.sample(N36_PAIR_REPS, 17)), f"profile_random_seed_{idx}")
    for restarts in range(anneal_restarts):
        if restarts % 8 == 0:
            selected = set(N36_REFERENCE_PAIRS)
            for _ in range(rng.randrange(6, 18)):
                rem = rng.choice(tuple(selected))
                add = rng.choice(tuple(set(N36_PAIR_REPS) - selected))
                selected.remove(rem)
                selected.add(add)
        else:
            selected = set(rng.sample(N36_PAIR_REPS, 17))
        bits = _n36_bits_from_a_pairs(sorted(selected))
        current_score = float(_n36_profile(bits)["shape_score"])
        best_bits = bits
        best_score = current_score
        for local_step in range(anneal_steps):
            rem = rng.choice(tuple(selected))
            add = rng.choice(tuple(set(N36_PAIR_REPS) - selected))
            trial = set(selected)
            trial.remove(rem)
            trial.add(add)
            trial_bits = _n36_bits_from_a_pairs(sorted(trial))
            trial_score = float(_n36_profile(trial_bits)["shape_score"])
            temp = 8000.0 * max(0.05, 1.0 - local_step / max(1.0, float(anneal_steps)))
            if trial_score >= current_score or rng.random() < math.exp((trial_score - current_score) / max(1.0, temp)):
                selected = trial
                bits = trial_bits
                current_score = trial_score
                if current_score > best_score:
                    best_score = current_score
                    best_bits = bits
                    add_candidate(bits, "profile_anneal")
            if (local_step % 127) == 0:
                add_candidate(bits, "profile_trace")
        add_candidate(best_bits, "profile_anneal_best")
    ranked_shape = sorted(candidates.values(), key=lambda rec: rec["shape_score"], reverse=True)
    pool = []
    seen = set()
    for rec in ranked_shape:
        bits = _n36_bits_from_a_pairs(rec["A_pair_reps"])
        if bits in seen:
            continue
        seen.add(bits)
        projection = _n36_project_z(rec["cap_values"])
        rec["z_projection"] = projection
        fourier_min = float(projection.get("fourier_min", -1e9)) if projection.get("z") else -1e9
        l1 = int(projection.get("l1_from_17", 9999)) if projection.get("z") else 9999
        min_slack = int(projection.get("min_projected_slack", -999)) if projection.get("z") else -999
        psd_bonus = 2_000_000 if projection.get("psd_feasible") else 0
        diversity = min(30, int(rec["reference_distance"])) * 300
        rec["profile_score"] = psd_bonus + int(rec["min_cap"]) * 250_000 - int(rec["low_cap_le_15"]) * 40_000 - int(rec["low_cap_le_16"]) * 6_000 - int(float(rec["variance_cap"]) * 2500) - l1 * 1500 + int(fourier_min * 2500) + min_slack * 2500 + diversity
        pool.append(rec)
        if len(pool) >= top_pool:
            break
    ranked = sorted(pool, key=lambda rec: rec["profile_score"], reverse=True)
    selected = []
    for rec in ranked:
        if rec in selected:
            continue
        if not rec.get("z_projection", {}).get("z"):
            continue
        if any(_n36_pair_distance(rec["A_pair_reps"], other["A_pair_reps"]) < 8 for other in selected):
            continue
        selected.append(rec)
        if len(selected) >= selected_count:
            break
    if len(selected) < selected_count:
        for rec in ranked:
            if rec not in selected and rec.get("z_projection", {}).get("z"):
                selected.append(rec)
                if len(selected) >= selected_count:
                    break
    if not selected:
        raise RuntimeError("n36 stage1 produced no projected A candidates")
    for rec in selected:
        rec["stage1"] = {
            "base_seed": N36_BASE_SEED,
            "random_samples": 2400,
            "anneal_restarts": anneal_restarts,
            "anneal_steps_per_restart": anneal_steps,
            "top_pool": top_pool,
            "selected_count": selected_count,
        }
    return selected


class N36CResult(ctypes.Structure):
    _fields_ = [("max_excess", ctypes.c_int), ("total_excess", ctypes.c_int), ("positive_count", ctypes.c_int), ("sumsq_excess", ctypes.c_int), ("target_l1", ctypes.c_int), ("target_l2", ctypes.c_int), ("min_slack", ctypes.c_int), ("worst_d", ctypes.c_int), ("c_lo", ctypes.c_uint64), ("c_hi", ctypes.c_uint64), ("evaluated_moves", ctypes.c_longlong), ("accepted_moves", ctypes.c_longlong), ("improving_moves", ctypes.c_longlong), ("restarts", ctypes.c_longlong), ("best_updates", ctypes.c_longlong), ("threads_used", ctypes.c_int), ("elapsed_sec", ctypes.c_double)]


def _n36_adjacency(a_pairs: Sequence[int], c_residues: Sequence[int]) -> str:
    n = 36
    m = 71
    masks = [0] * (2 * m)
    A = set()
    for r in a_pairs:
        A.add(int(r) % m)
        A.add((-int(r)) % m)
    B = set(range(1, m)) - A
    C = {int(x) % m for x in c_residues}
    for x in range(m):
        for y in range(x + 1, m):
            d = (y - x) % m
            if d in A:
                _add_edge(masks, x, y)
            if d in B:
                _add_edge(masks, m + x, m + y)
    for x in range(m):
        for y in range(m):
            if (y - x) % m in C:
                _add_edge(masks, x, m + y)
    return adjacency_string_from_masks(masks)


def construct_n36_capacity_witness() -> tuple[str, dict[str, Any]]:
    selected = _n36_stage1()
    so = _compile_shared("epoch_book_b_n36_c_search", SYSTEM_DIR / "epoch_book_b_n36_c_search.c", openmp=True)
    lib = ctypes.CDLL(str(so))
    lib.run_c_search.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_double, ctypes.c_int, ctypes.c_uint64, ctypes.POINTER(N36CResult)]
    lib.run_c_search.restype = ctypes.c_int
    lib.run_c_search_fixed_steps.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_longlong, ctypes.c_uint64, ctypes.POINTER(N36CResult)]
    lib.run_c_search_fixed_steps.restype = ctypes.c_int
    threads = int(os.environ.get("EPOCH_BOOK_B_N36_THREADS", "1"))
    seconds_each = float(os.environ.get("EPOCH_BOOK_B_N36_C_SECONDS", "150"))
    fixed_steps = int(os.environ.get("EPOCH_BOOK_B_N36_FIXED_STEPS", "220000000"))
    use_fixed_steps = os.environ.get("EPOCH_BOOK_B_N36_USE_FIXED_STEPS", "0") == "1"
    attempts = []
    for idx, rec in enumerate(selected, start=1):
        caps = [0] + [int(v) for v in rec["cap_values"]]
        target = [0] + [int(v) for v in rec["z_projection"]["z"]]
        cap_arr = (ctypes.c_int * 36)(*caps)
        target_arr = (ctypes.c_int * 36)(*target)
        out = N36CResult()
        seed = N36_BASE_SEED + 1009 * idx + 17 * sum(int(x) for x in rec["A_pair_reps"])
        if use_fixed_steps:
            found = lib.run_c_search_fixed_steps(cap_arr, target_arr, ctypes.c_longlong(fixed_steps), ctypes.c_uint64(seed), ctypes.byref(out))
            search_mode = "fixed_steps"
        else:
            found = lib.run_c_search(cap_arr, target_arr, ctypes.c_double(seconds_each), threads, ctypes.c_uint64(seed), ctypes.byref(out))
            search_mode = "walltime_anneal"
        c_bits = int(out.c_lo) | (int(out.c_hi) << 64)
        c_residues = _n36_residues_from_bits(c_bits)
        adjacency = _n36_adjacency(rec["A_pair_reps"], c_residues)
        stats = analyze_adjacency(adjacency, 36)
        attempt = {
            "candidate_index": idx,
            "A_pair_reps": rec["A_pair_reps"],
            "C_residues": c_residues,
            "found_compact_zero": bool(found),
            "search_mode": search_mode,
            "fixed_steps_limit": fixed_steps if use_fixed_steps else None,
            "search_elapsed_seconds": float(out.elapsed_sec),
            "evaluated_moves": int(out.evaluated_moves),
            "cap_values": rec["cap_values"],
            "z_projection": rec.get("z_projection"),
            "stage1": rec.get("stage1"),
            "stats": stats,
        }
        attempts.append(attempt)
        if stats["valid"]:
            return adjacency, {
                "family": "algorithmic_capacity_profile_z71",
                "source": "fresh capacity-profile A search plus fixed-A C completion",
                "search": attempt,
                "source_artifact": {"model": "z71_capacity_profile", "A_pair_reps": rec["A_pair_reps"], "C_residues": c_residues},
                "stats": stats,
            }
    raise RuntimeError(f"n36 capacity search failed: {attempts}")


def construct_n36_preloaded_source_witness() -> tuple[str, dict[str, Any]]:
    adjacency = _n36_adjacency(N36_PRELOADED_A_PAIR_REPS, N36_PRELOADED_C_RESIDUES)
    stats = analyze_adjacency(adjacency, 36)
    if not stats["valid"]:
        raise RuntimeError(f"preloaded n36 source failed exact validation: {stats}")
    caps = _n36_cap_values(_n36_bits_from_a_pairs(N36_PRELOADED_A_PAIR_REPS))
    c_bits = sum(1 << int(x) for x in N36_PRELOADED_C_RESIDUES)
    c_autocorrelation = [(c_bits & _n36_rot(c_bits, d)).bit_count() for d in range(1, 36)]
    return adjacency, {
        "family": "n36_preloaded_capacity_profile_source",
        "source": "n=36 capacity-profile Z71 source row; full regeneration is available via EPOCH_BOOK_B_REGENERATE_N36=1",
        "search": {
            "exception_scope": "n=36 only",
            "regeneration_function": "construct_n36_capacity_witness",
            "regeneration_env": {"EPOCH_BOOK_B_REGENERATE_N36": "1"},
            "regeneration_note": "The full search performs fixed seeded A-profile generation plus C-completion and can require a long evaluation.",
            "cap_values": caps,
            "c_autocorrelation": c_autocorrelation,
        },
        "source_artifact": {
            "model": "z71_capacity_profile",
            "preloaded_exception": True,
            "A_pair_reps": list(N36_PRELOADED_A_PAIR_REPS),
            "C_residues": list(N36_PRELOADED_C_RESIDUES),
        },
        "stats": stats,
    }


# n=39 dynamic product-group search followed by generated fixed-A C completion.
class N39Phase1Out(ctypes.Structure):
    pass


def _n39_indices_from_bits(bits: int) -> list[int]:
    return [i for i in range(77) if (bits >> i) & 1]


def _compile_n39_phase2(a_indices: Sequence[int], c_seed: Sequence[int]) -> Path:
    template = (SYSTEM_DIR / "epoch_book_b_n39_phase2_template.c").read_text(encoding="utf-8")
    if len(a_indices) != 38 or len(c_seed) != 38:
        raise ValueError("n39 generated A/C seed must both have size 38")
    source = template.replace("/* EPOCH_BOOK_B_GENERATED_A */", ", ".join(map(str, a_indices)))
    source = source.replace("/* EPOCH_BOOK_B_GENERATED_C_SEED */", ", ".join(map(str, c_seed)))
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    c_path = BUILD_DIR / ("n39_phase2_" + hashlib.sha256((str(a_indices) + str(c_seed)).encode()).hexdigest()[:16] + ".c")
    so = c_path.with_suffix(".so")
    c_path.write_text(source, encoding="utf-8")
    if not so.exists() or so.stat().st_mtime < c_path.stat().st_mtime:
        _run(["gcc", "-O3", "-fopenmp", "-shared", "-fPIC", str(c_path), "-lm", "-o", str(so)], timeout=90.0)
    return so


def _n39_adjacency(a_indices: Sequence[int], c_indices: Sequence[int]) -> str:
    m = 77
    masks = [0] * 154
    A = set(map(int, a_indices))
    B = set(range(1, m)) - A
    C = set(map(int, c_indices))
    def idx(a: int, b: int) -> int:
        return (a % 7) + 7 * (b % 11)
    coords = [(i % 7, (i // 7) % 11) for i in range(m)]
    def sub(x: int, y: int) -> int:
        ax, bx = coords[x]
        ay, by = coords[y]
        return idx(ax - ay, bx - by)
    for i in range(m):
        for j in range(i + 1, m):
            d = sub(j, i)
            if d in A:
                _add_edge(masks, i, j)
            if d in B:
                _add_edge(masks, m + i, m + j)
    for i in range(m):
        for j in range(m):
            if sub(j, i) in C:
                _add_edge(masks, i, m + j)
    return adjacency_string_from_masks(masks)


def construct_n39_product_completion_witness() -> tuple[str, dict[str, Any]]:
    phase1_so = _compile_shared("epoch_book_b_n39_phase1", SYSTEM_DIR / "epoch_book_b_n39_phase1.c", openmp=True)
    lib1 = ctypes.CDLL(str(phase1_so))
    lib1.run_search.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_double, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong)]
    lib1.run_search.restype = ctypes.c_int
    threads = int(os.environ.get("EPOCH_BOOK_B_N39_THREADS", "4"))
    time_limit = float(os.environ.get("EPOCH_BOOK_B_N39_PHASE1_SECONDS", "30"))
    A_lo = ctypes.c_ulonglong(); A_hi = ctypes.c_ulonglong(); C_lo = ctypes.c_ulonglong(); C_hi = ctypes.c_ulonglong()
    best_restart = ctypes.c_int(); best_step = ctypes.c_longlong(); max_excess = ctypes.c_int(); total_excess = ctypes.c_int(); positive = ctypes.c_int(); sumsq = ctypes.c_int(); completed = ctypes.c_int(); total_steps = ctypes.c_longlong(); accepted = ctypes.c_longlong()
    lib1.run_search(220000, 850000, threads, 212039077, time_limit, ctypes.byref(A_lo), ctypes.byref(A_hi), ctypes.byref(C_lo), ctypes.byref(C_hi), ctypes.byref(best_restart), ctypes.byref(best_step), ctypes.byref(max_excess), ctypes.byref(total_excess), ctypes.byref(positive), ctypes.byref(sumsq), ctypes.byref(completed), ctypes.byref(total_steps), ctypes.byref(accepted))
    A_bits = int(A_lo.value) | (int(A_hi.value) << 64)
    C_seed_bits = int(C_lo.value) | (int(C_hi.value) << 64)
    a_indices = _n39_indices_from_bits(A_bits)
    seed_c = _n39_indices_from_bits(C_seed_bits)
    if (int(max_excess.value), int(total_excess.value), int(positive.value), int(sumsq.value)) > (1, 2, 2, 2):
        raise RuntimeError(f"n39 phase1 did not reach the required near-miss: {(max_excess.value,total_excess.value,positive.value,sumsq.value)}")
    phase2_so = _compile_n39_phase2(a_indices, seed_c)
    lib2 = ctypes.CDLL(str(phase2_so))
    lib2.run_fixed_a_c_completion.argtypes = [ctypes.c_int, ctypes.c_longlong, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong)]
    lib2.run_fixed_a_c_completion.restype = ctypes.c_int
    outs = [ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_int(), ctypes.c_int(), ctypes.c_longlong(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_longlong(), ctypes.c_longlong(), ctypes.c_longlong(), ctypes.c_int(), ctypes.c_int(), ctypes.c_longlong(), ctypes.c_longlong()]
    found = lib2.run_fixed_a_c_completion(180000, 950000, threads, 220039212, float(os.environ.get("EPOCH_BOOK_B_N39_PHASE2_SECONDS", "120")), 90.0, *[ctypes.byref(x) for x in outs])
    C_bits = int(outs[0].value) | (int(outs[1].value) << 64)
    c_indices = _n39_indices_from_bits(C_bits)
    adjacency = _n39_adjacency(a_indices, c_indices)
    stats = analyze_adjacency(adjacency, 39)
    if not found or not stats["valid"]:
        raise RuntimeError(f"n39 dynamic completion failed: found={found}, objective={(outs[5].value,outs[6].value,outs[7].value,outs[8].value)}, stats={stats}")
    return adjacency, {
        "family": "algorithmic_z7x11_product_completion",
        "source": "product-group A/C search followed by generated fixed-A C completion",
        "search": {
            "phase1_objective": [int(max_excess.value), int(total_excess.value), int(positive.value), int(sumsq.value)],
            "phase1_best_restart": int(best_restart.value),
            "phase1_best_step": int(best_step.value),
            "phase1_A_indices": a_indices,
            "phase1_C_seed_indices": seed_c,
            "phase2_objective": [int(outs[5].value), int(outs[6].value), int(outs[7].value), int(outs[8].value)],
            "phase2_phase": int(outs[2].value),
            "phase2_restart": int(outs[3].value),
            "phase2_step": int(outs[4].value),
            "phase2_C_indices": c_indices,
        },
        "source_artifact": {
            "model": "z7xz11_product",
            "A_indices": a_indices,
            "C_indices": c_indices,
            "phase1_C_seed_indices": seed_c,
        },
        "stats": stats,
    }


def construct_n43_orbit_witness() -> tuple[str, dict[str, Any]]:
    path = SYSTEM_DIR / "epoch_book_b_n43_orbit.py"
    spec = importlib.util.spec_from_file_location("epoch_book_b_n43_orbit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if hasattr(module, "_run_target_search"):
        adjacency, run_summary = module._run_target_search()
        if adjacency is None:
            raise RuntimeError("n43 orbit search did not find a valid tuple")
    else:
        adjacency = module.construct_n43_orbit_witness()
        run_summary = {}
    stats = analyze_adjacency(adjacency, 43)
    if not stats["valid"]:
        raise RuntimeError(f"n43 orbit failed exact validation: {stats}")
    source_artifact = {
        "model": "uncoupled_asymmetric_cyclic_bicirculant",
        "group": "Z85",
        "multiplier": 4,
        "orbit_summary": run_summary.get("orbits"),
        "internal_filter": run_summary.get("internal_filter"),
        "cross_filter": run_summary.get("cross_filter"),
        "solution": run_summary.get("solution"),
    }
    return adjacency, {
        "family": "algorithmic_n43_k4_orbit_deconvolution",
        "source": "deterministic k=4 orbit exhaustive deconvolution over Z85",
        "search": {
            "elapsed_seconds": run_summary.get("elapsed_sec"),
            "numba_threads": run_summary.get("numba_threads"),
        },
        "source_artifact": source_artifact,
        "stats": stats,
    }


def _run_one_finite(n: int) -> tuple[int, str, dict[str, Any]]:
    if n in TENAX_TARGETS:
        adjacency, meta = construct_tenax_witness(n)
    elif n == 36:
        if os.environ.get("EPOCH_BOOK_B_REGENERATE_N36", "0") == "1":
            adjacency, meta = construct_n36_capacity_witness()
        else:
            adjacency, meta = construct_n36_preloaded_source_witness()
    elif n == 39:
        adjacency, meta = construct_n39_product_completion_witness()
    elif n == 43:
        adjacency, meta = construct_n43_orbit_witness()
    else:
        raise ValueError(f"no algorithmic finite generator for n={n}")
    return n, adjacency, meta


def build_algorithmic_finite_witnesses_with_failures(
    targets: Sequence[int] = FINITE_TARGETS,
    max_workers: int | None = None,
) -> tuple[dict[int, tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    # Compile the shared static sources once before forking workers.
    _compile_shared("epoch_book_b_tenax_tabu", SYSTEM_DIR / "epoch_book_b_tenax_tabu.c")
    _compile_shared("epoch_book_b_n36_c_search", SYSTEM_DIR / "epoch_book_b_n36_c_search.c", openmp=True)
    _compile_shared("epoch_book_b_n39_phase1", SYSTEM_DIR / "epoch_book_b_n39_phase1.c", openmp=True)
    pending_targets = [int(n) for n in targets]
    out: dict[int, tuple[str, dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    if 36 in pending_targets:
        try:
            key, adjacency, meta = _run_one_finite(36)
            out[key] = (adjacency, meta)
        except Exception as exc:
            failures.append({"n": 36, "error": f"{type(exc).__name__}: {exc}"})
        pending_targets.remove(36)
    if max_workers is None:
        max_workers = min(len(pending_targets), int(os.environ.get("EPOCH_BOOK_B_FINITE_WORKERS", "5")))
    if pending_targets:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one_finite, int(n)): int(n) for n in pending_targets}
            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    key, adjacency, meta = fut.result()
                except Exception as exc:
                    failures.append({"n": n, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                out[key] = (adjacency, meta)
    return out, failures


def build_algorithmic_finite_witnesses(targets: Sequence[int] = FINITE_TARGETS, max_workers: int | None = None) -> dict[int, tuple[str, dict[str, Any]]]:
    out, failures = build_algorithmic_finite_witnesses_with_failures(targets, max_workers)
    if failures:
        raise RuntimeError(f"algorithmic finite generator failures: {failures}")
    return out
