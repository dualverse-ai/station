from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
os.environ.setdefault("NUMBA_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "10"))

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads


TEST_RANGE = tuple(range(22, 51))
TARGET_N = 43
M = 85
TARGET_V = 2 * M
MULTIPLIER = 4
RED_TARGET = TARGET_N - 2
BLUE_TARGET = TARGET_N - 1
MASK_85 = (1 << M) - 1
LOW64_MASK = (1 << 64) - 1

THREAD_COUNT = int(os.environ.get("EVAL621_NUMBA_THREADS", os.environ.get("OMP_NUM_THREADS", "10")))
SUMMARY_PATH = os.environ.get("EPOCH_BOOK_B_N43_SUMMARY_PATH", "/tmp/epoch_book_b_n43_summary.json")

SUMMARY: dict[str, Any] = {
    "code_version": "eval621_n43_k4_uncoupled_exhaustive_v1",
    "target_n": TARGET_N,
    "M": M,
    "multiplier": MULTIPLIER,
    "model": "Uncoupled asymmetric cyclic bi-circulant, exhaustive over k=4 orbit choices",
    "requested_parameters": {
        "C_orbits": "under x -> 4x mod 85, include 0, strict |C| in [41, 46]",
        "P_orbits": "under x -> 4x and x -> -x mod 85 for nonzero residues, strict |A| and |B| in [36, 44]",
        "internal_edge_limit": 41,
        "internal_nonedge_limit": "2 * (same_block_size + C_size) - 126",
        "cross_edge_limit": 41,
        "cross_nonedge_limit": "A_size + B_size + 2 * C_size - 126",
    },
    "implementation_notes": [
        "The k=4 definition produces 14 paired P-orbits, not the 13 stated in the instruction; the exhaustive loop uses the definition directly.",
        "The first pass is Numba prange over all C choices and counts valid A choices under the internal constraints.",
        "The second pass recomputes valid A/B lists only for nonzero C rows, then checks the exact cross convolution.",
        "Any reduced-valid tuple is materialized as a 170-vertex graph and exact-validated before insertion.",
    ],
    "sources": {},
    "validation": {},
    "candidate_validation": {},
    "target_run": {},
    "notes": [],
}


def _expected_vertices(n: int) -> int:
    return 4 * n - 2


def _set_edge(masks: list[int], u: int, v: int) -> None:
    masks[u] |= 1 << v
    masks[v] |= 1 << u


def _adjacency_string_from_masks(masks: Sequence[int]) -> str:
    return "".join(
        "1" if (masks[j] >> i) & 1 else "0"
        for j in range(1, len(masks))
        for i in range(j)
    )


def _complete_bipartite(n: int) -> str:
    vertex_count = _expected_vertices(n)
    cut = vertex_count // 2
    masks = [0] * vertex_count
    for u in range(cut):
        for v in range(cut, vertex_count):
            _set_edge(masks, u, v)
    return _adjacency_string_from_masks(masks)


def _normalise_bitstring(n: int, bits: str) -> str:
    compact = "".join(str(bits).split())
    expected_len = _expected_vertices(n) * (_expected_vertices(n) - 1) // 2
    if len(compact) != expected_len:
        raise AssertionError(f"n={n}: expected bitstring length {expected_len}, got {len(compact)}")
    if any(ch not in "01" for ch in compact):
        raise AssertionError(f"n={n}: bitstring contains non-binary characters")
    return compact


def _masks_from_bitstring(n: int, bits: str) -> tuple[int, ...]:
    compact = _normalise_bitstring(n, bits)
    vertex_count = _expected_vertices(n)
    masks = [0] * vertex_count
    idx = 0
    for j in range(1, vertex_count):
        for i in range(j):
            if compact[idx] == "1":
                _set_edge(masks, i, j)
            idx += 1
    return tuple(masks)


def _analyze_bits(n: int, bits: str) -> dict[str, Any]:
    masks = _masks_from_bitstring(n, bits)
    vertex_count = len(masks)
    all_mask = (1 << vertex_count) - 1
    complement = tuple((all_mask ^ (1 << u)) ^ masks[u] for u in range(vertex_count))
    red_target = n - 2
    blue_target = n - 1
    edge_count = 0
    red_max = 0
    blue_max = 0
    red_bad = 0
    blue_bad = 0

    for j in range(1, vertex_count):
        mask_j = masks[j]
        comp_j = complement[j]
        for i in range(j):
            if (mask_j >> i) & 1:
                edge_count += 1
                common = (masks[i] & mask_j).bit_count()
                if common > red_max:
                    red_max = common
                if common > red_target:
                    red_bad += 1
            else:
                common = (complement[i] & comp_j).bit_count()
                if common > blue_max:
                    blue_max = common
                if common > blue_target:
                    blue_bad += 1

    degrees = [mask.bit_count() for mask in masks]
    total_pairs = vertex_count * (vertex_count - 1) // 2
    return {
        "n": n,
        "n_vertices": vertex_count,
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": sum(degrees) / len(degrees) if degrees else 0.0,
        "edge_count": edge_count,
        "density": edge_count / total_pairs if total_pairs else 0.0,
        "red_max": red_max,
        "blue_max": blue_max,
        "red_excess": max(0, red_max - red_target),
        "blue_excess": max(0, blue_max - blue_target),
        "red_bad_pairs": red_bad,
        "blue_bad_pairs": blue_bad,
        "valid": red_max <= red_target and blue_max <= blue_target,
    }


def _record(n: int, source: str, bits: str, *, candidate: bool = False) -> dict[str, Any]:
    stats = _analyze_bits(n, bits)
    bucket = "candidate_validation" if candidate else "validation"
    SUMMARY[bucket][str(n)] = {"source": source, **stats}
    if not candidate:
        SUMMARY["sources"][str(n)] = source
    print(
        (
            f"{'candidate' if candidate else 'batch'} n={n} source={source} valid={stats['valid']} "
            f"degree_min={stats['degree_min']} degree_max={stats['degree_max']} "
            f"red_max={stats['red_max']} red_target={n - 2} "
            f"blue_max={stats['blue_max']} blue_target={n - 1} "
            f"red_bad={stats['red_bad_pairs']} blue_bad={stats['blue_bad_pairs']}"
        ),
        flush=True,
    )
    return stats


def _pack_residues(residues: Sequence[int]) -> int:
    value = 0
    for residue in residues:
        value |= 1 << int(residue)
    return value


def _rotate_right_85(value: int, shift: int) -> int:
    shift %= M
    if shift == 0:
        return value & MASK_85
    return ((value >> shift) | ((value & ((1 << shift) - 1)) << (M - shift))) & MASK_85


def _make_c_orbits() -> list[tuple[int, ...]]:
    visited: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for seed in range(M):
        if seed in visited:
            continue
        cur: list[int] = []
        x = seed
        while x not in cur:
            cur.append(x)
            x = (MULTIPLIER * x) % M
        visited.update(cur)
        orbits.append(tuple(sorted(cur)))
    return orbits


def _make_p_orbits() -> list[tuple[int, ...]]:
    visited: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for seed in range(1, M):
        if seed in visited:
            continue
        stack = [seed]
        cur: set[int] = set()
        while stack:
            x = stack.pop()
            if x == 0 or x in cur:
                continue
            cur.add(x)
            stack.append((MULTIPLIER * x) % M)
            stack.append((-x) % M)
        visited.update(cur)
        orbits.append(tuple(sorted(cur)))
    return orbits


def _build_a_choices(orbits_p: Sequence[Sequence[int]], p_reps: Sequence[int]):
    started = time.time()
    rows: list[np.ndarray] = []
    sizes: list[int] = []
    edge_reps: list[list[int]] = []
    auto_reps: list[list[int]] = []
    packed_values: list[int] = []
    orbit_masks = [_pack_residues(orbit) for orbit in orbits_p]

    for mask in range(1 << len(orbits_p)):
        size = 0
        value = 0
        for orbit_idx, orbit in enumerate(orbits_p):
            if (mask >> orbit_idx) & 1:
                size += len(orbit)
                value |= orbit_masks[orbit_idx]
        if not (36 <= size <= 44):
            continue
        row = np.fromiter(((value >> idx) & 1 for idx in range(M)), dtype=np.uint8, count=M)
        rows.append(row)
        sizes.append(size)
        packed_values.append(value)
        edge_reps.append([int(row[rep]) for rep in p_reps])
        auto_reps.append([(value & _rotate_right_85(value, rep)).bit_count() for rep in p_reps])

    arr = np.vstack(rows).astype(np.uint8)
    size_arr = np.asarray(sizes, dtype=np.int16)
    edge_arr = np.asarray(edge_reps, dtype=np.uint8)
    auto_arr = np.asarray(auto_reps, dtype=np.int16)
    info = {
        "choice_count": int(arr.shape[0]),
        "size_distribution": {str(size): int(np.sum(size_arr == size)) for size in sorted(set(sizes))},
        "elapsed_sec": time.time() - started,
        "packed_values": packed_values,
    }
    print(
        f"eval621 built A/B choices count={arr.shape[0]} size_distribution={info['size_distribution']} "
        f"elapsed_sec={info['elapsed_sec']:.2f}",
        flush=True,
    )
    return arr, size_arr, edge_arr, auto_arr, info


def _build_c_choices(orbits_c: Sequence[Sequence[int]], p_reps: Sequence[int]):
    started = time.time()
    orbit_masks = [_pack_residues(orbit) for orbit in orbits_c]
    orbit_sizes = [len(orbit) for orbit in orbits_c]

    valid_masks: list[tuple[int, int]] = []
    for mask in range(1 << len(orbits_c)):
        size = 0
        for orbit_idx, orbit_size in enumerate(orbit_sizes):
            if (mask >> orbit_idx) & 1:
                size += orbit_size
        if 41 <= size <= 46:
            valid_masks.append((mask, size))

    count = len(valid_masks)
    lo = np.empty(count, dtype=np.uint64)
    hi = np.empty(count, dtype=np.uint64)
    size_arr = np.empty(count, dtype=np.int16)
    auto_arr = np.empty((count, len(p_reps)), dtype=np.int16)

    for row_idx, (mask, size) in enumerate(valid_masks):
        value = 0
        for orbit_idx, orbit_mask in enumerate(orbit_masks):
            if (mask >> orbit_idx) & 1:
                value |= orbit_mask
        lo[row_idx] = np.uint64(value & LOW64_MASK)
        hi[row_idx] = np.uint64(value >> 64)
        size_arr[row_idx] = size
        for rep_idx, rep in enumerate(p_reps):
            auto_arr[row_idx, rep_idx] = (value & _rotate_right_85(value, rep)).bit_count()

    distribution: dict[str, int] = {}
    for size in size_arr.tolist():
        distribution[str(int(size))] = distribution.get(str(int(size)), 0) + 1
    info = {
        "choice_count": int(count),
        "size_distribution": distribution,
        "elapsed_sec": time.time() - started,
    }
    print(
        f"eval621 built C choices count={count} size_distribution={distribution} "
        f"elapsed_sec={info['elapsed_sec']:.2f}",
        flush=True,
    )
    return lo, hi, size_arr, auto_arr, info


@njit(cache=False, parallel=True)
def _count_valid_a_by_c(A_edge_reps, A_auto_reps, A_sizes, C_auto_reps, C_sizes):
    num_c = C_auto_reps.shape[0]
    counts = np.zeros(num_c, dtype=np.int32)
    for c_idx in prange(num_c):
        c_size = int(C_sizes[c_idx])
        count = 0
        for a_idx in range(A_sizes.shape[0]):
            nonedge_limit = 2 * (int(A_sizes[a_idx]) + c_size) - 126
            ok = True
            for rep_idx in range(A_auto_reps.shape[1]):
                value = int(A_auto_reps[a_idx, rep_idx]) + int(C_auto_reps[c_idx, rep_idx])
                limit = 41 if A_edge_reps[a_idx, rep_idx] != 0 else nonedge_limit
                if value > limit:
                    ok = False
                    break
            if ok:
                count += 1
        counts[c_idx] = count
    return counts


@njit(cache=False)
def _has_packed_bit(lo, hi, idx):
    if idx < 64:
        return ((lo >> np.uint64(idx)) & np.uint64(1)) != np.uint64(0)
    return ((hi >> np.uint64(idx - 64)) & np.uint64(1)) != np.uint64(0)


@njit(cache=False)
def _find_cross_tuple(nonzero_c_indices, A_arr, A_edge_reps, A_auto_reps, A_sizes, C_lo, C_hi, C_sizes, C_auto_reps, c_reps):
    checked_pairs = 0
    for pos in range(nonzero_c_indices.shape[0]):
        c_idx = int(nonzero_c_indices[pos])
        c_size = int(C_sizes[c_idx])
        valid = np.empty(A_sizes.shape[0], dtype=np.int32)
        valid_count = 0

        for a_idx in range(A_sizes.shape[0]):
            nonedge_limit = 2 * (int(A_sizes[a_idx]) + c_size) - 126
            ok = True
            for rep_idx in range(A_auto_reps.shape[1]):
                value = int(A_auto_reps[a_idx, rep_idx]) + int(C_auto_reps[c_idx, rep_idx])
                limit = 41 if A_edge_reps[a_idx, rep_idx] != 0 else nonedge_limit
                if value > limit:
                    ok = False
                    break
            if ok:
                valid[valid_count] = a_idx
                valid_count += 1

        for ai_pos in range(valid_count):
            a_idx = int(valid[ai_pos])
            for bi_pos in range(valid_count):
                b_idx = int(valid[bi_pos])
                checked_pairs += 1
                cross_nonedge_limit = int(A_sizes[a_idx]) + int(A_sizes[b_idx]) + 2 * c_size - 126
                ok_cross = True
                for rep_pos in range(c_reps.shape[0]):
                    x = int(c_reps[rep_pos])
                    value = 0
                    for y in range(M):
                        idx = (x - y) % M
                        if _has_packed_bit(C_lo[c_idx], C_hi[c_idx], idx):
                            value += int(A_arr[a_idx, y]) + int(A_arr[b_idx, y])
                    limit = 41 if _has_packed_bit(C_lo[c_idx], C_hi[c_idx], x) else cross_nonedge_limit
                    if value > limit:
                        ok_cross = False
                        break
                if ok_cross:
                    return True, a_idx, b_idx, c_idx, checked_pairs
    return False, -1, -1, -1, checked_pairs


def _c_row_from_packed(lo: int, hi: int) -> np.ndarray:
    row = np.zeros(M, dtype=np.uint8)
    for idx in range(M):
        if idx < 64:
            row[idx] = (lo >> idx) & 1
        else:
            row[idx] = (hi >> (idx - 64)) & 1
    return row


def _build_n43_from_abc(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> str:
    masks = [0] * TARGET_V
    for i in range(M):
        for j in range(i + 1, M):
            diff = (j - i) % M
            if int(A[diff]):
                _set_edge(masks, i, j)
            if int(B[diff]):
                _set_edge(masks, M + i, M + j)
    for i in range(M):
        for j in range(M):
            diff = (j - i) % M
            if int(C[diff]):
                _set_edge(masks, i, M + j)
    return _adjacency_string_from_masks(masks)


def _write_summary() -> None:
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(SUMMARY, handle, indent=2, sort_keys=True)


def _run_target_search() -> tuple[str | None, dict[str, Any]]:
    started = time.time()
    set_num_threads(THREAD_COUNT)

    c_orbits = _make_c_orbits()
    p_orbits = _make_p_orbits()
    p_reps = [orbit[0] for orbit in p_orbits]
    c_reps = [orbit[0] for orbit in c_orbits]
    SUMMARY["target_run"]["orbits"] = {
        "C_count": len(c_orbits),
        "P_count": len(p_orbits),
        "C_sizes": [len(orbit) for orbit in c_orbits],
        "P_sizes": [len(orbit) for orbit in p_orbits],
        "C_orbits": [list(orbit) for orbit in c_orbits],
        "P_orbits": [list(orbit) for orbit in p_orbits],
    }
    print(
        (
            f"eval621 orbits C_count={len(c_orbits)} P_count={len(p_orbits)} "
            f"C_sizes={[len(orbit) for orbit in c_orbits]} "
            f"P_sizes={[len(orbit) for orbit in p_orbits]}"
        ),
        flush=True,
    )

    A_arr, A_sizes, A_edge_reps, A_auto_reps, a_info = _build_a_choices(p_orbits, p_reps)
    C_lo, C_hi, C_sizes, C_auto_reps, c_info = _build_c_choices(c_orbits, p_reps)

    SUMMARY["target_run"]["A_choices"] = {k: v for k, v in a_info.items() if k != "packed_values"}
    SUMMARY["target_run"]["B_choices"] = SUMMARY["target_run"]["A_choices"]
    SUMMARY["target_run"]["C_choices"] = c_info
    SUMMARY["target_run"]["numba_threads"] = get_num_threads()

    pass_started = time.time()
    valid_counts = _count_valid_a_by_c(A_edge_reps, A_auto_reps, A_sizes, C_auto_reps, C_sizes)
    pass_elapsed = time.time() - pass_started
    nonzero_c_indices = np.flatnonzero(valid_counts).astype(np.int64)
    SUMMARY["target_run"]["internal_filter"] = {
        "elapsed_sec": pass_elapsed,
        "total_valid_A_occurrences": int(valid_counts.sum()),
        "max_valid_A_per_C": int(valid_counts.max()) if valid_counts.size else 0,
        "nonzero_C_count": int(nonzero_c_indices.shape[0]),
        "nonzero_C_indices_prefix": [int(x) for x in nonzero_c_indices[:50].tolist()],
    }
    print(
        (
            f"eval621 internal_filter elapsed_sec={pass_elapsed:.2f} "
            f"valid_A_total={int(valid_counts.sum())} "
            f"max_valid_A_per_C={int(valid_counts.max()) if valid_counts.size else 0} "
            f"nonzero_C_count={int(nonzero_c_indices.shape[0])}"
        ),
        flush=True,
    )

    cross_started = time.time()
    found, a_idx, b_idx, c_idx, checked_pairs = _find_cross_tuple(
        nonzero_c_indices,
        A_arr,
        A_edge_reps,
        A_auto_reps,
        A_sizes,
        C_lo,
        C_hi,
        C_sizes,
        C_auto_reps,
        np.asarray(c_reps, dtype=np.int64),
    )
    cross_elapsed = time.time() - cross_started
    SUMMARY["target_run"]["cross_filter"] = {
        "elapsed_sec": cross_elapsed,
        "checked_AB_pairs": int(checked_pairs),
        "found": bool(found),
        "a_idx": int(a_idx),
        "b_idx": int(b_idx),
        "c_idx": int(c_idx),
    }
    print(
        (
            f"eval621 cross_filter elapsed_sec={cross_elapsed:.2f} checked_AB_pairs={int(checked_pairs)} "
            f"found={bool(found)} a_idx={int(a_idx)} b_idx={int(b_idx)} c_idx={int(c_idx)}"
        ),
        flush=True,
    )

    bits: str | None = None
    if found:
        C_arr = _c_row_from_packed(int(C_lo[c_idx]), int(C_hi[c_idx]))
        bits = _build_n43_from_abc(A_arr[a_idx], A_arr[b_idx], C_arr)
        candidate_stats = _record(TARGET_N, "eval621_k4_uncoupled_exhaustive_candidate", bits, candidate=True)
        SUMMARY["target_run"]["solution"] = {
            "A_size": int(A_sizes[a_idx]),
            "B_size": int(A_sizes[b_idx]),
            "C_size": int(C_sizes[c_idx]),
            "A_residues": [int(i) for i, value in enumerate(A_arr[a_idx].tolist()) if value],
            "B_residues": [int(i) for i, value in enumerate(A_arr[b_idx].tolist()) if value],
            "C_residues": [int(i) for i, value in enumerate(C_arr.tolist()) if value],
            "exact_stats": candidate_stats,
        }
        if not candidate_stats["valid"]:
            SUMMARY["notes"].append("Reduced tuple failed full exact validation; it was not inserted into the batch.")
            bits = None

    SUMMARY["target_run"]["elapsed_sec"] = time.time() - started
    print(f"eval621 target_search elapsed_sec={SUMMARY['target_run']['elapsed_sec']:.2f}", flush=True)
    return bits, SUMMARY["target_run"]




def construct_n43_orbit_witness() -> str:
    bits, _summary = _run_target_search()
    if bits is None:
        raise RuntimeError("n43 orbit search did not find a valid tuple")
    stats = _analyze_bits(TARGET_N, bits)
    if not stats.get("valid"):
        raise RuntimeError(f"n43 orbit tuple failed exact validation: {stats}")
    return bits


if __name__ == "__main__":
    print(construct_n43_orbit_witness())
