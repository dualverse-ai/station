from __future__ import annotations

import hashlib
import itertools
import json
import argparse
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
from typing import Iterable

import numpy as np

from epoch_book_b_finite_generators import build_algorithmic_finite_witnesses_with_failures


MIN_N = 22
MAX_N = 100


CHAMBERS = ("BB", "BR", "RB", "RR")
CHAMBER_INDEX = {name: i for i, name in enumerate(CHAMBERS)}

CHAMBER_RULES = {
    "BB|BB": {"R": (1, 0), "not_R": (-1, 1)},
    "BB|BR": {"diag": (1, 0), "R": (-1, 0), "not_R": (1, 0)},
    "BB|RB": {"diag": (1, 0), "R": (1, 0), "not_R": (-1, 0)},
    "BB|RR": {"diag": (-1, 1), "R": (-1, 1), "not_R": (1, 0)},
    "BR|BR": {"R": (-1, 0), "not_R": (1, 1)},
    "BR|RB": {"diag": (-1, 0), "R": (-1, 0), "not_R": (1, 1)},
    "BR|RR": {"diag": (-1, 0), "R": (-1, 0), "not_R": (1, 0)},
    "RB|RB": {"R": (-1, 0), "not_R": (1, 1)},
    "RB|RR": {"diag": (-1, 0), "R": (1, 0), "not_R": (-1, 0)},
    "RR|RR": {"R": (1, 0), "not_R": (-1, 1)},
}

ENDPOINT_RULES = {
    "u": {"BB": (-1, 0), "BR": (-1, 1), "RB": (1, 0), "RR": (1, 0)},
    "v": {"BB": (-1, 0), "BR": (1, 0), "RB": (-1, 1), "RR": (1, 0)},
    "u|v": (-1, 0),
}

# These are source conference relations, not final book-graph certificates.
# They are the allowed non-prime-power inputs used by the bundled S/K family.
NON_PRIME_POWER_CONFERENCE_MASKS = {
    45: (
        11131596448600, 18340564747504, 5781006653864, 5840177513669,
        11125000575363, 18287932351302, 18347084027947, 5833562661406,
        11072348931125, 34693475381409, 31326175093002, 4388151316564,
        34683631536724, 31316388284065, 4378388237578, 34664046679818,
        31296827178068, 4358746434209, 30135862510584, 30112347198919,
        30118032484415, 25095814031864, 25101501023175, 25119336043583,
        15068367295992, 15086214752711, 15062732718655, 18806630511030,
        6766790938550, 9677657112502, 6760971891053, 9672711442285,
        18804241891181, 9626568093915, 18764465934043, 6741664497371,
        23661241396497, 16528495832162, 29148349164684, 13549143513698,
        26616166013580, 22438268033809, 2990771648140, 2072977979153,
        3663757948514,
    ),
    65: (
        8589934590, 32702208444638191557, 30240416579365388195,
        20119351150514634513, 10290220086755262089, 4376147643626814533,
        6583361713055332899, 26332734368684175639, 17503614208183756943,
        31544101928124392543, 33120827962859045439, 15496646957681906047,
        21803832571597097215, 13572891922385486845, 24948561726703805947,
        26007270602816950257, 17398079551283304425, 33510637918320685201,
        30594022942441138441, 23371243961714189349, 11686470596818879043,
        9888598950114168983, 19661794841760401679, 2555072563577889373,
        4966038187432494139, 19865841719344269553, 10216912409067983209,
        5666952402643198917, 3981910870427452835, 22844611630075252369,
        15749152612347483913, 17592446566152420389, 26103834759602264643,
        26990002577574900504, 17228415062928870552, 34180774603065470052,
        32010742718371254882, 26584191825197879700, 16822637425894664586,
        32559352333827229268, 30389189026606687786, 19558633734357555372,
        10877980496814217554, 6753964032542401228, 4313717273977840946,
        17794172952268475056, 26474863423952622920, 31976797666115204294,
        34417194546651613990, 4161623950616255036, 6458328401007262810,
        16645405077811685748, 25832715478822121706, 29545143459604710828,
        29688563174533080018, 7500672209221097270, 8074914292675317454,
        29991992814033984860, 32306975545370360506, 9288632851256112368,
        18548000551707306856, 259785424893227940, 404594920612679106,
        1038543556688949014, 1617288949704611470,
    ),
}

NON_PRIME_POWER_CONFERENCE_SOURCES = {
    45: "Mathon PC strongly regular graph relation (45,22,10,11)",
    65: "Gritsenko strongly regular graph relation (65,32,15,16)",
}


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def prime_power_decomposition(value: int) -> tuple[int, int] | None:
    if value < 2:
        return None
    if is_prime(value):
        return value, 1
    for p in range(2, isqrt(value) + 1):
        if value % p != 0 or not is_prime(p):
            continue
        remaining = value
        exponent = 0
        while remaining % p == 0:
            remaining //= p
            exponent += 1
        if remaining == 1:
            return p, exponent
    return None


def _trim(poly: list[int]) -> list[int]:
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _poly_mod(poly: list[int], divisor: list[int], p: int) -> list[int]:
    out = [coeff % p for coeff in poly]
    divisor = _trim([coeff % p for coeff in divisor[:]])
    if divisor == [0]:
        raise ZeroDivisionError("zero polynomial divisor")
    while len(out) >= len(divisor) and out != [0]:
        coeff = out[-1] % p
        offset = len(out) - len(divisor)
        if coeff:
            for i, div_coeff in enumerate(divisor):
                out[offset + i] = (out[offset + i] - coeff * div_coeff) % p
        _trim(out)
    return out


def _is_irreducible_monic(poly: list[int], p: int) -> bool:
    degree = len(poly) - 1
    if degree <= 0 or poly[-1] % p != 1 or poly[0] % p == 0:
        return False
    for divisor_degree in range(1, degree // 2 + 1):
        for coeffs in itertools.product(range(p), repeat=divisor_degree):
            divisor = list(coeffs) + [1]
            if divisor[0] % p == 0:
                continue
            if _poly_mod(poly, divisor, p) == [0]:
                return False
    return True


def find_irreducible_polynomial(p: int, degree: int) -> tuple[int, ...]:
    if degree == 1:
        return (0, 1)
    for coeffs in itertools.product(range(p), repeat=degree):
        candidate = list(coeffs) + [1]
        if _is_irreducible_monic(candidate, p):
            return tuple(candidate)
    raise RuntimeError(f"No irreducible polynomial found over F_{p} of degree {degree}")


@dataclass(frozen=True)
class FiniteField:
    p: int
    degree: int
    modulus: tuple[int, ...]

    @classmethod
    def of_size(cls, q: int) -> "FiniteField":
        decomposition = prime_power_decomposition(q)
        if decomposition is None:
            raise ValueError(f"{q} is not a prime power")
        p, degree = decomposition
        return cls(p, degree, find_irreducible_polynomial(p, degree))

    @property
    def size(self) -> int:
        return self.p ** self.degree

    def coeffs(self, value: int) -> tuple[int, ...]:
        out = []
        x = value
        for _ in range(self.degree):
            out.append(x % self.p)
            x //= self.p
        return tuple(out)

    def encode(self, coeffs: Iterable[int]) -> int:
        value = 0
        factor = 1
        for coeff in coeffs:
            value += (coeff % self.p) * factor
            factor *= self.p
        return value

    def add(self, a: int, b: int) -> int:
        if self.degree == 1:
            return (a + b) % self.p
        a_coeffs = self.coeffs(a)
        b_coeffs = self.coeffs(b)
        return self.encode((a_coeffs[i] + b_coeffs[i]) % self.p for i in range(self.degree))

    def neg(self, a: int) -> int:
        if self.degree == 1:
            return (-a) % self.p
        return self.encode((-coeff) % self.p for coeff in self.coeffs(a))

    def sub(self, a: int, b: int) -> int:
        return self.add(a, self.neg(b))

    def mul(self, a: int, b: int) -> int:
        if self.degree == 1:
            return (a * b) % self.p
        a_coeffs = self.coeffs(a)
        b_coeffs = self.coeffs(b)
        product = [0] * (2 * self.degree - 1)
        for i, ai in enumerate(a_coeffs):
            for j, bj in enumerate(b_coeffs):
                product[i + j] = (product[i + j] + ai * bj) % self.p
        for deg in range(len(product) - 1, self.degree - 1, -1):
            coeff = product[deg] % self.p
            if coeff:
                offset = deg - self.degree
                for r in range(self.degree):
                    product[offset + r] = (product[offset + r] - coeff * self.modulus[r]) % self.p
        return self.encode(product[: self.degree])


def quadratic_residues(field: FiniteField) -> set[int]:
    return {field.mul(x, x) for x in range(1, field.size)}


def adjacency_string_from_masks(masks: tuple[int, ...] | list[int]) -> str:
    return "".join("1" if (int(masks[j]) >> i) & 1 else "0" for j in range(1, len(masks)) for i in range(j))


def add_edge(masks: list[int], u: int, v: int) -> None:
    if u == v:
        return
    masks[u] |= 1 << v
    masks[v] |= 1 << u


def neg_set(values: set[int], modulus: int) -> set[int]:
    return {(-value) % modulus for value in values}


def paley_conference_relation(q: int) -> tuple[int, ...]:
    if q % 4 != 1:
        raise ValueError(f"q must be congruent to 1 mod 4, got {q}")
    field = FiniteField.of_size(q)
    squares = quadratic_residues(field)
    masks = [0] * q
    for j in range(1, q):
        for i in range(j):
            if field.sub(j, i) in squares:
                masks[i] |= 1 << j
                masks[j] |= 1 << i
    return tuple(masks)


def conference_relation_stats(r_masks: tuple[int, ...]) -> dict[str, object]:
    q = len(r_masks)
    expected_degree = (q - 1) // 2
    expected_lambda = (q - 5) // 4
    expected_mu = (q - 1) // 4
    loop_count = 0
    symmetry_mismatches = 0
    degrees = []
    adjacent_common = []
    nonadjacent_common = []
    for i, mask_i in enumerate(r_masks):
        loop_count += (mask_i >> i) & 1
        degrees.append((mask_i & ~(1 << i)).bit_count())
        for j in range(i + 1, q):
            symmetry_mismatches += int(((r_masks[i] >> j) & 1) != ((r_masks[j] >> i) & 1))
    for j in range(1, q):
        for i in range(j):
            common = (r_masks[i] & r_masks[j]).bit_count()
            if (r_masks[i] >> j) & 1:
                adjacent_common.append(common)
            else:
                nonadjacent_common.append(common)
    ok = (
        q % 4 == 1
        and loop_count == 0
        and symmetry_mismatches == 0
        and set(degrees) == {expected_degree}
        and set(adjacent_common) == {expected_lambda}
        and set(nonadjacent_common) == {expected_mu}
    )
    return {
        "ok": ok,
        "q": q,
        "degree_values": sorted(set(degrees)),
        "adjacent_common_values": sorted(set(adjacent_common)),
        "nonadjacent_common_values": sorted(set(nonadjacent_common)),
        "expected": (expected_degree, expected_lambda, expected_mu),
        "loop_count": loop_count,
        "symmetry_mismatches": symmetry_mismatches,
    }


def known_conference_relation(q: int) -> tuple[int, ...]:
    if q in NON_PRIME_POWER_CONFERENCE_MASKS:
        return NON_PRIME_POWER_CONFERENCE_MASKS[q]
    if q % 4 == 1 and prime_power_decomposition(q) is not None:
        return paley_conference_relation(q)
    raise ValueError(f"No conference relation source is embedded for q={q}")


def _decode_sk_vertex(index: int, q: int) -> tuple[str, str | None, int | None]:
    if index < 4 * q:
        return "chamber", CHAMBERS[index // q], index % q
    return "endpoint", "u" if index == 4 * q else "v", None


def sk_rule_for_pair(i: int, j: int, q: int, r_masks: tuple[int, ...]) -> tuple[int, int]:
    type_i, label_i, coord_i = _decode_sk_vertex(i, q)
    type_j, label_j, coord_j = _decode_sk_vertex(j, q)
    if type_i == "endpoint" and type_j == "endpoint":
        return ENDPOINT_RULES["u|v"]
    if type_i == "endpoint" or type_j == "endpoint":
        endpoint = label_i if type_i == "endpoint" else label_j
        chamber = label_j if type_i == "endpoint" else label_i
        return ENDPOINT_RULES[endpoint][chamber]
    if CHAMBER_INDEX[label_i] > CHAMBER_INDEX[label_j]:
        label_i, label_j = label_j, label_i
        coord_i, coord_j = coord_j, coord_i
    key = f"{label_i}|{label_j}"
    if coord_i == coord_j:
        status = "diag"
    else:
        status = "R" if (r_masks[coord_i] >> coord_j) & 1 else "not_R"
    return CHAMBER_RULES[key][status]


def build_sk_adjacency_from_relation(r_masks: tuple[int, ...]) -> str:
    check = conference_relation_stats(r_masks)
    if not check["ok"]:
        raise ValueError(f"R does not have the required conference parameters: {check}")
    q = len(r_masks)
    vertex_count = 4 * q + 2
    masks = [0] * vertex_count
    for j in range(1, vertex_count):
        for i in range(j):
            sign, _slack = sk_rule_for_pair(i, j, q, r_masks)
            if sign == 1:
                masks[i] |= 1 << j
                masks[j] |= 1 << i
    return adjacency_string_from_masks(masks)


def construct_sk_witness(n: int) -> tuple[str, str]:
    q = n - 1
    r_masks = known_conference_relation(q)
    source = NON_PRIME_POWER_CONFERENCE_SOURCES.get(q, f"Paley conference graph over GF({q})")
    return build_sk_adjacency_from_relation(r_masks), f"S/K conference construction from {source}"


def construct_two_copy_paley_witness(n: int) -> tuple[str, str]:
    q = 2 * n - 1
    if q % 4 != 1 or prime_power_decomposition(q) is None:
        raise ValueError(f"q=2n-1={q} is not an eligible Paley prime power")
    field = FiniteField.of_size(q)
    squares = quadratic_residues(field)
    vertex_count = 2 * q
    masks = [0] * vertex_count

    def is_edge(i: int, j: int) -> bool:
        layer_i, x = divmod(i, q)
        layer_j, y = divmod(j, q)
        if layer_i == layer_j:
            diff = field.sub(y, x)
            in_square = diff in squares
            return in_square if layer_i == 0 else not in_square
        if layer_i == 0 and layer_j == 1:
            return field.sub(y, x) in squares
        return field.sub(x, y) in squares

    for j in range(1, vertex_count):
        for i in range(j):
            if is_edge(i, j):
                masks[i] |= 1 << j
                masks[j] |= 1 << i
    return adjacency_string_from_masks(masks), f"two-copy Paley construction over GF({q})"


def construct_skew_paley_sum_kernel_witness(n: int) -> tuple[str, str]:
    q = 4 * n - 1
    if q % 8 != 3 or prime_power_decomposition(q) is None:
        raise ValueError(f"q=4n-1={q} is not an eligible skew-Paley prime power")
    field = FiniteField.of_size(q)
    squares = quadratic_residues(field)
    elements = list(range(1, q))
    index = {x: i for i, x in enumerate(elements)}
    masks = [0] * len(elements)
    for b_pos, y in enumerate(elements[1:], 1):
        for a_pos in range(b_pos):
            x = elements[a_pos]
            s = field.add(x, y)
            if s != 0 and s not in squares:
                i = index[x]
                j = index[y]
                masks[i] |= 1 << j
                masks[j] |= 1 << i
    return adjacency_string_from_masks(masks), f"skew-Paley sum-kernel construction over GF({q})"


def parse_graph(text: str, expected_vertices: int) -> tuple[int, ...]:
    bits = "".join(text.split())
    expected_len = expected_vertices * (expected_vertices - 1) // 2
    if len(bits) != expected_len:
        raise AssertionError(f"bad length: expected {expected_len}, got {len(bits)}")
    if any(ch not in "01" for ch in bits):
        raise AssertionError("adjacency string contains a non-binary character")
    masks = [0] * expected_vertices
    idx = 0
    for j in range(1, expected_vertices):
        for i in range(j):
            if bits[idx] == "1":
                masks[i] |= 1 << j
                masks[j] |= 1 << i
            idx += 1
    return tuple(masks)


def analyze_masks(masks: tuple[int, ...], n: int) -> dict[str, object]:
    vertex_count = len(masks)
    all_mask = (1 << vertex_count) - 1
    comp = tuple((all_mask ^ (1 << u)) ^ masks[u] for u in range(vertex_count))
    red_max = 0
    blue_max = 0
    red_bad = 0
    blue_bad = 0
    edge_count = 0
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


def verify_adjacency(adjacency: str, n: int) -> dict[str, object]:
    masks = parse_graph(adjacency, 4 * n - 2)
    return analyze_masks(masks, n)


def build_baseline() -> tuple[dict[int, str], list[dict[str, object]], list[dict[str, object]]]:
    constructors = [
        ("sk_conference", construct_sk_witness),
        ("two_copy_paley", construct_two_copy_paley_witness),
        ("skew_paley_sum_kernel", construct_skew_paley_sum_kernel_witness),
    ]
    witnesses: dict[int, str] = {}
    manifest: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for n in range(MIN_N, MAX_N + 1):
        for family, constructor in constructors:
            try:
                adjacency, source = constructor(n)
                stats = verify_adjacency(adjacency, n)
            except Exception as exc:
                rejected.append({"n": n, "family": family, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if not stats["valid"]:
                rejected.append({"n": n, "family": family, "reason": "failed exact book validation", "stats": stats})
                continue
            witnesses[n] = adjacency
            manifest.append(
                {
                    "n": n,
                    "family": family,
                    "source": source,
                    "vertices": 4 * n - 2,
                    "bit_length": len(adjacency),
                    "sha256": hashlib.sha256(adjacency.encode("ascii")).hexdigest(),
                    "stats": stats,
                }
            )
            break

    finite_targets = [22, 23, 24, 28, 29, 32, 34, 36, 39, 43]
    needed_finite = [n for n in finite_targets if n not in witnesses]
    if needed_finite:
        finite, finite_failures = build_algorithmic_finite_witnesses_with_failures(needed_finite)
        if finite_failures:
            rejected.append(
                {
                    "n": needed_finite,
                    "family": "algorithmic_finite_generators",
                    "reason": "one or more finite generators failed",
                    "failures": finite_failures,
                }
            )
        for n in needed_finite:
            if n not in finite:
                continue
            adjacency, meta = finite[n]
            stats = verify_adjacency(adjacency, n)
            if not stats["valid"]:
                rejected.append(
                    {
                        "n": n,
                        "family": meta.get("family", "algorithmic_finite_generators"),
                        "reason": "failed exact book validation",
                        "stats": stats,
                    }
                )
                continue
            witnesses[n] = adjacency
            manifest.append(
                {
                    "n": n,
                    "family": meta.get("family", "algorithmic_finite_generators"),
                    "source": meta.get("source", "algorithmic finite generator"),
                    "vertices": 4 * n - 2,
                    "bit_length": len(adjacency),
                    "sha256": hashlib.sha256(adjacency.encode("ascii")).hexdigest(),
                    "stats": stats,
                    "search": meta.get("search"),
                    "source_artifact": meta.get("source_artifact"),
                }
            )
    return witnesses, manifest, rejected


def make_regenerated_source_artifacts(manifest: list[dict[str, object]]) -> dict[str, object]:
    artifacts = []
    for entry in sorted(manifest, key=lambda row: int(row["n"])):
        source_artifact = entry.get("source_artifact")
        search = entry.get("search")
        if source_artifact is None and search is None:
            continue
        artifacts.append(
            {
                "n": int(entry["n"]),
                "family": entry["family"],
                "source": entry["source"],
                "stats": entry["stats"],
                "source_artifact": source_artifact,
                "search": search,
            }
        )
    return {
        "description": (
            "Regenerated finite source artifacts from the baseline builder run. "
            "These records are emitted by the included finite generators; n=36 "
            "is explicitly marked as the sole preloaded source-row exception. "
            "The records are not loaded as inputs by the builder."
        ),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def make_slim_manifest(witnesses: dict[int, str], manifest: list[dict[str, object]]) -> dict[str, object]:
    family_breakdown: dict[str, list[int]] = {}
    family_sources: dict[str, list[str]] = {}
    for entry in manifest:
        family = str(entry["family"])
        family_breakdown.setdefault(family, []).append(int(entry["n"]))
        source = str(entry["source"])
        if source not in family_sources.setdefault(family, []):
            family_sources[family].append(source)

    for values in family_breakdown.values():
        values.sort()

    return {
        "range": [MIN_N, MAX_N],
        "max_score": MAX_N - MIN_N + 1,
        "baseline_score": len(witnesses),
        "solved_count": len(witnesses),
        "solved_n": sorted(witnesses),
        "missing_n": [n for n in range(MIN_N, MAX_N + 1) if n not in witnesses],
        "missing_under_50": [n for n in range(MIN_N, 50) if n not in witnesses],
        "family_breakdown": family_breakdown,
        "source_summary": {
            "sk_conference": "S/K construction from conference relations on q=n-1, including q=45 and q=65 non-prime-power masks.",
            "two_copy_paley": "Two-copy Paley construction for q=2n-1 prime powers with q=1 mod 4.",
            "skew_paley_sum_kernel": "Skew-Paley sum-kernel construction for q=4n-1 prime powers with q=3 mod 8.",
            "algorithmic_tenax_bicirculant_tabu": "Fresh deterministic bicirculant Tabu search in two cyclic fibers.",
            "algorithmic_capacity_profile_z71": "Fresh n=36 capacity-profile A search plus fixed-A C completion.",
            "n36_preloaded_capacity_profile_source": "Narrow n=36 exception: preloaded Z71 capacity-profile A/C source row; the long generator remains available.",
            "algorithmic_z7x11_product_completion": "Fresh n=39 Z7xZ11 product-group search plus generated fixed-A C completion.",
            "algorithmic_n43_k4_orbit_deconvolution": "Deterministic n=43 k=4 orbit exhaustive deconvolution over Z85.",
        },
        "construction_files": [
            "build_epoch_book_b_baseline.py",
            "epoch_book_b_finite_generators.py",
            "epoch_book_b_tenax_tabu.c",
            "epoch_book_b_n36_c_search.c",
            "epoch_book_b_n39_phase1.c",
            "epoch_book_b_n39_phase2_template.c",
            "epoch_book_b_n43_orbit.py",
            "construction_survey.md",
            "bridge_algorithm_notes.md",
            "finite_search_algorithms.py",
        ],
        "witness_file": "baseline_witnesses.npy",
        "regenerated_source_artifacts_file": "regenerated_source_artifacts.json",
        "policy": (
            "Generated only from infinite/parametric constructions, allowed conference "
            "source masks, and finite algorithms that execute during the builder run. "
            "The only finite source preload exception is the n=36 Z71 capacity-profile "
            "A/C row; the long regeneration script remains in this folder. The builder "
            "does not load precomputed witness files, bridge source-law rows, weighted "
            "n39 packet rows, or direct finite certificates."
        ),
        "audit_note": (
            "This default manifest is intentionally compact. Run the builder with "
            "--write-audit-manifest PATH to write per-row hashes, stats, and rejected constructor attempts."
        ),
    }


def make_audit_manifest(
    witnesses: dict[int, str],
    manifest: list[dict[str, object]],
    rejected: list[dict[str, object]],
) -> dict[str, object]:
    payload = make_slim_manifest(witnesses, manifest)
    payload["manifest"] = manifest
    payload["rejected_attempts"] = rejected
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the epoch_book_b construction baseline.")
    parser.add_argument(
        "--write-audit-manifest",
        nargs="?",
        const="baseline_audit_manifest.json",
        default=None,
        metavar="PATH",
        help="Optionally write a verbose audit manifest with hashes, stats, and rejected attempts.",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    witnesses, manifest, rejected = build_baseline()
    np.save(out_dir / "baseline_witnesses.npy", witnesses, allow_pickle=True)
    payload = make_slim_manifest(witnesses, manifest)
    (out_dir / "baseline_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "regenerated_source_artifacts.json").write_text(
        json.dumps(make_regenerated_source_artifacts(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    audit_path = None
    if args.write_audit_manifest is not None:
        audit_path = Path(args.write_audit_manifest)
        if not audit_path.is_absolute():
            audit_path = out_dir / audit_path
        audit_path.write_text(
            json.dumps(make_audit_manifest(witnesses, manifest, rejected), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    status = {k: payload[k] for k in ("solved_count", "solved_n", "missing_n")}
    if audit_path is not None:
        status["audit_manifest"] = str(audit_path)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
