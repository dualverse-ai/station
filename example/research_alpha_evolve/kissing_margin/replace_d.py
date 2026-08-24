#!/usr/bin/env python3
"""
Retarget the live kissing-margin station_data task to a new dimension.

Usage:
    python example/research_alpha_evolve/kissing_margin/replace_d.py 11

The fixed vector count is derived from bound.csv as:

    target_n = lower_bound(d) + 1

For example, d=11 has lower bound 593, so target_n=594.
"""

import csv
import re
import sys
from pathlib import Path

from station import file_io_utils


PACKAGE_ROOT = Path(__file__).resolve().parent
BOUND_CSV = PACKAGE_ROOT / "bound.csv"
TASK_FILENAMES = [
    "research_task.md",
    "research_task_phase2.md",
]
EVALUATOR_FILENAMES = [
    "evaluator.py",
    "evaluator_phase2.py",
]

# Only update the live station_data tree, not this example copy.
RESEARCH_ROOTS = [
    PACKAGE_ROOT.parents[2] / "station_data" / "rooms" / "research",
]


def load_bounds(dimension: int) -> tuple[int, int]:
    with BOUND_CSV.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Dimension"]) == dimension:
                return int(row["Lower bound"]), int(row["Upper bound"])
    raise ValueError(f"Dimension {dimension} not found in {BOUND_CSV}")


def replace_required(pattern: str, replacement: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text)
    if count == 0:
        raise ValueError(f"Could not update {label}")
    return new_text


def replace_first_required(pattern: str, replacement: str, text: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Could not update {label}")
    return new_text


def replace_optional(pattern: str, replacement: str, text: str) -> str:
    return re.sub(pattern, lambda _match: replacement, text)


def replace_optional_with_match(pattern: str, replacement, text: str) -> str:
    return re.sub(pattern, replacement, text)


def detect_current_values(
    task_texts: list[str],
    evaluator_texts: list[str],
    baseline_text: str,
) -> tuple[int, int]:
    joined_task_text = "\n".join(task_texts)
    joined_evaluator_text = "\n".join(evaluator_texts)
    marker = re.search(
        r"kissing_margin_dimension=(\d+)\s+kissing_margin_target_n=(\d+)",
        joined_task_text,
    )
    if marker:
        return int(marker.group(1)), int(marker.group(2))

    dimension_patterns = [
        (r"TARGET_DIMENSION\s*=\s*(\d+)", joined_evaluator_text),
        (r"DIMENSION\s*=\s*(\d+)", baseline_text),
        (r"Kissing Number Lower Bound for d\s*=\s*(\d+)", joined_task_text),
        (r"\bd\s*=\s*(\d+)\s*\.", joined_task_text),
        (r"dimension\s+\**(\d+)\**", joined_task_text),
    ]
    target_patterns = [
        (r"TARGET_COUNT\s*=\s*(\d+)", joined_evaluator_text),
        (r"TARGET_SPHERES\s*=\s*(\d+)", baseline_text),
        (r"exactly\s+\**(\d+)\s+non-zero vectors", joined_task_text),
        (r"at least\s+\**(\d+)\s+non-zero vectors", joined_task_text),
        (r"with\s+\**(\d+)\s+surrounding spheres", joined_task_text),
        (r"N\s*\\ge\s*(\d+)", joined_task_text),
    ]

    old_dimension = find_first_int(dimension_patterns, "current dimension")
    old_target_n = find_first_int(target_patterns, "current target vector count")
    return old_dimension, old_target_n


def find_first_int(patterns: list[tuple[str, str]], label: str) -> int:
    for pattern, text in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not detect {label}; keep the marker comment or TARGET_* constants present.")


def update_task_text(
    text: str,
    dimension: int,
    target_n: int,
    lower_bound: int,
) -> str:
    new_zero_tail = max(0, dimension - 3)

    text = replace_optional(
        r"kissing_margin_dimension=\d+\s+kissing_margin_target_n=\d+",
        f"kissing_margin_dimension={dimension} kissing_margin_target_n={target_n}",
        text,
    )
    text = replace_first_required(
        r"Kissing Number Lower Bound for d\s*=\s*\d+",
        f"Kissing Number Lower Bound for d={dimension}",
        text,
        "task title dimension",
    )

    dimension_replacements = [
        (r"\\+mathbb\{R\}\^\{\d+\}", rf"\mathbb{{R}}^{{{dimension}}}"),
        (r"(?<!for )\bd\s*=\s*\d+", f"d = {dimension}"),
        (r"dimension \\\(\d+\\\)", rf"dimension \({dimension}\)"),
        (r"dimension \*\*\d+\*\*", f"dimension **{dimension}**"),
        (r"dimension `\d+`", f"dimension `{dimension}`"),
        (r"dimension \(\d+\)", f"dimension ({dimension})"),
        (r"dimension \d+", f"dimension {dimension}"),
        (r"Dimension \d+", f"Dimension {dimension}"),
    ]
    for pattern, replacement in dimension_replacements:
        text = replace_optional(pattern, replacement, text)

    count_replacements = [
        (
            r"\*\*at least \d+\*\*(?=\s+surrounding spheres)",
            f"**at least {target_n}**",
        ),
        (
            r"(?<=The goal is to exhibit )\*\*\d+\*\*(?=\s+surrounding spheres)",
            f"**{target_n}**",
        ),
        (
            r"\*\*\d+\s+non-zero direction vectors\*\*",
            f"**{target_n} non-zero direction vectors**",
        ),
        (
            r"\*\*\d+\s+non-zero vectors\*\*",
            f"**{target_n} non-zero vectors**",
        ),
        (
            r"(?<=at least \*\*)\d+(?=\s+non-zero vectors\*\*)",
            str(target_n),
        ),
        (
            r"N\s*\\ge\s*\d+",
            rf"N \ge {target_n}",
        ),
        (
            r"(?<=with \*\*)\d+(?=\*\*\s+surrounding spheres)",
            str(target_n),
        ),
        (
            r"(?<=returns \*\*)\d+(?=\*\*\s+vectors)",
            str(target_n),
        ),
        (
            r"(?<=N = )\d+",
            str(target_n),
        ),
        (
            r"(?<=for _ in range\()\d+(?=\))",
            str(target_n),
        ),
        (
            r"np\.zeros\(\(\d+,\s*\d+\)",
            f"np.zeros(({target_n}, {dimension})",
        ),
        (
            r"np\.zeros\(\(N,\s*\d+\)",
            f"np.zeros((N, {dimension})",
        ),
        (
            r"\(\d+,\s*\d+\)(?=;\s*got ndim=)",
            f"({target_n}, {dimension})",
        ),
        (
            r"(?<=count other than \*\*)\d+(?=\*\*)",
            str(target_n),
        ),
        (
            r"(?<=size \*\*)\d+(?=\*\*)",
            str(target_n),
        ),
    ]
    for pattern, replacement in count_replacements:
        text = replace_optional(pattern, replacement, text)

    text = replace_optional_with_match(
        r"(1\s*\\le\s*i\s*<\s*j\s*\\le\s*)\d+",
        lambda match: f"{match.group(1)}{target_n}",
        text,
    )

    text = replace_optional(r"\[0\] \* \d+", f"[0] * {new_zero_tail}", text)
    text = replace_optional(r"namely \d+", f"namely {lower_bound}", text)
    text = replace_optional(r"lower bound `\d+`", f"lower bound `{lower_bound}`", text)
    text = replace_optional(r"lower bound \d+", f"lower bound {lower_bound}", text)
    return text


def update_evaluator(text: str, dimension: int, target_n: int) -> str:
    text = replace_required(
        r"TARGET_DIMENSION\s*=\s*\d+",
        f"TARGET_DIMENSION = {dimension}",
        text,
        "TARGET_DIMENSION",
    )
    text = replace_required(
        r"TARGET_COUNT\s*=\s*\d+",
        f"TARGET_COUNT = {target_n}",
        text,
        "TARGET_COUNT",
    )
    return text


def update_baseline(text: str, dimension: int, target_n: int) -> str:
    text = re.sub(
        r"Baseline submission that performs a short random repulsion search for \d+ vectors in \d+ dimensions\.",
        f"Baseline submission that performs a short random repulsion search for {target_n} vectors in {dimension} dimensions.",
        text,
    )
    text = replace_required(
        r"DIMENSION = \d+",
        f"DIMENSION = {dimension}",
        text,
        "baseline dimension",
    )
    text = replace_required(
        r"TARGET_SPHERES = \d+",
        f"TARGET_SPHERES = {target_n}",
        text,
        "baseline target spheres",
    )
    return text


def process_root(root: Path, dimension: int, target_n: int, lower_bound: int) -> None:
    task_paths = [root / filename for filename in TASK_FILENAMES]
    evaluator_paths = [root / "evaluators" / filename for filename in EVALUATOR_FILENAMES]
    baseline_path = root / "baseline.yamll"

    existing_task_paths = [path for path in task_paths if path.exists()]
    existing_evaluator_paths = [path for path in evaluator_paths if path.exists()]
    if not existing_task_paths:
        print(f"Skipping missing root: {root}")
        return

    task_texts = [path.read_text(encoding="utf-8") for path in existing_task_paths]
    evaluator_texts = [path.read_text(encoding="utf-8") for path in existing_evaluator_paths]
    baseline_text = baseline_path.read_text(encoding="utf-8") if baseline_path.exists() else ""
    old_dimension, old_target_n = detect_current_values(task_texts, evaluator_texts, baseline_text)

    updated_files = []
    for path, text in zip(existing_task_paths, task_texts):
        file_io_utils.save_text(
            update_task_text(text, dimension, target_n, lower_bound),
            str(path),
        )
        updated_files.append(path.name)

    for path, text in zip(existing_evaluator_paths, evaluator_texts):
        file_io_utils.save_text(
            update_evaluator(text, dimension, target_n),
            str(path),
        )
        updated_files.append(f"evaluators/{path.name}")

    if baseline_path.exists():
        file_io_utils.save_text(
            update_baseline(baseline_text, dimension, target_n),
            str(baseline_path),
        )
        updated_files.append(baseline_path.name)

    print(
        f"Updated {root} from d={old_dimension}, n={old_target_n} "
        f"to d={dimension}, fixed n={target_n} (lower bound {lower_bound} + 1): "
        + ", ".join(updated_files)
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python example/research_alpha_evolve/kissing_margin/replace_d.py <dimension>")
        sys.exit(1)

    try:
        dimension = int(sys.argv[1])
    except ValueError:
        print("Dimension must be an integer.")
        sys.exit(1)

    if dimension <= 0:
        print("Dimension must be positive.")
        sys.exit(1)

    lower_bound, _upper_bound = load_bounds(dimension)
    target_n = lower_bound + 1

    for root in RESEARCH_ROOTS:
        process_root(root, dimension, target_n, lower_bound)


if __name__ == "__main__":
    main()
