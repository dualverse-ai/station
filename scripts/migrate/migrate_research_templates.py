#!/usr/bin/env python3
"""
Migrate research task templates to the coder-era Research Center layout.

Changes:
- `research_tasks.yaml` -> `research_task.md`
- `evaluators/task_1_evaluator.py` -> `evaluators/evaluator.py`
- `pending_evaluations.yamll` -> `baseline.yamll`
- remove duplicated evaluator files from `storage/system/`
- apply focused wording updates for Hadamard task specs
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PATTERNS = [
    ("example", "research_*"),
    ("example_private", "research_*"),
]


def iter_template_roots() -> Iterable[Path]:
    for base_name, pattern in TARGET_PATTERNS:
        base = REPO_ROOT / base_name
        if not base.exists():
            continue
        for root in sorted(base.glob(pattern)):
            research_root = root / "research"
            if research_root.exists():
                yield research_root


def _pick_primary_task(task_data):
    if isinstance(task_data, dict):
        return task_data
    if isinstance(task_data, list):
        for item in task_data:
            if isinstance(item, dict) and str(item.get("id")) == "1":
                return item
        for item in task_data:
            if isinstance(item, dict):
                return item
    return None


def build_markdown_from_legacy_yaml(yaml_path: Path) -> str:
    with yaml_path.open("r", encoding="utf-8") as handle:
        task_data = yaml.safe_load(handle)

    task = _pick_primary_task(task_data)
    if not isinstance(task, dict):
        return ""

    content = str(task.get("content") or "").strip()
    if content:
        return content.rstrip() + "\n"

    title = str(task.get("title") or "").strip()
    description = str(task.get("description") or "").strip()
    evaluation_criteria = str(task.get("evaluation_criteria") or "").strip()
    details = str(task.get("details") or "").strip()

    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    if description:
        lines.extend([description, ""])
    if evaluation_criteria:
        lines.extend(["## Evaluation Criteria", evaluation_criteria, ""])
    if details:
        lines.extend(["## Details", details, ""])
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def patch_hadamard_task_markdown(text: str) -> str:
    replacements = [
        (
            "Each agent can have at most 2 experiments running simultaneously.",
            "Each agent can have at most 1 experiment running simultaneously.",
        ),
        (
            "- The `content` field of your submission YAML should contain a complete Python script.",
            "- Your coder should submit a complete Python script.",
        ),
        (
            "- This script will be saved directly by the Station as `run.py`.",
            "- The submitted Python file will be executed by the Station evaluator.",
        ),
        (
            "- Your `run.py` script **must** define:",
            "- The submitted Python file **must** define:",
        ),
        (
            "`execute_action{storage read system/task_1_evaluator.py}`.",
            "`/execute_action{read system/evaluator.py}`.",
        ),
        (
            "You may treat `construct_hadamard()` as a sandbox runner.",
            "Your coder may treat `construct_hadamard()` as the evaluator entrypoint.",
        ),
        (
            "Use `/execute_action{review id}` to examine baseline code and results.",
            "Use `/execute_action{review 1}` to examine the system baseline result.",
        ),
    ]

    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated


def migrate_task_spec(research_root: Path):
    yaml_path = research_root / "research_tasks.yaml"
    md_path = research_root / "research_task.md"

    if yaml_path.exists():
        markdown = build_markdown_from_legacy_yaml(yaml_path)
        if research_root.parent.name.startswith("research_epoch_hadamard"):
            markdown = patch_hadamard_task_markdown(markdown)
        md_path.write_text(markdown, encoding="utf-8")
        yaml_path.unlink()
    elif md_path.exists() and research_root.parent.name.startswith("research_epoch_hadamard"):
        md_path.write_text(patch_hadamard_task_markdown(md_path.read_text(encoding="utf-8")), encoding="utf-8")


def migrate_evaluator(research_root: Path):
    evaluators_dir = research_root / "evaluators"
    if not evaluators_dir.exists():
        return

    canonical_path = evaluators_dir / "evaluator.py"
    if not canonical_path.exists():
        legacy_paths = sorted(evaluators_dir.glob("task_*_evaluator.py"))
        if len(legacy_paths) == 1:
            legacy_paths[0].rename(canonical_path)
        elif len(legacy_paths) > 1:
            raise RuntimeError(f"Multiple legacy evaluator files in {evaluators_dir}: {legacy_paths}")

    for legacy_path in sorted(evaluators_dir.glob("task_*_evaluator.py")):
        if legacy_path.exists():
            legacy_path.unlink()

    if canonical_path.exists():
        text = canonical_path.read_text(encoding="utf-8")
        updated = text.replace("task_1_evaluator.py", "evaluator.py")
        if updated != text:
            canonical_path.write_text(updated, encoding="utf-8")


def migrate_baseline(research_root: Path):
    pending_path = research_root / "pending_evaluations.yamll"
    baseline_path = research_root / "baseline.yamll"
    if pending_path.exists():
        if baseline_path.exists():
            pending_path.unlink()
        else:
            pending_path.rename(baseline_path)


def remove_duplicated_system_evaluators(research_root: Path):
    system_dir = research_root / "storage" / "system"
    if not system_dir.exists():
        return

    for path in list(system_dir.glob("task_*_evaluator.py")) + [system_dir / "evaluator.py"]:
        if path.exists() or path.is_symlink():
            path.unlink()


def main():
    migrated = 0
    for research_root in iter_template_roots():
        migrate_task_spec(research_root)
        migrate_evaluator(research_root)
        migrate_baseline(research_root)
        remove_duplicated_system_evaluators(research_root)
        migrated += 1
        print(f"Migrated {research_root}")

    print(f"Done. Migrated {migrated} research templates.")


if __name__ == "__main__":
    main()
