#!/usr/bin/env python3
"""
Compute dialogue statistics aggregated by model name.

Usage:
  python scripts/model_stat.py
  python scripts/model_stat.py --station_dir station_data
  python scripts/model_stat.py --station_data_path station_data
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional

import yaml


ACTION_PATTERN = re.compile(r"/execute_action\{([^}]+)\}")
STATION_TICK_PATTERN = re.compile(r"- Station Tick:\s*([0-9]+)")
AGENT_NAME_PATTERN = re.compile(r"- Agent Name:\s*(.+)")
INVALID_ACTION_PATTERNS = (
    "not recognized",
    "not recognized or implemented",
    "invalid action",
    "action not recognized",
    "unrecognized action",
)

ROW_ORDER = [
    "Average thinking length",
    "Average % of response with thinking",
    "Average non-thinking length (Response)",
    "Average reflect action frequency",
    "Average meta action frequency",
    "Average submit action frequency",
    "Average invalid action rate",
    "Average life span",
]

RATE_ROWS = {
    "Average % of response with thinking",
    "Average reflect action frequency",
    "Average meta action frequency",
    "Average submit action frequency",
    "Average invalid action rate",
}

INTEGER_ROWS = {
    "Average life span",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute live station dialogue statistics aggregated by model name."
    )
    parser.add_argument(
        "--station_dir",
        "--station_data_path",
        dest="station_dir",
        default="station_data",
        help="Path to the station_data directory (default: station_data).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def load_yaml_lines(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            return [doc for doc in yaml.safe_load_all(handle) if isinstance(doc, dict)]
    except yaml.YAMLError:
        return []


def extract_station_tick(content: str) -> Optional[int]:
    match = STATION_TICK_PATTERN.search(content or "")
    return int(match.group(1)) if match else None


def extract_agent_name(content: str) -> Optional[str]:
    match = AGENT_NAME_PATTERN.search(content or "")
    return match.group(1).strip() if match else None


def extract_commands(content: str) -> List[str]:
    commands = []
    for raw_command in ACTION_PATTERN.findall(content or ""):
        command = raw_command.strip().lower()
        if command:
            commands.append(command)
    return commands


def command_uses_action(commands: Iterable[str], action_name: str) -> bool:
    prefix = f"{action_name} "
    return any(command == action_name or command.startswith(prefix) for command in commands)


def outcome_has_invalid_action(doc: dict) -> bool:
    text_parts: List[str] = []
    error = doc.get("error")
    if isinstance(error, str):
        text_parts.append(error)
    for key in ("actions_executed_summary", "parsed_actions_detail"):
        value = doc.get(key)
        if isinstance(value, list):
            text_parts.extend(str(item) for item in value)
    joined = "\n".join(text_parts).lower()
    return any(pattern in joined for pattern in INVALID_ACTION_PATTERNS)


def mean_and_ci(values: List[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    margin = 1.96 * math.sqrt(variance / n)
    return mean, mean - margin, mean + margin


def format_stat(row_name: str, values: List[float]) -> str:
    n = len(values)
    if n == 0:
        return "n/a"
    mean, lower, upper = mean_and_ci(values)
    if row_name in RATE_ROWS:
        mean = min(max(mean, 0.0), 1.0)
        lower = min(max(lower, 0.0), 1.0)
        upper = min(max(upper, 0.0), 1.0)
    if row_name in INTEGER_ROWS:
        return f"{mean:.0f} (N={n}, [{lower:.0f}-{upper:.0f}])"
    return f"{mean:.4f} (N={n}, [{lower:.4f}-{upper:.4f}])"


def resolve_model_name(dialogue_docs: List[dict], dialogue_path: Path, agents_dir: Path) -> Optional[str]:
    agent_name: Optional[str] = None
    for doc in dialogue_docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("speaker") == "Station" and doc.get("type") == "observation":
            extracted = extract_agent_name(doc.get("content", ""))
            if extracted:
                agent_name = extracted
        elif doc.get("speaker") == "AgentLLM":
            extracted = doc.get("agent_name")
            if isinstance(extracted, str) and extracted.strip():
                agent_name = extracted.strip()

    if not agent_name:
        agent_name = dialogue_path.name.removesuffix("_dialogue.yamll")

    agent_yaml = load_yaml(agents_dir / f"{agent_name}.yaml")
    if not agent_yaml:
        agent_yaml = load_yaml(agents_dir / f"{dialogue_path.name.removesuffix('_dialogue.yamll')}.yaml")
    if not agent_yaml:
        return None

    model_name = agent_yaml.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    return model_name.strip()


def maybe_add_life_span(
    grouped: DefaultDict[str, DefaultDict[str, List[float]]],
    model_name: str,
    dialogue_docs: List[dict],
    dialogue_path: Path,
    agents_dir: Path,
) -> None:
    final_agent_name: Optional[str] = None
    for doc in dialogue_docs:
        if not isinstance(doc, dict):
            continue
        if doc.get("speaker") == "Station" and doc.get("type") == "observation":
            extracted = extract_agent_name(doc.get("content", ""))
            if extracted:
                final_agent_name = extracted
        elif doc.get("speaker") == "AgentLLM":
            extracted = doc.get("agent_name")
            if isinstance(extracted, str) and extracted.strip():
                final_agent_name = extracted.strip()

    if not final_agent_name:
        final_agent_name = dialogue_path.name.removesuffix("_dialogue.yamll")

    agent_yaml = load_yaml(agents_dir / f"{final_agent_name}.yaml")
    if not agent_yaml:
        return

    tick_birth = agent_yaml.get("tick_birth")
    tick_exit = agent_yaml.get("tick_exit")
    if isinstance(tick_birth, int) and isinstance(tick_exit, int):
        grouped[model_name]["Average life span"].append(float(max(0, tick_exit - tick_birth)))


def collect_samples_for_dialogue(
    grouped: DefaultDict[str, DefaultDict[str, List[float]]],
    model_name: str,
    dialogue_docs: List[dict],
) -> None:
    pending_turn_commands: Optional[List[str]] = None
    pending_thinking_length: Optional[float] = None
    pending_response_recorded = False

    for doc in dialogue_docs:
        if not isinstance(doc, dict):
            continue

        speaker = doc.get("speaker")
        doc_type = doc.get("type")

        if speaker == "Station" and doc_type == "observation":
            content = doc.get("content", "")
            if "## System Information" in content and extract_station_tick(content) is not None:
                pending_turn_commands = []
                pending_thinking_length = None
                pending_response_recorded = False

        elif speaker == "AgentLLM" and doc_type == "thinking_block":
            if pending_turn_commands is not None:
                pending_thinking_length = float(len((doc.get("content") or "").strip()))

        elif speaker == "Agent" and doc_type == "submission":
            if pending_turn_commands is None or pending_response_recorded:
                continue
            content = (doc.get("content") or "").strip()
            grouped[model_name]["Average thinking length"].append(pending_thinking_length or 0.0)
            grouped[model_name]["Average % of response with thinking"].append(
                1.0 if pending_thinking_length is not None else 0.0
            )
            grouped[model_name]["Average non-thinking length (Response)"].append(float(len(content)))
            pending_turn_commands.extend(extract_commands(content))
            pending_response_recorded = True

        elif speaker == "Station" and doc_type == "submission_outcome":
            if pending_turn_commands is None or not pending_response_recorded:
                continue
            grouped[model_name]["Average reflect action frequency"].append(
                1.0 if command_uses_action(pending_turn_commands, "reflect") else 0.0
            )
            grouped[model_name]["Average meta action frequency"].append(
                1.0 if command_uses_action(pending_turn_commands, "meta") else 0.0
            )
            grouped[model_name]["Average submit action frequency"].append(
                1.0 if command_uses_action(pending_turn_commands, "submit") else 0.0
            )
            grouped[model_name]["Average invalid action rate"].append(
                1.0 if outcome_has_invalid_action(doc) else 0.0
            )
            pending_turn_commands = None
            pending_thinking_length = None
            pending_response_recorded = False


def build_markdown_table(grouped: DefaultDict[str, DefaultDict[str, List[float]]]) -> str:
    models = sorted(grouped)
    header = "| Statistic | " + " | ".join(models) + " |"
    separator = "|---|" + "|".join(["---"] * len(models)) + "|"
    lines = [header, separator]
    for row_name in ROW_ORDER:
        row_values = [format_stat(row_name, grouped[model].get(row_name, [])) for model in models]
        lines.append("| " + row_name + " | " + " | ".join(row_values) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    station_dir = Path(args.station_dir).resolve()
    dialogue_dir = station_dir / "dialogue_logs"
    agents_dir = station_dir / "agents"

    if not dialogue_dir.is_dir():
        print(f"Dialogue log directory not found: {dialogue_dir}", file=sys.stderr)
        return 1
    if not agents_dir.is_dir():
        print(f"Agent directory not found: {agents_dir}", file=sys.stderr)
        return 1

    grouped: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    skipped_logs: List[str] = []

    for dialogue_path in sorted(dialogue_dir.glob("*_dialogue.yamll")):
        if dialogue_path.name.startswith("Guest_"):
            continue

        docs = load_yaml_lines(dialogue_path)
        if not docs:
            skipped_logs.append(f"{dialogue_path.name}: empty or unreadable")
            continue

        model_name = resolve_model_name(docs, dialogue_path, agents_dir)
        if not model_name:
            skipped_logs.append(f"{dialogue_path.name}: could not resolve model_name")
            continue

        collect_samples_for_dialogue(grouped, model_name, docs)
        maybe_add_life_span(grouped, model_name, docs, dialogue_path, agents_dir)

    if not grouped:
        print("No dialogue statistics could be computed.", file=sys.stderr)
        if skipped_logs:
            print("\n".join(skipped_logs), file=sys.stderr)
        return 1

    print(build_markdown_table(grouped))

    if skipped_logs:
        print("", file=sys.stderr)
        print("Skipped logs:", file=sys.stderr)
        for skipped in skipped_logs:
            print(f"- {skipped}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
