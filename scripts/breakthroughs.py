#!/usr/bin/env python3
"""Analyze Research Center evaluations for canonical breakthrough events."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from station import constants
from station.eval_research import breakthroughs as research_breakthroughs
from station.eval_research.evaluation_manager import EvaluationManager
from station.file_io_utils import load_yaml


DEFAULT_CSV_PATH = "/tmp/breakthroughs.csv"
DEFAULT_BREAKTHROUGH_EPS = float(getattr(constants, "BREAKTHROUGH_EPS", 1e-2))
PENDING_SCORE_VALUES = {"pending", "n.a.", "na", "n/a"}


@dataclass(frozen=True)
class ScoredEvaluation:
    eval_id: str
    eval_num: int
    agent_name: str
    score: float
    sort_key: Tuple[Any, ...]
    submitted_tick: Optional[int]
    title: str
    abstract: str
    tags: List[str]
    task_id: Optional[Any]
    source_path: Path


@dataclass(frozen=True)
class Breakthrough:
    kind: str
    track: str
    eval_id: str
    agent_name: str
    score: Any
    value: Any
    previous_value: Any
    tick: Optional[int]
    title: str
    abstract: str
    tags: List[str]
    sort_key: Tuple[Any, ...]
    previous_sort_key: Optional[Tuple[Any, ...]]
    task_id: Optional[Any]
    is_breakthrough: bool = True

    @property
    def improvement(self) -> Optional[float]:
        current = _coerce_float(self.value)
        previous = _coerce_float(self.previous_value)
        if current is None or previous is None:
            return None
        return current - previous


def _is_pending_score(score: Any) -> bool:
    if score is None:
        return True
    if isinstance(score, str):
        return score.strip().lower() in PENDING_SCORE_VALUES
    return False


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_sort_key_component(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.startswith("*"):
        return None

    try:
        return int(text)
    except (TypeError, ValueError):
        pass

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_sort_key(sort_key: Any, score: Any) -> Optional[Tuple[Any, ...]]:
    """Match EvaluationManager sort-key normalization for breakthrough comparison."""
    if sort_key is not None:
        raw_items = tuple(sort_key) if isinstance(sort_key, (list, tuple)) else (sort_key,)
        normalized_items = []
        for item in raw_items:
            normalized_item = _normalize_sort_key_component(item)
            if normalized_item is None:
                normalized_items = []
                break
            normalized_items.append(normalized_item)
        if normalized_items:
            return tuple(normalized_items)

    score_float = _coerce_float(score)
    if score_float is None:
        return None
    return (score_float,)


def _normalize_tags(tags: Any) -> List[str]:
    if isinstance(tags, str):
        return [part.strip() for part in tags.split(",") if part.strip()]
    if isinstance(tags, Sequence) and not isinstance(tags, (bytes, bytearray)):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    return []


def _coerce_tick(tick: Any) -> Optional[int]:
    try:
        return int(tick)
    except (TypeError, ValueError):
        return None


def _coerce_eval_num(eval_id: Any) -> Optional[int]:
    try:
        return int(str(eval_id))
    except (TypeError, ValueError):
        return None


def _load_agent_model(agents_dir: Path, agent_name: str) -> str:
    if not agent_name or os.sep in agent_name or (os.altsep and os.altsep in agent_name):
        return "N/A"

    agent_path = agents_dir / f"{agent_name}{constants.YAML_EXTENSION}"
    agent_data = load_yaml(str(agent_path))
    if not isinstance(agent_data, dict):
        return "N/A"

    model_name = str(agent_data.get(constants.AGENT_MODEL_NAME_KEY) or "").strip()
    return model_name or "N/A"


def _build_scored_evaluation(
    evaluations_dir: Path,
    display_info: Dict[str, Any],
    tag_filter: Optional[str] = None,
) -> Optional[ScoredEvaluation]:
    eval_id = str(display_info.get(constants.EVALUATION_ID_KEY) or "").strip()
    eval_num = _coerce_eval_num(eval_id)
    if eval_num is None:
        return None

    score = display_info.get(constants.EVALUATION_SCORE_KEY, constants.RESEARCH_SCORE_NA)
    if _is_pending_score(score):
        return None
    sort_key = normalize_sort_key(display_info.get("sort_key"), score)
    if sort_key is None:
        return None

    score_float = _coerce_float(score)
    if score_float is None:
        return None

    agent_name = str(display_info.get(constants.EVALUATION_AUTHOR_KEY) or "").strip()
    if not agent_name:
        return None

    tags = _normalize_tags(display_info.get(constants.EVALUATION_TAGS_KEY, []))
    if tag_filter and tag_filter not in tags:
        return None

    return ScoredEvaluation(
        eval_id=eval_id,
        eval_num=eval_num,
        agent_name=agent_name,
        score=score_float,
        sort_key=sort_key,
        submitted_tick=_coerce_tick(display_info.get(constants.EVALUATION_SUBMITTED_TICK_KEY)),
        title=str(display_info.get(constants.EVALUATION_TITLE_KEY) or "Untitled"),
        abstract=str(display_info.get(constants.EVALUATION_ABSTRACT_KEY) or ""),
        tags=tags,
        task_id=None,
        source_path=evaluations_dir / f"{eval_id}{constants.RESEARCH_EVALUATION_FILE_EXTENSION}",
    )


def collect_scored_evaluations(station_data_path: str, tag_filter: Optional[str] = None) -> List[ScoredEvaluation]:
    evaluations_dir = Path(station_data_path) / "rooms" / "research" / "evaluations"
    if not evaluations_dir.exists():
        raise FileNotFoundError(f"Research evaluations directory not found: {evaluations_dir}")

    eval_manager = EvaluationManager(str(evaluations_dir))
    scored_evaluations: List[ScoredEvaluation] = []
    for display_info in eval_manager.get_compact_display_infos():
        scored = _build_scored_evaluation(
            evaluations_dir,
            display_info,
            tag_filter=tag_filter,
        )
        if scored is None:
            continue
        scored_evaluations.append(scored)

    scored_evaluations.sort(key=lambda item: item.eval_num)
    return scored_evaluations


def _is_numeric_singleton_tuple(key: Any) -> bool:
    return isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], (int, float))


def _is_breakthrough(candidate_key: Tuple[Any, ...], current_key: Optional[Tuple[Any, ...]], eps: float) -> bool:
    if current_key is None:
        return True
    if _is_numeric_singleton_tuple(candidate_key) and _is_numeric_singleton_tuple(current_key):
        return candidate_key[0] > current_key[0] + eps
    return candidate_key > current_key


def find_top_submissions(
    scored_evaluations: Iterable[ScoredEvaluation],
    eps: float = DEFAULT_BREAKTHROUGH_EPS,
) -> List[Breakthrough]:
    """Return every exact global top change, marked by breakthrough threshold."""
    top_submissions: List[Breakthrough] = []
    current_top_key: Optional[Tuple[Any, ...]] = None
    current_top_score: Optional[float] = None
    breakthrough_key: Optional[Tuple[Any, ...]] = None

    for evaluation in sorted(scored_evaluations, key=lambda item: item.eval_num):
        if not _is_breakthrough(evaluation.sort_key, current_top_key, 0.0):
            continue

        is_breakthrough = _is_breakthrough(evaluation.sort_key, breakthrough_key, eps)
        top_submissions.append(
            Breakthrough(
                kind="top_submission",
                track=research_breakthroughs.GLOBAL_BREAKTHROUGH_TRACK,
                eval_id=evaluation.eval_id,
                agent_name=evaluation.agent_name,
                score=evaluation.score,
                value=evaluation.score,
                previous_value=current_top_score,
                tick=evaluation.submitted_tick,
                title=evaluation.title,
                abstract=evaluation.abstract,
                tags=evaluation.tags,
                sort_key=evaluation.sort_key,
                previous_sort_key=current_top_key,
                task_id=evaluation.task_id,
                is_breakthrough=is_breakthrough,
            )
        )
        current_top_key = evaluation.sort_key
        current_top_score = evaluation.score
        if is_breakthrough:
            breakthrough_key = evaluation.sort_key

    return top_submissions


def find_breakthroughs(scored_evaluations: Iterable[ScoredEvaluation], eps: float = DEFAULT_BREAKTHROUGH_EPS) -> List[Breakthrough]:
    """Return only threshold-qualified global breakthroughs."""
    return [item for item in find_top_submissions(scored_evaluations, eps=eps) if item.is_breakthrough]


def collect_breakthrough_events(
    station_data_path: str,
    tag_filter: Optional[str] = None,
    eps: float = DEFAULT_BREAKTHROUGH_EPS,
) -> List[Breakthrough]:
    evaluations_dir = Path(station_data_path) / "rooms" / "research" / "evaluations"
    if not evaluations_dir.exists():
        raise FileNotFoundError(f"Research evaluations directory not found: {evaluations_dir}")

    events = research_breakthroughs.get_breakthrough_events(
        str(evaluations_dir),
        tag_filter=tag_filter,
        eps=eps,
    )
    breakthroughs: List[Breakthrough] = []
    for event in events:
        breakthroughs.append(
            Breakthrough(
                kind=event.kind,
                track=event.track,
                eval_id=event.evaluation_id,
                agent_name=event.agent_name,
                score=event.score,
                value=event.value,
                previous_value=event.previous_value,
                tick=event.submitted_tick,
                title=event.title,
                abstract=event.abstract,
                tags=event.tags,
                sort_key=event.rank_key,
                previous_sort_key=event.previous_rank_key,
                task_id=None,
                is_breakthrough=True,
            )
        )
    return breakthroughs


def _format_float(value: Optional[float], precision: int = 9, prefix_plus: bool = False) -> str:
    if value is None:
        return "N/A"
    if prefix_plus:
        return f"{value:+.{precision}f}"
    return f"{value:.{precision}f}"


def _format_breakthrough_value(value: Any, precision: int = 9) -> str:
    numeric = _coerce_float(value)
    if numeric is not None:
        return f"{numeric:.{precision}f}"
    if value is None:
        return "N/A"
    return str(value)


def _load_breakthrough_agent_models(agents_dir: Path, breakthroughs: Sequence[Breakthrough]) -> Dict[str, str]:
    agent_models: Dict[str, str] = {}
    for item in breakthroughs:
        if item.agent_name not in agent_models:
            agent_models[item.agent_name] = _load_agent_model(agents_dir, item.agent_name)
    return agent_models


def _agent_model_for(agent_models: Dict[str, str], agent_name: str) -> str:
    return agent_models.get(agent_name) or "N/A"


def _best_numeric_value(items: Sequence[Breakthrough]) -> Optional[float]:
    values = [_coerce_float(item.value) for item in items]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _best_numeric_sort_value(items: Sequence[Breakthrough]) -> float:
    value = _best_numeric_value(items)
    return value if value is not None else float("-inf")


def _truncate_text(value: Any, max_width: int) -> str:
    text = str(value)
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return text[: max_width - 3] + "..."


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], max_widths: Optional[Dict[str, int]] = None):
    max_widths = max_widths or {}
    string_rows = [[str(value) for value in row] for row in rows]
    widths = []
    for index, header in enumerate(headers):
        column_values = [row[index] for row in string_rows]
        width = max([len(header), *(len(value) for value in column_values)])
        max_width = max_widths.get(header)
        if max_width is not None:
            width = min(width, max_width)
        widths.append(width)

    def format_row(row: Sequence[Any]) -> str:
        return "  ".join(_truncate_text(value, widths[index]).ljust(widths[index]) for index, value in enumerate(row))

    header_line = format_row(headers)
    separator = "-" * len(header_line)
    print("=" * len(header_line))
    print(header_line)
    print(separator)
    for row in string_rows:
        print(format_row(row))
    print(separator)


def save_breakthroughs_to_csv(
    breakthroughs: List[Breakthrough],
    csv_path: str,
    agent_models: Optional[Dict[str, str]] = None,
):
    """Save breakthroughs data to CSV."""
    agent_models = agent_models or {}
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Eval ID",
                    "Kind",
                    "Track",
                    "Breakthrough",
                    "Tick",
                    "Agent Name",
                    "Agent Model",
                    "Value",
                    "Improvement",
                    "Title",
                    "Abstract",
                    "Tags",
                    "Sort Key",
                ]
            )

            for item in breakthroughs:
                writer.writerow(
                    [
                        item.eval_id,
                        item.kind,
                        item.track,
                        "yes" if item.is_breakthrough else "no",
                        str(item.tick) if item.tick is not None else "N/A",
                        item.agent_name,
                        _agent_model_for(agent_models, item.agent_name),
                        _format_breakthrough_value(item.value),
                        _format_float(item.improvement, prefix_plus=True),
                        item.title,
                        item.abstract,
                        ", ".join(item.tags),
                        repr(item.sort_key),
                    ]
                )

        print(f"Successfully saved {len(breakthroughs)} top/breakthrough rows to {csv_path}")
    except Exception as exc:
        print(f"Error saving CSV file: {exc}")


def analyze_research_breakthroughs(
    station_data_path: str,
    tag_filter: Optional[str] = None,
    csv_path: str = DEFAULT_CSV_PATH,
    eps: float = DEFAULT_BREAKTHROUGH_EPS,
):
    """
    Analyze exact top-submission changes and breakthrough achievements.

    Every exact global top change is displayed. A breakthrough is a global top
    or task-defined progress record that improves its breakthrough frontier by
    the configured epsilon in numeric evaluation-ID order.
    """
    evaluations_dir = Path(station_data_path) / "rooms" / "research" / "evaluations"
    if not evaluations_dir.exists():
        print(f"Research evaluations directory not found: {evaluations_dir}")
        return

    eval_manager = EvaluationManager(str(evaluations_dir))
    print(f"Processing {len(eval_manager.get_all_evaluation_ids())} evaluations...")

    scored_evaluations = collect_scored_evaluations(station_data_path, tag_filter=tag_filter)
    top_submissions = find_top_submissions(scored_evaluations, eps=eps)
    progress_breakthroughs = [
        item
        for item in collect_breakthrough_events(station_data_path, tag_filter=tag_filter, eps=eps)
        if item.track != research_breakthroughs.GLOBAL_BREAKTHROUGH_TRACK
    ]
    analysis_rows = sorted(
        [*top_submissions, *progress_breakthroughs],
        key=lambda item: (
            _coerce_eval_num(item.eval_id) if _coerce_eval_num(item.eval_id) is not None else float("inf"),
            item.track,
        ),
    )
    breakthroughs = [item for item in analysis_rows if item.is_breakthrough]
    agents_dir = Path(station_data_path) / constants.AGENTS_DIR_NAME
    agent_models = _load_breakthrough_agent_models(agents_dir, analysis_rows)

    save_breakthroughs_to_csv(analysis_rows, csv_path, agent_models=agent_models)
    print(f"\nTop submission and breakthrough rows saved to: {csv_path}")

    agent_breakthroughs: Dict[str, List[Breakthrough]] = defaultdict(list)
    for item in breakthroughs:
        agent_breakthroughs[item.agent_name].append(item)

    breakthrough_rows = []
    for item in analysis_rows:
        tick_str = str(item.tick) if item.tick is not None else "N/A"
        breakthrough_rows.append(
            [
                item.eval_id,
                item.kind,
                item.track,
                "yes" if item.is_breakthrough else "no",
                tick_str,
                item.agent_name,
                _agent_model_for(agent_models, item.agent_name),
                _format_breakthrough_value(item.value),
                _format_float(item.improvement, prefix_plus=True),
                item.title,
            ]
        )

    print("\nResearch Top Submission and Breakthrough Analysis")
    _print_table(
        ["Eval ID", "Kind", "Track", "Breakthrough", "Tick", "Agent Name", "Agent Model", "Value", "Improvement", "Title"],
        breakthrough_rows,
        max_widths={"Track": 36, "Title": 100},
    )

    sorted_agents = sorted(
        agent_breakthroughs.items(),
        key=lambda item: (len(item[1]), _best_numeric_sort_value(item[1])),
        reverse=True,
    )

    agent_rows = []
    total_breakthroughs = 0
    for agent_name, breakthrough_list in sorted_agents:
        num_breakthroughs = len(breakthrough_list)
        agent_model = _agent_model_for(agent_models, agent_name)
        best_score = _best_numeric_value(breakthrough_list)
        total_improvement = sum(item.improvement or 0.0 for item in breakthrough_list)
        agent_rows.append(
            [
                agent_name,
                agent_model,
                str(num_breakthroughs),
                _format_float(best_score),
                f"{total_improvement:+.9f}",
            ]
        )
        total_breakthroughs += num_breakthroughs

    print("\nBreakthroughs by Agent")
    _print_table(
        ["Agent Name", "Agent Model", "Breakthroughs", "Best Score", "Total Improvement"],
        agent_rows,
    )

    current_value = breakthroughs[-1].value if breakthroughs else None
    print("\nOverall Statistics:")
    print(f"- Total evaluations with valid scores: {len(scored_evaluations)}")
    print(f"- Total breakthrough achievements: {total_breakthroughs}")
    print(f"- Number of agents with breakthroughs: {len(agent_breakthroughs)}")
    print(f"- Latest breakthrough value: {_format_breakthrough_value(current_value)}")

    if breakthroughs:
        first = breakthroughs[0]
        last = breakthroughs[-1]
        print("\nBreakthrough Timeline:")
        print(f"- First breakthrough: {_format_breakthrough_value(first.value)} on {first.track} by {first.agent_name} (Eval #{first.eval_id})")
        print(f"- Latest breakthrough: {_format_breakthrough_value(last.value)} on {last.track} by {last.agent_name} (Eval #{last.eval_id})")

        if sorted_agents:
            top_agent = sorted_agents[0]
            print("\nMost Breakthrough Achievements:")
            print(f"- {top_agent[0]}: {len(top_agent[1])} breakthroughs")
            for index, breakthrough in enumerate(top_agent[1], 1):
                print(
                    f"  {index}. Eval #{breakthrough.eval_id}: "
                    f"{breakthrough.track}={_format_breakthrough_value(breakthrough.value)} "
                    f"({_format_float(breakthrough.improvement, prefix_plus=True)})"
                )


def _default_station_data_path() -> str:
    script_dir = Path(__file__).resolve().parent
    return str(script_dir.parent / "station_data")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze exact research top submissions and threshold-qualified breakthroughs"
    )
    parser.add_argument(
        "station_data_path",
        nargs="?",
        default=None,
        help="Path to station_data directory (defaults to ../station_data from script location)",
    )
    parser.add_argument("--tag", type=str, default=None, help="Filter submissions by tag")
    parser.add_argument("--csv-out", default=DEFAULT_CSV_PATH, help=f"CSV output path (default: {DEFAULT_CSV_PATH})")
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_BREAKTHROUGH_EPS,
        help=f"Minimum numeric score improvement to count as a breakthrough (default: {DEFAULT_BREAKTHROUGH_EPS})",
    )

    args = parser.parse_args(argv)
    station_data_path = os.path.abspath(args.station_data_path or _default_station_data_path())

    if not os.path.exists(station_data_path):
        print(f"Error: station_data path does not exist: {station_data_path}")
        return 1

    print(f"Using station_data path: {station_data_path}")
    if args.tag:
        print(f"Filtering submissions by tag: {args.tag}")
    analyze_research_breakthroughs(station_data_path, tag_filter=args.tag, csv_path=args.csv_out, eps=args.eps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
