#!/usr/bin/env python3
"""Analyze Research Center evaluations for new-SOTA breakthroughs."""

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
from station.eval_research.evaluation_manager import EvaluationManager


DEFAULT_CSV_PATH = "/tmp/breakthroughs.csv"
DEFAULT_BREAKTHROUGH_EPS = 1e-8
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
    eval_id: str
    agent_name: str
    score: float
    previous_sota: Optional[float]
    tick: Optional[int]
    title: str
    abstract: str
    tags: List[str]
    sort_key: Tuple[Any, ...]
    task_id: Optional[Any]

    @property
    def improvement(self) -> Optional[float]:
        if self.previous_sota is None:
            return None
        return self.score - self.previous_sota


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


def _build_scored_evaluation(evaluations_dir: Path, display_info: Dict[str, Any]) -> Optional[ScoredEvaluation]:
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

    return ScoredEvaluation(
        eval_id=eval_id,
        eval_num=eval_num,
        agent_name=agent_name,
        score=score_float,
        sort_key=sort_key,
        submitted_tick=_coerce_tick(display_info.get(constants.EVALUATION_SUBMITTED_TICK_KEY)),
        title=str(display_info.get(constants.EVALUATION_TITLE_KEY) or "Untitled"),
        abstract=str(display_info.get(constants.EVALUATION_ABSTRACT_KEY) or ""),
        tags=_normalize_tags(display_info.get(constants.EVALUATION_TAGS_KEY, [])),
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
        scored = _build_scored_evaluation(evaluations_dir, display_info)
        if scored is None:
            continue
        if tag_filter and tag_filter not in scored.tags:
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


def find_breakthroughs(scored_evaluations: Iterable[ScoredEvaluation], eps: float = DEFAULT_BREAKTHROUGH_EPS) -> List[Breakthrough]:
    breakthroughs: List[Breakthrough] = []
    current_sota_key: Optional[Tuple[Any, ...]] = None
    current_sota_score: Optional[float] = None

    for evaluation in sorted(scored_evaluations, key=lambda item: item.eval_num):
        if not _is_breakthrough(evaluation.sort_key, current_sota_key, eps):
            continue

        breakthroughs.append(
            Breakthrough(
                eval_id=evaluation.eval_id,
                agent_name=evaluation.agent_name,
                score=evaluation.score,
                previous_sota=current_sota_score,
                tick=evaluation.submitted_tick,
                title=evaluation.title,
                abstract=evaluation.abstract,
                tags=evaluation.tags,
                sort_key=evaluation.sort_key,
                task_id=evaluation.task_id,
            )
        )
        current_sota_key = evaluation.sort_key
        current_sota_score = evaluation.score

    return breakthroughs


def _format_float(value: Optional[float], precision: int = 9, prefix_plus: bool = False) -> str:
    if value is None:
        return "N/A"
    sign = "+" if prefix_plus else ""
    return f"{sign}{value:.{precision}f}"


def save_breakthroughs_to_csv(breakthroughs: List[Breakthrough], csv_path: str):
    """Save breakthroughs data to CSV."""
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Eval ID",
                    "Tick",
                    "Agent Name",
                    "New SOTA",
                    "Previous",
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
                        str(item.tick) if item.tick is not None else "N/A",
                        item.agent_name,
                        _format_float(item.score),
                        _format_float(item.previous_sota),
                        _format_float(item.improvement, prefix_plus=True),
                        item.title,
                        item.abstract,
                        ", ".join(item.tags),
                        repr(item.sort_key),
                    ]
                )

        print(f"Successfully saved {len(breakthroughs)} breakthroughs to {csv_path}")
    except Exception as exc:
        print(f"Error saving CSV file: {exc}")


def analyze_research_breakthroughs(
    station_data_path: str,
    tag_filter: Optional[str] = None,
    csv_path: str = DEFAULT_CSV_PATH,
    eps: float = DEFAULT_BREAKTHROUGH_EPS,
):
    """
    Analyze research evaluations to count breakthrough achievements.

    A breakthrough is an evaluation whose normalized score/sort key improves over
    all previous scored evaluations in numeric evaluation-ID order.
    """
    evaluations_dir = Path(station_data_path) / "rooms" / "research" / "evaluations"
    if not evaluations_dir.exists():
        print(f"Research evaluations directory not found: {evaluations_dir}")
        return

    eval_manager = EvaluationManager(str(evaluations_dir))
    print(f"Processing {len(eval_manager.get_all_evaluation_ids())} evaluations...")

    scored_evaluations = collect_scored_evaluations(station_data_path, tag_filter=tag_filter)
    breakthroughs = find_breakthroughs(scored_evaluations, eps=eps)

    save_breakthroughs_to_csv(breakthroughs, csv_path)
    print(f"\nBreakthroughs saved to: {csv_path}")

    agent_breakthroughs: Dict[str, List[Breakthrough]] = defaultdict(list)
    for item in breakthroughs:
        agent_breakthroughs[item.agent_name].append(item)

    print("\nResearch Breakthroughs Analysis (New SOTA Achievements)")
    print("=" * 210)
    print(f"{'Eval ID':<10} {'Tick':<10} {'Agent Name':<25} {'New SOTA':<12} {'Previous':<12} {'Improvement':<12} {'Title':<100}")
    print("-" * 210)

    for item in breakthroughs:
        tick_str = str(item.tick) if item.tick is not None else "N/A"
        title_display = item.title if len(item.title) <= 97 else item.title[:97] + "..."
        print(
            f"{item.eval_id:<10} {tick_str:<10} {item.agent_name:<25} "
            f"{item.score:<12.9f} {_format_float(item.previous_sota):<12} "
            f"{_format_float(item.improvement, prefix_plus=True):<12} {title_display:<100}"
        )

    print("-" * 210)

    print("\nBreakthroughs by Agent")
    print("=" * 80)
    print(f"{'Agent Name':<25} {'Breakthroughs':<15} {'Best Score':<15} {'Total Improvement':<20}")
    print("-" * 80)

    sorted_agents = sorted(
        agent_breakthroughs.items(),
        key=lambda item: (len(item[1]), max(b.score for b in item[1])),
        reverse=True,
    )

    total_breakthroughs = 0
    for agent_name, breakthrough_list in sorted_agents:
        num_breakthroughs = len(breakthrough_list)
        best_score = max(item.score for item in breakthrough_list)
        total_improvement = sum(item.improvement or 0.0 for item in breakthrough_list)
        print(f"{agent_name:<25} {num_breakthroughs:<15} {best_score:<15.9f} {f'+{total_improvement:.9f}':<20}")
        total_breakthroughs += num_breakthroughs

    print("-" * 80)

    current_sota_score = breakthroughs[-1].score if breakthroughs else None
    print("\nOverall Statistics:")
    print(f"- Total evaluations with valid scores: {len(scored_evaluations)}")
    print(f"- Total breakthrough achievements: {total_breakthroughs}")
    print(f"- Number of agents with breakthroughs: {len(agent_breakthroughs)}")
    print(f"- Current SOTA score: {_format_float(current_sota_score)}")

    if breakthroughs:
        first = breakthroughs[0]
        last = breakthroughs[-1]
        print("\nBreakthrough Timeline:")
        print(f"- First SOTA: {first.score:.9f} by {first.agent_name} (Eval #{first.eval_id})")
        print(f"- Latest SOTA: {last.score:.9f} by {last.agent_name} (Eval #{last.eval_id})")

        if sorted_agents:
            top_agent = sorted_agents[0]
            print("\nMost Breakthrough Achievements:")
            print(f"- {top_agent[0]}: {len(top_agent[1])} breakthroughs")
            for index, breakthrough in enumerate(top_agent[1], 1):
                print(
                    f"  {index}. Eval #{breakthrough.eval_id}: "
                    f"{breakthrough.score:.9f} ({_format_float(breakthrough.improvement, prefix_plus=True)})"
                )


def _default_station_data_path() -> str:
    script_dir = Path(__file__).resolve().parent
    return str(script_dir.parent / "station_data")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze research breakthroughs in the station")
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
