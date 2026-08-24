"""Canonical Research Center breakthrough detection.

Evaluation YAML remains authoritative, but normal breakthrough queries read the
Research Center SQLite index. A breakthrough is an improvement on one named
track. The built-in global track is derived from the final score/sort_key;
tasks may add extra tracks through final.progress_records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from station import constants
from station.eval_research import evaluation_index


GLOBAL_BREAKTHROUGH_TRACK = "global"


@dataclass(frozen=True)
class BreakthroughEvent:
    kind: str
    track: str
    evaluation_id: str
    eval_id_num: int
    agent_name: str
    lineage: str
    submitted_tick: Optional[int]
    title: str
    abstract: str
    tags: List[str]
    score: Any
    value: Any
    rank_key: Tuple[Any, ...]
    previous_rank_key: Optional[Tuple[Any, ...]]
    previous_value: Any = None
    label: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["rank_key"] = list(self.rank_key)
        result["previous_rank_key"] = list(self.previous_rank_key) if self.previous_rank_key is not None else None
        return result


def normalize_rank_key(rank_key: Any, fallback_score: Any = None) -> Optional[Tuple[Any, ...]]:
    if rank_key is not None:
        raw_items = tuple(rank_key) if isinstance(rank_key, (list, tuple)) else (rank_key,)
        normalized_items = []
        for item in raw_items:
            normalized_item = normalize_rank_key_component(item)
            if normalized_item is None:
                normalized_items = []
                break
            normalized_items.append(normalized_item)
        if normalized_items:
            return tuple(normalized_items)

    if fallback_score is None:
        return None
    normalized_score = normalize_rank_key_component(fallback_score)
    if normalized_score is None:
        return None
    return (normalized_score,)


def normalize_rank_key_component(value: Any) -> Optional[Any]:
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


def normalize_progress_records(raw_records: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_records, list):
        return []

    records: List[Dict[str, Any]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        track = str(raw.get("track") or "").strip()
        if not track:
            continue
        rank_key = normalize_rank_key(raw.get("rank_key"))
        if rank_key is None:
            continue
        record: Dict[str, Any] = {
            "track": track,
            "rank_key": list(rank_key),
        }
        if "value" in raw:
            record["value"] = _to_json_safe(raw.get("value"))
        if raw.get("label") is not None:
            record["label"] = str(raw.get("label"))
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            record["metadata"] = _to_json_safe(metadata)
        records.append(record)
    return records


def get_breakthrough_events(
    evaluations_dir: str,
    *,
    include_progress_records: bool = True,
    tag_filter: Optional[str] = None,
    eps: Optional[float] = None,
) -> List[BreakthroughEvent]:
    rows = evaluation_index.list_breakthrough_source_rows(evaluations_dir)
    events: List[BreakthroughEvent] = []
    frontier_by_track: Dict[str, BreakthroughEvent] = {}
    eps_value = float(getattr(constants, "BREAKTHROUGH_EPS", 1e-2) if eps is None else eps)

    for row in rows:
        tags = [str(tag) for tag in row.get("tags", [])]
        if tag_filter and tag_filter not in tags:
            continue
        for candidate in _iter_candidates(row, include_progress_records=include_progress_records):
            current = frontier_by_track.get(candidate.track)
            if not rank_key_improved(candidate.rank_key, current.rank_key if current else None, eps_value):
                continue
            event = BreakthroughEvent(
                kind=candidate.kind,
                track=candidate.track,
                evaluation_id=candidate.evaluation_id,
                eval_id_num=candidate.eval_id_num,
                agent_name=candidate.agent_name,
                lineage=candidate.lineage,
                submitted_tick=candidate.submitted_tick,
                title=candidate.title,
                abstract=candidate.abstract,
                tags=candidate.tags,
                score=candidate.score,
                value=candidate.value,
                rank_key=candidate.rank_key,
                previous_rank_key=current.rank_key if current else None,
                previous_value=current.value if current else None,
                label=candidate.label,
                metadata=candidate.metadata,
            )
            frontier_by_track[candidate.track] = event
            events.append(event)

    return events


def get_latest_breakthrough_summary(
    evaluations_dir: str,
    *,
    include_progress_records: bool = True,
    eps: Optional[float] = None,
) -> Dict[str, Any]:
    events = get_breakthrough_events(
        evaluations_dir,
        include_progress_records=include_progress_records,
        eps=eps,
    )
    frontiers: Dict[str, Dict[str, Any]] = {}
    latest_event: Optional[BreakthroughEvent] = None
    last_tick = 1
    for event in events:
        frontiers[event.track] = event.to_dict()
        if event.submitted_tick is not None:
            last_tick = max(last_tick, int(event.submitted_tick))
        if latest_event is None:
            latest_event = event
            continue
        latest_tick = latest_event.submitted_tick if latest_event.submitted_tick is not None else -1
        event_tick = event.submitted_tick if event.submitted_tick is not None else -1
        if (event_tick, event.eval_id_num, event.track) >= (latest_tick, latest_event.eval_id_num, latest_event.track):
            latest_event = event

    return {
        "last_breakthrough_tick": last_tick,
        "latest_event": latest_event.to_dict() if latest_event else None,
        "frontiers": frontiers,
        "event_count": len(events),
    }


def rank_key_improved(candidate_key: Tuple[Any, ...], current_key: Optional[Tuple[Any, ...]], eps: float) -> bool:
    if current_key is None:
        return True
    if _is_numeric_singleton_tuple(candidate_key) and _is_numeric_singleton_tuple(current_key):
        return candidate_key[0] > current_key[0] + eps
    return candidate_key > current_key


@dataclass(frozen=True)
class _Candidate:
    kind: str
    track: str
    evaluation_id: str
    eval_id_num: int
    agent_name: str
    lineage: str
    submitted_tick: Optional[int]
    title: str
    abstract: str
    tags: List[str]
    score: Any
    value: Any
    rank_key: Tuple[Any, ...]
    label: str = ""
    metadata: Optional[Dict[str, Any]] = None


def _iter_candidates(row: Dict[str, Any], *, include_progress_records: bool) -> Iterable[_Candidate]:
    score = row.get("score")
    global_rank_key = normalize_rank_key(row.get("sort_key"), score)
    if global_rank_key is not None:
        yield _Candidate(
            kind="top_submission",
            track=GLOBAL_BREAKTHROUGH_TRACK,
            evaluation_id=str(row.get("eval_id")),
            eval_id_num=int(row.get("eval_id_num") or 0),
            agent_name=str(row.get("author") or "Unknown"),
            lineage=str(row.get("lineage") or ""),
            submitted_tick=_coerce_tick(row.get("submitted_tick")),
            title=str(row.get("title") or "Untitled"),
            abstract=str(row.get("abstract") or ""),
            tags=[str(tag) for tag in row.get("tags", [])],
            score=score,
            value=score,
            rank_key=global_rank_key,
            label="Global score",
            metadata=None,
        )

    if not include_progress_records:
        return
    for record in normalize_progress_records(row.get("progress_records")):
        rank_key = normalize_rank_key(record.get("rank_key"))
        if rank_key is None:
            continue
        yield _Candidate(
            kind="progress_record",
            track=str(record.get("track")),
            evaluation_id=str(row.get("eval_id")),
            eval_id_num=int(row.get("eval_id_num") or 0),
            agent_name=str(row.get("author") or "Unknown"),
            lineage=str(row.get("lineage") or ""),
            submitted_tick=_coerce_tick(row.get("submitted_tick")),
            title=str(row.get("title") or "Untitled"),
            abstract=str(row.get("abstract") or ""),
            tags=[str(tag) for tag in row.get("tags", [])],
            score=score,
            value=record.get("value"),
            rank_key=rank_key,
            label=str(record.get("label") or record.get("track") or ""),
            metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else None,
        )


def _is_numeric_singleton_tuple(key: Any) -> bool:
    return isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], (int, float))


def _coerce_tick(tick: Any) -> Optional[int]:
    try:
        return int(tick)
    except (TypeError, ValueError):
        return None


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
