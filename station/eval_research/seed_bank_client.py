"""Frozen, read-only client copied into seed-enabled Research workspaces.

This module intentionally depends only on the Python standard library and
NumPy. Station owns all writes; coding agents only query and load candidates.
"""

from __future__ import annotations

import json
import heapq
import os
import random
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np


API_HELP_OVERVIEW = """Research Seed Bank read-only API

Use focused help so you only load the documentation needed for the operation:
  SeedBank.help("top")              best official-ranking records
  SeedBank.help("sample")           uniform filtered sampling
  SeedBank.help("from_evaluations") records from named evaluation IDs
  SeedBank.help("load")             load numerical seeds from records
  SeedBank.help("load_population")  select and load top seeds directly
  SeedBank.help("iter_all")         stream all qualifying metadata records
  SeedBank.help("iter_batches")     load qualifying seeds in bounded batches
  SeedBank.help("rank")             rerank using numerical seed data
  SeedBank.help("rank_metadata")    rerank using indexed metadata only
  SeedBank.help("distinct")         caller-defined greedy diversity selection
  SeedBank.help("filters")          shared indexed filters
  SeedBank.help("metrics")          discover and query secondary metrics
  SeedBank.help("metadata")         SeedRecord and metadata fields
  SeedBank.help("summary")          counts and best official score

Shell equivalent:
  python storage/system/seed_bank.py --help TOPIC

All queries automatically use the access scope supplied by Station. The API
has no argument that can widen that scope. Station owns all writes.
"""


API_HELP_TOPICS = {
    "filters": """Shared indexed query filters

Pass these as keyword arguments to iter_all(), top(), sample(),
from_evaluations(), load_population(), rank(), or rank_metadata():
  eval_ids=["12", "19"]  restrict to named official evaluations
  attempt_numbers=[1, 2] restrict to named official attempt numbers
  min_score=0.7          official score >= value
  max_score=0.9          official score <= value
  score_gt=0.7           official score > value
  score_lt=0.9           official score < value
  winners_only=True      only the official winner from each submitted batch
  metric_filters={       indexed numeric secondary-metric conditions
      "NormRatio": {"gt": 0.85},
      "Steps": {"gte": 2000, "lte": 10000},
  }
  order_metric="NormRatio"  order by this secondary metric instead of score
  order_metric_desc=False   use ascending secondary-metric order

Metric operators are eq, gt, gte, min, lt, lte, and max. Multiple named
metrics are combined with AND. Candidates lacking a requested metric are
excluded. Use bank.metrics() to discover indexed metric names and ranges.
""",
    "metrics": """bank.metrics() -> list[dict]

Lists indexed numeric secondary metrics within the enforced access scope.
Each entry contains name, candidate count, minimum, and maximum.

Native metric filtering works in top(), sample(), iter_all(),
from_evaluations(), load_population(), rank(), and rank_metadata(). Secondary-
metric ordering works in all of those selection paths except uniform sample().

Examples:
  print(bank.metrics())
  records = bank.top(
      limit=64,
      order_metric="NormRatio",
      metric_filters={"Steps": {"gte": 2000}},
  )
  records = bank.sample(
      128,
      metric_filters={"NonzeroFraction": {"gt": 0.9}},
  )
""",
    "summary": """bank.summary() -> dict

Returns candidate count, exact-distinct content count, evaluation count, best
official score, and the access scope enforced for this coder session.
""",
    "iter_all": """bank.iter_all(batch_size=512, **filters) -> iterator[SeedRecord]

Streams qualifying records in best-first official evaluator order without
loading numerical seeds. SQLite rows are fetched in chunks of batch_size. See
SeedBank.help("filters") for accepted indexed filters.
""",
    "top": """bank.top(limit=64, exact_distinct=True,
         max_per_evaluation=None, order_metric=None,
         order_metric_desc=True, metric_filters=None, **filters)
    -> list[SeedRecord]

Selects records by the evaluator's official sort key, best first.
exact_distinct removes identical numerical content. max_per_evaluation limits
dominance by any one official evaluation. Use bank.load(records) to load the
numerical seeds. Score filters still apply to the displayed numeric score.

Example:
  records = bank.top(limit=64, max_per_evaluation=4, score_gt=0.7)
  seeds, metadata = bank.load(records)

  records = bank.top(
      limit=64,
      order_metric="NormRatio",
      metric_filters={"Steps": {"gte": 2000, "lte": 10000}},
  )
""",
    "sample": """bank.sample(count, exact_distinct=True,
            max_per_evaluation=None, rng=None, **filters) -> list[SeedRecord]

Uniformly samples qualifying records with bounded-memory reservoir sampling.
exact_distinct removes identical numerical content. rng may be a seeded
random.Random instance. Use bank.load(records) to load the numerical seeds.

Example:
  records = bank.sample(
      128,
      metric_filters={"NonzeroFraction": {"gt": 0.9}},
      max_per_evaluation=8,
  )
  seeds, metadata = bank.load(records)
""",
    "from_evaluations": """bank.from_evaluations(eval_ids, **filters) -> list[SeedRecord]

Returns score-ordered records belonging to the named official evaluation IDs.
All preserved successful attempts are included unless `attempt_numbers=[...]`
is supplied. The normal Station access scope still applies.

Example:
  records = bank.from_evaluations(["12", "19"], attempt_numbers=[1, 2], score_gt=0.7)
  seeds, metadata = bank.load(records)
""",
    "load": """bank.load(records) -> (list_of_seeds, list_of_metadata)

Loads only the NPZ members referenced by the supplied SeedRecord objects.
Metadata aligns positionally with seeds. Prefer iter_batches() when loading a
large selection.
""",
    "load_population": """bank.load_population(limit=64, **filters)
    -> (list_of_seeds, list_of_metadata)

Convenience form equivalent to bank.load(bank.top(limit=limit, **filters)).
It uses the evaluator's official sort key and exact-deduplicates by default.

Example:
  seeds, metadata = bank.load_population(
      limit=64,
      order_metric="NormRatio",
      metric_filters={"Steps": {"gte": 2000}},
  )
""",
    "iter_batches": """bank.iter_batches(records=None, batch_size=64, **filters)
    -> iterator[(list_of_seeds, list_of_metadata)]

Loads records in bounded batches. Supply a prior record selection, or omit
records to stream all qualifying records using the indexed filters.

Example:
  for seeds, metadata in bank.iter_batches(bank.top(limit=4096), batch_size=64):
      process(seeds, metadata)
""",
    "rank": """bank.rank(key, limit=64, pool_limit=None, reverse=True,
          exact_distinct=True, load_batch_size=8, **filters) -> list[SeedRecord]

Reranks candidates with key(seed, metadata). pool_limit bounds the initial
best-first official evaluator pool; use it for large banks. Only a small
loading batch and the requested frontier are retained in memory.

Example:
  records = bank.rank(
      lambda seed, meta: my_metric(seed), limit=64, pool_limit=4096
  )
  seeds, metadata = bank.load(records)
""",
    "rank_metadata": """bank.rank_metadata(key, limit=64, pool_limit=None,
                   reverse=True, exact_distinct=True, **filters)
    -> list[SeedRecord]

Reranks with key(metadata) without loading numerical seeds. Prefer this over
rank() when the new metric is already present in indexed evaluator details.

Example:
  records = bank.rank_metadata(
      lambda meta: meta["details"]["Metric"], limit=64, pool_limit=4096
  )
""",
    "distinct": """bank.distinct(records, limit=64, distance=distance_fn,
              min_distance=threshold, feature=None, load_batch_size=8)
    -> list[SeedRecord]

Greedily keeps a candidate only when its caller-defined distance from every
already selected candidate is >= min_distance. feature(seed) may create a
cheaper representation. This compares against the small selected set and does
not construct a full all-pairs distance matrix.

Example:
  pool = bank.top(limit=4096)
  records = bank.distinct(
      pool, limit=64,
      distance=lambda a, b: float(np.linalg.norm(a - b)),
      min_distance=1e-8,
  )
  seeds, metadata = bank.load(records)
""",
    "metadata": """SeedRecord fields and record.metadata()

candidate_id, eval_id, attempt_number, batch_index, lineage, author, score, sort_key, details,
batch_rank, is_winner, content_id, fingerprint, artifact_path, descriptor.
record.metadata() returns the public fields except artifact_path and descriptor.
""",
}


@dataclass(frozen=True)
class SeedRecord:
    candidate_id: int
    eval_id: str
    attempt_number: int
    batch_index: int
    lineage: str
    author: str
    score: float
    sort_key: Any
    details: Any
    batch_rank: int
    is_winner: bool
    content_id: str
    fingerprint: str
    artifact_path: str
    descriptor: dict

    def metadata(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "eval_id": self.eval_id,
            "attempt_number": self.attempt_number,
            "batch_index": self.batch_index,
            "lineage": self.lineage,
            "author": self.author,
            "score": self.score,
            "sort_key": self.sort_key,
            "details": self.details,
            "batch_rank": self.batch_rank,
            "is_winner": self.is_winner,
            "content_id": self.content_id,
            "fingerprint": self.fingerprint,
        }


class SeedBank:
    """Read-only query and selective-loading interface for submitted seeds."""

    @staticmethod
    def help(topic: Optional[str] = None) -> str:
        """Return a compact index or focused API help for one operation."""
        if topic is None:
            return API_HELP_OVERVIEW
        normalized = str(topic).strip().lower()
        if normalized in API_HELP_TOPICS:
            return API_HELP_TOPICS[normalized]
        topics = ", ".join(sorted(API_HELP_TOPICS))
        return f"Unknown Seed Bank help topic {topic!r}. Available topics: {topics}"

    def __init__(self, bank_root: Path):
        self.bank_root = bank_root.resolve()
        self.index_path = self.bank_root / "index.sqlite"
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Seed Bank index not found: {self.index_path}")
        self._lineage = str(os.environ.get("STATION_LINEAGE", "unknown")).strip().lower()
        self._phase = str(os.environ.get("STATION_ACCESS_PHASE", "mature")).strip().lower()
        if self._phase not in {"immature", "mature"}:
            self._phase = "mature"

    @classmethod
    def open(cls, root: Optional[str] = None) -> "SeedBank":
        if root:
            bank_root = Path(root)
        else:
            research_root = Path(os.environ.get("STATION_RESEARCH_ROOT", os.getcwd()))
            bank_root = research_root / "storage" / "shared" / "seed_bank"
        return cls(bank_root)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "seed_sort_key",
            1,
            self._sortable_sort_key,
            deterministic=True,
        )
        return connection

    @staticmethod
    def _sortable_sort_key(sort_key_json: str) -> bytes:
        """Encode a finite numeric tuple so SQLite BLOB order matches tuple order."""
        try:
            values = json.loads(sort_key_json)
            if not isinstance(values, list) or not values:
                return b""
            encoded = bytearray()
            for value in values:
                numeric = float(value)
                if not np.isfinite(numeric):
                    return b""
                if numeric == 0.0:
                    numeric = 0.0
                bits = struct.unpack(">Q", struct.pack(">d", numeric))[0]
                if bits & (1 << 63):
                    bits = (~bits) & ((1 << 64) - 1)
                else:
                    bits ^= 1 << 63
                encoded.extend(struct.pack(">Q", bits))
            return bytes(encoded)
        except (TypeError, ValueError, json.JSONDecodeError):
            return b""

    def _visibility_clause(self) -> tuple[str, list[Any]]:
        if self._phase == "immature":
            return " AND c.lineage = ?", [self._lineage]
        return "", []

    def _build_query(
        self,
        *,
        eval_ids: Optional[Sequence[str]] = None,
        attempt_numbers: Optional[Sequence[int]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        score_gt: Optional[float] = None,
        score_lt: Optional[float] = None,
        winners_only: bool = False,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        limit: Optional[int] = None,
        order: str = "score",
    ) -> tuple[str, list[Any]]:
        sql = """
            SELECT c.candidate_id, c.eval_id, c.attempt_number, c.batch_index, c.lineage,
                   c.author, c.score, c.sort_key_json, c.details_json,
                   c.batch_rank, c.is_winner, c.content_id,
                   x.fingerprint, x.artifact_path, x.descriptor_json
              FROM seed_candidates c
              JOIN seed_contents x ON x.content_id = c.content_id
        """
        params: list[Any] = []
        normalized_order_metric = str(order_metric or "").strip()
        if normalized_order_metric:
            sql += (
                " JOIN seed_candidate_metrics ordered_metric "
                "ON ordered_metric.candidate_id = c.candidate_id "
                "AND ordered_metric.metric_name = ?"
            )
            params.append(normalized_order_metric)
        sql += " WHERE 1 = 1"
        visibility_sql, visibility_params = self._visibility_clause()
        sql += visibility_sql
        params.extend(visibility_params)
        if eval_ids:
            normalized = [str(value) for value in eval_ids]
            sql += " AND c.eval_id IN (" + ",".join("?" for _ in normalized) + ")"
            params.extend(normalized)
        if attempt_numbers:
            normalized_attempts = [max(1, int(value)) for value in attempt_numbers]
            sql += " AND c.attempt_number IN (" + ",".join("?" for _ in normalized_attempts) + ")"
            params.extend(normalized_attempts)
        if min_score is not None:
            sql += " AND c.score >= ?"
            params.append(float(min_score))
        if max_score is not None:
            sql += " AND c.score <= ?"
            params.append(float(max_score))
        if score_gt is not None:
            sql += " AND c.score > ?"
            params.append(float(score_gt))
        if score_lt is not None:
            sql += " AND c.score < ?"
            params.append(float(score_lt))
        if winners_only:
            sql += " AND c.is_winner = 1"
        for index, (metric_name, conditions) in enumerate(
            self._normalize_metric_filters(metric_filters)
        ):
            alias = f"metric_filter_{index}"
            sql += (
                f" AND EXISTS (SELECT 1 FROM seed_candidate_metrics {alias} "
                f"WHERE {alias}.candidate_id = c.candidate_id "
                f"AND {alias}.metric_name = ?"
            )
            params.append(metric_name)
            for operator, numeric_value in conditions:
                sql += f" AND {alias}.numeric_value {operator} ?"
                params.append(numeric_value)
            sql += ")"
        if normalized_order_metric:
            direction = "DESC" if bool(order_metric_desc) else "ASC"
            sql += (
                f" ORDER BY ordered_metric.numeric_value {direction}, "
                "seed_sort_key(c.sort_key_json) DESC, c.candidate_id ASC"
            )
        elif order == "score":
            sql += " ORDER BY seed_sort_key(c.sort_key_json) DESC, c.candidate_id ASC"
        elif order == "random":
            sql += " ORDER BY RANDOM()"
        else:
            sql += " ORDER BY c.candidate_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))

        return sql, params

    @staticmethod
    def _normalize_metric_filters(
        metric_filters: Optional[dict[str, Any]],
    ) -> list[tuple[str, list[tuple[str, float]]]]:
        if metric_filters is None:
            return []
        if not isinstance(metric_filters, dict):
            raise TypeError("metric_filters must be a mapping from metric names to numeric conditions.")
        operator_map = {
            "eq": "=",
            "gt": ">",
            "gte": ">=",
            "min": ">=",
            "lt": "<",
            "lte": "<=",
            "max": "<=",
        }
        normalized = []
        for raw_name, raw_conditions in metric_filters.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Secondary metric names must be non-empty.")
            if isinstance(raw_conditions, bool) or isinstance(raw_conditions, (int, float, np.number)):
                conditions = {"eq": raw_conditions}
            elif isinstance(raw_conditions, dict) and raw_conditions:
                conditions = raw_conditions
            else:
                raise TypeError(
                    f"Conditions for secondary metric {name!r} must be a number or non-empty mapping."
                )
            normalized_conditions = []
            for raw_operator, raw_value in conditions.items():
                operator_name = str(raw_operator).strip().lower()
                if operator_name not in operator_map:
                    allowed = ", ".join(sorted(operator_map))
                    raise ValueError(
                        f"Unknown secondary metric operator {raw_operator!r}; use one of: {allowed}."
                    )
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, np.number)):
                    raise TypeError(
                        f"Secondary metric condition {name!r}.{operator_name} must be numeric."
                    )
                numeric_value = float(raw_value)
                if not np.isfinite(numeric_value):
                    raise ValueError(
                        f"Secondary metric condition {name!r}.{operator_name} must be finite."
                    )
                normalized_conditions.append((operator_map[operator_name], numeric_value))
            normalized.append((name, normalized_conditions))
        return normalized

    def _iter_query(self, *, fetch_size: int = 512, **filters) -> Iterator[SeedRecord]:
        sql, params = self._build_query(**filters)
        connection = self._connect()
        try:
            cursor = connection.execute(sql, params)
            while True:
                rows = cursor.fetchmany(max(1, int(fetch_size)))
                if not rows:
                    return
                for row in rows:
                    yield self._row_to_record(row)
        finally:
            connection.close()

    def _query(self, **filters) -> list[SeedRecord]:
        return list(self._iter_query(**filters))

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> SeedRecord:
        return SeedRecord(
            candidate_id=int(row["candidate_id"]),
            eval_id=str(row["eval_id"]),
            attempt_number=int(row["attempt_number"]),
            batch_index=int(row["batch_index"]),
            lineage=str(row["lineage"]),
            author=str(row["author"]),
            score=float(row["score"]),
            sort_key=json.loads(row["sort_key_json"]),
            details=json.loads(row["details_json"]),
            batch_rank=int(row["batch_rank"]),
            is_winner=bool(row["is_winner"]),
            content_id=str(row["content_id"]),
            fingerprint=str(row["fingerprint"]),
            artifact_path=str(row["artifact_path"]),
            descriptor=json.loads(row["descriptor_json"]),
        )

    def summary(self) -> dict:
        visibility_sql, params = self._visibility_clause()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS candidates, COUNT(DISTINCT c.content_id) AS contents, "
                "COUNT(DISTINCT c.eval_id) AS evaluations "
                "FROM seed_candidates c WHERE 1 = 1" + visibility_sql,
                params,
            ).fetchone()
            best = connection.execute(
                "SELECT c.score FROM seed_candidates c WHERE 1 = 1"
                + visibility_sql
                + " ORDER BY seed_sort_key(c.sort_key_json) DESC, c.candidate_id ASC LIMIT 1",
                params,
            ).fetchone()
        return {
            "candidates": int(row["candidates"] or 0),
            "distinct_contents": int(row["contents"] or 0),
            "evaluations": int(row["evaluations"] or 0),
            "best_score": None if best is None else float(best["score"]),
            "visibility": "current lineage" if self._phase == "immature" else "station-wide",
        }

    def metrics(self) -> list[dict]:
        """List indexed numeric secondary metrics within the enforced access scope."""
        visibility_sql, params = self._visibility_clause()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT m.metric_name, COUNT(*) AS candidate_count, "
                "MIN(m.numeric_value) AS min_value, MAX(m.numeric_value) AS max_value "
                "FROM seed_candidate_metrics m "
                "JOIN seed_candidates c ON c.candidate_id = m.candidate_id "
                "WHERE 1 = 1" + visibility_sql + " "
                "GROUP BY m.metric_name ORDER BY m.metric_name ASC",
                params,
            ).fetchall()
        return [
            {
                "name": str(row["metric_name"]),
                "candidates": int(row["candidate_count"]),
                "min": float(row["min_value"]),
                "max": float(row["max_value"]),
            }
            for row in rows
        ]

    def iter_all(
        self,
        *,
        batch_size: int = 512,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> Iterator[SeedRecord]:
        # SQLite remains the hot-path index; no evaluation YAML is scanned.
        yield from self._iter_query(
            fetch_size=batch_size,
            order="score",
            metric_filters=metric_filters,
            order_metric=order_metric,
            order_metric_desc=order_metric_desc,
            **filters,
        )

    def top(
        self,
        limit: int = 64,
        *,
        exact_distinct: bool = True,
        max_per_evaluation: Optional[int] = None,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> list[SeedRecord]:
        requested = max(0, int(limit))
        if requested == 0:
            return []
        selected: list[SeedRecord] = []
        seen_contents: set[str] = set()
        per_eval: dict[str, int] = {}
        for record in self._iter_query(
            order="score",
            metric_filters=metric_filters,
            order_metric=order_metric,
            order_metric_desc=order_metric_desc,
            **filters,
        ):
            if exact_distinct and record.content_id in seen_contents:
                continue
            if max_per_evaluation is not None and per_eval.get(record.eval_id, 0) >= int(max_per_evaluation):
                continue
            selected.append(record)
            seen_contents.add(record.content_id)
            per_eval[record.eval_id] = per_eval.get(record.eval_id, 0) + 1
            if len(selected) >= requested:
                break
        return selected

    def sample(
        self,
        count: int,
        *,
        exact_distinct: bool = True,
        max_per_evaluation: Optional[int] = None,
        rng: Optional[random.Random] = None,
        metric_filters: Optional[dict[str, Any]] = None,
        **filters,
    ) -> list[SeedRecord]:
        requested = max(0, int(count))
        if requested == 0:
            return []
        per_eval_limit = None if max_per_evaluation is None else max(0, int(max_per_evaluation))
        if per_eval_limit == 0:
            return []
        random_source = rng or random
        seen_contents: set[str] = set()

        if per_eval_limit is None:
            reservoir: list[tuple[float, int, SeedRecord]] = []
            for record in self._iter_query(
                order="id", metric_filters=metric_filters, **filters
            ):
                if exact_distinct and record.content_id in seen_contents:
                    continue
                if exact_distinct:
                    seen_contents.add(record.content_id)
                item = (float(random_source.random()), record.candidate_id, record)
                if len(reservoir) < requested:
                    heapq.heappush(reservoir, item)
                elif item > reservoir[0]:
                    heapq.heapreplace(reservoir, item)
            return [item[2] for item in sorted(reservoir, reverse=True)]

        per_eval: dict[str, list[tuple[float, int, SeedRecord]]] = {}
        for record in self._iter_query(
            order="id", metric_filters=metric_filters, **filters
        ):
            if exact_distinct and record.content_id in seen_contents:
                continue
            if exact_distinct:
                seen_contents.add(record.content_id)
            item = (float(random_source.random()), record.candidate_id, record)
            reservoir = per_eval.setdefault(record.eval_id, [])
            if len(reservoir) < per_eval_limit:
                heapq.heappush(reservoir, item)
            elif item > reservoir[0]:
                heapq.heapreplace(reservoir, item)
        eligible = [item for reservoir in per_eval.values() for item in reservoir]
        return [item[2] for item in heapq.nlargest(requested, eligible)]

    def from_evaluations(
        self,
        eval_ids: Sequence[str],
        *,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> list[SeedRecord]:
        return self._query(
            eval_ids=eval_ids,
            order="score",
            metric_filters=metric_filters,
            order_metric=order_metric,
            order_metric_desc=order_metric_desc,
            **filters,
        )

    def load(self, records: Iterable[SeedRecord]) -> tuple[list[Any], list[dict]]:
        records = list(records)
        open_artifacts: dict[str, Any] = {}
        seeds: list[Any] = []
        try:
            for record in records:
                artifact = open_artifacts.get(record.artifact_path)
                if artifact is None:
                    artifact = np.load(self.bank_root / record.artifact_path, allow_pickle=False)
                    open_artifacts[record.artifact_path] = artifact
                seeds.append(self._restore(record.descriptor, artifact))
        finally:
            for artifact in open_artifacts.values():
                artifact.close()
        return seeds, [record.metadata() for record in records]

    def load_population(
        self,
        limit: int = 64,
        *,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> tuple[list[Any], list[dict]]:
        return self.load(
            self.top(
                limit=limit,
                metric_filters=metric_filters,
                order_metric=order_metric,
                order_metric_desc=order_metric_desc,
                **filters,
            )
        )

    def iter_batches(
        self,
        records: Optional[Iterable[SeedRecord]] = None,
        *,
        batch_size: int = 64,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> Iterator[tuple[list[Any], list[dict]]]:
        size = max(1, int(batch_size))
        iterator = iter(records) if records is not None else self.iter_all(
            metric_filters=metric_filters,
            order_metric=order_metric,
            order_metric_desc=order_metric_desc,
            **filters,
        )
        pending: list[SeedRecord] = []
        for record in iterator:
            pending.append(record)
            if len(pending) >= size:
                yield self.load(pending)
                pending = []
        if pending:
            yield self.load(pending)

    def rank(
        self,
        key: Callable[[Any, dict], Any],
        *,
        limit: int = 64,
        pool_limit: Optional[int] = None,
        reverse: bool = True,
        exact_distinct: bool = True,
        load_batch_size: int = 8,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> list[SeedRecord]:
        requested = max(0, int(limit))
        if requested == 0:
            return []
        ranked: list[tuple[Any, SeedRecord]] = []
        seen_contents: set[str] = set()
        pending: list[SeedRecord] = []

        def consume(batch: list[SeedRecord]) -> None:
            seeds, metadata = self.load(batch)
            for record, seed, meta in zip(batch, seeds, metadata):
                ranked.append((key(seed, meta), record))
                # Keep only the requested frontier. This intentionally trades
                # O(limit log limit) metadata work for bounded seed memory.
                ranked.sort(key=lambda item: item[0], reverse=bool(reverse))
                if len(ranked) > requested:
                    del ranked[requested:]

        for record in self._iter_query(
            limit=pool_limit,
            order="score",
            metric_filters=metric_filters,
            order_metric=order_metric,
            order_metric_desc=order_metric_desc,
            **filters,
        ):
            if exact_distinct and record.content_id in seen_contents:
                continue
            if exact_distinct:
                seen_contents.add(record.content_id)
            pending.append(record)
            if len(pending) >= max(1, int(load_batch_size)):
                consume(pending)
                pending = []
        if pending:
            consume(pending)
        return [item[1] for item in ranked]

    def rank_metadata(
        self,
        key: Callable[[dict], Any],
        *,
        limit: int = 64,
        pool_limit: Optional[int] = None,
        reverse: bool = True,
        exact_distinct: bool = True,
        metric_filters: Optional[dict[str, Any]] = None,
        order_metric: Optional[str] = None,
        order_metric_desc: bool = True,
        **filters,
    ) -> list[SeedRecord]:
        """Rerank from indexed metadata without loading numerical seed arrays."""
        requested = max(0, int(limit))
        if requested == 0:
            return []
        ranked: list[tuple[Any, SeedRecord]] = []
        seen_contents: set[str] = set()
        for record in self._iter_query(
            limit=pool_limit,
            order="score",
            metric_filters=metric_filters,
            order_metric=order_metric,
            order_metric_desc=order_metric_desc,
            **filters,
        ):
            if exact_distinct and record.content_id in seen_contents:
                continue
            if exact_distinct:
                seen_contents.add(record.content_id)
            ranked.append((key(record.metadata()), record))
            ranked.sort(key=lambda item: item[0], reverse=bool(reverse))
            if len(ranked) > requested:
                del ranked[requested:]
        return [item[1] for item in ranked]

    def distinct(
        self,
        records: Iterable[SeedRecord],
        *,
        limit: int,
        distance: Callable[[Any, Any], float],
        min_distance: float,
        feature: Optional[Callable[[Any], Any]] = None,
        load_batch_size: int = 8,
    ) -> list[SeedRecord]:
        """Greedy streaming diversity selection without a full pairwise matrix."""
        requested = max(0, int(limit))
        if requested == 0:
            return []
        selected_records: list[SeedRecord] = []
        selected_features: list[Any] = []
        pending: list[SeedRecord] = []
        for record in records:
            pending.append(record)
            if len(pending) < max(1, int(load_batch_size)):
                continue
            seeds, _metadata = self.load(pending)
            for pending_record, seed in zip(pending, seeds):
                candidate_feature = feature(seed) if feature else seed
                if all(float(distance(candidate_feature, prior)) >= float(min_distance) for prior in selected_features):
                    selected_records.append(pending_record)
                    selected_features.append(candidate_feature)
                    if len(selected_records) >= requested:
                        return selected_records
            pending = []
        if pending:
            seeds, _metadata = self.load(pending)
            for pending_record, seed in zip(pending, seeds):
                candidate_feature = feature(seed) if feature else seed
                if all(float(distance(candidate_feature, prior)) >= float(min_distance) for prior in selected_features):
                    selected_records.append(pending_record)
                    selected_features.append(candidate_feature)
                    if len(selected_records) >= requested:
                        break
        return selected_records

    @classmethod
    def _restore(cls, descriptor: dict, artifact) -> Any:
        kind = descriptor["kind"]
        if kind == "array":
            return np.asarray(artifact[descriptor["member"]])
        if kind == "scalar":
            return np.asarray(artifact[descriptor["member"]]).item()
        if kind == "tuple":
            return tuple(cls._restore(item, artifact) for item in descriptor["items"])
        if kind == "list":
            return [cls._restore(item, artifact) for item in descriptor["items"]]
        if kind == "dict":
            return {item["key"]: cls._restore(item["value"], artifact) for item in descriptor["items"]}
        raise ValueError(f"Unknown seed descriptor kind: {kind}")


__all__ = ["SeedBank", "SeedRecord"]


if __name__ == "__main__":
    import sys

    arguments = [argument for argument in sys.argv[1:] if argument != "--help"]
    print(SeedBank.help(arguments[0] if arguments else None))
