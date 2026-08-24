"""Station-owned persistence for optional Research Seed Bank candidates."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from filelock import FileLock

from station import file_io_utils
from station.eval_research.base_evaluator import SeedBatchEvaluation


SEED_BANK_DIRNAME = "seed_bank"
SEED_BANK_SCHEMA_VERSION = 3
SEED_BANK_CLIENT_FILENAME = "seed_bank.py"


@dataclass(frozen=True)
class RankedSeedBatch:
    batch: SeedBatchEvaluation
    ranked_indices: list[int]
    winner_index: int
    runner_up_index: Optional[int]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _numeric_secondary_metrics(details: Any) -> list[tuple[str, float]]:
    """Extract task-defined top-level numeric metrics from evaluator details."""
    if not isinstance(details, dict):
        return []
    metrics: list[tuple[str, float]] = []
    for raw_name, raw_value in details.items():
        name = str(raw_name).strip()
        if not name or name.lower() == "message":
            continue
        value = raw_value
        # Formatted evaluator details use [display_string, raw_value].
        if isinstance(value, (list, tuple)) and len(value) == 2:
            value = value[1]
        if isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, (int, float, np.number)):
            numeric = float(value)
        else:
            continue
        if np.isfinite(numeric):
            metrics.append((name, numeric))
    return metrics


def validate_and_rank_seed_batch(batch: SeedBatchEvaluation, consts) -> RankedSeedBatch:
    seeds = list(batch.seeds)
    scores = np.asarray(batch.scores)
    valid = np.asarray(batch.valid, dtype=bool)
    sort_keys = list(batch.sort_keys)
    details = list(batch.details)
    errors = list(batch.errors)
    count = len(seeds)
    max_candidates = int(getattr(consts, "RESEARCH_SEED_BANK_MAX_CANDIDATES", 64))
    if count < 1:
        raise ValueError("Seed-enabled submissions must contain at least one candidate.")
    if count > max_candidates:
        raise ValueError(f"Seed batch has {count} candidates; maximum is {max_candidates}.")
    if scores.shape != (count,):
        raise ValueError(f"Seed batch scores must have shape ({count},), got {scores.shape}.")
    if valid.shape != (count,):
        raise ValueError(f"Seed batch valid flags must have shape ({count},), got {valid.shape}.")
    for name, values in (("sort_keys", sort_keys), ("details", details), ("errors", errors)):
        if len(values) != count:
            raise ValueError(f"Seed batch {name} must contain {count} entries, got {len(values)}.")

    ranked_indices = [index for index in range(count) if bool(valid[index])]
    if not ranked_indices:
        combined_errors = "; ".join(str(error) for error in errors if error) or "all candidates were invalid"
        raise ValueError(f"Seed batch has no valid candidates: {combined_errors}")
    normalized_sort_keys = list(sort_keys)
    for index in ranked_indices:
        if not np.isfinite(float(scores[index])):
            raise ValueError(f"Valid seed candidate {index} has a non-finite score.")
        if not isinstance(sort_keys[index], (tuple, list)):
            raise ValueError(f"Seed candidate {index} sort key must be a tuple or list.")
        try:
            normalized_key = tuple(float(component) for component in sort_keys[index])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Seed candidate {index} sort key must be numeric.") from exc
        if not normalized_key or not all(np.isfinite(component) for component in normalized_key):
            raise ValueError(f"Seed candidate {index} sort key must be finite and non-empty.")
        normalized_sort_keys[index] = normalized_key
    ranked_indices.sort(key=lambda index: normalized_sort_keys[index], reverse=True)
    return RankedSeedBatch(
        batch=SeedBatchEvaluation(seeds, scores, valid, [tuple(key) for key in normalized_sort_keys], details, errors),
        ranked_indices=ranked_indices,
        winner_index=ranked_indices[0],
        runner_up_index=ranked_indices[1] if len(ranked_indices) > 1 else None,
    )


class SeedBankStore:
    def __init__(self, paths, consts):
        self.paths = paths
        self.consts = consts
        self.root = Path(paths.shared_storage) / SEED_BANK_DIRNAME
        self.manifests_dir = self.root / "manifests"
        self.artifacts_dir = self.root / "artifacts"
        self.index_path = self.root / "index.sqlite"
        self.lock_path = self.root / "index.lock"
        self._layout_ready = False

    def ensure_layout(self) -> None:
        for path in (self.root, self.manifests_dir, self.artifacts_dir):
            file_io_utils.ensure_dir_exists(str(path))
        with FileLock(str(self.lock_path)):
            with self._connect() as connection:
                self._create_schema(connection)
                signature = self._task_signature()
                existing = connection.execute(
                    "SELECT value FROM seed_meta WHERE key = 'task_signature'"
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO seed_meta(key, value) VALUES('task_signature', ?)",
                        (signature,),
                    )
                elif str(existing[0]) != signature:
                    candidate_count = connection.execute(
                        "SELECT COUNT(*) FROM seed_candidates"
                    ).fetchone()[0]
                    if int(candidate_count or 0) > 0:
                        raise RuntimeError(
                            "Seed Bank task/evaluator signature changed while the bank contains candidates. "
                            "Archive or rebuild the task-specific bank before starting this task."
                        )
                    connection.execute(
                        "UPDATE seed_meta SET value = ? WHERE key = 'task_signature'",
                        (signature,),
                    )
                connection.commit()
        self._layout_ready = True

    def _task_signature(self) -> str:
        hasher = hashlib.sha256()
        research_root = Path(self.paths.research_root)
        sources = [
            research_root / "research_task.md",
            research_root / "evaluators" / "evaluator.py",
        ]
        found = False
        for source in sources:
            if not source.is_file():
                continue
            found = True
            hasher.update(source.name.encode("utf-8") + b"\0")
            hasher.update(source.read_bytes())
        return hasher.hexdigest() if found else "unavailable"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.index_path), timeout=60)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS seed_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS seed_contents (
                content_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                artifact_path TEXT NOT NULL,
                descriptor_json TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS seed_candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                batch_index INTEGER NOT NULL,
                lineage TEXT NOT NULL,
                author TEXT NOT NULL,
                score REAL NOT NULL,
                sort_key_json TEXT NOT NULL,
                details_json TEXT NOT NULL,
                error TEXT,
                batch_rank INTEGER NOT NULL,
                is_winner INTEGER NOT NULL,
                content_id TEXT NOT NULL REFERENCES seed_contents(content_id),
                created_at REAL NOT NULL,
                UNIQUE(eval_id, attempt_number, batch_index)
            );
            CREATE TABLE IF NOT EXISTS seed_candidate_metrics (
                candidate_id INTEGER NOT NULL REFERENCES seed_candidates(candidate_id) ON DELETE CASCADE,
                metric_name TEXT NOT NULL,
                numeric_value REAL NOT NULL,
                PRIMARY KEY(candidate_id, metric_name)
            );
            CREATE INDEX IF NOT EXISTS idx_seed_candidates_score
                ON seed_candidates(score DESC, candidate_id ASC);
            CREATE INDEX IF NOT EXISTS idx_seed_candidates_lineage_score
                ON seed_candidates(lineage, score DESC, candidate_id ASC);
            CREATE INDEX IF NOT EXISTS idx_seed_candidates_eval
                ON seed_candidates(eval_id, attempt_number, batch_rank);
            CREATE INDEX IF NOT EXISTS idx_seed_candidates_content
                ON seed_candidates(content_id);
            CREATE INDEX IF NOT EXISTS idx_seed_candidate_metrics_name_value
                ON seed_candidate_metrics(metric_name, numeric_value DESC, candidate_id ASC);
            CREATE INDEX IF NOT EXISTS idx_seed_candidate_metrics_candidate
                ON seed_candidate_metrics(candidate_id, metric_name);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO seed_meta(key, value) VALUES('schema_version', ?)",
            (str(SEED_BANK_SCHEMA_VERSION),),
        )
        connection.commit()

    @classmethod
    def _replace_secondary_metrics(cls, connection: sqlite3.Connection, candidates: list[dict]) -> None:
        if not candidates:
            return
        eval_ids = sorted({str(candidate["eval_id"]) for candidate in candidates})
        placeholders = ",".join("?" for _ in eval_ids)
        rows = connection.execute(
            "SELECT candidate_id, eval_id, attempt_number, batch_index FROM seed_candidates "
            f"WHERE eval_id IN ({placeholders})",
            eval_ids,
        ).fetchall()
        candidate_ids = {
            (str(eval_id), int(attempt_number), int(batch_index)): int(candidate_id)
            for candidate_id, eval_id, attempt_number, batch_index in rows
        }
        metric_rows = []
        for candidate in candidates:
            candidate_id = candidate_ids.get(
                (
                    str(candidate["eval_id"]),
                    int(candidate.get("attempt_number", 1)),
                    int(candidate["batch_index"]),
                )
            )
            if candidate_id is None:
                continue
            for metric_name, numeric_value in _numeric_secondary_metrics(candidate.get("details")):
                metric_rows.append((candidate_id, metric_name, numeric_value))
        if metric_rows:
            connection.executemany(
                "INSERT OR REPLACE INTO seed_candidate_metrics "
                "(candidate_id, metric_name, numeric_value) VALUES (?, ?, ?)",
                metric_rows,
            )

    def save_batch(
        self,
        *,
        eval_id: str,
        attempt_number: int = 1,
        lineage: str,
        author: str,
        ranked: RankedSeedBatch,
    ) -> dict:
        if not self._layout_ready:
            self.ensure_layout()
        eval_id = str(eval_id)
        attempt_number = max(1, int(attempt_number))
        lineage = str(lineage or "unknown").lower()
        author = str(author or "Unknown")
        artifact_name = f"eval_{eval_id}_attempt_{attempt_number}.npz"
        if (self.artifacts_dir / artifact_name).exists():
            # Keep content artifacts immutable: later successful debug attempts
            # for the same evaluation must not invalidate exact-duplicate rows
            # that still reference the earlier numerical payload.
            artifact_name = (
                f"eval_{eval_id}_attempt_{attempt_number}_{uuid.uuid4().hex[:12]}.npz"
            )
        artifact_path = self.artifacts_dir / artifact_name
        artifact_rel_path = str(Path("artifacts") / artifact_name)
        max_bytes = int(getattr(self.consts, "RESEARCH_SEED_BANK_MAX_BATCH_BYTES", 1_000_000_000))

        encoded_candidates: dict[int, tuple[dict, dict[str, np.ndarray], str, int]] = {}
        total_bytes = 0
        for batch_index in ranked.ranked_indices:
            encoded = self._encode_seed(ranked.batch.seeds[batch_index], f"c{batch_index:03d}")
            encoded_candidates[batch_index] = encoded
            total_bytes += encoded[3]
            if total_bytes > max_bytes:
                raise ValueError(
                    f"Canonical seed batch uses {total_bytes} bytes; maximum is {max_bytes}."
                )

        with FileLock(str(self.lock_path)):
            fingerprints = sorted({encoded[2] for encoded in encoded_candidates.values()})
            with self._connect() as connection:
                placeholders = ",".join("?" for _ in fingerprints)
                existing_rows = connection.execute(
                    "SELECT content_id, fingerprint, artifact_path, descriptor_json, byte_count, created_at "
                    f"FROM seed_contents WHERE fingerprint IN ({placeholders})",
                    fingerprints,
                ).fetchall()
            existing = {
                row[1]: {
                    "content_id": row[0],
                    "fingerprint": row[1],
                    "artifact_path": row[2],
                    "descriptor": json.loads(row[3]),
                    "byte_count": int(row[4]),
                    "created_at": float(row[5]),
                }
                for row in existing_rows
            }

            arrays_to_write: dict[str, np.ndarray] = {}
            candidate_content: dict[int, dict] = {}
            for batch_index in ranked.ranked_indices:
                descriptor, arrays, fingerprint, byte_count = encoded_candidates[batch_index]
                record = existing.get(fingerprint)
                if record is None:
                    content_id = fingerprint
                    record = {
                        "content_id": content_id,
                        "fingerprint": fingerprint,
                        "artifact_path": artifact_rel_path,
                        "descriptor": descriptor,
                        "byte_count": byte_count,
                        "created_at": time.time(),
                    }
                    existing[fingerprint] = record
                    arrays_to_write.update(arrays)
                candidate_content[batch_index] = record

            self._atomic_write_npz(artifact_path, arrays_to_write)
            rank_by_index = {batch_index: rank + 1 for rank, batch_index in enumerate(ranked.ranked_indices)}
            candidates = []
            for batch_index in ranked.ranked_indices:
                content = candidate_content[batch_index]
                candidates.append(
                    {
                        "eval_id": eval_id,
                        "attempt_number": attempt_number,
                        "batch_index": batch_index,
                        "lineage": lineage,
                        "author": author,
                        "score": float(ranked.batch.scores[batch_index]),
                        "sort_key": _jsonable(ranked.batch.sort_keys[batch_index]),
                        "details": _jsonable(ranked.batch.details[batch_index]),
                        "error": ranked.batch.errors[batch_index],
                        "batch_rank": rank_by_index[batch_index],
                        "is_winner": batch_index == ranked.winner_index,
                        "content_id": content["content_id"],
                        "created_at": time.time(),
                    }
                )
            referenced_contents = {
                record["content_id"]: record for record in candidate_content.values()
            }
            manifest = {
                "schema_version": SEED_BANK_SCHEMA_VERSION,
                "eval_id": eval_id,
                "attempt_number": attempt_number,
                "created_at": time.time(),
                "artifact_path": artifact_rel_path,
                "candidates": candidates,
                "contents": list(referenced_contents.values()),
            }
            manifest_path = self.manifests_dir / (
                f"eval_{eval_id}_attempt_{attempt_number}.json"
            )
            file_io_utils.save_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                str(manifest_path),
            )

            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM seed_candidates WHERE eval_id = ? AND attempt_number = ?",
                    (eval_id, attempt_number),
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO seed_contents "
                    "(content_id, fingerprint, artifact_path, descriptor_json, byte_count, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            record["content_id"],
                            record["fingerprint"],
                            record["artifact_path"],
                            json.dumps(record["descriptor"], sort_keys=True),
                            int(record["byte_count"]),
                            float(record["created_at"]),
                        )
                        for record in referenced_contents.values()
                    ],
                )
                connection.executemany(
                    "INSERT INTO seed_candidates "
                    "(eval_id, attempt_number, batch_index, lineage, author, score, sort_key_json, details_json, "
                    "error, batch_rank, is_winner, content_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            candidate["eval_id"], candidate["attempt_number"],
                            candidate["batch_index"], candidate["lineage"],
                            candidate["author"], candidate["score"],
                            json.dumps(candidate["sort_key"], sort_keys=True),
                            json.dumps(candidate["details"], sort_keys=True), candidate["error"],
                            candidate["batch_rank"], int(candidate["is_winner"]),
                            candidate["content_id"], candidate["created_at"],
                        )
                        for candidate in candidates
                    ],
                )
                self._replace_secondary_metrics(connection, candidates)
                connection.commit()
        return manifest

    def rebuild_index(self) -> None:
        self.ensure_layout()
        with FileLock(str(self.lock_path)):
            temporary = self.index_path.with_name(f"index.{uuid.uuid4().hex}.sqlite")
            connection = sqlite3.connect(str(temporary))
            try:
                self._create_schema(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO seed_meta(key, value) VALUES('task_signature', ?)",
                    (self._task_signature(),),
                )
                for manifest_path in sorted(self.manifests_dir.glob("eval_*.json")):
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    for record in manifest.get("contents", []):
                        connection.execute(
                            "INSERT OR IGNORE INTO seed_contents "
                            "(content_id, fingerprint, artifact_path, descriptor_json, byte_count, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                record["content_id"], record["fingerprint"], record["artifact_path"],
                                json.dumps(record["descriptor"], sort_keys=True),
                                int(record["byte_count"]), float(record["created_at"]),
                            ),
                        )
                    candidates = list(manifest.get("candidates", []))
                    manifest_attempt_number = int(manifest["attempt_number"])
                    for candidate in candidates:
                        attempt_number = int(candidate["attempt_number"])
                        if attempt_number != manifest_attempt_number:
                            raise ValueError(
                                f"Seed manifest attempt mismatch in {manifest_path}: "
                                f"manifest={manifest_attempt_number}, candidate={attempt_number}"
                            )
                        connection.execute(
                            "INSERT OR REPLACE INTO seed_candidates "
                            "(eval_id, attempt_number, batch_index, lineage, author, score, sort_key_json, details_json, "
                            "error, batch_rank, is_winner, content_id, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                candidate["eval_id"], attempt_number,
                                candidate["batch_index"], candidate["lineage"],
                                candidate["author"], candidate["score"],
                                json.dumps(candidate["sort_key"], sort_keys=True),
                                json.dumps(candidate["details"], sort_keys=True), candidate.get("error"),
                                candidate["batch_rank"], int(candidate["is_winner"]),
                                candidate["content_id"], candidate["created_at"],
                            ),
                        )
                    self._replace_secondary_metrics(
                        connection,
                        candidates,
                    )
                connection.commit()
            finally:
                connection.close()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.index_path) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            os.replace(temporary, self.index_path)

    @classmethod
    def _encode_seed(cls, seed: Any, prefix: str) -> tuple[dict, dict[str, np.ndarray], str, int]:
        arrays: dict[str, np.ndarray] = {}
        hasher = hashlib.sha256()
        byte_count = 0

        def encode(value: Any, path: str) -> dict:
            nonlocal byte_count
            if isinstance(value, dict):
                items = []
                for key in sorted(value):
                    if not isinstance(key, str):
                        raise TypeError("Seed dictionaries must use string keys.")
                    hasher.update(b"dict-key\0" + key.encode("utf-8") + b"\0")
                    items.append({"key": key, "value": encode(value[key], f"{path}_{key}")})
                return {"kind": "dict", "items": items}
            if isinstance(value, tuple):
                hasher.update(b"tuple\0")
                return {"kind": "tuple", "items": [encode(item, f"{path}_{i:03d}") for i, item in enumerate(value)]}
            if isinstance(value, list) and not cls._is_numeric_array(value):
                hasher.update(b"list\0")
                return {"kind": "list", "items": [encode(item, f"{path}_{i:03d}") for i, item in enumerate(value)]}

            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError("Seed candidates cannot contain object-dtype arrays.")
            if array.dtype.kind in {"U", "S", "V"}:
                raise TypeError("Seed candidates must contain numeric or boolean arrays/scalars.")
            array = np.ascontiguousarray(array)
            member = path
            arrays[member] = array
            byte_count += int(array.nbytes)
            kind = "scalar" if array.ndim == 0 else "array"
            hasher.update(kind.encode("ascii") + b"\0")
            hasher.update(array.dtype.str.encode("ascii") + b"\0")
            hasher.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
            hasher.update(array.tobytes(order="C"))
            return {"kind": kind, "member": member}

        descriptor = encode(seed, prefix)
        return descriptor, arrays, hasher.hexdigest(), byte_count

    @staticmethod
    def _is_numeric_array(value: list) -> bool:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return False
        return not array.dtype.hasobject and array.dtype.kind not in {"U", "S", "V"}

    @staticmethod
    def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        file_io_utils.ensure_dir_exists(str(path.parent))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temporary, "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def ensure_seed_bank_layout(paths, consts) -> Optional[SeedBankStore]:
    if not bool(getattr(consts, "RESEARCH_SEED_BANK_ENABLED", False)):
        return None
    store = SeedBankStore(paths, consts)
    store.ensure_layout()
    return store


def install_seed_bank_client(paths, consts, *, snapshot_dirname: str = "_internal") -> None:
    """Install the frozen client inside this station's resolved storage allocation.

    ``snapshot_dirname`` is retained for compatibility with older callers.  Older
    releases installed the client below the branch-local ``_internal`` directory
    and exposed it through a relative symlink from ``storage/system``.  That link
    becomes stale when multistart relocates Research storage to a per-seed remote
    allocation or moves the selected branch back to ``station_data``.

    Installing a regular file in ``paths.system_storage`` keeps the client and its
    branch-private Seed Bank data in the same allocation.  ``save_text`` replaces
    legacy or read-only symlinks atomically through the writable system directory.
    """
    if not bool(getattr(consts, "RESEARCH_SEED_BANK_ENABLED", False)):
        return
    _ = snapshot_dirname
    source = Path(__file__).with_name("seed_bank_client.py")
    content = source.read_text(encoding="utf-8")
    system_storage = Path(paths.system_storage)
    file_io_utils.ensure_dir_exists(str(system_storage))
    client = system_storage / SEED_BANK_CLIENT_FILENAME

    current_content = None
    current_is_writable = False
    if client.is_file() and not client.is_symlink():
        current_content = client.read_text(encoding="utf-8")
        current_is_writable = bool(client.stat().st_mode & 0o222)
    if client.is_symlink() or current_content != content or current_is_writable:
        file_io_utils.save_text(content, str(client), file_mode=0o444)


__all__ = [
    "RankedSeedBatch",
    "SeedBankStore",
    "ensure_seed_bank_layout",
    "install_seed_bank_client",
    "validate_and_rank_seed_batch",
]
