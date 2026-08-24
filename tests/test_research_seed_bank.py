import importlib.util
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from station import constants
from station.eval_research.base_evaluator import SeedBatchEvaluation
from station.eval_research.auto_evaluator import AutoResearchEvaluator
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.eval_research.seed_bank import (
    SeedBankStore,
    validate_and_rank_seed_batch,
)
from station.eval_research.seed_bank_client import SeedBank
from station.eval_research.coder_helpers import render_coder_access_policy
from station.eval_research.coder_manager import ResearchCoderManager
from station.eval_research.evaluation_manager import EvaluationManager
from station.rooms.research_center import ResearchCenter


class ResearchSeedBankTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="station-seed-bank-test-")
        self.addCleanup(shutil.rmtree, self.temp_dir, True)

    def _runtime_patches(self, enabled):
        return patch.multiple(
            constants,
            BASE_STATION_DATA_PATH=self.temp_dir,
            RESEARCH_STORAGE_BASE_PATH=None,
            RESEARCH_SEED_BANK_ENABLED=enabled,
            RESEARCH_SEED_BANK_MAX_CANDIDATES=64,
            RESEARCH_SEED_BANK_MAX_BATCH_BYTES=1_000_000,
        )

    def _store(self):
        research_root = Path(self.temp_dir) / "research"
        shared = research_root / "storage" / "shared"
        system = research_root / "storage" / "system"
        shared.mkdir(parents=True, exist_ok=True)
        system.mkdir(parents=True, exist_ok=True)
        paths = SimpleNamespace(
            research_root=str(research_root),
            shared_storage=str(shared),
            system_storage=str(system),
        )
        consts = SimpleNamespace(
            RESEARCH_SEED_BANK_ENABLED=True,
            RESEARCH_SEED_BANK_MAX_CANDIDATES=64,
            RESEARCH_SEED_BANK_MAX_BATCH_BYTES=1_000_000,
        )
        store = SeedBankStore(paths, consts)
        store.ensure_layout()
        return store, paths, consts

    @staticmethod
    def _batch(seeds, scores, valid=None):
        valid = valid if valid is not None else [True] * len(seeds)
        return SeedBatchEvaluation(
            seeds=list(seeds),
            scores=np.asarray(scores, dtype=float),
            valid=np.asarray(valid, dtype=bool),
            sort_keys=[(float(score),) for score in scores],
            details=[{"Message": f"candidate {index}", "Metric": float(score)} for index, score in enumerate(scores)],
            errors=[None if flag else "invalid" for flag in valid],
        )

    def test_disabled_runtime_creates_no_seed_bank_artifacts(self):
        with self._runtime_patches(False):
            paths = ensure_runtime_layout(constants)
        self.assertFalse(Path(paths.shared_storage, "seed_bank").exists())
        self.assertFalse(Path(paths.system_storage, "seed_bank.py").exists())
        self.assertFalse(Path(paths.research_root, "_internal", "seed_bank_client_snapshot.py").exists())

    def test_enabled_runtime_installs_frozen_client_and_index(self):
        with self._runtime_patches(True):
            paths = ensure_runtime_layout(constants)
        bank_root = Path(paths.shared_storage, "seed_bank")
        client = Path(paths.system_storage, "seed_bank.py")
        snapshot = Path(paths.research_root, "_internal", "seed_bank_client_snapshot.py")
        self.assertTrue(bank_root.joinpath("index.sqlite").is_file())
        self.assertTrue(bank_root.joinpath("manifests").is_dir())
        self.assertTrue(bank_root.joinpath("artifacts").is_dir())
        self.assertTrue(client.is_file())
        self.assertFalse(client.is_symlink())
        self.assertEqual(
            Path(__file__).parents[1].joinpath("station", "eval_research", "seed_bank_client.py").read_text(),
            client.read_text(),
        )
        self.assertFalse(snapshot.exists())

    def test_remote_client_upgrades_legacy_link_and_survives_selected_branch_move(self):
        origin = Path(self.temp_dir, "origin_station_data")
        remote_base = Path(self.temp_dir, "remote_storage")
        branch = Path(self.temp_dir, "job", "station_data_s1")
        live = Path(self.temp_dir, "station_data")
        client_source = Path(__file__).parents[1] / "station" / "eval_research" / "seed_bank_client.py"

        with patch.multiple(
            constants,
            BASE_STATION_DATA_PATH=str(origin),
            RESEARCH_STORAGE_BASE_PATH=None,
            RESEARCH_SEED_BANK_ENABLED=True,
            RESEARCH_SEED_BANK_MAX_CANDIDATES=64,
            RESEARCH_SEED_BANK_MAX_BATCH_BYTES=1_000_000,
        ), patch.dict(os.environ, {"RESEARCH_STORAGE_BASE_PATH": ""}):
            origin_paths = ensure_runtime_layout(constants)

        # Recreate the legacy layout that a pre-fix station contributes to a
        # multistart copy: a read-only branch-local snapshot exposed through a
        # relative link from local storage/system.
        origin_system = Path(origin_paths.system_storage)
        origin_client = origin_system / "seed_bank.py"
        legacy_snapshot = Path(origin_paths.research_root, "_internal", "seed_bank_client_snapshot.py")
        os.chmod(origin_system, 0o755)
        legacy_snapshot.write_text(client_source.read_text(encoding="utf-8"), encoding="utf-8")
        legacy_snapshot.chmod(0o444)
        origin_client.unlink()
        origin_client.symlink_to(os.path.relpath(legacy_snapshot, origin_client.parent))
        os.chmod(origin_system, 0o555)

        branch.parent.mkdir(parents=True)
        shutil.copytree(origin, branch, symlinks=True)
        remote_base.mkdir()
        allocation = remote_base / "seed-1-allocation"
        branch_storage = branch / "rooms" / "research" / "storage"
        branch_storage.rename(allocation)
        branch_storage.symlink_to(allocation, target_is_directory=True)

        stale_client = allocation / "system" / "seed_bank.py"
        self.assertTrue(stale_client.is_symlink())
        self.assertFalse(stale_client.exists())
        self.assertEqual(0o444, (branch / "rooms" / "research" / "_internal" / legacy_snapshot.name).stat().st_mode & 0o777)

        real_chmod = os.chmod

        def reject_remote_chmod(path, mode, *args, **kwargs):
            try:
                Path(path).resolve(strict=False).relative_to(allocation)
            except ValueError:
                return real_chmod(path, mode, *args, **kwargs)
            raise PermissionError(1, "remote filesystem rejects chmod", str(path))

        with patch.multiple(
            constants,
            BASE_STATION_DATA_PATH=str(branch),
            RESEARCH_STORAGE_BASE_PATH=str(remote_base),
            RESEARCH_SEED_BANK_ENABLED=True,
            RESEARCH_SEED_BANK_MAX_CANDIDATES=64,
            RESEARCH_SEED_BANK_MAX_BATCH_BYTES=1_000_000,
        ), patch.dict(os.environ, {"RESEARCH_STORAGE_BASE_PATH": str(remote_base)}), patch(
            "station.eval_research.runtime_paths.os.chmod",
            side_effect=reject_remote_chmod,
        ):
            branch_paths = ensure_runtime_layout(constants)

        installed_client = allocation / "system" / "seed_bank.py"
        self.assertEqual(allocation, branch_storage.resolve())
        self.assertEqual(Path(branch_paths.system_storage), installed_client.parent)
        self.assertTrue(installed_client.is_file())
        self.assertFalse(installed_client.is_symlink())
        self.assertEqual(client_source.read_text(encoding="utf-8"), installed_client.read_text(encoding="utf-8"))
        self.assertEqual(0o444, installed_client.stat().st_mode & 0o777)

        # Multistart finalization moves station_data_sN to station_data while
        # promoting the same private remote allocation.  No path inside the
        # installed client depends on the old branch root.  Simulate a copied
        # regular client that acquired an owner-write bit during an immutable
        # system-tree repair; startup must replace it even when its content is
        # already current and the remote filesystem rejects chmod.
        installed_client.chmod(0o644)
        branch.rename(live)
        with patch.multiple(
            constants,
            BASE_STATION_DATA_PATH=str(live),
            RESEARCH_STORAGE_BASE_PATH=str(remote_base),
            RESEARCH_SEED_BANK_ENABLED=True,
            RESEARCH_SEED_BANK_MAX_CANDIDATES=64,
            RESEARCH_SEED_BANK_MAX_BATCH_BYTES=1_000_000,
        ), patch.dict(os.environ, {"RESEARCH_STORAGE_BASE_PATH": str(remote_base)}), patch(
            "station.eval_research.runtime_paths.os.chmod",
            side_effect=reject_remote_chmod,
        ):
            live_paths = ensure_runtime_layout(constants)

        live_storage = live / "rooms" / "research" / "storage"
        self.assertEqual(allocation, live_storage.resolve())
        self.assertEqual(str(allocation / "shared"), live_paths.shared_storage)
        self.assertTrue(installed_client.is_file())
        self.assertFalse(installed_client.is_symlink())
        self.assertEqual(client_source.read_text(encoding="utf-8"), installed_client.read_text(encoding="utf-8"))
        self.assertEqual(0o444, installed_client.stat().st_mode & 0o777)
        self.assertEqual(
            1,
            len(list(allocation.glob(".system_read_only_legacy*"))),
        )

    def test_client_help_is_focused_by_operation(self):
        overview = SeedBank.help()
        self.assertIn('SeedBank.help("top")', overview)
        self.assertNotIn("max_per_evaluation=4", overview)
        top_help = SeedBank.help("top")
        self.assertIn("max_per_evaluation", top_help)
        self.assertIn("score_gt=0.7", top_help)
        self.assertNotIn("lambda seed, meta: my_metric", top_help)
        rank_help = SeedBank.help("rank")
        self.assertIn("key(seed, metadata)", rank_help)
        self.assertIn("pool_limit=4096", rank_help)
        metrics_help = SeedBank.help("metrics")
        self.assertIn('order_metric="NormRatio"', metrics_help)
        self.assertIn('"NonzeroFraction": {"gt": 0.9}', metrics_help)
        self.assertIn("Unknown Seed Bank help topic", SeedBank.help("missing"))

    def test_none_result_completes_without_saving_a_seed(self):
        store, _store_paths, _consts = self._store()
        run_root = Path(self.temp_dir) / "none-run"
        stdout_dir = run_root / "stdout"
        stderr_dir = run_root / "stderr"
        stdout_dir.mkdir(parents=True)
        stderr_dir.mkdir(parents=True)
        submission_path = run_root / "submission.py"
        submission_path.write_text("def construct_solution():\n    return None\n", encoding="utf-8")
        request_path = run_root / "77_attempt_1.yaml"
        request_path.write_text("eval_id: '77'\nattempt: 1\n", encoding="utf-8")

        class NoneEvaluator:
            batch_called = False

            @staticmethod
            def get_execution_mode():
                return "function"

            @staticmethod
            def validate_submission_code(_content, _author, _agent_module):
                return True, None

            def evaluate_seed_batch_with_formatting(self, *_args, **_kwargs):
                self.batch_called = True
                raise AssertionError("None must bypass seed batch evaluation")

        class FakeManager:
            def __init__(self):
                self.completed = None

            @staticmethod
            def get_evaluation(_eval_id):
                return {
                    "id": "77",
                    "instruction": "Run an analysis-only attempt.",
                    "author": "Agent",
                    "lineage": "alpha",
                    "coder": {"max_attempts": 5},
                    "coder_access": {"phase": "immature"},
                    "attempts": [{"attempt": 1, "submission_path": str(submission_path)}],
                }

            @staticmethod
            def mark_attempt_running(*_args, **_kwargs):
                return None

            def complete_attempt(self, eval_id, attempt_number, **kwargs):
                self.completed = {
                    "eval_id": eval_id,
                    "attempt_number": attempt_number,
                    **kwargs,
                }

        task_evaluator = NoneEvaluator()
        manager = FakeManager()
        auto = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
        auto.eval_manager = manager
        auto.station = SimpleNamespace(_get_current_tick=lambda: 8, agent_module=None)
        auto.paths = SimpleNamespace(stdout_dir=str(stdout_dir), stderr_dir=str(stderr_dir))
        auto.seed_bank_store = store
        auto._get_evaluator = lambda: task_evaluator
        auto._initialize_attempt_logs = lambda *_args, **_kwargs: "attempt start"
        auto._execute_submission = lambda **_kwargs: {
            "success": True,
            "result": None,
            "stdout": "analysis finished",
            "stderr": "",
        }

        with patch.object(constants, "RESEARCH_SEED_BANK_ENABLED", True):
            auto._execute_run_request("77", 1, str(request_path))

        self.assertFalse(task_evaluator.batch_called)
        self.assertFalse(request_path.exists())
        self.assertIsNotNone(manager.completed)
        self.assertTrue(manager.completed["success"])
        self.assertEqual(manager.completed["status"], "completed")
        self.assertEqual(manager.completed["score"], constants.RESEARCH_SCORE_NA)
        self.assertEqual(manager.completed["details"]["SeedBankCandidates"], 0)
        self.assertIn("ATTEMPT_STATUS: completed", manager.completed["stdout"])
        self.assertIn("PRIMARY_SCORE: n.a.", manager.completed["stdout"])
        self.assertEqual(list(Path(store.manifests_dir).glob("*.json")), [])
        self.assertEqual(list(Path(store.artifacts_dir).glob("*.npz")), [])

    def test_auto_evaluator_passes_official_attempt_number_to_seed_store(self):
        run_root = Path(self.temp_dir) / "attempt-provenance-run"
        stdout_dir = run_root / "stdout"
        stderr_dir = run_root / "stderr"
        stdout_dir.mkdir(parents=True)
        stderr_dir.mkdir(parents=True)
        submission_path = run_root / "submission.py"
        submission_path.write_text("def construct_solution():\n    return [1.0]\n", encoding="utf-8")
        request_path = run_root / "78_attempt_2.yaml"
        request_path.write_text("eval_id: '78'\nattempt: 2\n", encoding="utf-8")

        class BatchEvaluator:
            @staticmethod
            def get_execution_mode():
                return "function"

            @staticmethod
            def validate_submission_code(_content, _author, _agent_module):
                return True, None

            def evaluate_seed_batch_with_formatting(self, *_args, **_kwargs):
                return ResearchSeedBankTest._batch([np.array([1.0])], [0.75])

        class FakeManager:
            completed = None

            @staticmethod
            def get_evaluation(_eval_id):
                return {
                    "id": "78",
                    "instruction": "Return one construction.",
                    "author": "Agent",
                    "lineage": "alpha",
                    "coder": {"max_attempts": 5},
                    "coder_access": {"phase": "immature"},
                    "attempts": [{"attempt": 2, "submission_path": str(submission_path)}],
                }

            @staticmethod
            def mark_attempt_running(*_args, **_kwargs):
                return None

            def complete_attempt(self, _eval_id, _attempt_number, **kwargs):
                self.completed = kwargs

        class FakeStore:
            saved = None

            def save_batch(self, **kwargs):
                self.saved = kwargs

        manager = FakeManager()
        seed_store = FakeStore()
        auto = AutoResearchEvaluator.__new__(AutoResearchEvaluator)
        auto.eval_manager = manager
        auto.station = SimpleNamespace(_get_current_tick=lambda: 8, agent_module=None)
        auto.paths = SimpleNamespace(stdout_dir=str(stdout_dir), stderr_dir=str(stderr_dir))
        auto.seed_bank_store = seed_store
        auto._get_evaluator = lambda: BatchEvaluator()
        auto._initialize_attempt_logs = lambda *_args, **_kwargs: "attempt start"
        auto._execute_submission = lambda **_kwargs: {
            "success": True,
            "result": [1.0],
            "stdout": "constructed",
            "stderr": "",
        }

        with patch.object(constants, "RESEARCH_SEED_BANK_ENABLED", True):
            auto._execute_run_request("78", 2, str(request_path))

        self.assertIsNotNone(seed_store.saved)
        self.assertEqual(seed_store.saved["eval_id"], "78")
        self.assertEqual(seed_store.saved["attempt_number"], 2)
        self.assertEqual(seed_store.saved["lineage"], "alpha")
        self.assertEqual(manager.completed["status"], "completed")

    def test_rank_persist_dedupe_visibility_and_client_apis(self):
        store, _paths, consts = self._store()
        first = self._batch(
            [np.array([1.0, 0.0]), np.array([0.5, 1.0]), np.array([1.0, 0.0])],
            [0.5, 0.9, 0.4],
        )
        for detail, preference in zip(first.details, [2.0, 1.0, 3.0]):
            detail["Preference"] = [f"{preference:.1f}", preference]
        ranked = validate_and_rank_seed_batch(first, consts)
        self.assertEqual(ranked.winner_index, 1)
        self.assertEqual(ranked.runner_up_index, 0)
        manifest = store.save_batch(
            eval_id="10", lineage="alpha", author="Agent A", ranked=ranked
        )
        self.assertEqual([item["batch_rank"] for item in manifest["candidates"]], [1, 2, 3])
        self.assertEqual(len({item["content_id"] for item in manifest["candidates"]}), 2)

        second = self._batch(
            [np.array([0.2, 0.8]), np.array([1.0, 0.0])],
            [0.8, 0.7],
        )
        for detail, preference in zip(second.details, [5.0, 4.0]):
            detail["Preference"] = [f"{preference:.1f}", preference]
        store.save_batch(
            eval_id="11",
            lineage="beta",
            author="Agent B",
            ranked=validate_and_rank_seed_batch(second, consts),
        )

        with sqlite3.connect(store.index_path) as connection:
            candidate_count = connection.execute("SELECT COUNT(*) FROM seed_candidates").fetchone()[0]
            content_count = connection.execute("SELECT COUNT(*) FROM seed_contents").fetchone()[0]
            metric_count = connection.execute("SELECT COUNT(*) FROM seed_candidate_metrics").fetchone()[0]
        self.assertEqual(candidate_count, 5)
        self.assertEqual(content_count, 3)
        self.assertEqual(metric_count, 10)

        with patch.dict(
            os.environ,
            {"STATION_LINEAGE": "alpha", "STATION_ACCESS_PHASE": "immature"},
            clear=False,
        ):
            bank = SeedBank.open(str(store.root))
            self.assertEqual(bank.summary()["candidates"], 3)
            records = bank.top(limit=64)
            self.assertEqual(len(records), 2)
            seeds, metadata = bank.load(records)
            self.assertTrue(np.allclose(seeds[0], [0.5, 1.0]))
            self.assertEqual(metadata[0]["eval_id"], "10")

        with patch.dict(
            os.environ,
            {"STATION_LINEAGE": "alpha", "STATION_ACCESS_PHASE": "mature"},
            clear=False,
        ):
            bank = SeedBank.open(str(store.root))
            self.assertEqual(bank.summary()["candidates"], 5)
            metrics = {item["name"]: item for item in bank.metrics()}
            self.assertEqual(metrics["Preference"]["candidates"], 5)
            self.assertEqual(metrics["Preference"]["min"], 1.0)
            self.assertEqual(metrics["Preference"]["max"], 5.0)
            self.assertEqual(len(bank.from_evaluations(["11"])), 2)
            self.assertEqual(len(bank.sample(2, min_score=0.7)), 2)
            self.assertTrue(all(record.score > 0.8 for record in bank.sample(2, score_gt=0.8)))
            metric_ordered = bank.top(
                limit=5,
                exact_distinct=False,
                order_metric="Preference",
            )
            self.assertEqual(
                [(record.eval_id, record.batch_index) for record in metric_ordered],
                [("11", 0), ("11", 1), ("10", 2), ("10", 0), ("10", 1)],
            )
            metric_filtered = bank.top(
                limit=5,
                exact_distinct=False,
                metric_filters={
                    "Preference": {"gt": 2.0},
                    "Metric": {"gte": 0.7},
                },
            )
            self.assertEqual(
                {(record.eval_id, record.batch_index) for record in metric_filtered},
                {("11", 0), ("11", 1)},
            )
            sampled_metric = bank.sample(
                5,
                exact_distinct=False,
                metric_filters={"Preference": {"gte": 4.0}},
            )
            self.assertEqual(
                {(record.eval_id, record.batch_index) for record in sampled_metric},
                {("11", 0), ("11", 1)},
            )
            ranked_custom = bank.rank(
                lambda seed, _metadata: float(np.sum(seed)), limit=2
            )
            self.assertEqual(len(ranked_custom), 2)
            ranked_metadata = bank.rank_metadata(
                lambda metadata: float(metadata["details"]["Metric"]), limit=2
            )
            self.assertEqual([record.score for record in ranked_metadata], [0.9, 0.8])
            diverse = bank.distinct(
                bank.top(limit=10),
                limit=2,
                distance=lambda left, right: float(np.linalg.norm(left - right)),
                min_distance=0.25,
            )
            self.assertEqual(len(diverse), 2)
            distance_calls = 0

            def counted_distance(left, right):
                nonlocal distance_calls
                distance_calls += 1
                return float(np.linalg.norm(left - right))

            pool = bank.top(limit=10)
            bank.distinct(pool, limit=3, distance=counted_distance, min_distance=0.01)
            self.assertLessEqual(distance_calls, len(pool) * 3)
            self.assertEqual(
                bank.distinct(pool, limit=0, distance=counted_distance, min_distance=0.01),
                [],
            )
            batches = list(bank.iter_batches(bank.top(limit=3), batch_size=2))
            self.assertEqual([len(seeds) for seeds, _metadata in batches], [2, 1])

    def test_successful_attempts_are_preserved_independently(self):
        store, _paths, consts = self._store()
        first = self._batch(
            [np.array([1.0, 0.0]), np.array([0.5, 0.5])],
            [0.7, 0.6],
        )
        second = self._batch(
            [np.array([0.0, 1.0]), np.array([0.2, 0.8])],
            [0.9, 0.8],
        )
        first_manifest = store.save_batch(
            eval_id="12",
            attempt_number=1,
            lineage="alpha",
            author="Agent",
            ranked=validate_and_rank_seed_batch(first, consts),
        )
        second_manifest = store.save_batch(
            eval_id="12",
            attempt_number=2,
            lineage="alpha",
            author="Agent",
            ranked=validate_and_rank_seed_batch(second, consts),
        )
        self.assertEqual(first_manifest["attempt_number"], 1)
        self.assertEqual(second_manifest["attempt_number"], 2)
        self.assertTrue((store.manifests_dir / "eval_12_attempt_1.json").is_file())
        self.assertTrue((store.manifests_dir / "eval_12_attempt_2.json").is_file())
        with sqlite3.connect(store.index_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM seed_candidates WHERE eval_id = '12'"
                ).fetchone()[0],
                4,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(DISTINCT attempt_number) FROM seed_candidates "
                    "WHERE eval_id = '12'"
                ).fetchone()[0],
                2,
            )
        with patch.dict(
            os.environ,
            {"STATION_LINEAGE": "alpha", "STATION_ACCESS_PHASE": "immature"},
            clear=False,
        ):
            bank = SeedBank.open(str(store.root))
            records = bank.from_evaluations(["12"])
            self.assertEqual(len(records), 4)
            self.assertEqual({record.attempt_number for record in records}, {1, 2})
            second_records = bank.from_evaluations(["12"], attempt_numbers=[2])
            self.assertEqual(len(second_records), 2)
            self.assertTrue(all(record.attempt_number == 2 for record in second_records))

    def test_client_uses_full_official_sort_key_for_minimization_and_tuples(self):
        store, _paths, consts = self._store()
        batch = SeedBatchEvaluation(
            seeds=[np.array([index], dtype=float) for index in range(4)],
            scores=np.asarray([9.0, 8.0, 7.0, 6.0]),
            valid=np.asarray([True, True, True, True]),
            sort_keys=[(-2.0,), (-1.0,), (-1.0, 5.0), (-1.0, -5.0)],
            details=[{"Message": f"candidate {index}"} for index in range(4)],
            errors=[None, None, None, None],
        )
        store.save_batch(
            eval_id="13",
            lineage="alpha",
            author="Agent",
            ranked=validate_and_rank_seed_batch(batch, consts),
        )

        with patch.dict(
            os.environ,
            {"STATION_LINEAGE": "alpha", "STATION_ACCESS_PHASE": "immature"},
            clear=False,
        ):
            bank = SeedBank.open(str(store.root))
            self.assertEqual(bank.summary()["best_score"], 7.0)
            records = bank.top(limit=4, exact_distinct=False)
            self.assertEqual([record.batch_index for record in records], [2, 3, 1, 0])
            self.assertEqual(
                [record.batch_index for record in bank.from_evaluations(["13"])],
                [2, 3, 1, 0],
            )

    def test_seed_bank_prompt_surfaces_are_strictly_optional(self):
        disabled = SimpleNamespace(
            RESEARCH_SEED_BANK_ENABLED=False,
            RESEARCH_SEED_BANK_MAX_CANDIDATES=17,
            RESEARCH_TASK_SUFFIX="",
            RESEARCH_TASK_CODER_ONLY_BEGIN_MARKER="__CODER_ONLY_BEGIN__",
            RESEARCH_TASK_CODER_ONLY_END_MARKER="__CODER_ONLY_END__",
        )
        enabled = SimpleNamespace(**{**disabled.__dict__, "RESEARCH_SEED_BANK_ENABLED": True})
        self.assertEqual(ResearchCenter._build_agent_task_markdown("Task", disabled), "Task")
        self.assertNotIn(
            "seed_bank",
            render_coder_access_policy({"coder_access": {"phase": "immature"}}, "alpha", "1", disabled),
        )
        self.assertIn("Submitted-Solution Seed Bank", ResearchCenter._build_agent_task_markdown("Task", enabled))
        agent_task = ResearchCenter._build_agent_task_markdown("Task", enabled)
        self.assertNotIn("visible", agent_task.lower())
        self.assertIn("all saved seeds whose lineage is your current lineage", agent_task)
        self.assertIn("all saved seeds from every lineage in the station", agent_task)
        self.assertEqual(agent_task.count("(Mature only.)"), 1)
        self.assertNotIn("if I am", agent_task)
        self.assertNotIn("Unless you specify another ranking metric", agent_task)
        self.assertIn("up to **17** candidates per attempt", agent_task)
        self.assertIn("maximum, not a target or suggested optimizer population size", agent_task)
        self.assertIn("return `None` for runs that produce no meaningful construction", agent_task)
        self.assertIn('A raw "top K" query will usually return variations from one leading basin', agent_task)
        self.assertIn("nearest-neighbor distances", agent_task)
        self.assertIn(
            "sorted by [official score / secondary metric / agent-defined metric]",
            agent_task,
        )
        self.assertIn("require normalized RMS distance at least `0.02`", agent_task)
        self.assertIn("exploitation, not basin discovery", agent_task)
        self.assertIn("publish the reusable search program", agent_task)
        self.assertIn("the evaluations that led to the discovery", agent_task)
        self.assertIn("task-specific population, ancestry, checkpoints", agent_task)
        self.assertNotIn("top 64", agent_task)
        self.assertNotIn("64 best", agent_task)
        enabled_policy = render_coder_access_policy(
            {"coder_access": {"phase": "immature"}}, "alpha", "1", enabled
        )
        self.assertIn("storage/shared/seed_bank", enabled_policy)
        self.assertIn("Other paths under `storage/shared", enabled_policy)

    def test_full_coder_prompt_is_consistent_when_seed_bank_enabled(self):
        class DummyStation:
            station_id = "seed-prompt-test"

        with self._runtime_patches(True), patch.object(
            constants, "RESEARCH_SEED_BANK_MAX_CANDIDATES", 17
        ):
            paths = ensure_runtime_layout(constants)
            manager = EvaluationManager(paths.evaluations_dir)
            coder = ResearchCoderManager(DummyStation(), manager, paths=paths)
            runtime_env = coder._build_runtime_env()
            self.assertEqual(os.path.abspath(paths.research_root), runtime_env["STATION_RESEARCH_ROOT"])
            immature = coder._build_prompt(
                {
                    "id": "41",
                    "lineage": "alpha",
                    "instruction": "Reuse diverse prior seeds.",
                    "coder": {"max_attempts": 5},
                    "coder_access": {"phase": "immature"},
                }
            )
            mature = coder._build_prompt(
                {
                    "id": "42",
                    "lineage": "alpha",
                    "instruction": "Rerank prior seeds.",
                    "coder": {"max_attempts": 5},
                    "coder_access": {"phase": "mature"},
                }
            )

        for prompt in (immature, mature):
            self.assertEqual(prompt.count("Seed Bank (read-only)"), 1)
            self.assertIn('sys.path.insert(0, "storage/system")', prompt)
            self.assertNotIn("load_population(limit=64)", prompt)
            self.assertIn("up to 17 candidates per official attempt", prompt)
            self.assertIn("17 is a maximum, not a target", prompt)
            self.assertIn("never add tiny perturbations", prompt)
            self.assertIn("custom reranking", prompt)
            self.assertIn("caller-defined structural diversity selection", prompt)
            self.assertIn("`metrics`", prompt)
            self.assertNotIn("visible scope", prompt)
            self.assertIn("structural comparison over a large pool is linear", prompt)
            self.assertIn("Do not write to the Seed Bank or its SQLite/NPZ files directly", prompt)
            self.assertIn("--help TOPIC", prompt)
            self.assertIn("Read only the topics needed for this experiment", prompt)
            self.assertIn("Exact-content deduplication removes only numerically identical", prompt)
            self.assertIn("Do not substitute or invent another structural-distance rule", prompt)
            self.assertIn("loaded seeds may still be variations from the same optimization basin", prompt)
            self.assertIn("nearest-neighbor distance statistics", prompt)
            self.assertNotIn("storage/shared/seed_bank/...` (read/write", prompt)
        self.assertIn("Phase: immature", immature)
        self.assertIn("Other paths under `storage/shared/...`", immature)
        self.assertIn("requested by an immature agent from lineage `alpha`", immature)
        self.assertIn("automatically restricts every query to seeds submitted by lineage `alpha`", immature)
        self.assertIn("first obtain the `SeedRecord` through the helper", immature)
        self.assertIn("read only the NPZ members named by that returned record's `descriptor`", immature)
        self.assertNotIn("requested by a mature agent", immature)
        self.assertIn("Phase: mature", mature)
        self.assertIn("`storage/shared/...` except `storage/shared/seed_bank/...`", mature)
        self.assertIn("requested by a mature agent", mature)
        self.assertIn("automatically queries the entire station-wide Seed Bank", mature)
        self.assertIn("directly read underlying NPZ artifacts when needed", mature)
        self.assertNotIn("requested by an immature agent", mature)

        disabled_root = tempfile.mkdtemp(prefix="station-seed-bank-disabled-prompt-")
        self.addCleanup(shutil.rmtree, disabled_root, True)
        with patch.multiple(
            constants,
            BASE_STATION_DATA_PATH=disabled_root,
            RESEARCH_STORAGE_BASE_PATH=None,
            RESEARCH_SEED_BANK_ENABLED=False,
        ):
            disabled_paths = ensure_runtime_layout(constants)
            disabled_manager = EvaluationManager(disabled_paths.evaluations_dir)
            disabled_coder = ResearchCoderManager(DummyStation(), disabled_manager, paths=disabled_paths)
            disabled_prompt = disabled_coder._build_prompt(
                {
                    "id": "43",
                    "lineage": "alpha",
                    "instruction": "Run normally.",
                    "coder": {"max_attempts": 5},
                    "coder_access": {"phase": "immature"},
                }
            )
        self.assertNotIn("Seed Bank", disabled_prompt)
        self.assertNotIn("seed_bank", disabled_prompt)
        self.assertNotIn("fabricating a fallback seed", disabled_prompt)
        self.assertIn("- `storage/shared/...`", disabled_prompt)

    def test_npz_stores_candidates_as_separate_members(self):
        store, _paths, consts = self._store()
        batch = self._batch(
            [np.arange(8, dtype=np.float64), np.arange(6, dtype=np.float64)],
            [1.0, 2.0],
        )
        manifest = store.save_batch(
            eval_id="20", lineage="alpha", author="Agent", ranked=validate_and_rank_seed_batch(batch, consts)
        )
        artifact = store.root / manifest["artifact_path"]
        with np.load(artifact, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {"c000", "c001"})
        descriptors = [item["descriptor"] for item in manifest["contents"]]
        self.assertEqual({item["member"] for item in descriptors}, {"c000", "c001"})

    def test_rebuild_index_uses_manifests(self):
        store, _paths, consts = self._store()
        batch = self._batch([np.array([1.0, 2.0])], [0.75])
        store.save_batch(
            eval_id="30", lineage="alpha", author="Agent", ranked=validate_and_rank_seed_batch(batch, consts)
        )
        store.index_path.unlink()
        store.rebuild_index()
        with sqlite3.connect(store.index_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM seed_candidates").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM seed_candidate_metrics").fetchone()[0], 1)

    def test_batch_limits_and_no_valid_candidate(self):
        _store, _paths, consts = self._store()
        too_many = self._batch([np.array([1.0])] * 65, list(range(65)))
        with self.assertRaisesRegex(ValueError, "maximum is 64"):
            validate_and_rank_seed_batch(too_many, consts)
        invalid = self._batch([None], [float("-inf")], valid=[False])
        with self.assertRaisesRegex(ValueError, "no valid candidates"):
            validate_and_rank_seed_batch(invalid, consts)

    def test_autocorr_evaluator_single_and_batch(self):
        evaluator_path = (
            Path(__file__).resolve().parents[1]
            / "example"
            / "research_alpha_evolve"
            / "autocorr_6-3"
            / "research"
            / "evaluators"
            / "evaluator.py"
        )
        spec = importlib.util.spec_from_file_location("autocorr_seed_evaluator", evaluator_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        evaluator = module.Task1Evaluator()

        single = evaluator.evaluate_seed_batch([1.0, 1.0, 1.0])
        self.assertEqual(single.scores.shape, (1,))
        self.assertTrue(single.valid[0])
        encoded = evaluator.evaluate_seed_batch("[1.0, 1.0, 1.0]")
        self.assertEqual(encoded.valid.tolist(), [False])

        batch = evaluator.evaluate_seed_batch(
            [np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0]), np.array([-1.0, 1.0])]
        )
        self.assertEqual(batch.scores.shape, (3,))
        self.assertEqual(batch.valid.tolist(), [True, True, False])
        self.assertIsNone(batch.errors[0])
        self.assertIn("nonnegative", batch.errors[2])


if __name__ == "__main__":
    unittest.main()
