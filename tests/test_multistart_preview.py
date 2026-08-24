import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import yaml

from station import constants
from station.multistart import paths, state
from web_interface import multistart_preview


class MultistartPreviewTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="station_multistart_preview_test_", dir="/tmp"))
        self.root = paths.multistart_root(self.repo)
        self.root.mkdir()
        self.job_dir = self.root / "10_job"
        self.seed_one = self.job_dir / "station_data_s1"
        self.seed_two = self.job_dir / "station_data_s2"
        self.seed_one.mkdir(parents=True)
        self.seed_two.mkdir(parents=True)
        self._write_config(self.seed_one, tick=14, name="Seed One")
        self._write_config(self.seed_two, tick=15, name="Seed Two")
        state.save_current_job(
            self.repo,
            {
                "job_id": "job",
                "mode": "stagnation",
                "status": "running",
                "job_dir": str(self.job_dir),
                "branch_tick": 10,
                "roll_ticks": 20,
                "seed_count": 2,
            },
        )
        state.save_job_state(
            self.job_dir,
            {
                "job_id": "job",
                "mode": "stagnation",
                "status": "running",
                "branch_tick": 10,
                "roll_ticks": 20,
                "seed_count": 2,
                "branches": [
                    {
                        "seed": 1,
                        "status": "running",
                        "data_root": str(self.seed_one),
                        "current_tick": 14,
                        "target_tick": 30,
                    },
                    {
                        "seed": 2,
                        "status": "completed",
                        "data_root": str(self.seed_two),
                        "current_tick": 30,
                        "target_tick": 30,
                    },
                ],
            },
        )

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    @staticmethod
    def _write_config(root: Path, *, tick: int, name: str):
        (root / "station_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "station_id": f"station-{name.lower().replace(' ', '-')}",
                    "station_name": name,
                    "station_description": "Preview branch",
                    "station_status": "Running",
                    "current_tick": tick,
                    "agent_turn_order": ["Ada I"],
                }
            ),
            encoding="utf-8",
        )

    def test_preview_resolves_seed_one_in_place_without_creating_live_station_data(self):
        context = multistart_preview.get_preview_context(self.repo)

        self.assertIsNotNone(context)
        self.assertEqual(self.seed_one.resolve(), context.data_root)
        self.assertFalse((self.repo / "station_data").exists())

        payload = multistart_preview.dashboard_context(self.repo)
        status = multistart_preview.orchestrator_status(self.repo)
        self.assertEqual(1, payload["seed"])
        self.assertEqual("Seed One", payload["branch"]["station_name"])
        self.assertEqual(1, payload["completed_count"])
        self.assertTrue(status["read_only"])
        self.assertEqual("running", status["branch_status"])
        self.assertEqual(14, status["multistart"]["branch"]["current_tick"])
        self.assertEqual(30, status["target_tick"])

    def test_preview_rejects_branch_root_outside_active_job(self):
        outside = self.repo / "outside"
        outside.mkdir()
        self._write_config(outside, tick=99, name="Outside")
        detail = state.load_job_state(self.job_dir)
        detail["branches"][0]["data_root"] = str(outside)
        state.save_job_state(self.job_dir, detail)

        self.assertIsNone(multistart_preview.get_preview_context(self.repo))

    def test_preview_request_policy_locks_station_mutations_but_keeps_job_controls(self):
        self.assertTrue(multistart_preview.request_allowed("GET", "/api/agents"))
        self.assertTrue(multistart_preview.request_allowed("POST", "/api/multistart/pause"))
        self.assertTrue(multistart_preview.request_allowed("POST", "/api/multistart/resume"))
        self.assertFalse(multistart_preview.request_allowed("POST", "/api/orchestrator/start_loop"))
        self.assertFalse(multistart_preview.request_allowed("PUT", "/api/station/config"))

    def test_preview_reads_lightweight_agents_statistics_task_and_capsule_index(self):
        config = state.read_station_config(self.seed_one)
        config.update(
            {
                constants.STATION_CONFIG_STAGNATION_COUNTER: 186,
                constants.STATION_CONFIG_TOP_EVALUATION_ID: "8",
                constants.STATION_CONFIG_TOP_TITLE: "Config-cached top",
                constants.STATION_CONFIG_TOP_SCORE: 9.0,
                constants.STATION_CONFIG_TOP_SORT_KEY: [9.0],
                constants.STATION_CONFIG_TOP_TICK: 13,
                constants.STATION_CONFIG_TOP_AGENT_NAME: "Ada I",
            }
        )
        state.save_yaml_mapping(self.seed_one / constants.STATION_CONFIG_FILENAME, config)
        agents_dir = self.seed_one / constants.AGENTS_DIR_NAME
        agents_dir.mkdir()
        (agents_dir / "Ada I.yaml").write_text(
            yaml.safe_dump(
                {
                    "status": "Recursive Agent",
                    "model_name": "test-model",
                    "model_provider_class": "TestProvider",
                    "awaiting_human_intervention": False,
                }
            ),
            encoding="utf-8",
        )

        research_root = self.seed_one / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH
        evaluations_dir = research_root / constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
        evaluations_dir.mkdir(parents=True)
        (research_root / constants.RESEARCH_TASK_SPEC_FILENAME).write_text("# Preview Task\n", encoding="utf-8")
        self._write_index(evaluations_dir)

        agents = multistart_preview.agents(self.repo)
        stats = multistart_preview.statistics(self.repo)
        task = multistart_preview.task_spec_snapshot(self.repo)
        capsules = multistart_preview.capsule_view(self.repo)

        self.assertEqual(["Ada I"], [item["name"] for item in agents])
        self.assertTrue(stats["read_only"])
        self.assertEqual("8", stats["top_research_submission"]["evaluation_id"])
        self.assertEqual(186, stats["ticks_since_last_breakthrough"])
        self.assertEqual(1, stats["running_jobs_count"])
        self.assertEqual("# Preview Task\n", task["raw_markdown"])
        archive_rows = capsules.list_capsules(constants.CAPSULE_TYPE_ARCHIVE)
        self.assertEqual(["archive_3"], [row[constants.CAPSULE_ID_KEY] for row in archive_rows])
        self.assertFalse((self.repo / "station_data").exists())

    def test_handoff_statistics_are_empty_when_preview_context_is_gone(self):
        state.clear_current_job(self.repo)

        self.assertIsNone(multistart_preview.statistics(self.repo))
        stats = multistart_preview.handoff_statistics()

        self.assertTrue(stats["handoff_pending"])
        self.assertEqual(0, stats["running_jobs_count"])
        self.assertEqual(0, stats["queued_jobs_count"])
        self.assertFalse(stats["pending_research_evaluations"])
        self.assertFalse(stats["pending_archive_evaluations"])

    def _write_index(self, evaluations_dir: Path):
        index_dir = self.seed_one / constants.STATION_INDEX_DIR_NAME
        index_dir.mkdir()
        db_path = index_dir / constants.STATION_INDEX_DB_FILENAME
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE research_evaluations (
                    evaluations_dir TEXT,
                    eval_id TEXT,
                    eval_id_num INTEGER,
                    author TEXT,
                    title TEXT,
                    submitted_tick INTEGER,
                    start_timestamp REAL,
                    display_status TEXT,
                    top_level_status TEXT,
                    latest_attempt_status TEXT,
                    coder_active INTEGER,
                    execution_source TEXT,
                    system_baseline INTEGER,
                    is_active INTEGER
                );
                CREATE TABLE capsule_metadata (
                    capsule_type TEXT,
                    lineage_key TEXT,
                    numeric_id INTEGER,
                    capsule_id TEXT,
                    author_name TEXT,
                    author_lineage TEXT,
                    author_generation INTEGER,
                    created_at_tick INTEGER,
                    last_updated_at_tick INTEGER,
                    title TEXT,
                    abstract TEXT,
                    word_count_total INTEGER,
                    total_message_count INTEGER,
                    is_deleted INTEGER,
                    reviewer_score REAL,
                    question_status TEXT,
                    question_net_upvote INTEGER,
                    question_solved_by_message_id TEXT
                );
                """
            )
            conn.execute(
                "INSERT INTO research_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(evaluations_dir.resolve()),
                    "8",
                    8,
                    "Ada I",
                    "Running",
                    14,
                    1.0,
                    "coder_running",
                    "running",
                    "",
                    1,
                    "coder",
                    0,
                    1,
                ),
            )
            conn.execute(
                "INSERT INTO capsule_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    constants.CAPSULE_TYPE_ARCHIVE,
                    "",
                    3,
                    "archive_3",
                    "Ada I",
                    "Ada",
                    1,
                    11,
                    11,
                    "Paper",
                    "Abstract",
                    100,
                    1,
                    0,
                    8.0,
                    None,
                    None,
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
