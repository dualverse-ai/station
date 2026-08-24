import csv
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import breakthroughs
from station.eval_research import evaluation_index


def _write_eval(evaluations_dir: Path, eval_id: int, **overrides):
    data = {
        "schema_version": 2,
        "id": str(eval_id),
        "author": "Logos I",
        "title": f"Eval {eval_id}",
        "tags": ["baseline"],
        "abstract": "",
        "submitted_tick": eval_id,
        "status": "completed",
        "final": {
            "primary_score": 1.0,
            "sort_key": None,
        },
    }
    data.update(overrides)
    path = evaluations_dir / f"{eval_id}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_agent(agents_dir: Path, agent_name: str, model_name: str):
    agents_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "agent_name": agent_name,
        "model_name": model_name,
    }
    path = agents_dir / f"{agent_name}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class BreakthroughScriptTests(unittest.TestCase):
    def test_exact_top_submission_is_independent_from_breakthrough_eps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)

            _write_eval(evaluations_dir, 1, final={"primary_score": 1.0, "sort_key": None})
            _write_eval(evaluations_dir, 2, final={"primary_score": 1.005, "sort_key": None})

            manager = breakthroughs.EvaluationManager(str(evaluations_dir))
            self.assertEqual("2", manager.get_top_submission()["evaluation_id"])

            top_rows = breakthroughs.find_top_submissions(
                breakthroughs.collect_scored_evaluations(str(station_data))
            )
            self.assertEqual(["1", "2"], [item.eval_id for item in top_rows])
            self.assertEqual([True, False], [item.is_breakthrough for item in top_rows])

            found = breakthroughs.collect_breakthrough_events(str(station_data))
            global_ids = [item.eval_id for item in found if item.track == "global"]
            self.assertEqual(["1"], global_ids)

    def test_restart_refreshes_legacy_top_cache_from_sql_without_yaml_rebuild(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)
            _write_eval(evaluations_dir, 1, final={"primary_score": 1.0, "sort_key": None})
            _write_eval(evaluations_dir, 2, final={"primary_score": 1.005, "sort_key": None})

            manager = breakthroughs.EvaluationManager(str(evaluations_dir))
            self.assertEqual("2", manager.get_top_submission()["evaluation_id"])
            scope = str(evaluations_dir.resolve())
            db_path = evaluation_index.get_database_path(str(evaluations_dir))
            legacy_top = dict(manager.get_top_submission())
            legacy_top.update({"evaluation_id": "1", "score": 1.0, "sort_key": [1.0]})
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE research_evaluation_scopes SET top_submission_json = ? WHERE evaluations_dir = ?",
                    (json.dumps(legacy_top), scope),
                )
            evaluation_index._PROCESS_TOP_REFRESHED_SCOPES.discard(scope)

            with patch.object(
                evaluation_index,
                "rebuild_research_evaluation_index",
                side_effect=AssertionError("top refresh must not rebuild from YAML"),
            ):
                restarted = breakthroughs.EvaluationManager(str(evaluations_dir))

            self.assertEqual("2", restarted.get_top_submission()["evaluation_id"])

    def test_collects_current_numeric_yaml_records_and_applies_eps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)

            _write_eval(evaluations_dir, 1, final={"primary_score": 1.0, "sort_key": None})
            _write_eval(evaluations_dir, 2, final={"primary_score": 1.0 + 5e-9, "sort_key": None})
            _write_eval(evaluations_dir, 3, final={"primary_score": 1.1, "sort_key": None})
            _write_eval(
                evaluations_dir,
                4,
                status="running",
                final={},
            )

            scored = breakthroughs.collect_scored_evaluations(str(station_data))
            found = breakthroughs.find_breakthroughs(scored)

            self.assertEqual([item.eval_id for item in scored], ["1", "2", "3"])
            self.assertEqual([item.eval_id for item in found], ["1", "3"])

    def test_sort_key_controls_breakthrough_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)

            _write_eval(evaluations_dir, 1, final={"primary_score": 10.0, "sort_key": [10, 0]})
            _write_eval(evaluations_dir, 2, final={"primary_score": 9.0, "sort_key": [10, 1]})
            _write_eval(evaluations_dir, 3, final={"primary_score": 11.0, "sort_key": [9, 99]})

            scored = breakthroughs.collect_scored_evaluations(str(station_data))
            found = breakthroughs.find_breakthroughs(scored)

            self.assertEqual([item.eval_id for item in found], ["1", "2"])
            self.assertEqual(found[-1].score, 9.0)

    def test_collects_progress_record_breakthroughs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)

            _write_eval(
                evaluations_dir,
                1,
                final={
                    "primary_score": 1.0,
                    "sort_key": [1.0],
                    "progress_records": [
                        {"track": "dimension:d4", "rank_key": [-0.9], "value": 0.9}
                    ],
                },
            )
            _write_eval(
                evaluations_dir,
                2,
                final={
                    "primary_score": 0.5,
                    "sort_key": [0.5],
                    "progress_records": [
                        {"track": "dimension:d4", "rank_key": [-0.8], "value": 0.8}
                    ],
                },
            )

            found = breakthroughs.collect_breakthrough_events(str(station_data))

            self.assertIn(("1", "global"), [(item.eval_id, item.track) for item in found])
            self.assertIn(("1", "dimension:d4"), [(item.eval_id, item.track) for item in found])
            self.assertIn(("2", "dimension:d4"), [(item.eval_id, item.track) for item in found])

    def test_tag_filter_accepts_comma_separated_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)

            _write_eval(evaluations_dir, 1, tags="book, accumulator")
            _write_eval(evaluations_dir, 2, tags=["other"], final={"primary_score": 2.0, "sort_key": None})

            scored = breakthroughs.collect_scored_evaluations(str(station_data), tag_filter="accumulator")

            self.assertEqual([item.eval_id for item in scored], ["1"])

    def test_analysis_prints_and_exports_agent_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            agents_dir = station_data / "agents"
            evaluations_dir.mkdir(parents=True)
            _write_agent(agents_dir, "Logos I", "gpt-test-model")
            _write_eval(evaluations_dir, 1, final={"primary_score": 1.0, "sort_key": None})

            csv_path = station_data / "breakthroughs.csv"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                breakthroughs.analyze_research_breakthroughs(str(station_data), csv_path=str(csv_path))

            output = stdout.getvalue()
            self.assertIn("Agent Model", output)
            self.assertIn("gpt-test-model", output)

            with csv_path.open(newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                self.assertNotIn("Previous", reader.fieldnames or [])
                self.assertIn("Breakthrough", reader.fieldnames or [])
            self.assertEqual("gpt-test-model", rows[0]["Agent Model"])

    def test_analysis_exports_sub_epsilon_top_as_not_breakthrough(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)
            _write_eval(evaluations_dir, 1, final={"primary_score": 1.0, "sort_key": None})
            _write_eval(evaluations_dir, 2, final={"primary_score": 1.005, "sort_key": None})

            csv_path = station_data / "breakthroughs.csv"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                breakthroughs.analyze_research_breakthroughs(str(station_data), csv_path=str(csv_path))

            with csv_path.open(newline="", encoding="utf-8") as csvfile:
                rows = {row["Eval ID"]: row for row in csv.DictReader(csvfile)}
            self.assertEqual("yes", rows["1"]["Breakthrough"])
            self.assertEqual("no", rows["2"]["Breakthrough"])
            self.assertIn("Research Top Submission and Breakthrough Analysis", stdout.getvalue())

    def test_collection_does_not_load_agent_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            agents_dir = station_data / "agents"
            evaluations_dir.mkdir(parents=True)
            _write_agent(agents_dir, "Logos I", "gpt-test-model")
            _write_eval(evaluations_dir, 1, final={"primary_score": 1.0, "sort_key": None})
            _write_eval(evaluations_dir, 2, final={"primary_score": 2.0, "sort_key": None})

            load_calls = []
            original_load_yaml = breakthroughs.load_yaml

            def counting_load_yaml(path):
                load_calls.append(path)
                return original_load_yaml(path)

            with patch.object(breakthroughs, "load_yaml", side_effect=counting_load_yaml):
                scored = breakthroughs.collect_scored_evaluations(str(station_data))

            self.assertEqual([item.eval_id for item in scored], ["1", "2"])
            self.assertEqual(load_calls, [])

    def test_analysis_loads_agent_metadata_only_for_breakthrough_authors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            agents_dir = station_data / "agents"
            evaluations_dir.mkdir(parents=True)
            _write_agent(agents_dir, "Logos I", "gpt-test-model")
            _write_agent(agents_dir, "NoSota I", "unused-model")
            _write_eval(evaluations_dir, 1, author="Logos I", final={"primary_score": 1.0, "sort_key": None})
            _write_eval(evaluations_dir, 2, author="NoSota I", final={"primary_score": 0.5, "sort_key": None})

            load_calls = []
            original_load_yaml = breakthroughs.load_yaml

            def counting_load_yaml(path):
                load_calls.append(path)
                return original_load_yaml(path)

            stdout = io.StringIO()
            with patch.object(breakthroughs, "load_yaml", side_effect=counting_load_yaml):
                with redirect_stdout(stdout):
                    breakthroughs.analyze_research_breakthroughs(str(station_data), csv_path=str(station_data / "out.csv"))

            self.assertEqual(len(load_calls), 1)
            self.assertIn("Logos I.yaml", load_calls[0])


if __name__ == "__main__":
    unittest.main()
