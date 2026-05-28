import tempfile
import unittest
from pathlib import Path

import yaml

from scripts import breakthroughs


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


class BreakthroughScriptTests(unittest.TestCase):
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

    def test_tag_filter_accepts_comma_separated_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            station_data = Path(temp_dir)
            evaluations_dir = station_data / "rooms" / "research" / "evaluations"
            evaluations_dir.mkdir(parents=True)

            _write_eval(evaluations_dir, 1, tags="book, accumulator")
            _write_eval(evaluations_dir, 2, tags=["other"], final={"primary_score": 2.0, "sort_key": None})

            scored = breakthroughs.collect_scored_evaluations(str(station_data), tag_filter="accumulator")

            self.assertEqual([item.eval_id for item in scored], ["1"])


if __name__ == "__main__":
    unittest.main()
