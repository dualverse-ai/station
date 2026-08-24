import os
import shutil
import tempfile
import unittest

from station import constants
from station import file_io_utils
from station.eval_archive import evaluation_index


class ArchiveEvaluationIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_archive_eval_index_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_log(self, filename, agent_name, result, score):
        evaluations_dir = evaluation_index.get_archive_evaluations_dir()
        file_io_utils.ensure_dir_exists(evaluations_dir)
        path = os.path.join(evaluations_dir, filename)
        file_io_utils.save_yaml(
            {
                "evaluation_id": filename,
                "agent_name": agent_name,
                "result": result,
                "extracted_result": {"score": score},
            },
            path,
        )
        return path

    def test_counts_high_quality_archive_evaluation_logs_by_lineage(self):
        self._write_log("evaluation_high.yaml", "Test I", "accepted", 8.5)
        self._write_log("evaluation_low.yaml", "Test II", "accepted", 7.9)
        self._write_log("evaluation_rejected.yaml", "Other I", "rejected", 9.0)

        evaluation_index.rebuild_archive_evaluation_index()
        counts = evaluation_index.count_high_quality_papers_by_lineage(score_threshold=8.0)

        self.assertEqual({"Test": 1}, counts)

    def test_upsert_archive_evaluation_log(self):
        path = self._write_log("evaluation_one.yaml", "Test I", "accepted", 7.0)
        evaluation_index.rebuild_archive_evaluation_index()
        self.assertEqual({}, evaluation_index.count_high_quality_papers_by_lineage(score_threshold=8.0))

        log_data = file_io_utils.load_yaml(path)
        log_data["extracted_result"]["score"] = 8.0
        file_io_utils.save_yaml(log_data, path)
        evaluation_index.upsert_archive_evaluation(log_data, path)

        self.assertEqual({"Test": 1}, evaluation_index.count_high_quality_papers_by_lineage(score_threshold=8.0))


if __name__ == "__main__":
    unittest.main()
