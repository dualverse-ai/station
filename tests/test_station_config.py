import unittest
from unittest import mock

from station import constants, station_config
from station.station import Station


class StationConfigTests(unittest.TestCase):
    def test_default_config_includes_top_sort_key(self):
        config = station_config.build_default_station_config(git_commit="test")

        self.assertIn(constants.STATION_CONFIG_TOP_SORT_KEY, config)
        self.assertIsNone(config[constants.STATION_CONFIG_TOP_SORT_KEY])

    def test_sync_top_submission_persists_sort_key(self):
        station = Station.__new__(Station)
        station.config = {}
        station._save_config = mock.Mock()
        submission = {
            "evaluation_id": "7",
            "title": "Candidate",
            "score": 3.0,
            "sort_key": [1.0, 5.0, -2.0],
            "submitted_tick": 11,
            "agent_name": "Agent I",
            "tags": ["exact-top"],
            "abstract": "A compact cached abstract.",
        }

        station.sync_top_research_submission_config(submission)

        self.assertEqual([1.0, 5.0, -2.0], station.config[constants.STATION_CONFIG_TOP_SORT_KEY])
        cached = station_config.top_submission_from_config(station.config)
        self.assertEqual(["exact-top"], cached["tags"])
        self.assertEqual("A compact cached abstract.", cached["abstract"])
        submission["sort_key"][0] = 99.0
        submission["tags"].append("mutated")
        self.assertEqual([1.0, 5.0, -2.0], station.config[constants.STATION_CONFIG_TOP_SORT_KEY])
        self.assertEqual(["exact-top"], station.config[constants.STATION_CONFIG_TOP_TAGS])
        station._save_config.assert_called_once_with()

    def test_sync_top_submission_defaults_sort_key_to_score(self):
        station = Station.__new__(Station)
        station.config = {}
        station._save_config = mock.Mock()

        station.sync_top_research_submission_config({"score": 4.5})

        self.assertEqual([4.5], station.config[constants.STATION_CONFIG_TOP_SORT_KEY])


if __name__ == "__main__":
    unittest.main()
