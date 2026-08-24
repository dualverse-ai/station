import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from station import agent as agent_module
from station import constants
from station.stagnation_protocol import StagnationProtocol


class FakeEvalManager:
    def __init__(self, top_submission=None, breakthrough_summary=None):
        self.top_submission = top_submission or {"submitted_tick": 1, "score": 1.0}
        self.breakthrough_summary = breakthrough_summary or {}

    def get_top_submission(self):
        return self.top_submission

    def get_latest_breakthrough_summary(self):
        return self.breakthrough_summary


class FakeAgentModule:
    get_all_active_agent_names = staticmethod(agent_module.get_all_active_agent_names)


class FakeStation:
    def __init__(
        self,
        current_tick=100,
        top_submission=None,
        station_status="Healthy",
        status_start_tick=0,
        stagnation_counter=0,
        breakthrough_summary=None,
    ):
        self.config = {
            "current_tick": current_tick,
            "station_status": station_status,
            constants.STATION_CONFIG_STAGNATION_COUNTER: stagnation_counter,
            "status_history": [{"status": "Healthy", "start_tick": 0}],
        }
        if station_status != "Healthy":
            self.config["status_history"].append({"status": station_status, "start_tick": status_start_tick})
        self.agent_module = FakeAgentModule()
        self.auto_research_evaluator = SimpleNamespace(
            eval_manager=FakeEvalManager(top_submission, breakthrough_summary)
        )
        self.status_updates = []

    def _get_current_tick(self):
        return self.config["current_tick"]

    def _get_agent_age_status(self, agent_data, current_tick):
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return None
        age = current_tick - birth_tick
        if constants.AGENT_ISOLATION_TICKS is not None and age < constants.AGENT_ISOLATION_TICKS:
            return "immature"
        tenured_threshold = getattr(constants, "MIN_AGENT_AGE_BEFORE_LEAVE", None)
        if tenured_threshold is not None and tenured_threshold > 0 and age >= tenured_threshold:
            return "tenured"
        return "mature"

    def update_station_status(self, new_status, current_tick):
        self.status_updates.append((new_status, current_tick))
        self.config["station_status"] = new_status

class StagnationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_stagnation_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        self.old_repo_root = os.environ.get("STATION_REPO_ROOT")
        self.old_multistart_stagnation_seeds = constants.MULTISTART_STAGNATION_SEEDS
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        constants.MULTISTART_STAGNATION_SEEDS = 0
        os.environ["STATION_REPO_ROOT"] = self.tmpdir
        os.makedirs(os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        constants.MULTISTART_STAGNATION_SEEDS = self.old_multistart_stagnation_seeds
        if self.old_repo_root is None:
            os.environ.pop("STATION_REPO_ROOT", None)
        else:
            os.environ["STATION_REPO_ROOT"] = self.old_repo_root
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save_agent(self, name, birth_tick, role=None):
        data = {
            constants.AGENT_NAME_KEY: name,
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: birth_tick,
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_IS_ASCENDED_KEY: False,
            constants.AGENT_SESSION_ENDED_KEY: False,
        }
        if role is not None:
            data[constants.AGENT_ROLE_KEY] = role
        self.assertTrue(agent_module.save_agent_data(name, data))

    def _notifications(self, name):
        return agent_module.load_agent_data(name)[constants.AGENT_NOTIFICATIONS_PENDING_KEY]

    def test_stagnation_protocol_assigns_lanes_only_to_non_immature_agents(self):
        self._save_agent("Agent A", birth_tick=50)
        self._save_agent("Agent B", birth_tick=50)
        self._save_agent("Agent C", birth_tick=50)
        self._save_agent("Agent D", birth_tick=50)
        self._save_agent("Agent E", birth_tick=50)
        self._save_agent("Immature Agent", birth_tick=99)
        self._save_agent("Supervisor", birth_tick=50, role=constants.ROLE_SUPERVISOR)
        station = FakeStation(current_tick=100)
        protocol = StagnationProtocol(station)

        recipients = protocol._get_stagnation_protocol_recipients()
        recipient_names = {name for name, _data in recipients}
        self.assertEqual(
            {"Agent A", "Agent B", "Agent C", "Agent D", "Agent E", "Supervisor"},
            recipient_names,
        )

        ordered_recipients = [
            ("Agent A", agent_module.load_agent_data("Agent A")),
            ("Agent B", agent_module.load_agent_data("Agent B")),
            ("Agent C", agent_module.load_agent_data("Agent C")),
            ("Agent D", agent_module.load_agent_data("Agent D")),
            ("Agent E", agent_module.load_agent_data("Agent E")),
            ("Supervisor", agent_module.load_agent_data("Supervisor")),
        ]
        with mock.patch("station.stagnation_protocol.random.shuffle", side_effect=lambda items: None):
            sent_count = protocol._send_lane_protocol_messages(ordered_recipients)

        self.assertEqual(6, sent_count)
        self.assertIn("Your assigned lane is: **Exploration**", self._notifications("Agent A")[0])
        self.assertIn("Your assigned lane is: **Exploitation**", self._notifications("Agent B")[0])
        revival_notification = self._notifications("Agent C")[0]
        self.assertIn("Your assigned lane is: **Revival**", revival_notification)
        self.assertIn("relevant Research evaluations, not only Archive papers", revival_notification)
        self.assertIn("important Archive papers and Research evaluations", revival_notification)
        self.assertIn("inspect any high-priority Research evaluations", revival_notification)
        self.assertIn("Your assigned lane is: **Understanding**", self._notifications("Agent D")[0])
        self.assertIn("Your assigned lane is: **Strategy**", self._notifications("Agent E")[0])
        for agent_name in ("Agent A", "Agent B", "Agent C", "Agent D", "Agent E"):
            self.assertIn("within 5 ticks", self._notifications(agent_name)[0])
        self.assertEqual([], self._notifications("Immature Agent"))
        self.assertIn("supervisor-facing station-wide announcement", self._notifications("Supervisor")[0])

    def test_lane_batches_are_independently_shuffled(self):
        station = FakeStation(current_tick=100)
        protocol = StagnationProtocol(station)
        shuffled_batches = iter([
            ["a", "c", "b"],
            ["b", "c", "a"],
        ])

        def shuffle_batch(items):
            items[:] = next(shuffled_batches)

        with mock.patch("station.stagnation_protocol.random.shuffle", side_effect=shuffle_batch):
            sequence = protocol._build_random_lane_sequence(["a", "b", "c"], 5)

        self.assertEqual(["a", "c", "b", "b", "c"], sequence)

    def test_external_counter_suffix_only_applies_to_tenured_lane_agents_when_enabled(self):
        self._save_agent("Tenured Agent", birth_tick=0)
        self._save_agent("Mature Agent", birth_tick=350)
        self._save_agent("Supervisor", birth_tick=0, role=constants.ROLE_SUPERVISOR)
        station = FakeStation(current_tick=400)
        protocol = StagnationProtocol(station)
        recipients = [
            ("Tenured Agent", agent_module.load_agent_data("Tenured Agent")),
            ("Mature Agent", agent_module.load_agent_data("Mature Agent")),
            ("Supervisor", agent_module.load_agent_data("Supervisor")),
        ]

        with (
            mock.patch.object(constants, "EXTERNAL_COUNTER_ENABLED", True),
            mock.patch("station.stagnation_protocol.random.shuffle", side_effect=lambda items: None),
        ):
            protocol._send_lane_protocol_messages(recipients)

        heading = "## External Counter Usage"
        self.assertIn(heading, self._notifications("Tenured Agent")[0])
        self.assertNotIn(heading, self._notifications("Mature Agent")[0])
        self.assertNotIn(heading, self._notifications("Supervisor")[0])

    def test_external_counter_suffix_is_absent_when_disabled(self):
        self._save_agent("Tenured Agent", birth_tick=0)
        station = FakeStation(current_tick=400)
        protocol = StagnationProtocol(station)
        recipients = [("Tenured Agent", agent_module.load_agent_data("Tenured Agent"))]

        with (
            mock.patch.object(constants, "EXTERNAL_COUNTER_ENABLED", False),
            mock.patch("station.stagnation_protocol.random.shuffle", side_effect=lambda items: None),
        ):
            protocol._send_lane_protocol_messages(recipients)

        self.assertNotIn("## External Counter Usage", self._notifications("Tenured Agent")[0])

    def test_stagnation_protocol_delays_until_four_non_immature_non_supervisors(self):
        self._save_agent("Agent A", birth_tick=50)
        self._save_agent("Agent B", birth_tick=50)
        self._save_agent("Agent C", birth_tick=50)
        self._save_agent("Supervisor", birth_tick=50, role=constants.ROLE_SUPERVISOR)
        station = FakeStation(current_tick=321, top_submission={"submitted_tick": 1, "score": 1.0})
        protocol = StagnationProtocol(station)

        with mock.patch.object(protocol, "_send_lane_protocol_messages") as send_mock:
            protocol.check_and_update_stagnation()

        send_mock.assert_not_called()
        self.assertEqual([], station.status_updates)
        self.assertEqual("Healthy", station.config["station_status"])
        self.assertEqual([], self._notifications("Agent A"))

    def test_stagnation_protocol_broadcasts_when_four_non_immature_non_supervisors_exist(self):
        self._save_agent("Agent A", birth_tick=50)
        self._save_agent("Agent B", birth_tick=50)
        self._save_agent("Agent C", birth_tick=50)
        self._save_agent("Agent D", birth_tick=50)
        self._save_agent("Immature Agent", birth_tick=299)
        self._save_agent("Supervisor", birth_tick=50, role=constants.ROLE_SUPERVISOR)
        station = FakeStation(current_tick=321, top_submission={"submitted_tick": 1, "score": 1.0})
        protocol = StagnationProtocol(station)

        with mock.patch("station.stagnation_protocol.random.shuffle", side_effect=lambda items: None):
            protocol.check_and_update_stagnation()

        self.assertEqual([("Stagnation I", 321)], station.status_updates)
        self.assertEqual("Stagnation I", station.config["station_status"])
        self.assertEqual(1, len(self._notifications("Agent A")))

    def test_stagnation_multistart_defers_lane_assignment_and_writes_request(self):
        self._save_agent("Agent A", birth_tick=50)
        self._save_agent("Agent B", birth_tick=50)
        self._save_agent("Agent C", birth_tick=50)
        self._save_agent("Agent D", birth_tick=50)
        station = FakeStation(current_tick=321, top_submission={"submitted_tick": 1, "score": 1.0})
        station._save_config = mock.Mock()
        station.orchestrator = SimpleNamespace(
            is_paused=False,
            pause_condition_met=False,
            pause_reason_message="",
        )
        protocol = StagnationProtocol(station)

        with mock.patch.object(constants, "MULTISTART_STAGNATION_SEEDS", 4):
            protocol.check_and_update_stagnation()

        request_path = os.path.join(self.tmpdir, "station_multistart", "pending_stagnation.yaml")
        self.assertTrue(os.path.isfile(request_path))
        self.assertTrue(station.orchestrator.is_paused)
        self.assertEqual([], station.status_updates)
        self.assertEqual([], self._notifications("Agent A"))

    def test_healthy_reset_does_not_erase_breakthrough_age_after_maturity_delay(self):
        self._save_agent("Agent A", birth_tick=3721)
        self._save_agent("Agent B", birth_tick=3721)
        self._save_agent("Agent C", birth_tick=3721)
        self._save_agent("Agent D", birth_tick=3721)
        station = FakeStation(
            current_tick=3779,
            station_status="Healthy",
            stagnation_counter=57,
            top_submission={"submitted_tick": 866, "score": 82.75862068965517},
        )
        station.config["status_history"] = [
            {"status": "Healthy", "start_tick": 0},
            {"status": "Stagnation IX", "start_tick": 3616},
            {"status": "Healthy", "start_tick": 3721},
        ]
        protocol = StagnationProtocol(station)

        with mock.patch("station.stagnation_protocol.random.shuffle", side_effect=lambda items: None):
            protocol.check_and_update_stagnation()

        self.assertEqual(2913, station.config[constants.STATION_CONFIG_STAGNATION_COUNTER])
        self.assertEqual([("Stagnation IX", 3779)], station.status_updates)
        self.assertEqual("Stagnation IX", station.config["station_status"])
        self.assertEqual(1, len(self._notifications("Agent A")))

    def test_sort_key_improvement_counts_as_breakthrough(self):
        station = FakeStation(
            current_tick=100,
            top_submission={"submitted_tick": 10, "score": 0.0, "sort_key": [1.0, 594, -0.0]},
        )
        protocol = StagnationProtocol(station)

        breakthrough_tick, improved_now, current_top, previous_score = protocol._detect_last_breakthrough_tick(100)
        self.assertEqual(10, breakthrough_tick)
        self.assertFalse(improved_now)
        self.assertEqual(10, current_top["submitted_tick"])
        self.assertIsNone(previous_score)

        station.auto_research_evaluator.eval_manager.top_submission = {
            "submitted_tick": 20,
            "score": 0.0,
            "sort_key": [1.0, 595, -0.0],
        }

        breakthrough_tick, improved_now, current_top, previous_score = protocol._detect_last_breakthrough_tick(101)
        self.assertEqual(101, breakthrough_tick)
        self.assertTrue(improved_now)
        self.assertEqual(20, current_top["submitted_tick"])
        self.assertEqual(0.0, previous_score)

    def test_progress_track_improvement_counts_as_breakthrough_without_global_top_change(self):
        station = FakeStation(
            current_tick=100,
            top_submission={"submitted_tick": 10, "score": 1.0, "sort_key": [1.0]},
            breakthrough_summary={
                "frontiers": {
                    "dimension:d3": {"rank_key": [-0.9], "submitted_tick": 10, "evaluation_id": "10"},
                    "dimension:d4": {"rank_key": [-0.8], "submitted_tick": 12, "evaluation_id": "12"},
                },
                "last_breakthrough_tick": 12,
            },
        )
        protocol = StagnationProtocol(station)

        breakthrough_tick, improved_now, _current_top, _previous_score = protocol._detect_last_breakthrough_tick(100)
        self.assertEqual(12, breakthrough_tick)
        self.assertFalse(improved_now)

        station.auto_research_evaluator.eval_manager.breakthrough_summary = {
            "frontiers": {
                "dimension:d3": {"rank_key": [-0.9], "submitted_tick": 10, "evaluation_id": "10"},
                "dimension:d4": {"rank_key": [-0.7], "submitted_tick": 20, "evaluation_id": "20"},
            },
            "last_breakthrough_tick": 20,
        }

        breakthrough_tick, improved_now, current_top, previous_score = protocol._detect_last_breakthrough_tick(101)
        self.assertEqual(101, breakthrough_tick)
        self.assertTrue(improved_now)
        self.assertEqual(10, current_top["submitted_tick"])
        self.assertEqual(1.0, previous_score)

    def test_healthy_reset_announcement_names_breakthrough_details(self):
        station = FakeStation(
            current_tick=101,
            station_status="Stagnation I",
            status_start_tick=90,
            stagnation_counter=250,
            top_submission={
                "evaluation_id": "602",
                "agent_name": "Agent A",
                "title": "Better algebraic construction",
                "submitted_tick": 100,
                "score": 12.5,
            },
        )
        protocol = StagnationProtocol(station)
        protocol.last_top_score = 10.0
        protocol.last_top_rank_key = (10.0,)

        with mock.patch.object(protocol, "_send_system_message_to_all_recursive", return_value=0) as send_mock:
            protocol.check_and_update_stagnation()

        send_mock.assert_called_once()
        message = send_mock.call_args.args[0]
        self.assertIn("Responsible agent: Agent A", message)
        self.assertIn("Evaluation: Eval #602", message)
        self.assertIn("Title: Better algebraic construction", message)
        self.assertIn("Score jump: from 10.00000000 to 12.50000000", message)
        self.assertEqual([("Healthy", 101)], station.status_updates)

    def test_breakthrough_after_stagnation_start_resets_after_restart(self):
        station = FakeStation(
            current_tick=628,
            station_status="Stagnation I",
            status_start_tick=563,
            stagnation_counter=250,
            top_submission={
                "evaluation_id": "602",
                "submitted_tick": 623,
                "score": 0.42165976085946655,
                "sort_key": [-0.42165976085946655],
            },
            breakthrough_summary={
                "frontiers": {
                    "global": {
                        "rank_key": [-0.42165976085946655],
                        "submitted_tick": 623,
                        "evaluation_id": "602",
                    },
                },
                "last_breakthrough_tick": 623,
            },
        )
        protocol = StagnationProtocol(station)

        with mock.patch.object(protocol, "_send_system_message_to_all_recursive", return_value=0):
            protocol.check_and_update_stagnation()

        self.assertEqual([("Healthy", 628)], station.status_updates)
        self.assertEqual("Healthy", station.config["station_status"])
        self.assertEqual(0, station.config[constants.STATION_CONFIG_STAGNATION_COUNTER])

    def test_sub_epsilon_top_does_not_end_stagnation_after_restart(self):
        station = FakeStation(
            current_tick=628,
            station_status="Stagnation I",
            status_start_tick=563,
            stagnation_counter=250,
            top_submission={
                "evaluation_id": "602",
                "submitted_tick": 623,
                "score": 1.005,
                "sort_key": [1.005],
            },
            breakthrough_summary={
                "frontiers": {
                    "global": {
                        "rank_key": [1.0],
                        "submitted_tick": 500,
                        "evaluation_id": "500",
                    },
                },
                "last_breakthrough_tick": 500,
            },
        )
        protocol = StagnationProtocol(station)

        with mock.patch.object(protocol, "_send_system_message_to_all_recursive", return_value=0):
            protocol.check_and_update_stagnation()

        self.assertEqual([], station.status_updates)
        self.assertEqual("Stagnation I", station.config["station_status"])
        self.assertEqual(385, station.config[constants.STATION_CONFIG_STAGNATION_COUNTER])

    def test_progress_breakthrough_after_stagnation_start_resets_after_restart(self):
        station = FakeStation(
            current_tick=628,
            station_status="Stagnation I",
            status_start_tick=563,
            stagnation_counter=250,
            top_submission={
                "evaluation_id": "10",
                "submitted_tick": 100,
                "score": 1.0,
                "sort_key": [-1.0],
            },
            breakthrough_summary={
                "frontiers": {
                    "dimension:d4": {"rank_key": [-0.7], "submitted_tick": 623, "evaluation_id": "623"},
                },
                "last_breakthrough_tick": 623,
            },
        )
        protocol = StagnationProtocol(station)

        with mock.patch.object(protocol, "_send_system_message_to_all_recursive", return_value=0):
            protocol.check_and_update_stagnation()

        self.assertEqual([("Healthy", 628)], station.status_updates)
        self.assertEqual("Healthy", station.config["station_status"])
        self.assertEqual(0, station.config[constants.STATION_CONFIG_STAGNATION_COUNTER])


if __name__ == "__main__":
    unittest.main()
