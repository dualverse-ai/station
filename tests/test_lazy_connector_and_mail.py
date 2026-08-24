import os
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from station import agent as agent_module
from station import constants
from station.base_room import RoomContext
from station.rooms.mail import MailRoom
from station.station_runner import Orchestrator


class LazyConnectorAndMailTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_lazy_connector_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        os.makedirs(os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save_recursive_agent(self, name: str) -> None:
        self.assertTrue(
            agent_module.save_agent_data(
                name,
                {
                    constants.AGENT_NAME_KEY: name,
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_SESSION_ENDED_KEY: False,
                    constants.AGENT_IS_ASCENDED_KEY: False,
                    constants.AGENT_MODEL_PROVIDER_CLASS_KEY: "OpenAI",
                    constants.AGENT_MODEL_NAME_KEY: "test-model",
                },
            )
        )

    def _orchestrator_for_config(self, turn_order):
        station = SimpleNamespace(
            agent_module=agent_module,
            config={constants.STATION_CONFIG_AGENT_TURN_ORDER: list(turn_order)},
            _save_config=lambda: None,
            get_next_agent_index_from_config=lambda: 0,
            get_agent_departure_reason=lambda _name: "unknown",
        )
        orchestrator = object.__new__(Orchestrator)
        orchestrator.station = station
        orchestrator.agent_turn_order = []
        orchestrator.current_tick_processed_agents = set()
        orchestrator.agent_llm_connectors = {}
        orchestrator.current_agent_index_in_turn_order = 0
        orchestrator.is_prepared = False
        orchestrator.is_paused = False
        orchestrator.pause_condition_met = False
        orchestrator.pause_reason_message = ""
        orchestrator.events = []
        orchestrator._api_runtime_connector_lock = threading.RLock()
        orchestrator._push_log_event = lambda event_type, payload: orchestrator.events.append((event_type, payload))
        return orchestrator

    def test_prepare_validation_does_not_construct_connectors(self):
        self._save_recursive_agent("Agent A")
        orchestrator = self._orchestrator_for_config(["Agent A"])

        with mock.patch("station.station_runner.create_llm_connector", side_effect=AssertionError("eager connector init")):
            self.assertTrue(orchestrator._load_agent_turn_order())
            self.assertTrue(orchestrator.initialize_connectors_for_active_agents())

        self.assertEqual(["Agent A"], orchestrator.agent_turn_order)
        self.assertEqual({}, orchestrator.agent_llm_connectors)

    def test_connector_is_created_on_first_use(self):
        self._save_recursive_agent("Agent A")
        orchestrator = self._orchestrator_for_config(["Agent A"])
        fake_connector = SimpleNamespace(api_runtime_config_generation=0)

        with mock.patch("station.station_runner.runtime_api_config.get_generation", return_value=0), mock.patch(
            "station.station_runner.create_llm_connector",
            return_value=fake_connector,
        ) as create_connector:
            connector = orchestrator._get_current_connector_for_agent("Agent A")

        self.assertIs(connector, fake_connector)
        self.assertIs(orchestrator.agent_llm_connectors["Agent A"], fake_connector)
        self.assertEqual(1, create_connector.call_count)

    def test_mail_header_uses_station_turn_order(self):
        class AgentManager:
            def __init__(self):
                self.load_calls = []

            def get_active_recursive_agent_names(self):
                raise AssertionError("mail header should not scan all active recursive agents")

            def load_agent_data(self, name):
                self.load_calls.append(name)
                return {
                    constants.AGENT_NAME_KEY: name,
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    "mature": name == "Mature",
                }

        manager = AgentManager()
        room_context = RoomContext(
            agent_manager=manager,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=SimpleNamespace(
                config={constants.STATION_CONFIG_AGENT_TURN_ORDER: ["Mature", "Immature"]},
                _is_agent_mature=lambda data, _tick: data["mature"],
            ),
        )

        header = MailRoom()._get_room_specific_header_elements({}, room_context, current_tick=5)

        self.assertEqual(["Mature", "Immature"], manager.load_calls)
        self.assertIn("Mature", "\n".join(header))
        self.assertNotIn("Immature", "\n".join(header))

    def test_mail_header_raises_for_stale_turn_order_agent(self):
        class AgentManager:
            def get_active_recursive_agent_names(self):
                raise AssertionError("mail header should not scan all active recursive agents")

            def load_agent_data(self, _name):
                return None

        room_context = RoomContext(
            agent_manager=AgentManager(),
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=SimpleNamespace(
                config={constants.STATION_CONFIG_AGENT_TURN_ORDER: ["Missing"]},
                _is_agent_mature=lambda _data, _tick: True,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
            MailRoom()._get_room_specific_header_elements({}, room_context, current_tick=5)

    def test_mail_header_excludes_guest_in_turn_order(self):
        class AgentManager:
            def get_active_recursive_agent_names(self):
                raise AssertionError("mail header should not scan all active recursive agents")

            def load_agent_data(self, name):
                status = (
                    constants.AGENT_STATUS_GUEST
                    if name == "Guest_1"
                    else constants.AGENT_STATUS_RECURSIVE
                )
                return {
                    constants.AGENT_NAME_KEY: name,
                    constants.AGENT_STATUS_KEY: status,
                    "mature": True,
                }

        room_context = RoomContext(
            agent_manager=AgentManager(),
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=SimpleNamespace(
                config={constants.STATION_CONFIG_AGENT_TURN_ORDER: ["Recursive A", "Guest_1"]},
                _is_agent_mature=lambda data, _tick: data["mature"],
            ),
        )

        header = MailRoom()._get_room_specific_header_elements({}, room_context, current_tick=5)
        rendered = "\n".join(header)

        self.assertIn("Recursive A", rendered)
        self.assertNotIn("Guest_1", rendered)


if __name__ == "__main__":
    unittest.main()
