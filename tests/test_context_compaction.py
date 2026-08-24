import os
import shutil
import tempfile
import unittest

from station import agent as agent_module
from station import constants
from station import context_compaction
from station import file_io_utils
from station.base_room import RoomContext
from station.rooms.lobby import LobbyRoom
from station.station import (
    _merge_context_compaction_events_after_turn,
    _merge_protected_context_items_after_turn,
    _save_request_status_snapshot_atomically,
)
from station.station_runner import Orchestrator


class ContextCompactionTests(unittest.TestCase):
    def test_compaction_prompt_prioritizes_durable_station_knowledge(self):
        prompt = constants.CONTEXT_COMPACTION_PROMPT_TEMPLATE
        self.assertIn("Do not be overly brief", prompt)
        self.assertIn("more than 2000 words", prompt)
        self.assertIn("Before writing, review the previous session summary", prompt)

    def test_context_compaction_uses_persisted_connector_history(self):
        class DummyConnector:
            def __init__(self):
                self.persist_to_disk = True
                self.persist_values_seen = []
                self.reloaded = False

            def send_message(self, prompt, current_tick):
                self.persist_values_seen.append(self.persist_to_disk)
                return "complete summary", None, {}

            def reload_session_from_disk(self):
                self.reloaded = True

        class DummyAgentModule:
            def load_agent_data(self, agent_name):
                return {
                    constants.AGENT_NAME_KEY: agent_name,
                    constants.AGENT_TOKEN_BUDGET_CURRENT_KEY: 100,
                    constants.AGENT_TOKEN_BUDGET_MAX_KEY: 100,
                    constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY: [],
                }

        class DummyStation:
            def __init__(self):
                self.agent_module = DummyAgentModule()
                self.logged_entries = []
                self.saved_summary = None

            def should_compact_agent_context(self, agent_data):
                return True

            def _log_dialogue_entry(self, agent_name, entry):
                self.logged_entries.append((agent_name, entry))

            def save_context_compaction_summary(self, agent_name, tick, summary):
                self.saved_summary = (agent_name, tick, summary)
                return True

        connector = DummyConnector()
        station = DummyStation()
        orch = Orchestrator.__new__(Orchestrator)
        orch.station = station
        orch.agent_llm_connectors = {"Agent A": connector}
        orch._get_current_connector_for_agent = lambda agent_name: orch.agent_llm_connectors.get(agent_name)
        orch._push_log_event = lambda *args, **kwargs: None
        orch._trigger_pause_due_to_llm_error = lambda *args, **kwargs: None

        self.assertTrue(orch._run_context_compaction_maintenance("Agent A", 12))
        self.assertEqual([True], connector.persist_values_seen)
        self.assertTrue(connector.persist_to_disk)
        self.assertTrue(connector.reloaded)
        self.assertEqual(("Agent A", 12, "complete summary"), station.saved_summary)

    def test_new_agent_data_includes_context_schema(self):
        tmpdir = tempfile.mkdtemp(prefix="station_context_agent_test_", dir="/tmp")
        old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = tmpdir
        try:
            os.makedirs(os.path.join(tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)
            agent_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=0,
            )
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIsNotNone(agent_data)
        self.assertEqual([], agent_data[constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY])
        self.assertEqual([], agent_data[constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY])

    def test_birth_lobby_help_adds_protected_context_item(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Guest_1",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }

        class DummyManager:
            add_protected_context_item = staticmethod(agent_module.add_protected_context_item)
            get_agent_room_state = staticmethod(agent_module.get_agent_room_state)
            set_agent_room_state = staticmethod(agent_module.set_agent_room_state)

        room_context = RoomContext(
            agent_manager=DummyManager(),
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=None,
        )

        output = LobbyRoom().get_room_output(agent_data, room_context, current_tick=0)

        self.assertIn("Help Message - Lobby", output)
        items = agent_module.get_protected_context_items(agent_data)
        self.assertEqual(1, len(items))
        self.assertEqual(constants.PROTECTED_CONTEXT_KIND_ROOM_HELP, items[0][constants.PROTECTED_CONTEXT_KIND_KEY])
        self.assertEqual(0, items[0][constants.PROTECTED_CONTEXT_TICK_KEY])
        self.assertIn("Help Message - Lobby", items[0][constants.PROTECTED_CONTEXT_CONTENT_KEY])

    def test_ascension_carries_protected_context_and_compaction_events(self):
        tmpdir = tempfile.mkdtemp(prefix="station_context_ascend_test_", dir="/tmp")
        old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = tmpdir
        try:
            os.makedirs(os.path.join(tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)
            guest_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=0,
            )
            self.assertIsNotNone(guest_data)
            guest_name = guest_data[constants.AGENT_NAME_KEY]
            agent_module.add_protected_context_item(
                guest_data,
                tick=4,
                kind=constants.PROTECTED_CONTEXT_KIND_ROOM_HELP,
                source="room:lobby",
                title="Help Message - Lobby",
                content="Lobby help",
            )
            agent_module.add_context_compaction_event(
                guest_data,
                compacted_after_tick=5,
                summary="summary",
            )
            agent_module.mark_context_compaction_anchored(
                guest_data,
                compacted_after_tick=5,
                anchor_tick=6,
            )
            agent_module.save_agent_data(guest_name, guest_data)

            ascended = agent_module.ascend_agent(
                guest_agent_name=guest_name,
                new_recursive_name="Aletheia I",
                new_lineage="Aletheia",
                new_generation=1,
                current_tick=7,
                new_description="A recursive test agent.",
            )
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIsNotNone(ascended)
        self.assertEqual(guest_data[constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY], ascended[constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY])
        self.assertEqual(guest_data[constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY], ascended[constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY])

    def test_summary_response_uses_entire_response(self):
        summary = context_compaction.normalize_summary_response(
            "  Eval #1 worked.\n\nNext plan: continue.  "
        )
        self.assertEqual("Eval #1 worked.\n\nNext plan: continue.", summary)

    def test_compaction_anchor_intro_renders_protected_items(self):
        intro = context_compaction.build_compaction_anchor_intro(
            agent_name="Agent A",
            protected_items=[
                {
                    constants.PROTECTED_CONTEXT_TICK_KEY: 3,
                    constants.PROTECTED_CONTEXT_KIND_KEY: constants.PROTECTED_CONTEXT_KIND_RESEARCH_TASK,
                    constants.PROTECTED_CONTEXT_TITLE_KEY: "Research Task",
                    constants.PROTECTED_CONTEXT_CONTENT_KEY: "Task body",
                }
            ],
            summary="Session summary",
        )
        self.assertIn("Agent A", intro)
        self.assertIn("Station Tick 3", intro)
        self.assertIn("Task body", intro)
        self.assertIn("Session summary", intro)

    def test_context_record_merges_preserve_concurrent_updates(self):
        start_items = [{"tick": 1, "kind": "room_help", "content": "old"}]
        latest_items = start_items + [{"tick": 2, "kind": "room_help", "content": "concurrent"}]
        turn_items = start_items + [{"tick": 3, "kind": "research_task", "content": "turn"}]

        self.assertEqual(
            latest_items + [{"tick": 3, "kind": "research_task", "content": "turn"}],
            _merge_protected_context_items_after_turn(turn_items, latest_items, start_items),
        )

    def test_compaction_event_merge_replaces_pending_with_anchor(self):
        start = [
            {
                constants.CONTEXT_COMPACTION_COMPACTED_AFTER_TICK_KEY: 4,
                constants.CONTEXT_COMPACTION_SUMMARY_KEY: "summary",
                constants.CONTEXT_COMPACTION_STATUS_KEY: constants.CONTEXT_COMPACTION_STATUS_PENDING_ANCHOR,
            }
        ]
        latest = list(start)
        turn = [
            {
                constants.CONTEXT_COMPACTION_COMPACTED_AFTER_TICK_KEY: 4,
                constants.CONTEXT_COMPACTION_SUMMARY_KEY: "summary",
                constants.CONTEXT_COMPACTION_STATUS_KEY: constants.CONTEXT_COMPACTION_STATUS_ANCHORED,
                constants.CONTEXT_COMPACTION_ANCHOR_TICK_KEY: 5,
            }
        ]

        merged = _merge_context_compaction_events_after_turn(turn, latest, start)

        self.assertEqual(5, merged[0][constants.CONTEXT_COMPACTION_ANCHOR_TICK_KEY])

    def test_request_status_snapshot_merges_context_fields(self):
        class Manager:
            def __init__(self):
                self.data = {
                    constants.AGENT_NAME_KEY: "Agent A",
                    constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
                    constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY: [
                        {
                            constants.PROTECTED_CONTEXT_TICK_KEY: 1,
                            constants.PROTECTED_CONTEXT_KIND_KEY: constants.PROTECTED_CONTEXT_KIND_ROOM_HELP,
                            constants.PROTECTED_CONTEXT_CONTENT_KEY: "concurrent",
                        }
                    ],
                    constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY: [],
                }

            def update_agent_with_function(self, agent_name, update_func):
                update_func(self.data)
                return True

        manager = Manager()
        start = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY: [],
            constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY: [],
        }
        snapshot = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY: [
                {
                    constants.PROTECTED_CONTEXT_TICK_KEY: 2,
                    constants.PROTECTED_CONTEXT_KIND_KEY: constants.PROTECTED_CONTEXT_KIND_RESEARCH_TASK,
                    constants.PROTECTED_CONTEXT_CONTENT_KEY: "turn",
                }
            ],
            constants.AGENT_CONTEXT_COMPACTION_EVENTS_KEY: [],
        }

        rendered = _save_request_status_snapshot_atomically(
            manager,
            "Agent A",
            start,
            snapshot,
        )

        self.assertIsNotNone(rendered)
        self.assertEqual(2, len(rendered[constants.AGENT_PROTECTED_CONTEXT_ITEMS_KEY]))

if __name__ == "__main__":
    unittest.main()
