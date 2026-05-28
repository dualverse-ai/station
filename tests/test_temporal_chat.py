import copy
import os
import shutil
import tempfile
import threading
import unittest

from station import constants
from station import file_io_utils
from station.base_room import RoomContext
from station.rooms.token_management import TokenManagementRoom
from station.station_runner import Orchestrator


class _FakeAgentModule:
    def __init__(self, agent_data):
        self.agent_data = copy.deepcopy(agent_data)

    def load_agent_data(self, agent_name, include_ended=False, include_ascended=False):
        if agent_name != self.agent_data.get(constants.AGENT_NAME_KEY):
            return None
        return copy.deepcopy(self.agent_data)

    def get_agent_role_definition(self, agent_data):
        return agent_data.get("role_definition", "")


class _FakeStation:
    def __init__(self, agent_data, current_tick):
        self.agent_module = _FakeAgentModule(agent_data)
        self.current_tick = current_tick

    def _get_current_tick(self):
        return self.current_tick


class TemporalChatBranchingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_temporal_chat_branch_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent_data(self):
        return {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_MODEL_PROVIDER_CLASS_KEY: "openai",
            constants.AGENT_MODEL_NAME_KEY: "test-model",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY: [
                {
                    constants.PRUNE_TICKS_KEY: "1-2",
                    constants.PRUNE_SUMMARY_KEY: "older pruned context",
                    constants.PRUNE_PRUNED_AT_TICK_KEY: 2,
                },
                {
                    constants.PRUNE_TICKS_KEY: "3",
                    constants.PRUNE_SUMMARY_KEY: "future prune relative to branch",
                    constants.PRUNE_PRUNED_AT_TICK_KEY: 4,
                },
                {
                    constants.PRUNE_TICKS_KEY: "1-10",
                    constants.PRUNE_SUMMARY_KEY: "legacy block without timestamp",
                },
            ],
        }

    def _orchestrator(self, current_tick=10):
        orch = Orchestrator.__new__(Orchestrator)
        orch.station = _FakeStation(self._agent_data(), current_tick)
        orch._temporal_chat_lock = threading.Lock()
        return orch

    def _write_history(self):
        history_dir = os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Agent A")
        file_io_utils.ensure_dir_exists(history_dir)
        history_path = os.path.join(history_dir, "llm_chat_history.yamll")
        for tick, role, content in [
            (1, "user", "station observation tick 1"),
            (1, "model", "agent response tick 1"),
            (3, "user", "internal prompt tick 3"),
            (3, "model", "internal response tick 3"),
            (4, "user", "station observation tick 4"),
            (4, "model", "agent response tick 4"),
        ]:
            file_io_utils.append_yaml_line(
                {"tick": tick, "role": role, "text_content": content},
                history_path,
            )

    def test_branch_from_selected_tick_filters_history_and_prune_blocks(self):
        self._write_history()
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=3)

        self.assertIsNone(error_msg)
        self.assertEqual(3, chat_state["base_tick"])

        internal_history = file_io_utils.load_yaml_lines(orch._temporal_chat_internal_history_path("Agent A"))
        self.assertEqual([1, 1, 3, 3], [entry["tick"] for entry in internal_history])

        meta = file_io_utils.load_yaml(orch._temporal_chat_internal_meta_path("Agent A"))
        self.assertEqual(
            [
                {
                    constants.PRUNE_TICKS_KEY: "1-2",
                    constants.PRUNE_SUMMARY_KEY: "older pruned context",
                    constants.PRUNE_PRUNED_AT_TICK_KEY: 2,
                }
            ],
            meta["model"]["prune_blocks"],
        )

    def test_future_branch_tick_is_rejected(self):
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=11)

        self.assertIsNone(chat_state)
        self.assertIn("future", error_msg)
        self.assertFalse(os.path.exists(orch._temporal_chat_transcript_path("Agent A")))

    def test_current_branch_keeps_live_prune_blocks_without_legacy_fallback(self):
        orch = self._orchestrator(current_tick=10)

        chat_state, error_msg = orch.refresh_temporal_chat("Agent A", base_tick=10)

        self.assertIsNone(error_msg)
        self.assertEqual(10, chat_state["base_tick"])
        meta = file_io_utils.load_yaml(orch._temporal_chat_internal_meta_path("Agent A"))
        self.assertEqual(
            self._agent_data()[constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY],
            meta["model"]["prune_blocks"],
        )

    def test_legacy_prune_blocks_restore_last_twenty_ticks_from_branch(self):
        orch = self._orchestrator(current_tick=100)
        effective = orch._build_temporal_chat_effective_prune_blocks(
            [
                {
                    constants.PRUNE_TICKS_KEY: "1-50",
                    constants.PRUNE_SUMMARY_KEY: "legacy",
                },
                {
                    constants.PRUNE_TICKS_KEY: "31-50",
                    constants.PRUNE_SUMMARY_KEY: "recent legacy",
                },
                {
                    constants.PRUNE_TICKS_KEY: "51-60",
                    constants.PRUNE_SUMMARY_KEY: "future legacy",
                },
                {
                    constants.PRUNE_TICKS_KEY: "10-12",
                    constants.PRUNE_SUMMARY_KEY: "recorded after branch",
                    constants.PRUNE_PRUNED_AT_TICK_KEY: 60,
                },
            ],
            base_tick=50,
        )

        self.assertEqual(
            [
                {
                    constants.PRUNE_TICKS_KEY: "1-30",
                    constants.PRUNE_SUMMARY_KEY: "legacy",
                }
            ],
            effective,
        )

    def test_new_prune_blocks_record_pruned_at_tick(self):
        room = TokenManagementRoom()
        agent_data = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
        }
        room_context = RoomContext(
            agent_manager=None,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=None,
        )

        actions, handler = room.handle_action(
            agent_data,
            constants.ACTION_PRUNE_RESPONSE,
            None,
            {
                constants.PRUNE_BLOCKS_KEY: [
                    {
                        constants.PRUNE_TICKS_KEY: "1",
                        constants.PRUNE_SUMMARY_KEY: "old context",
                    }
                ]
            },
            room_context,
            current_tick=5,
        )

        self.assertIsNone(handler)
        self.assertIn("Successfully added", actions[0])
        self.assertEqual(
            5,
            agent_data[constants.AGENT_PRUNED_DIALOGUE_TICKS_KEY][0][constants.PRUNE_PRUNED_AT_TICK_KEY],
        )


if __name__ == "__main__":
    unittest.main()
