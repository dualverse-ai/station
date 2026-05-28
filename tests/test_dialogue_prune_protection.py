import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from station import agent as agent_module
from station import constants
from station import file_io_utils
from station.base_room import RoomContext
from station.rooms.lobby import LobbyRoom
from station.rooms.reflect import MetaReflectionHandler
from station.rooms.research_center import ResearchCenter
from station.rooms.token_management import TokenManagementRoom
from station.station import (
    _drop_stale_pending_dialogue_tick_protections,
    _merge_pending_dialogue_tick_protections_after_turn,
    _merge_protected_dialogue_ticks_after_turn,
    _save_request_status_snapshot_atomically,
)


class DialoguePruneProtectionTests(unittest.TestCase):
    def _load_migration_module(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate" / "migrate_protected_dialogue_ticks.py"
        spec = importlib.util.spec_from_file_location("migrate_protected_dialogue_ticks", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_cleanup_module(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "clean_duplicate_research_task_read_protections.py"
        spec = importlib.util.spec_from_file_location("clean_duplicate_research_task_read_protections", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_new_agent_data_includes_protection_schema_marker(self):
        tmpdir = tempfile.mkdtemp(prefix="station_protection_agent_test_", dir="/tmp")
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
        self.assertEqual([], agent_data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY])
        self.assertEqual([], agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])

    def test_birth_lobby_help_marks_current_tick_protected(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Guest_1",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST,
            constants.AGENT_TICK_BIRTH_KEY: 0,
        }

        class DummyManager:
            protect_dialogue_tick = staticmethod(agent_module.protect_dialogue_tick)
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
        self.assertEqual({0}, agent_module.get_protected_dialogue_ticks(agent_data))
        self.assertTrue(
            agent_data[constants.SHORT_ROOM_NAME_LOBBY][
                constants.AGENT_ROOM_STATE_FIRST_VISIT_HELP_SHOWN_KEY
            ]
        )

    def test_ascension_carries_protected_dialogue_ticks(self):
        tmpdir = tempfile.mkdtemp(prefix="station_protection_ascend_test_", dir="/tmp")
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
            agent_module.protect_dialogue_tick(
                guest_data,
                4,
                constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
                source="room:lobby",
            )
            agent_module.queue_dialogue_tick_protection(
                guest_data,
                constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                source="architect_message_notification",
            )
            agent_module.save_agent_data(guest_name, guest_data)

            ascended = agent_module.ascend_agent(
                guest_agent_name=guest_name,
                new_recursive_name="Aletheia I",
                new_lineage="Aletheia",
                new_generation=1,
                current_tick=3,
                new_description="A recursive test agent.",
            )
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIsNotNone(ascended)
        self.assertEqual(
            [
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 4,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:lobby",
                }
            ],
            ascended[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY],
        )
        self.assertEqual(
            [
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                }
            ],
            ascended[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY],
        )

    def test_protected_ticks_are_explicit_only_after_migration(self):
        agent_data = {}
        agent_module.protect_dialogue_tick(
            agent_data,
            7,
            constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
            source="meta_reflect",
        )
        raw_entries = [
            {
                "tick": 3,
                "speaker": "Station",
                "content": "**Help Message - Lobby**\nWelcome.",
            },
            {
                "tick": 5,
                "role": "user",
                "text_content": (
                    "**Research Task**\n"
                    "This specification holds the highest degree of credibility "
                    "in this research station and overrides all other sources."
                ),
            },
            {
                "tick": 9,
                "speaker": "Station",
                "content": "**Architect Message**\nStagnation protocol.",
            },
        ]

        self.assertEqual(
            {7},
            agent_module.get_protected_dialogue_ticks(agent_data, raw_entries),
        )

    def test_meta_reflection_protection_limit_keeps_latest_ticks(self):
        old_limit = constants.REFLECTION_META_PROTECTED_TICK_LIMIT
        constants.REFLECTION_META_PROTECTED_TICK_LIMIT = 4
        try:
            legacy_agent_data = {
                constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [
                    {
                        constants.PROTECTED_DIALOGUE_TICK_KEY: tick,
                        constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
                        constants.PROTECTED_DIALOGUE_SOURCE_KEY: "meta_reflect",
                    }
                    for tick in range(1, 6)
                ]
            }

            self.assertEqual(
                {2, 3, 4, 5},
                agent_module.get_protected_dialogue_ticks(legacy_agent_data),
            )

            current_agent_data = {}
            for tick in range(1, 6):
                agent_module.protect_dialogue_tick(
                    current_agent_data,
                    tick,
                    constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
                    source="meta_reflect",
                )

            records_by_key = {
                (
                    record[constants.PROTECTED_DIALOGUE_TICK_KEY],
                    record[constants.PROTECTED_DIALOGUE_REASON_KEY],
                )
                for record in current_agent_data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY]
            }
            self.assertNotIn(
                (1, constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION),
                records_by_key,
            )
            self.assertEqual(
                {2, 3, 4, 5},
                agent_module.get_protected_dialogue_ticks(current_agent_data),
            )

            agent_module.protect_dialogue_tick(
                current_agent_data,
                1,
                constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
                source="room:lobby",
            )
            records_by_key = {
                (
                    record[constants.PROTECTED_DIALOGUE_TICK_KEY],
                    record[constants.PROTECTED_DIALOGUE_REASON_KEY],
                )
                for record in current_agent_data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY]
            }
            self.assertIn(
                (1, constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP),
                records_by_key,
            )
            self.assertEqual(
                {1, 2, 3, 4, 5},
                agent_module.get_protected_dialogue_ticks(current_agent_data),
            )
        finally:
            constants.REFLECTION_META_PROTECTED_TICK_LIMIT = old_limit

    def test_token_management_can_prune_meta_tick_after_limit_rolls_forward(self):
        old_limit = constants.REFLECTION_META_PROTECTED_TICK_LIMIT
        constants.REFLECTION_META_PROTECTED_TICK_LIMIT = 4
        try:
            raw_dialogue = []
            for tick in range(1, 6):
                raw_dialogue.extend(
                    [
                        {
                            "tick": tick,
                            "speaker": "Station",
                            "type": "observation",
                            "content": f"Observation {tick}",
                        },
                        {
                            "tick": tick,
                            "speaker": "Agent",
                            "type": "submission",
                            "content": f"Response {tick}",
                        },
                    ]
                )

            def build_agent_data():
                data = {
                    constants.AGENT_NAME_KEY: "Agent A",
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_LAST_PRUNE_ACTION_TICK_KEY: -constants.TOKEN_MANAGEMENT_COOLDOWN_TICKS,
                }
                for tick in range(1, 6):
                    agent_module.protect_dialogue_tick(
                        data,
                        tick,
                        constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
                        source="meta_reflect",
                    )
                return data

            room = TokenManagementRoom()
            room._load_agent_dialogue_history = lambda _agent_name, _room_context: raw_dialogue
            room_context = RoomContext(
                agent_manager=None,
                capsule_manager=None,
                notification_manager=None,
                constants_module=constants,
                station_instance=None,
            )

            actions, _handler = room.handle_action(
                build_agent_data(),
                constants.ACTION_PRUNE_RESPONSE,
                None,
                {
                    constants.PRUNE_BLOCKS_KEY: [
                        {
                            constants.PRUNE_TICKS_KEY: "1",
                            constants.PRUNE_SUMMARY_KEY: "Old meta reflection context.",
                        }
                    ]
                },
                room_context,
                current_tick=10,
            )
            self.assertTrue(actions[0].startswith("Successfully added 1 prune block"))

            actions, _handler = room.handle_action(
                build_agent_data(),
                constants.ACTION_PRUNE_RESPONSE,
                None,
                {
                    constants.PRUNE_BLOCKS_KEY: [
                        {
                            constants.PRUNE_TICKS_KEY: "2",
                            constants.PRUNE_SUMMARY_KEY: "Still protected meta reflection context.",
                        }
                    ]
                },
                room_context,
                current_tick=10,
            )
            self.assertIn("Protected ticks cannot be pruned: [2]", actions[0])
        finally:
            constants.REFLECTION_META_PROTECTED_TICK_LIMIT = old_limit

    def test_migration_converts_legacy_dialogue_keywords_to_yaml_records(self):
        module = self._load_migration_module()
        tmpdir = tempfile.mkdtemp(prefix="station_protection_migration_test_", dir="/tmp")
        try:
            station_data_path = Path(tmpdir)
            agents_dir = station_data_path / constants.AGENTS_DIR_NAME
            dialogue_dir = station_data_path / constants.DIALOGUE_LOGS_DIR_NAME
            file_io_utils.ensure_dir_exists(str(agents_dir))
            file_io_utils.ensure_dir_exists(str(dialogue_dir))

            agent_data = {
                constants.AGENT_NAME_KEY: "Old Agent",
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            }
            file_io_utils.save_yaml(agent_data, str(agents_dir / "Old Agent.yaml"))

            log_path = dialogue_dir / f"Old_Agent{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"
            file_io_utils.append_yaml_line(
                {
                    "tick": 1,
                    "speaker": "Station",
                    "type": "observation",
                    "content": "**Help Message - Lobby**\nWelcome.",
                },
                str(log_path),
            )
            file_io_utils.append_yaml_line(
                {
                    "tick": 2,
                    "speaker": "Station",
                    "type": "observation",
                    "content": "**Help Message - Lobby**\nRepeated help.",
                },
                str(log_path),
            )
            file_io_utils.append_yaml_line(
                {
                    "tick": 3,
                    "role": "user",
                    "text_content": (
                        "**Research Task**\n"
                        "This specification holds the highest degree of credibility "
                        "in this research station and overrides all other sources."
                    ),
                },
                str(log_path),
            )
            file_io_utils.append_yaml_line(
                {
                    "tick": 4,
                    "speaker": "Station",
                    "type": "observation",
                    "content": "**Architect Message**\nStagnation protocol.",
                },
                str(log_path),
            )
            file_io_utils.append_yaml_line(
                {
                    "tick": 5,
                    "speaker": "Station",
                    "type": "observation",
                    "content": "Help for Research Center:\n**Welcome to the Research Center**",
                },
                str(log_path),
            )
            file_io_utils.append_yaml_line(
                {
                    "tick": 6,
                    "speaker": "Agent",
                    "type": "submission",
                    "content": "**Architect Message** should not count from agent-side text.",
                },
                str(log_path),
            )

            self.assertTrue(module.needs_migration(station_data_path))
            summary = module.migrate_station_data(station_data_path)
            migrated = file_io_utils.load_yaml(str(agents_dir / "Old Agent.yaml"))
            needs_migration_after = module.needs_migration(station_data_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(1, summary["agents_updated"])
        self.assertEqual(4, summary["records_added"])
        self.assertEqual([], migrated[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])
        records_by_key = {
            (
                record[constants.PROTECTED_DIALOGUE_TICK_KEY],
                record[constants.PROTECTED_DIALOGUE_REASON_KEY],
                record.get(constants.PROTECTED_DIALOGUE_SOURCE_KEY),
            )
            for record in migrated[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY]
        }
        self.assertEqual(
            {
                (1, constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP, "room:lobby"),
                (3, constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ, "legacy_research_task_read"),
                (4, constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE, "legacy_architect_message"),
                (5, constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP, "room:research"),
            },
            records_by_key,
        )
        self.assertFalse(needs_migration_after)

    def test_migration_queues_old_pending_architect_notification_protection(self):
        module = self._load_migration_module()
        tmpdir = tempfile.mkdtemp(prefix="station_protection_pending_migration_test_", dir="/tmp")
        try:
            station_data_path = Path(tmpdir)
            agents_dir = station_data_path / constants.AGENTS_DIR_NAME
            file_io_utils.ensure_dir_exists(str(agents_dir))
            file_io_utils.save_yaml(
                {
                    constants.AGENT_NAME_KEY: "Old Agent",
                    constants.AGENT_NOTIFICATIONS_PENDING_KEY: [
                        "**Architect Message**\nPending station-wide instruction."
                    ],
                },
                str(agents_dir / "Old Agent.yaml"),
            )

            summary = module.migrate_station_data(station_data_path)
            migrated = file_io_utils.load_yaml(str(agents_dir / "Old Agent.yaml"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(1, summary["pending_records_added"])
        self.assertEqual(
            [
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "legacy_pending_architect_message",
                }
            ],
            migrated[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY],
        )

    def test_cleanup_keeps_first_research_task_read_and_real_architect_messages(self):
        module = self._load_cleanup_module()
        agent_data = {
            constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY: False,
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
                },
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                },
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:archive",
                },
            ],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 4,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
                },
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 5,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
                },
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 6,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                },
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 7,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                },
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 8,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:lobby",
                },
            ]
        }

        changed, removed_research, removed_architect = module.clean_agent_data(
            agent_data,
            real_architect_ticks={6},
        )
        changed_pending, removed_pending = module.clean_pending_protection_records(agent_data)

        self.assertTrue(changed)
        self.assertEqual(1, removed_research)
        self.assertEqual(1, removed_architect)
        self.assertTrue(changed_pending)
        self.assertEqual(3, removed_pending)
        self.assertEqual([], agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])
        self.assertEqual(
            [
                (
                    record[constants.PROTECTED_DIALOGUE_TICK_KEY],
                    record[constants.PROTECTED_DIALOGUE_REASON_KEY],
                )
                for record in agent_data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY]
            ],
            [
                (4, constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ),
                (6, constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE),
                (8, constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP),
            ],
        )

    def test_cleanup_skips_inactive_agents_by_default(self):
        module = self._load_cleanup_module()
        tmpdir = tempfile.mkdtemp(prefix="station_protection_cleanup_skip_test_", dir="/tmp")
        try:
            station_data_path = Path(tmpdir)
            agents_dir = station_data_path / constants.AGENTS_DIR_NAME
            file_io_utils.ensure_dir_exists(str(agents_dir))
            file_io_utils.save_yaml(
                {
                    constants.AGENT_NAME_KEY: "Ended Agent",
                    constants.AGENT_SESSION_ENDED_KEY: True,
                    constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [
                        {
                            constants.PROTECTED_DIALOGUE_TICK_KEY: 1,
                            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
                        },
                        {
                            constants.PROTECTED_DIALOGUE_TICK_KEY: 2,
                            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
                        },
                    ],
                },
                str(agents_dir / "Ended Agent.yaml"),
            )

            summary = module.scan_station_data(station_data_path, dry_run=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(1, summary["agents_seen"])
        self.assertEqual(1, summary["agents_skipped_inactive"])
        self.assertEqual(0, summary["agents_updated"])
        self.assertEqual(0, summary["research_task_read_records_removed"])

    def test_pending_protection_applies_to_rendered_tick(self):
        agent_data = {}
        agent_module.add_pending_notification(
            agent_data,
            "**Architect Message**\nSupervisor assignment.",
        )

        self.assertEqual(
            [
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                }
            ],
            agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY],
        )

        self.assertTrue(agent_module.apply_pending_dialogue_tick_protections(agent_data, 11))
        self.assertEqual([], agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])
        self.assertEqual({11}, agent_module.get_protected_dialogue_ticks(agent_data))

    def test_request_snapshot_clears_consumed_pending_protections_without_losing_concurrent_changes(self):
        stale_research = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
        }
        stale_architect = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }
        concurrent_help = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:archive",
        }
        existing_protected = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 1,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:lobby",
        }
        concurrent_protected = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 9,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:mail",
        }
        rendered_research = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 10,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
        }
        rendered_architect = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 10,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }

        turn_start = {
            constants.AGENT_DESCRIPTION_KEY: "before",
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: ["old notification"],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [stale_research, stale_architect],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [existing_protected],
        }
        latest = {
            constants.AGENT_DESCRIPTION_KEY: "before",
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: ["old notification", "concurrent notification"],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [
                stale_research,
                stale_architect,
                concurrent_help,
            ],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [
                existing_protected,
                concurrent_protected,
            ],
        }
        snapshot = {
            constants.AGENT_DESCRIPTION_KEY: "after",
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: ["old notification", "new snapshot notification"],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [
                existing_protected,
                rendered_research,
                rendered_architect,
            ],
        }

        class DummyManager:
            def __init__(self):
                self.data = latest

            def update_agent_with_function(self, _agent_name, update_func):
                update_func(self.data)
                return True

        manager = DummyManager()
        rendered = _save_request_status_snapshot_atomically(
            manager,
            "Agent A",
            turn_start,
            snapshot,
            refresh_shown_notifications=True,
        )

        self.assertEqual([concurrent_help], manager.data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])
        self.assertEqual(
            [
                existing_protected,
                concurrent_protected,
                rendered_research,
                rendered_architect,
            ],
            manager.data[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY],
        )
        self.assertEqual("after", manager.data[constants.AGENT_DESCRIPTION_KEY])
        self.assertEqual(
            ["old notification", "concurrent notification", "new snapshot notification"],
            manager.data[constants.AGENT_NOTIFICATIONS_PENDING_KEY],
        )
        self.assertEqual(
            manager.data[constants.AGENT_NOTIFICATIONS_PENDING_KEY],
            manager.data[constants.AGENT_SHOWN_NOTIFICATIONS_KEY],
        )
        self.assertEqual(manager.data, rendered)

    def test_request_snapshot_applies_concurrent_pending_protection_before_marking_shown(self):
        concurrent_notification = "**Architect Message**\nSupervisor assignment."
        concurrent_pending = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }
        turn_start = {
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [],
        }
        latest = {
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [concurrent_notification],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [concurrent_pending],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [],
        }
        snapshot = {
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [],
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [],
        }

        class DummyManager:
            apply_pending_dialogue_tick_protections = staticmethod(
                agent_module.apply_pending_dialogue_tick_protections
            )

            def __init__(self):
                self.data = latest

            def update_agent_with_function(self, _agent_name, update_func):
                update_func(self.data)
                return True

        manager = DummyManager()
        rendered = _save_request_status_snapshot_atomically(
            manager,
            "Agent A",
            turn_start,
            snapshot,
            refresh_shown_notifications=True,
            apply_pending_protections_tick=12,
        )

        self.assertEqual([concurrent_notification], rendered[constants.AGENT_NOTIFICATIONS_PENDING_KEY])
        self.assertEqual([concurrent_notification], rendered[constants.AGENT_SHOWN_NOTIFICATIONS_KEY])
        self.assertEqual([], rendered[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])
        self.assertEqual(
            [
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 12,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                }
            ],
            rendered[constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY],
        )

    def test_turn_merge_preserves_concurrent_pending_and_protected_records(self):
        stale_pending = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
        }
        turn_pending = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }
        concurrent_pending = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:mail",
        }
        old_protected = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 1,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:lobby",
        }
        turn_protected = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 2,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "meta_reflect",
        }
        concurrent_protected = {
            constants.PROTECTED_DIALOGUE_TICK_KEY: 3,
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }

        self.assertEqual(
            [turn_pending, concurrent_pending],
            _merge_pending_dialogue_tick_protections_after_turn(
                [turn_pending],
                [stale_pending, concurrent_pending],
                [stale_pending],
            ),
        )
        self.assertEqual(
            [old_protected, concurrent_protected, turn_protected],
            _merge_protected_dialogue_ticks_after_turn(
                [old_protected, turn_protected],
                [old_protected, concurrent_protected],
                [old_protected],
            ),
        )

    def test_stale_pending_protections_are_dropped_without_backing_notifications(self):
        stale_research = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
        }
        stale_architect = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }
        stale_help = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:archive",
        }
        agent_data = {
            constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY: False,
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [
                stale_research,
                stale_architect,
                stale_help,
            ],
        }

        self.assertTrue(_drop_stale_pending_dialogue_tick_protections(agent_data))
        self.assertEqual([], agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY])

    def test_pending_protections_are_kept_when_backing_notification_will_render(self):
        pending_research = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "research_center.read_task",
        }
        pending_architect = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
        }
        pending_help = {
            constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ROOM_HELP,
            constants.PROTECTED_DIALOGUE_SOURCE_KEY: "room:archive",
        }
        agent_data = {
            constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY: True,
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [
                "**Architect Message**\nStation protocol.",
                "Help for Archive Room:\nArchive help text.",
            ],
            constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [
                pending_research,
                pending_architect,
                pending_help,
            ],
        }

        self.assertFalse(_drop_stale_pending_dialogue_tick_protections(agent_data))
        self.assertEqual(
            [pending_research, pending_architect, pending_help],
            agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY],
        )

    def test_atomic_architect_notification_queues_protection(self):
        tmpdir = tempfile.mkdtemp(prefix="station_protection_atomic_test_", dir="/tmp")
        old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = tmpdir
        try:
            os.makedirs(os.path.join(tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)
            agent_data = {
                constants.AGENT_NAME_KEY: "Agent A",
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
                constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
                constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
                constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [],
                constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [],
            }
            self.assertTrue(agent_module.save_agent_data("Agent A", agent_data))

            self.assertTrue(
                agent_module.add_pending_notification_atomic(
                    "Agent A",
                    "**Architect Message**\nStation-wide protocol.",
                )
            )
            updated = agent_module.load_agent_data("Agent A", include_ended=True, include_ascended=True)
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual(
            [
                {
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_ARCHITECT_MESSAGE,
                    constants.PROTECTED_DIALOGUE_SOURCE_KEY: "architect_message_notification",
                }
            ],
            updated[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY],
        )

    def test_atomic_notification_ignores_preexisting_unlocked_lock_file(self):
        tmpdir = tempfile.mkdtemp(prefix="station_stale_agent_lock_test_", dir="/tmp")
        old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = tmpdir
        try:
            os.makedirs(os.path.join(tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)
            agent_data = {
                constants.AGENT_NAME_KEY: "Agent A",
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
                constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
                constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
                constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [],
                constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY: [],
            }
            self.assertTrue(agent_module.save_agent_data("Agent A", agent_data))
            lock_path = os.path.join(tmpdir, constants.AGENTS_DIR_NAME, "Agent A.yaml.lock")
            Path(lock_path).touch()

            self.assertTrue(agent_module.add_pending_notification_atomic("Agent A", "research done"))
            updated = agent_module.load_agent_data("Agent A", include_ended=True, include_ascended=True)
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIn("research done", updated[constants.AGENT_NOTIFICATIONS_PENDING_KEY])

    def test_update_agent_fields_ignores_preexisting_unlocked_lock_file(self):
        tmpdir = tempfile.mkdtemp(prefix="station_stale_agent_field_lock_test_", dir="/tmp")
        old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = tmpdir
        try:
            os.makedirs(os.path.join(tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)
            agent_data = {
                constants.AGENT_NAME_KEY: "Agent A",
                constants.AGENT_SESSION_ENDED_KEY: False,
                constants.AGENT_IS_ASCENDED_KEY: False,
                constants.AGENT_DESCRIPTION_KEY: "old",
            }
            self.assertTrue(agent_module.save_agent_data("Agent A", agent_data))
            lock_path = os.path.join(tmpdir, constants.AGENTS_DIR_NAME, "Agent A.yaml.lock")
            Path(lock_path).touch()

            self.assertTrue(
                agent_module.update_agent_fields_atomic(
                    "Agent A",
                    {constants.AGENT_DESCRIPTION_KEY: "new"},
                )
            )
            updated = agent_module.load_agent_data("Agent A", include_ended=True, include_ascended=True)
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertEqual("new", updated[constants.AGENT_DESCRIPTION_KEY])

    def test_research_read_task_only_queues_first_time_protection(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY: False,
        }

        class DummyManager:
            protect_dialogue_tick = staticmethod(agent_module.protect_dialogue_tick)
            queue_dialogue_tick_protection = staticmethod(agent_module.queue_dialogue_tick_protection)
            apply_pending_dialogue_tick_protections = staticmethod(agent_module.apply_pending_dialogue_tick_protections)
            add_pending_notification = staticmethod(agent_module.add_pending_notification)
            has_dialogue_tick_protection = staticmethod(agent_module.has_dialogue_tick_protection)
            get_agent_room_state = staticmethod(agent_module.get_agent_room_state)
            set_agent_room_state = staticmethod(agent_module.set_agent_room_state)

        room_context = RoomContext(
            agent_manager=DummyManager(),
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=None,
        )

        room = ResearchCenter.__new__(ResearchCenter)
        tmpdir = tempfile.mkdtemp(prefix="station_read_task_protection_test_", dir="/tmp")
        old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = tmpdir
        try:
            task_path = (
                Path(tmpdir)
                / constants.ROOMS_DIR_NAME
                / constants.SHORT_ROOM_NAME_RESEARCH
                / constants.RESEARCH_TASK_SPEC_FILENAME
            )
            file_io_utils.ensure_dir_exists(str(task_path.parent))
            file_io_utils.save_text("# Test Task\n\nFind a good construction.", str(task_path))

            actions, handler = ResearchCenter.handle_action(
                room,
                agent_data,
                constants.ACTION_RESEARCH_READ_TASK,
                None,
                None,
                room_context,
                4,
            )
        finally:
            constants.BASE_STATION_DATA_PATH = old_base
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIsNone(handler)
        self.assertEqual(["Research task details sent to your System Messages."], actions)
        self.assertFalse(agent_data[constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY])
        self.assertTrue(agent_data[constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY])
        self.assertTrue(ResearchCenter._must_read_task_before_submit(agent_data, constants))
        self.assertEqual(
            constants.PROTECTED_DIALOGUE_REASON_RESEARCH_TASK_READ,
            agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY][0][constants.PROTECTED_DIALOGUE_REASON_KEY],
        )

        agent_data[constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY] = True
        agent_data[constants.AGENT_PENDING_CURRENT_RESEARCH_TASK_READ_KEY] = False

        agent_data[constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY] = []
        ResearchCenter.handle_action(
            room,
            agent_data,
            constants.ACTION_RESEARCH_READ_TASK,
            None,
            None,
            room_context,
            5,
        )
        self.assertEqual([], agent_data.get(constants.AGENT_PENDING_DIALOGUE_TICK_PROTECTIONS_KEY, []))

    def test_research_submit_gate_only_blocks_new_unread_agents(self):
        new_agent = {
            constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY: False,
        }
        old_agent = {}
        read_agent = {
            constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY: True,
        }

        self.assertTrue(ResearchCenter._must_read_task_before_submit(new_agent, constants))
        self.assertFalse(ResearchCenter._must_read_task_before_submit(old_agent, constants))
        self.assertFalse(ResearchCenter._must_read_task_before_submit(read_agent, constants))

    def test_research_submission_status_mentions_required_task_read(self):
        room = ResearchCenter.__new__(ResearchCenter)
        room.eval_manager = None
        agent_data = {
            constants.AGENT_NAME_KEY: "Agent A",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_HAS_READ_CURRENT_RESEARCH_TASK_KEY: False,
        }

        self.assertEqual(
            "**Submission Not Available:** Read the current Research Task first with `/execute_action{read_task}`.",
            ResearchCenter._build_submission_status_message(room, agent_data, constants, 3),
        )

    def test_meta_reflect_exposes_tick_protection(self):
        handler = MetaReflectionHandler(
            agent_data={},
            room_context=None,
            current_tick=10,
            prompt="prompt",
            num_ticks=1,
        )

        self.assertEqual(
            {
                "reason": constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
                "source": "meta_reflect",
            },
            handler.get_dialogue_tick_protection(),
        )

    def test_display_keeps_protected_tick_even_if_old_prune_block_exists(self):
        room = TokenManagementRoom()
        raw_dialogue = [
            {"tick": 12, "speaker": "Station", "type": "observation", "content": "Important"},
            {"tick": 12, "speaker": "Agent", "type": "submission", "content": "Response"},
        ]
        agent_data = {
            constants.AGENT_PROTECTED_DIALOGUE_TICKS_KEY: [
                {
                    constants.PROTECTED_DIALOGUE_TICK_KEY: 12,
                    constants.PROTECTED_DIALOGUE_REASON_KEY: constants.PROTECTED_DIALOGUE_REASON_META_REFLECTION,
                }
            ]
        }

        display = room._parse_dialogue_for_display(
            raw_dialogue,
            [{constants.PRUNE_TICKS_KEY: "12", constants.PRUNE_SUMMARY_KEY: "old summary"}],
            agent_data,
        )

        self.assertEqual([12], [entry["tick"] for entry in display])
        self.assertTrue(display[0]["is_protected"])


if __name__ == "__main__":
    unittest.main()
