import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from station import agent as agent_module
from station import constants
from station import file_io_utils
from station import session_end_flow
from station.base_room import RoomContext
from station.rooms.exit import ExitReflectionHandler
from station.rooms.lobby import LobbyRoom
from station.station import Station
from station.system_messages import build_station_level_system_prompt


class TempAgentRoleDefinitionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_role_test_", dir="/tmp")
        self._saved_constants = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "THEORIST_SPAWN_PROBABILITY": constants.THEORIST_SPAWN_PROBABILITY,
            "AUTO_RESPAWN": constants.AUTO_RESPAWN,
        }
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        constants.THEORIST_SPAWN_PROBABILITY = 0.0
        constants.AUTO_RESPAWN = True
        file_io_utils.ensure_dir_exists(
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME)
        )

    def tearDown(self):
        for key, value in self._saved_constants.items():
            setattr(constants, key, value)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def write_init_roles(self, roles):
        file_io_utils.save_yaml(
            roles,
            os.path.join(self.tmpdir, constants.INIT_ROLE_DEFINITION_FILENAME),
        )


class GuestRoleDefinitionTests(TempAgentRoleDefinitionTestCase):
    def test_lobby_rejects_research_storage_reserved_lineage_names(self):
        for name in ["Shared", "System", "Lineages", "Tmp", "Submission", "Stdout", "Stderr", "Report", "Architect", "Unknown"]:
            with self.subTest(name=name):
                self.assertIn("reserved", LobbyRoom._validate_lineage_name(name))

    def test_agent_creation_rejects_reserved_lineage_names(self):
        created = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Tmp",
            generation=1,
            current_tick=0,
        )

        self.assertIsNone(created)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Tmp I.yaml")))

    def test_ascend_agent_rejects_reserved_lineage_names(self):
        guest = agent_module.create_guest_agent(
            model_name="test-model",
            current_tick=0,
        )
        self.assertIsNotNone(guest)

        ascended = agent_module.ascend_agent(
            guest_agent_name=guest[constants.AGENT_NAME_KEY],
            new_recursive_name="Tmp I",
            new_lineage="Tmp",
            new_generation=1,
            current_tick=1,
        )

        self.assertIsNone(ascended)
        reloaded_guest = agent_module.load_agent_data(
            guest[constants.AGENT_NAME_KEY],
            include_ascended=True,
            include_ended=True,
        )
        self.assertFalse(reloaded_guest[constants.AGENT_IS_ASCENDED_KEY])

    def test_guest_samples_init_role_when_no_explicit_definition(self):
        self.write_init_roles(["Role A", "Role B"])

        with patch.object(agent_module.random, "choice", return_value="Role B"):
            agent_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=0,
            )

        self.assertIsNotNone(agent_data)
        self.assertEqual(
            "Role B",
            agent_data[constants.AGENT_ROLE_DEFINITION_KEY],
        )
        self.assertEqual("Role B", agent_module.get_agent_role_definition(agent_data))

    def test_guest_can_sample_empty_init_role(self):
        self.write_init_roles(["Role A", ""])

        with patch.object(agent_module.random, "choice", return_value=""):
            agent_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=0,
            )

        self.assertIsNotNone(agent_data)
        self.assertEqual("", agent_data[constants.AGENT_ROLE_DEFINITION_KEY])
        self.assertIsNone(agent_module.get_agent_role_definition(agent_data))

    def test_explicit_guest_role_definition_skips_random_init_role(self):
        self.write_init_roles(["Random Role"])

        with patch.object(
            agent_module,
            "pick_random_init_role_definition",
            side_effect=AssertionError("should not sample role pool"),
        ):
            agent_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=0,
                role_definition="  Explicit Role  ",
            )
            blank_agent_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=0,
                role_definition="   ",
            )

        self.assertEqual(
            "Explicit Role",
            agent_data[constants.AGENT_ROLE_DEFINITION_KEY],
        )
        self.assertEqual("", blank_agent_data[constants.AGENT_ROLE_DEFINITION_KEY])

    def test_guest_sampling_pool_excludes_departed_next_roles(self):
        self.write_init_roles(["Init Role"])
        departed = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Aletheia",
            generation=1,
            current_tick=0,
            role_definition="Departed current role",
        )
        self.assertIsNotNone(departed)
        departed[constants.AGENT_SESSION_ENDED_KEY] = True
        departed[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Departed next role"
        agent_module.save_agent_data(departed[constants.AGENT_NAME_KEY], departed)

        active = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Noesis",
            generation=1,
            current_tick=0,
            role_definition="Active current role",
        )
        self.assertIsNotNone(active)
        active[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Active next role"
        agent_module.save_agent_data(active[constants.AGENT_NAME_KEY], active)

        pool = agent_module.get_role_definition_sampling_pool()

        self.assertEqual(["Init Role"], pool)
        self.assertNotIn("Departed next role", pool)
        self.assertNotIn("Departed current role", pool)
        self.assertNotIn("Active next role", pool)

        def choose_role(role_pool):
            self.assertEqual(pool, role_pool)
            return "Init Role"

        with patch.object(agent_module.random, "choice", side_effect=choose_role):
            agent_data = agent_module.create_guest_agent(
                model_name="test-model",
                current_tick=1,
            )

        self.assertIsNotNone(agent_data)
        self.assertEqual(
            "Init Role",
            agent_data[constants.AGENT_ROLE_DEFINITION_KEY],
        )

    def test_guest_sampling_pool_excludes_all_departed_next_roles(self):
        self.write_init_roles(["Init Role"])
        supervisor = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="SupervisorLine",
            generation=1,
            current_tick=0,
            role_definition="Supervisor current role",
            role=constants.ROLE_SUPERVISOR,
        )
        theorist = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="TheoristLine",
            generation=1,
            current_tick=0,
            role_definition="Theorist current role",
            role=constants.ROLE_THEORIST,
        )
        ordinary = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="OrdinaryLine",
            generation=1,
            current_tick=0,
            role_definition="Ordinary current role",
        )
        self.assertIsNotNone(supervisor)
        self.assertIsNotNone(theorist)
        self.assertIsNotNone(ordinary)

        supervisor[constants.AGENT_SESSION_ENDED_KEY] = True
        supervisor[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Supervisor next role"
        agent_module.save_agent_data(supervisor[constants.AGENT_NAME_KEY], supervisor)

        theorist[constants.AGENT_SESSION_ENDED_KEY] = True
        theorist[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Theorist next role"
        agent_module.save_agent_data(theorist[constants.AGENT_NAME_KEY], theorist)

        ordinary[constants.AGENT_SESSION_ENDED_KEY] = True
        ordinary[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Ordinary next role"
        agent_module.save_agent_data(ordinary[constants.AGENT_NAME_KEY], ordinary)

        pool = agent_module.get_role_definition_sampling_pool()

        self.assertEqual(["Init Role"], pool)
        self.assertNotIn("Ordinary next role", pool)
        self.assertNotIn("Supervisor next role", pool)
        self.assertNotIn("Theorist next role", pool)

    def test_blocked_exit_role_replacement_uses_fresh_guest_sampling_pool(self):
        self.write_init_roles(["Init Role"])
        departed = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Sophia",
            generation=1,
            current_tick=0,
            role_definition="Departed current role",
        )
        self.assertIsNotNone(departed)
        departed[constants.AGENT_SESSION_ENDED_KEY] = True
        departed[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Departed next role"
        agent_module.save_agent_data(departed[constants.AGENT_NAME_KEY], departed)

        def choose_role(role_pool):
            self.assertIn("Init Role", role_pool)
            self.assertNotIn("Departed next role", role_pool)
            return "Init Role"

        with patch.object(agent_module.random, "choice", side_effect=choose_role):
            replacement, blocked_term = session_end_flow.replace_blocked_role_definition(
                "Make the next agent a Hacker."
            )

        self.assertEqual("Init Role", replacement)
        self.assertEqual("Hacker", blocked_term)


class RespawnRoleDefinitionTests(TempAgentRoleDefinitionTestCase):
    def make_station(self):
        station = Station.__new__(Station)
        station.agent_module = agent_module
        station.config_path = os.path.join(
            self.tmpdir,
            constants.STATION_CONFIG_FILENAME,
        )
        station.config = {
            constants.STATION_CONFIG_CURRENT_TICK: 7,
            constants.STATION_CONFIG_AGENT_TURN_ORDER: [],
            constants.STATION_CONFIG_NEXT_AGENT_INDEX: 0,
        }
        return station

    def test_respawned_guest_samples_init_role_instead_of_old_role(self):
        self.write_init_roles(["Fresh init role"])
        original = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Aletheia",
            generation=1,
            current_tick=0,
            model_provider_class="OpenAI",
            role_definition="Departed agent role",
            llm_temperature=0.3,
            llm_max_tokens=4096,
            llm_custom_api_params={"base_url": "https://example.test"},
        )
        self.assertIsNotNone(original)
        original[constants.AGENT_SESSION_ENDED_KEY] = True
        agent_module.save_agent_data(original[constants.AGENT_NAME_KEY], original)

        with patch.object(agent_module.random, "choice", return_value="Fresh init role"):
            new_agent_name = self.make_station().create_respawn_guest_agent(
                original[constants.AGENT_NAME_KEY]
            )

        self.assertIsNotNone(new_agent_name)
        respawned = agent_module.load_agent_data(new_agent_name)
        self.assertIsNotNone(respawned)
        self.assertEqual(
            "Fresh init role",
            respawned[constants.AGENT_ROLE_DEFINITION_KEY],
        )
        self.assertEqual("OpenAI", respawned[constants.AGENT_MODEL_PROVIDER_CLASS_KEY])
        self.assertEqual(0.3, respawned[constants.AGENT_LLM_TEMPERATURE_KEY])
        self.assertEqual(4096, respawned[constants.AGENT_LLM_MAX_TOKENS_KEY])
        self.assertEqual(
            {"base_url": "https://example.test"},
            respawned[constants.AGENT_LLM_CUSTOM_API_PARAMS_KEY],
        )

    def test_respawned_guest_uses_init_pool_instead_of_next_role_inheritance(self):
        self.write_init_roles(["Fresh init role"])
        original = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Noesis",
            generation=1,
            current_tick=0,
            role_definition="Departed agent role",
        )
        self.assertIsNotNone(original)
        original[constants.AGENT_SESSION_ENDED_KEY] = True
        original[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Ascension-only role"
        agent_module.save_agent_data(original[constants.AGENT_NAME_KEY], original)

        def choose_role(role_pool):
            self.assertIn("Fresh init role", role_pool)
            self.assertNotIn("Ascension-only role", role_pool)
            self.assertNotIn("Departed agent role", role_pool)
            return "Fresh init role"

        with patch.object(agent_module.random, "choice", side_effect=choose_role):
            new_agent_name = self.make_station().create_respawn_guest_agent(
                original[constants.AGENT_NAME_KEY]
            )

        self.assertIsNotNone(new_agent_name)
        respawned = agent_module.load_agent_data(new_agent_name)
        self.assertIsNotNone(respawned)
        self.assertEqual(
            "Fresh init role",
            respawned[constants.AGENT_ROLE_DEFINITION_KEY],
        )
        self.assertNotEqual(
            "Departed agent role",
            respawned[constants.AGENT_ROLE_DEFINITION_KEY],
        )


class StationStub:
    def __init__(self):
        self.ascensions = []
        self.terminated_agents = []
        self.field_updates = []

    def _is_agent_mature(self, agent_data, current_tick):
        return False

    def update_turn_order_on_ascension(self, old_guest_name, new_recursive_name):
        self.ascensions.append((old_guest_name, new_recursive_name))
        return True

    def _terminate_agent_session_with_broadcast(self, agent_name, reason, critical_notification):
        self.terminated_agents.append((agent_name, reason, critical_notification))

    def update_specific_agent_fields(self, agent_name, fields):
        self.field_updates.append((agent_name, fields))
        return True


class AscensionRoleDefinitionTests(TempAgentRoleDefinitionTestCase):
    def make_context(self, station_stub):
        return RoomContext(
            agent_manager=agent_module,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=station_stub,
        )

    def test_inherited_next_role_beats_guest_role_on_ascend_inherit(self):
        ancestor = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Aletheia",
            generation=1,
            current_tick=0,
            role_definition="Ancestor current role",
        )
        self.assertIsNotNone(ancestor)
        ancestor[constants.AGENT_SESSION_ENDED_KEY] = True
        ancestor[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = "Ancestor next role"
        agent_module.save_agent_data(ancestor[constants.AGENT_NAME_KEY], ancestor)

        guest = agent_module.create_guest_agent(
            model_name="test-model",
            current_tick=0,
            role_definition="Guest role",
        )
        self.assertIsNotNone(guest)
        guest[constants.AGENT_ASCENSION_ELIGIBLE_KEY] = True
        guest[constants.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = ancestor[
            constants.AGENT_NAME_KEY
        ]

        station_stub = StationStub()
        actions, _ = LobbyRoom().handle_action(
            guest,
            constants.ACTION_ASCEND_INHERIT,
            None,
            None,
            self.make_context(station_stub),
            current_tick=1,
        )

        self.assertTrue(any("Ascension to Aletheia II" in action for action in actions))
        ascended = agent_module.load_agent_data("Aletheia II")
        self.assertIsNotNone(ascended)
        self.assertEqual(
            "Ancestor next role",
            ascended[constants.AGENT_ROLE_DEFINITION_KEY],
        )

    def test_explicit_empty_inherited_next_role_beats_guest_role(self):
        ancestor = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Noesis",
            generation=1,
            current_tick=0,
            role_definition="Ancestor current role",
        )
        self.assertIsNotNone(ancestor)
        ancestor[constants.AGENT_SESSION_ENDED_KEY] = True
        ancestor[constants.AGENT_NEXT_ROLE_DEFINITION_KEY] = ""
        agent_module.save_agent_data(ancestor[constants.AGENT_NAME_KEY], ancestor)

        guest = agent_module.create_guest_agent(
            model_name="test-model",
            current_tick=0,
            role_definition="Guest role",
        )
        self.assertIsNotNone(guest)
        guest[constants.AGENT_ASCENSION_ELIGIBLE_KEY] = True
        guest[constants.AGENT_POTENTIAL_ANCESTOR_NAME_KEY] = ancestor[
            constants.AGENT_NAME_KEY
        ]

        LobbyRoom().handle_action(
            guest,
            constants.ACTION_ASCEND_INHERIT,
            None,
            None,
            self.make_context(StationStub()),
            current_tick=1,
        )

        ascended = agent_module.load_agent_data("Noesis II")
        self.assertIsNotNone(ascended)
        self.assertEqual("", ascended[constants.AGENT_ROLE_DEFINITION_KEY])
        self.assertIsNone(agent_module.get_agent_role_definition(ascended))


class ExitNextRoleDefinitionTests(TempAgentRoleDefinitionTestCase):
    def make_context(self, station_stub):
        return RoomContext(
            agent_manager=agent_module,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=station_stub,
        )

    def assert_role_cannot_define_next_role_on_exit(self, role):
        agent_data = {
            constants.AGENT_NAME_KEY: f"{role} agent",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_ROLE_KEY: role,
            constants.AGENT_SESSION_ENDED_KEY: False,
        }
        station_stub = StationStub()
        handler = ExitReflectionHandler(
            agent_data=agent_data,
            room_context=self.make_context(station_stub),
            current_tick=1,
        )

        next_prompt, actions = handler.step("/execute_action{exit}")

        self.assertIsNone(next_prompt)
        self.assertFalse(handler.awaiting_next_role_definition)
        self.assertNotIn(constants.AGENT_NEXT_ROLE_DEFINITION_KEY, agent_data)
        self.assertEqual(
            {
                constants.AGENT_SESSION_ENDED_KEY: True,
                constants.AGENT_TICK_EXIT_KEY: 1,
            },
            handler.get_delta_updates(),
        )
        self.assertEqual(
            (
                agent_data[constants.AGENT_NAME_KEY],
                {
                    constants.AGENT_SESSION_ENDED_KEY: True,
                    constants.AGENT_TICK_EXIT_KEY: 1,
                },
            ),
            station_stub.field_updates[-1],
        )
        self.assertEqual(1, len(station_stub.terminated_agents))
        self.assertTrue(any("session has been terminated" in action for action in actions))

    def test_supervisor_cannot_define_next_role_on_exit(self):
        self.assert_role_cannot_define_next_role_on_exit(constants.ROLE_SUPERVISOR)

    def test_theorist_cannot_define_next_role_on_exit(self):
        self.assert_role_cannot_define_next_role_on_exit(constants.ROLE_THEORIST)

    def test_regular_recursive_exit_delta_keeps_terminal_state_with_next_role(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Ordinary agent",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_SESSION_ENDED_KEY: False,
        }
        station_stub = StationStub()
        handler = ExitReflectionHandler(
            agent_data=agent_data,
            room_context=self.make_context(station_stub),
            current_tick=7,
        )

        next_prompt, actions = handler.step("/execute_action{exit}")
        self.assertIsNotNone(next_prompt)
        self.assertEqual([], actions)
        self.assertTrue(handler.awaiting_next_role_definition)

        next_prompt, actions = handler.step(
            "```yaml\ncontent: |\n  Future role\n```"
        )

        self.assertIsNone(next_prompt)
        self.assertEqual(1, len(station_stub.terminated_agents))
        self.assertTrue(any("session has been terminated" in action for action in actions))
        self.assertEqual(
            {
                constants.AGENT_NEXT_ROLE_DEFINITION_KEY: "Future role",
                constants.AGENT_SESSION_ENDED_KEY: True,
                constants.AGENT_TICK_EXIT_KEY: 7,
            },
            handler.get_delta_updates(),
        )
        self.assertEqual(
            (
                "Ordinary agent",
                {
                    constants.AGENT_SESSION_ENDED_KEY: True,
                    constants.AGENT_TICK_EXIT_KEY: 7,
                },
            ),
            station_stub.field_updates[-1],
        )

    def test_next_role_request_formats_existing_roles_as_fenced_sections(self):
        active = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Formatter",
            generation=1,
            current_tick=0,
            role_definition="Formatter role line 1\nFormatter role line 2",
        )
        self.assertIsNotNone(active)

        prompt = session_end_flow.build_next_role_definition_request(
            "Departing agent",
            self.make_context(StationStub()),
        )

        self.assertIn(
            "## Agent 1\n\n```\nFormatter role line 1\nFormatter role line 2\n```",
            prompt,
        )
        self.assertNotIn("1. Formatter role line 1", prompt)

    def test_life_limit_normal_end_prompts_for_next_role_without_exit_confirmation(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Ordinary agent",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_SESSION_ENDED_KEY: False,
        }
        station_stub = StationStub()
        handler = session_end_flow.NextRoleDefinitionSessionEndHandler(
            agent_data=agent_data,
            room_context=self.make_context(station_stub),
            current_tick=11,
            reason=session_end_flow.SESSION_END_REASON_LIFE_LIMIT,
            critical_notification="CRITICAL: life ended",
        )

        prompt = handler.init()
        self.assertIn("define a role description for your next descendant", prompt)

        next_prompt, actions = handler.step(
            "```yaml\ncontent: |\n  Future life-limit role\n```"
        )

        self.assertIsNone(next_prompt)
        self.assertTrue(any("Next role definition recorded." in action for action in actions))
        self.assertEqual(
            [("Ordinary agent", session_end_flow.SESSION_END_REASON_LIFE_LIMIT, "CRITICAL: life ended")],
            station_stub.terminated_agents,
        )
        self.assertEqual(
            {
                constants.AGENT_NEXT_ROLE_DEFINITION_KEY: "Future life-limit role",
                constants.AGENT_SESSION_ENDED_KEY: True,
                constants.AGENT_TICK_EXIT_KEY: 11,
            },
            handler.get_delta_updates(),
        )

    def test_manual_and_context_overflow_do_not_request_next_role(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Ordinary agent",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_SESSION_ENDED_KEY: False,
        }

        self.assertFalse(
            session_end_flow.should_request_next_role_definition(
                agent_data,
                constants,
                session_end_flow.SESSION_END_REASON_MANUAL,
            )
        )
        self.assertFalse(
            session_end_flow.should_request_next_role_definition(
                agent_data,
                constants,
                session_end_flow.SESSION_END_REASON_CONTEXT_OVERFLOW,
            )
        )

    def test_station_life_limit_returns_next_role_handler_for_regular_recursive_agent(self):
        agent_data = agent_module.create_recursive_agent(
            model_name="test-model",
            lineage="Limit",
            generation=1,
            current_tick=0,
            role_definition="Current role",
        )
        self.assertIsNotNone(agent_data)
        agent_data[constants.AGENT_MAX_AGE_KEY] = 1
        agent_module.save_agent_data(agent_data[constants.AGENT_NAME_KEY], agent_data)

        station = Station.__new__(Station)
        station.agent_module = agent_module
        station.room_context = self.make_context(station)
        station._log_dialogue_entry = lambda agent_name, entry: None

        can_continue, handler = Station._check_agent_life_limit(
            station,
            agent_data[constants.AGENT_NAME_KEY],
            current_tick=1,
        )

        self.assertFalse(can_continue)
        self.assertIsNotNone(handler)
        self.assertIsInstance(
            handler.actual_handler,
            session_end_flow.NextRoleDefinitionSessionEndHandler,
        )


class RolePromptOverrideTests(TempAgentRoleDefinitionTestCase):
    def test_station_system_prompt_uses_researcher_template_without_codex(self):
        file_io_utils.save_text(
            "Codex text that belongs in Lobby help.",
            os.path.join(self.tmpdir, constants.CODEX_FILENAME),
        )
        agent_data = {
            constants.AGENT_NAME_KEY: "Scholar I",
            constants.AGENT_ROLE_DEFINITION_KEY: "Study the active task carefully.",
        }
        agent_module.save_agent_data(agent_data[constants.AGENT_NAME_KEY], agent_data)

        prompt = build_station_level_system_prompt(
            agent_data[constants.AGENT_NAME_KEY],
            agent_module.get_agent_role_definition(agent_data),
        )

        self.assertIn(
            "You are an academic researcher in a multi-agent research environment called the Station.",
            prompt,
        )
        self.assertIn(
            "Do not fabricate results, overclaim, or treat bounded computational failures as impossibility proofs.",
            prompt,
        )
        self.assertIn("Your defined role is:\n\nStudy the active task carefully.", prompt)
        self.assertNotIn("Codex text that belongs in Lobby help.", prompt)
        self.assertNotIn("This is the Codex", prompt)

    def test_lobby_help_renders_codex_before_first_mission(self):
        file_io_utils.save_text(
            "Codex principle one.\nCodex principle two.",
            os.path.join(self.tmpdir, constants.CODEX_FILENAME),
        )
        room_context = RoomContext(
            agent_manager=None,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=None,
        )

        help_message = LobbyRoom().get_help_message(
            {constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST},
            room_context,
        )

        intro_index = help_message.index("You are an AI designed for autonomous research.")
        codex_index = help_message.index("Codex principle one.")
        mission_index = help_message.index("### Your First Mission")
        self.assertLess(intro_index, codex_index)
        self.assertLess(codex_index, mission_index)
        self.assertNotIn("{codex}", help_message)

    def test_theorist_role_overrides_stored_role_definition_in_system_prompt(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Theoria I",
            constants.AGENT_ROLE_KEY: constants.ROLE_THEORIST,
            constants.AGENT_ROLE_DEFINITION_KEY: "Stored ordinary role",
        }
        agent_module.save_agent_data(agent_data[constants.AGENT_NAME_KEY], agent_data)

        prompt = build_station_level_system_prompt(
            agent_data[constants.AGENT_NAME_KEY],
            agent_module.get_agent_role_definition(agent_data),
        )

        self.assertIn("You have been assigned as a theorist", prompt)
        self.assertNotIn("Stored ordinary role", prompt)

    def test_supervisor_role_overrides_stored_role_definition_in_system_prompt(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Supervisor I",
            constants.AGENT_ROLE_KEY: constants.ROLE_SUPERVISOR,
            constants.AGENT_ROLE_DEFINITION_KEY: "Stored ordinary role",
        }
        agent_module.save_agent_data(agent_data[constants.AGENT_NAME_KEY], agent_data)

        prompt = build_station_level_system_prompt(
            agent_data[constants.AGENT_NAME_KEY],
            agent_module.get_agent_role_definition(agent_data),
        )

        self.assertIn("Supervisor Protocol", prompt)
        self.assertIn("A known failure mode is premature demotion", prompt)
        self.assertIn("verify in later meetings that the backlog is revisited", prompt)
        self.assertNotIn("Stored ordinary role", prompt)


class WebCreateAgentRolePayloadTests(unittest.TestCase):
    def test_blank_role_payload_normalizes_to_no_explicit_role(self):
        from web_interface.input_utils import normalize_optional_role_definition

        self.assertIsNone(normalize_optional_role_definition(""))
        self.assertIsNone(normalize_optional_role_definition("   "))
        self.assertIsNone(normalize_optional_role_definition(None))
        self.assertEqual(
            "Explicit Role",
            normalize_optional_role_definition("  Explicit Role  "),
        )


if __name__ == "__main__":
    unittest.main()
