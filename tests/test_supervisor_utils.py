import unittest

from station import constants
from station.supervisor_utils import _get_supervisor_candidates


class _AgentManager:
    def __init__(self, agents):
        self.agents = agents

    def get_all_active_agent_names(self):
        return list(self.agents)

    def load_agent_data(self, name):
        return self.agents[name]


class _ArchiveRoom:
    def count_agent_archive_capsules(self, agent_name, _room_context):
        return 1


class _Station:
    rooms = {constants.ROOM_ARCHIVE: _ArchiveRoom()}


class _RoomContext:
    constants_module = constants
    station_instance = _Station()


class SupervisorCandidateTests(unittest.TestCase):
    def test_default_model_patterns_accept_gpt_and_claude_opus(self):
        agents = {
            "gpt": {
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_MODEL_NAME_KEY: "gpt-5.5",
                constants.AGENT_ROLE_KEY: None,
            },
            "opus5": {
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_MODEL_NAME_KEY: "claude-opus-5",
                constants.AGENT_ROLE_KEY: None,
            },
            "opus48": {
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_MODEL_NAME_KEY: "claude-opus-4-8",
                constants.AGENT_ROLE_KEY: None,
            },
            "other": {
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_MODEL_NAME_KEY: "claude-sonnet-4",
                constants.AGENT_ROLE_KEY: None,
            },
        }
        candidates = _get_supervisor_candidates(
            _AgentManager(agents), _RoomContext(), current_tick=100
        )
        self.assertEqual(candidates, ["gpt", "opus5"])

    def test_single_string_override_remains_supported(self):
        original = constants.SUPERVISOR_REQUIRED_MODEL_NAME
        try:
            constants.SUPERVISOR_REQUIRED_MODEL_NAME = "gpt-*"
            agents = {
                "gpt": {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_MODEL_NAME_KEY: "gpt-5.5",
                    constants.AGENT_ROLE_KEY: None,
                },
                "opus5": {
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_MODEL_NAME_KEY: "claude-opus-5",
                    constants.AGENT_ROLE_KEY: None,
                },
            }
            candidates = _get_supervisor_candidates(
                _AgentManager(agents), _RoomContext(), current_tick=100
            )
            self.assertEqual(candidates, ["gpt"])
        finally:
            constants.SUPERVISOR_REQUIRED_MODEL_NAME = original

    def test_assignment_cooldown_default_is_zero(self):
        self.assertEqual(constants.SUPERVISOR_ASSIGNMENT_COOLDOWN_TICKS, 0)


if __name__ == "__main__":
    unittest.main()
