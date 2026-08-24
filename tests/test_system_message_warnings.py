import unittest
import types

from station import constants
from station import system_messages


class FakeAgentModule:
    def add_pending_notification(self, agent_data, notification_message):
        agent_data.setdefault(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []).append(notification_message)

    def get_agent_room_state(self, agent_data, room_name, state_key, default=None):
        return True


class FakeStation:
    def __init__(self):
        self.agent_module = FakeAgentModule()

    def _get_current_tick(self):
        return 20

    def is_holiday_tick(self, current_tick):
        return False

    def _get_agent_age_status(self, agent_data, current_tick):
        return "mature"


class SystemMessageWarningTests(unittest.TestCase):
    def test_life_warning_requires_abandoned_route_handoff(self):
        agent_data = {
            constants.AGENT_TICK_BIRTH_KEY: 0,
            constants.AGENT_MAX_AGE_KEY: 30,
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
        }
        station = FakeStation()

        system_messages.check_and_apply_life_warnings(station, agent_data, current_tick=20)

        notification = agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY][0]
        self.assertIn("leave a clear draft and handoff", notification)
        self.assertIn("your descendant can polish and publish it", notification)
        self.assertIn("every significant direction you demoted or abandoned", notification)
        self.assertIn("what evidence could revive it", notification)
        self.assertIn("where the relevant artifacts can be found", notification)

    def test_tenured_prompt_audience(self):
        self.assertTrue(
            system_messages.prompt_audience_matches(
                {"tenured"},
                is_supervisor=False,
                is_tenured=True,
            )
        )
        self.assertFalse(
            system_messages.prompt_audience_matches(
                {"tenured"},
                is_supervisor=False,
                is_tenured=False,
            )
        )

        eligible = [("tenured tip", {"tenured"}), ("general tip", {"all"})]
        agent_data = {constants.AGENT_NAME_KEY: "Ada"}
        self.assertEqual(
            system_messages.select_prompt_candidates_for_agent(
                eligible,
                [],
                agent_data,
                is_tenured=False,
            ),
            ["general tip"],
        )
        self.assertEqual(
            system_messages.select_prompt_candidates_for_agent(
                eligible,
                [],
                agent_data,
                is_tenured=True,
            ),
            ["tenured tip", "general tip"],
        )

    def test_prompt_conditions_support_numeric_comparisons(self):
        constants_module = types.SimpleNamespace(RESEARCH_MAX_CONCURRENT_SUBMISSIONS=2)

        self.assertTrue(
            system_messages.is_prompt_condition_met(
                "RESEARCH_MAX_CONCURRENT_SUBMISSIONS >= 2",
                constants_module,
            )
        )
        self.assertFalse(
            system_messages.is_prompt_condition_met(
                "RESEARCH_MAX_CONCURRENT_SUBMISSIONS > 2",
                constants_module,
            )
        )

        constants_module.RESEARCH_MAX_CONCURRENT_SUBMISSIONS = 1
        self.assertFalse(
            system_messages.is_prompt_condition_met(
                "RESEARCH_MAX_CONCURRENT_SUBMISSIONS >= 2",
                constants_module,
            )
        )

    def test_goto_only_warning_is_sent_once_per_agent_lifetime(self):
        agent_data = {
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: 0,
            constants.AGENT_LAST_PARSED_ACTIONS_RAW_KEY: [
                {"command": constants.ACTION_GO_TO, "args": constants.SHORT_ROOM_NAME_LOBBY}
            ],
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
        }
        station = FakeStation()

        system_messages.check_and_apply_inactivity_warning(station, agent_data, current_tick=20)

        self.assertIn(
            constants.GOTO_ONLY_ACTION_WARNING,
            agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY],
        )
        self.assertTrue(agent_data[constants.AGENT_GOTO_ONLY_WARNING_SENT_KEY])

        agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY] = []
        system_messages.check_and_apply_inactivity_warning(station, agent_data, current_tick=21)

        self.assertNotIn(
            constants.GOTO_ONLY_ACTION_WARNING,
            agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY],
        )


if __name__ == "__main__":
    unittest.main()
