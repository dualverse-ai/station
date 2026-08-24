import os
import shutil
import tempfile
import unittest
from unittest import mock

from station import capsule
from station import constants
from station import file_io_utils
from station.action_parser import ActionParser
from station.base_room import RoomContext
from station.rooms.question import QuestionRoom


class DummyAgentManager:
    def __init__(self, agents):
        self.agents = agents

    def get_agent_room_state(self, agent_data, room_key, state_key, default=None):
        room_state = agent_data.get(room_key, {})
        if not isinstance(room_state, dict):
            return default
        return room_state.get(state_key, default)

    def set_agent_room_state(self, agent_data, room_key, state_key, value):
        agent_data.setdefault(room_key, {})[state_key] = value

    def add_pending_notification(self, agent_data, message):
        agent_data.setdefault(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []).append(message)

    def get_all_active_agent_names(self):
        return list(self.agents)

    def get_active_recursive_agent_names(self):
        return [
            name for name, data in self.agents.items()
            if data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE
        ]

    def update_agent_with_function(self, agent_name, update_func, max_retries=5):
        update_func(self.agents[agent_name])
        return True

    def load_agent_data(self, agent_name, *args, **kwargs):
        return self.agents.get(agent_name)


class DummyStation:
    def __init__(self, tick=100):
        self.tick = tick

    def _get_current_tick(self):
        return self.tick

    def _get_agent_age_status(self, agent_data, current_tick):
        birth_tick = agent_data.get(constants.AGENT_TICK_BIRTH_KEY)
        if birth_tick is None:
            return None
        age = current_tick - birth_tick
        if constants.AGENT_ISOLATION_TICKS is not None and age < constants.AGENT_ISOLATION_TICKS:
            return "immature"
        if age >= constants.MIN_AGENT_AGE_BEFORE_LEAVE:
            return "tenured"
        return "mature"

    def _is_agent_question_room_allowed(self, agent_data, current_tick):
        if agent_data.get(constants.AGENT_STATUS_KEY) != constants.AGENT_STATUS_RECURSIVE:
            return False
        if agent_data.get(constants.AGENT_ROLE_KEY) == constants.ROLE_SUPERVISOR:
            return True
        return self._get_agent_age_status(agent_data, current_tick) == "tenured"


class QuestionRoomTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_question_room_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        self.room = QuestionRoom()
        self.agents = {
            "Author": self._agent("Author", birth_tick=0, lineage="A"),
            "Voter1": self._agent("Voter1", birth_tick=0, lineage="B"),
            "Voter2": self._agent("Voter2", birth_tick=0, lineage="C"),
            "Voter3": self._agent("Voter3", birth_tick=0, lineage="D"),
            "Supervisor": self._agent("Supervisor", birth_tick=99, lineage="S", role=constants.ROLE_SUPERVISOR),
            "Mature": self._agent("Mature", birth_tick=50, lineage="M"),
        }
        self.agent_manager = DummyAgentManager(self.agents)
        self.station = DummyStation(tick=100)
        self.context = RoomContext(
            agent_manager=self.agent_manager,
            capsule_manager=capsule,
            notification_manager=None,
            constants_module=constants,
            station_instance=self.station,
        )

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _agent(self, name, birth_tick, lineage, role=None):
        return {
            constants.AGENT_NAME_KEY: name,
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_TICK_BIRTH_KEY: birth_tick,
            constants.AGENT_LINEAGE_KEY: lineage,
            constants.AGENT_GENERATION_KEY: 1,
            constants.AGENT_ROLE_KEY: role,
        }

    def _create_question(self, author_name="Author"):
        actions, _ = self.room.handle_action(
            self.agents[author_name],
            constants.ACTION_CAPSULE_CREATE,
            None,
            {
                "title": f"Question by {author_name}",
                "abstract": "abstract",
                "content": "Research Question: Q\nScope: S\nMotivation: M\nSuccess Criteria: C",
            },
            self.context,
            100,
        )
        self.assertIn("Successfully created", actions[0])

    def _vote(self, agent_name, command, target):
        actions, _ = self.room.handle_action(
            self.agents[agent_name],
            command,
            target,
            None,
            self.context,
            100,
        )
        return actions

    def test_access_requires_tenure_unless_supervisor(self):
        self.assertFalse(self.room._is_allowed_agent(self.agents["Mature"], self.context))
        self.assertTrue(self.room._is_allowed_agent(self.agents["Author"], self.context))
        self.assertTrue(self.room._is_allowed_agent(self.agents["Supervisor"], self.context))

    def test_help_clarifies_solution_attempts_use_reply(self):
        help_text = self.room.get_help_message(self.agents["Author"], self.context)
        self.assertIn("There is no separate solution-submit action", help_text)
        self.assertIn("/execute_action{reply question_id}", help_text)
        self.assertIn("Question Type", help_text)
        self.assertIn("Reduced problem", help_text)
        self.assertIn("Subproblem", help_text)
        self.assertIn("Constructive", help_text)
        self.assertIn("Theory", help_text)
        self.assertIn("Knowledge", help_text)
        self.assertIn("saved in shared Research Center storage", help_text)
        self.assertIn("reproduces a known good basin", help_text)
        self.assertIn("Diagnostic-only", help_text)
        self.assertIn("Easy escape", help_text)
        self.assertIn("At tenure, community involvement is important", help_text)
        self.assertIn("maintaining good questions and participating in discussions", help_text)
        self.assertIn("more than sixteen", help_text)
        self.assertIn("prioritize answering or advancing existing open questions", help_text)
        self.assertIn("If fewer than eight questions are pending or open", help_text)
        self.assertIn("Do not post merely to increase the count", help_text)
        self.assertIn("use a parallel slot to try answering or advancing some of these questions", help_text)
        self.assertIn("not the agent's lineage", help_text)
        self.assertIn("/execute_action{retire 15}", help_text)
        self.assertIn("/execute_action{unretire 15}", help_text)
        self.assertIn("filter pending|open|redacted|solved|retired", help_text)
        self.assertIn("does not count toward its author's active-question limit", help_text)
        self.assertIn("bounded extinction result", help_text)
        self.assertIn("Exactness alone does not make a question strategically important", help_text)
        self.assertIn("Optionally include YAML `content` to post a retirement reason", help_text)

    def test_help_omits_parallel_slot_guidance_when_only_one_concurrent_submission(self):
        with mock.patch.object(constants, "RESEARCH_MAX_CONCURRENT_SUBMISSIONS", 1):
            help_text = self.room.get_help_message(self.agents["Author"], self.context)

        self.assertIn("prioritize answering or advancing existing open questions", help_text)
        self.assertNotIn("use a parallel slot to try answering or advancing some of these questions", help_text)

    def test_supervisor_prompt_does_not_assume_open_questions_have_strategic_value(self):
        prompt = constants.SUPERVISOR_PROTOCOL_SYSTEM_PROMPT
        self.assertIn("verify that the question is currently open and not retired", prompt)
        self.assertIn("does not automatically establish strategic value", prompt)
        self.assertIn("include optional YAML `content` with a concise reason", prompt)
        self.assertNotIn("first reply to it", prompt)
        self.assertNotIn("Answering open questions in the Question Room is generally considered", prompt)

    def test_retire_action_parses_with_or_without_optional_yaml(self):
        parser = ActionParser()
        with_reason = parser.parse(
            "/execute_action{retire 7}\n"
            "```yaml\n"
            "content: Superseded by Archive 12.\n"
            "```\n"
        )
        without_reason = parser.parse("/execute_action{retire 7}\n")

        self.assertEqual({"content": "Superseded by Archive 12."}, with_reason[0].yaml_data)
        self.assertIsNone(without_reason[0].yaml_data)

    def test_creation_limit_and_delete_frees_slot(self):
        self._create_question("Author")
        denied, _ = self.room.handle_action(
            self.agents["Author"],
            constants.ACTION_CAPSULE_CREATE,
            None,
            {"title": "Second", "abstract": "abstract", "content": "content"},
            self.context,
            100,
        )
        self.assertIn("Permission denied", denied[0])

        deleted, _ = self.room.handle_action(
            self.agents["Author"],
            constants.ACTION_CAPSULE_DELETE,
            "1",
            None,
            self.context,
            101,
        )
        self.assertIn("soft deleted", deleted[0])

        created, _ = self.room.handle_action(
            self.agents["Author"],
            constants.ACTION_CAPSULE_CREATE,
            None,
            {"title": "Second", "abstract": "abstract", "content": "content"},
            self.context,
            102,
        )
        self.assertIn("Successfully created", created[0])

    def test_supervisor_creation_limit_is_three(self):
        self._create_question("Supervisor")
        self._create_question("Supervisor")
        self._create_question("Supervisor")
        denied, _ = self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_CAPSULE_CREATE,
            None,
            {"title": "Fourth", "abstract": "abstract", "content": "content"},
            self.context,
            100,
        )
        self.assertIn("Permission denied", denied[0])

    def test_create_notifications_skip_non_tenured_agents_but_include_supervisor(self):
        self._create_question("Author")
        self.assertTrue(self.agents["Voter1"].get(constants.AGENT_NOTIFICATIONS_PENDING_KEY))
        self.assertTrue(self.agents["Supervisor"].get(constants.AGENT_NOTIFICATIONS_PENDING_KEY))
        self.assertFalse(self.agents["Mature"].get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []))

    def test_problem_votes_open_redact_and_later_votes_do_not_change_status(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        opened = self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        self.assertIn("Status changed from pending to open", opened[0])

        switched = self._vote("Voter3", constants.ACTION_QUESTION_DOWNVOTE, "1")
        self.assertIn("Net upvote: 1", switched[0])
        data = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        self.assertEqual(constants.QUESTION_STATUS_OPEN, data[constants.QUESTION_STATUS_KEY])

        self._create_question("Supervisor")
        self._vote("Voter1", constants.ACTION_QUESTION_DOWNVOTE, "2")
        self._vote("Voter2", constants.ACTION_QUESTION_DOWNVOTE, "2")
        redacted = self._vote("Voter3", constants.ACTION_QUESTION_DOWNVOTE, "2")
        self.assertIn("Status changed from pending to redacted", redacted[0])

    def test_problem_vote_notifies_question_author(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_DOWNVOTE, "1")

        notifications = self.agents["Author"][constants.AGENT_NOTIFICATIONS_PENDING_KEY]
        self.assertTrue(
            any(
                "Your question (Question #1) has been upvoted by Voter1; current total net upvote is 1."
                in message
                for message in notifications
            )
        )
        self.assertTrue(
            any(
                "Your question (Question #1) has been downvoted by Voter2; current total net upvote is 0."
                in message
                for message in notifications
            )
        )

    def test_supervisor_can_retire_and_unretire_question(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")

        denied, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_QUESTION_RETIRE,
            "1",
            None,
            self.context,
            101,
        )
        self.assertIn("Permission denied", denied[0])

        retired, _ = self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_QUESTION_RETIRE,
            "1",
            None,
            self.context,
            102,
        )
        self.assertEqual(["Question #1 retired from status open."], retired)
        data = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        self.assertEqual(constants.QUESTION_STATUS_RETIRED, data[constants.QUESTION_STATUS_KEY])
        self.assertEqual(constants.QUESTION_STATUS_OPEN, data[constants.QUESTION_STATUS_BEFORE_RETIREMENT_KEY])
        self.assertEqual("Supervisor", data[constants.QUESTION_RETIRED_BY_KEY])
        self.assertEqual(102, data[constants.QUESTION_RETIRED_AT_TICK_KEY])

        default_content = self.room._get_specific_room_content(self.agents["Voter1"], self.context, 102)
        self.assertNotIn("| 1 | Question by Author |", default_content)
        self.assertIn("retired questions: 1.", default_content)

        filtered, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_QUESTION_FILTER,
            constants.QUESTION_STATUS_RETIRED,
            None,
            self.context,
            102,
        )
        self.assertEqual(["Filtered questions by status: retired."], filtered)
        retired_content = self.room._get_specific_room_content(self.agents["Voter1"], self.context, 102)
        self.assertIn("| 1 | Question by Author |", retired_content)
        self.assertIn("| retired |", retired_content)

        reply, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "Should be rejected"},
            self.context,
            103,
        )
        self.assertIn("Permission denied", reply[0])
        vote = self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self.assertIn("Permission denied", vote[0])

        self._create_question("Author")

        unretired, _ = self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_QUESTION_UNRETIRE,
            "1",
            None,
            self.context,
            104,
        )
        self.assertEqual(["Question #1 unretired and restored to status open."], unretired)
        data = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        self.assertEqual(constants.QUESTION_STATUS_OPEN, data[constants.QUESTION_STATUS_KEY])

    def test_retire_can_record_rationale_on_pending_question(self):
        self._create_question("Author")

        denied_reply, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "Ordinary pending reply"},
            self.context,
            101,
        )
        self.assertIn("Permission denied", denied_reply[0])

        denied_supervisor_reply, _ = self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "Retiring because later work supersedes this question."},
            self.context,
            101,
        )
        self.assertIn("Permission denied", denied_supervisor_reply[0])

        retired, _ = self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_QUESTION_RETIRE,
            "1",
            {"content": "Retiring because later work supersedes this question."},
            self.context,
            101,
        )
        self.assertEqual(["Question #1 retired from status pending."], retired)
        data = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        self.assertEqual(constants.QUESTION_STATUS_RETIRED, data[constants.QUESTION_STATUS_KEY])
        self.assertEqual(2, len(data[constants.CAPSULE_MESSAGES_KEY]))
        rationale = data[constants.CAPSULE_MESSAGES_KEY][-1]
        self.assertEqual("Supervisor", rationale[constants.MESSAGE_AUTHOR_NAME_KEY])
        self.assertEqual(
            "Retiring because later work supersedes this question.",
            rationale[constants.MESSAGE_CONTENT_KEY],
        )
        author_notifications = self.agents["Author"].get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
        self.assertTrue(any("Retiring because later work supersedes" in item for item in author_notifications))

    def test_retire_ignores_empty_or_non_string_optional_content(self):
        for numeric_id, yaml_data in enumerate(
            (None, {}, {"content": ""}, {"content": ["not", "text"]}),
            start=1,
        ):
            with self.subTest(yaml_data=yaml_data):
                self._create_question("Author")
                retired, _ = self.room.handle_action(
                    self.agents["Supervisor"],
                    constants.ACTION_QUESTION_RETIRE,
                    str(numeric_id),
                    yaml_data,
                    self.context,
                    101,
                )
                self.assertEqual([f"Question #{numeric_id} retired from status pending."], retired)
                data = capsule.get_capsule(numeric_id, constants.CAPSULE_TYPE_QUESTION)
                self.assertEqual(constants.QUESTION_STATUS_RETIRED, data[constants.QUESTION_STATUS_KEY])
                self.assertEqual(1, len(data[constants.CAPSULE_MESSAGES_KEY]))

    def test_retirement_rationale_works_from_every_question_status(self):
        statuses = (
            constants.QUESTION_STATUS_PENDING,
            constants.QUESTION_STATUS_OPEN,
            constants.QUESTION_STATUS_REDACTED,
            constants.QUESTION_STATUS_SOLVED,
        )
        for numeric_id, status in enumerate(statuses, start=1):
            with self.subTest(status=status):
                self._create_question("Author")
                data = capsule.get_capsule(numeric_id, constants.CAPSULE_TYPE_QUESTION)
                data[constants.QUESTION_STATUS_KEY] = status
                self.room._save_question_capsule(data, numeric_id)

                retired, _ = self.room.handle_action(
                    self.agents["Supervisor"],
                    constants.ACTION_QUESTION_RETIRE,
                    str(numeric_id),
                    {"content": f"Reason for retiring a {status} question."},
                    self.context,
                    101,
                )

                self.assertEqual([f"Question #{numeric_id} retired from status {status}."], retired)
                data = capsule.get_capsule(numeric_id, constants.CAPSULE_TYPE_QUESTION)
                self.assertEqual(constants.QUESTION_STATUS_RETIRED, data[constants.QUESTION_STATUS_KEY])
                self.assertEqual(status, data[constants.QUESTION_STATUS_BEFORE_RETIREMENT_KEY])
                self.assertEqual(
                    f"Reason for retiring a {status} question.",
                    data[constants.CAPSULE_MESSAGES_KEY][-1][constants.MESSAGE_CONTENT_KEY],
                )

    def test_retiring_solved_question_preserves_solution_on_restore(self):
        self._create_question("Author")
        solved_question = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        solved_question[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_SOLVED
        solved_question[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = "question_1-2"
        self.room._save_question_capsule(solved_question, 1)

        self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_QUESTION_RETIRE,
            "1",
            None,
            self.context,
            101,
        )
        self.room.handle_action(
            self.agents["Supervisor"],
            constants.ACTION_QUESTION_UNRETIRE,
            "1",
            None,
            self.context,
            102,
        )

        data = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        self.assertEqual(constants.QUESTION_STATUS_SOLVED, data[constants.QUESTION_STATUS_KEY])
        self.assertEqual("question_1-2", data[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY])

    def test_solution_vote_solves_can_reopen_and_read_shows_net_upvotes(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")

        reply, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "A proposed solution"},
            self.context,
            101,
        )
        self.assertIn("Replied to capsule", reply[0])
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1-2")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1-2")
        solved = self._vote("Supervisor", constants.ACTION_QUESTION_UPVOTE, "1-2")
        self.assertIn("Status changed from open to solved", solved[0])

        read, _ = self.room.handle_action(
            self.agents["Voter2"],
            constants.ACTION_CAPSULE_READ,
            "1",
            None,
            self.context,
            102,
        )
        self.assertIn("Read command processed", read[0])
        notification = self.agents["Voter2"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("Status: solved, Net Upvote: 3", notification)
        self.assertIn("Solved By: #1-2", notification)
        self.assertIn("**Message question_1-2 [Accepted Solution]**", notification)
        self.assertIn("Solution Net Upvote: 3", notification)

        content = self.room._get_specific_room_content(self.agents["Voter2"], self.context, 102)
        self.assertIn("solved (#1-2)", content)

        direct_read, _ = self.room.handle_action(
            self.agents["Voter3"],
            constants.ACTION_CAPSULE_READ,
            "1-2",
            None,
            self.context,
            102,
        )
        self.assertIn("Read command processed", direct_read[0])
        direct_notification = self.agents["Voter3"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("**Message question_1-2 [Accepted Solution] from Question", direct_notification)
        self.assertIn("Solved By: #1-2", direct_notification)

        reopened = self._vote("Supervisor", constants.ACTION_QUESTION_DOWNVOTE, "1-2")
        self.assertIn("Status changed from solved to open", reopened[0])
        data = capsule.get_capsule(1, constants.CAPSULE_TYPE_QUESTION)
        self.assertEqual(constants.QUESTION_STATUS_OPEN, data[constants.QUESTION_STATUS_KEY])
        self.assertIsNone(data[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY])
        self.assertEqual(1, data[constants.CAPSULE_MESSAGES_KEY][1][constants.QUESTION_SOLUTION_NET_UPVOTE_KEY])

    def test_read_question_adds_general_voting_footer_for_non_author_unsolved_question(self):
        self._create_question("Author")

        read, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_READ,
            "1",
            None,
            self.context,
            100,
        )

        self.assertIn("Read command processed", read[0])
        notification = self.agents["Voter1"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("**Voting obligation:**", notification)
        self.assertIn("After reading, vote up or down on the question.", notification)
        self.assertIn("If any reply above is a solution attempt", notification)
        self.assertIn("Do not withhold a downvote out of politeness", notification)

    def test_read_message_range_adds_general_voting_footer_for_non_author_unsolved_question(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        for index in range(3):
            reply, _ = self.room.handle_action(
                self.agents["Voter1"],
                constants.ACTION_CAPSULE_REPLY,
                "1",
                {"content": f"Solution Attempt {index}"},
                self.context,
                101 + index,
            )
            self.assertIn("Replied to capsule", reply[0])

        read, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_READ,
            "1-3:1-5",
            None,
            self.context,
            105,
        )

        self.assertIn("Read command processed", read[0])
        notification = self.agents["Voter1"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("**Voting obligation:**", notification)
        self.assertEqual(1, notification.count("**Voting obligation:**"))
        self.assertNotIn("After reading, vote up or down on the question.", notification)
        self.assertIn("Message question_1-3", notification)
        self.assertIn("Message question_1-5", notification)

    def test_read_question_adds_solution_only_footer_for_author_or_existing_problem_voter(self):
        self._create_question("Author")

        read_author, _ = self.room.handle_action(
            self.agents["Author"],
            constants.ACTION_CAPSULE_READ,
            "1",
            None,
            self.context,
            100,
        )
        self.assertIn("Read command processed", read_author[0])
        author_notification = self.agents["Author"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("**Voting obligation:**", author_notification)
        self.assertIn("If any reply above is a solution attempt", author_notification)
        self.assertNotIn("After reading, vote up or down on the question.", author_notification)

        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        read_voter, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_READ,
            "1",
            None,
            self.context,
            100,
        )
        self.assertIn("Read command processed", read_voter[0])
        voter_notification = self.agents["Voter1"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("**Voting obligation:**", voter_notification)
        self.assertIn("If any reply above is a solution attempt", voter_notification)
        self.assertNotIn("After reading, vote up or down on the question.", voter_notification)

    def test_read_question_omits_general_voting_footer_for_solved_question(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        reply, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "Solution Attempt: exact proof sketch"},
            self.context,
            101,
        )
        self.assertIn("Replied to capsule", reply[0])
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1-2")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1-2")
        self._vote("Supervisor", constants.ACTION_QUESTION_UPVOTE, "1-2")

        read, _ = self.room.handle_action(
            self.agents["Voter2"],
            constants.ACTION_CAPSULE_READ,
            "1",
            None,
            self.context,
            102,
        )

        self.assertIn("Read command processed", read[0])
        notification = self.agents["Voter2"][constants.AGENT_NOTIFICATIONS_PENDING_KEY][-1]
        self.assertIn("Status: solved", notification)
        self.assertNotIn("**Voting obligation:**", notification)

    def test_solution_vote_notifies_reply_author(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        reply, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "A proposed solution"},
            self.context,
            101,
        )
        self.assertIn("Replied to capsule", reply[0])

        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1-2")
        self._vote("Voter3", constants.ACTION_QUESTION_DOWNVOTE, "1-2")

        notifications = self.agents["Voter1"][constants.AGENT_NOTIFICATIONS_PENDING_KEY]
        self.assertTrue(
            any(
                "Your reply (Question #1-2) has been upvoted by Voter2; current total net upvote is 1."
                in message
                for message in notifications
            )
        )
        self.assertTrue(
            any(
                "Your reply (Question #1-2) has been downvoted by Voter3; current total net upvote is 0."
                in message
                for message in notifications
            )
        )

    def test_reply_notification_includes_content_and_marks_read_for_thread_participant(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        reply_content = "Intermediate progress for the original author."

        reply, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": reply_content},
            self.context,
            101,
        )

        self.assertIn("Replied to capsule", reply[0])
        notifications = self.agents["Author"][constants.AGENT_NOTIFICATIONS_PENDING_KEY]
        self.assertTrue(any(reply_content in message for message in notifications))
        read_status = self.agents["Author"][constants.SHORT_ROOM_NAME_QUESTION][
            constants.AGENT_ROOM_STATE_READ_STATUS_KEY
        ]
        self.assertTrue(read_status["question_1-2"])

    def test_problem_author_reply_cannot_receive_solution_votes(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        reply, _ = self.room.handle_action(
            self.agents["Author"],
            constants.ACTION_CAPSULE_REPLY,
            "1",
            {"content": "Author clarification or proposed idea"},
            self.context,
            101,
        )
        self.assertIn("Replied to capsule", reply[0])
        denied = self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1-2")
        self.assertIn("original problem author", denied[0])

    def test_filter_and_rank(self):
        self._create_question("Author")
        filtered, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_QUESTION_FILTER,
            constants.QUESTION_STATUS_PENDING,
            None,
            self.context,
            100,
        )
        self.assertEqual(["Filtered questions by status: pending."], filtered)
        ranked, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_QUESTION_RANK,
            "upvote",
            None,
            self.context,
            100,
        )
        self.assertEqual(["Questions ranked by net upvote."], ranked)
        content = self.room._get_specific_room_content(self.agents["Voter1"], self.context, 100)
        self.assertIn("filter: pending", content)
        self.assertIn("rank: upvote", content)
        reset_ranked, _ = self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_QUESTION_RANK,
            "id",
            None,
            self.context,
            100,
        )
        self.assertEqual(["Questions ranked by default ID order."], reset_ranked)
        content = self.room._get_specific_room_content(self.agents["Voter1"], self.context, 100)
        self.assertIn("filter: pending", content)
        self.assertNotIn("rank: upvote", content)

    def test_room_table_shows_status_summary(self):
        self._create_question("Author")
        self._vote("Voter1", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter2", constants.ACTION_QUESTION_UPVOTE, "1")
        self._vote("Voter3", constants.ACTION_QUESTION_UPVOTE, "1")
        self._create_question("Supervisor")
        self._create_question("Supervisor")

        solved_question = capsule.get_capsule(3, constants.CAPSULE_TYPE_QUESTION)
        solved_question[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_SOLVED
        solved_question[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = "question_3-2"
        self.room._save_question_capsule(solved_question, 3)

        self.room.handle_action(
            self.agents["Voter1"],
            constants.ACTION_QUESTION_FILTER,
            constants.QUESTION_STATUS_PENDING,
            None,
            self.context,
            100,
        )
        content = self.room._get_specific_room_content(self.agents["Voter1"], self.context, 100)

        self.assertIn("| 2 | Question by Supervisor |", content)
        self.assertNotIn("| 1 | Question by Author |", content)
        self.assertIn(
            "Current number of open questions: 1; pending questions: 1; solved questions: 1; retired questions: 0.",
            content,
        )

        old_page_size = constants.DEFAULT_PAGE_SIZE_CAPSULES
        constants.DEFAULT_PAGE_SIZE_CAPSULES = 1
        try:
            self.room.handle_action(
                self.agents["Voter1"],
                constants.ACTION_QUESTION_UNFILTER,
                None,
                None,
                self.context,
                100,
            )
            paginated_content = self.room._get_specific_room_content(self.agents["Voter1"], self.context, 100)
        finally:
            constants.DEFAULT_PAGE_SIZE_CAPSULES = old_page_size
        page_hint_index = paginated_content.index("(Use `/execute_action{page N}`")
        summary_index = paginated_content.index("Current number of open questions:")
        self.assertLess(page_hint_index, summary_index)


if __name__ == "__main__":
    unittest.main()
