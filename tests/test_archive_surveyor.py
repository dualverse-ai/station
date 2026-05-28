from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import types
import time
import unittest
from unittest import mock

from station import constants
from station import capsule as capsule_module
from station import file_io_utils
from station import index_paths
from station.action_parser import ActionParser
from station.base_room import RoomContext
from station.rooms.archive import ArchiveRoom
from station.rooms.mail import MailRoom
from station.eval_archive.surveyor import (
    ActiveSurveySession,
    ArchiveSurveySubmissionService,
    AutoArchiveSurveyor,
    ensure_archive_surveyor_layout,
    mark_archive_survey_committed,
    queue_archive_survey_request,
    rollback_provisional_archive_surveys,
)
from station.eval_research.runtime_paths import ensure_runtime_layout
from station.sync.parallel_state import ParallelTickState


class _AgentModuleStub:
    def __init__(self):
        self.messages = []
        self.agents = {}

    def load_agent_data(self, author):
        return self.agents.setdefault(author, {constants.AGENT_NAME_KEY: author})

    def save_agent_data(self, author, agent_data):
        self.agents[author] = agent_data
        return True

    def add_pending_notification(self, agent_data, message):
        agent_data.setdefault(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []).append(message)
        self.messages.append((agent_data.get(constants.AGENT_NAME_KEY), message))

    def add_pending_notification_atomic(
        self,
        author,
        message,
        protection_reason=None,
        protection_source="",
    ):
        def update_func(agent_data):
            agent_data.setdefault(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []).append(message)

        return self.update_agent_with_function(author, update_func)

    def update_agent_with_function(self, author, update_func):
        agent_data = self.agents.setdefault(author, {constants.AGENT_NAME_KEY: author})
        before = list(agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, []))
        update_func(agent_data)
        after = agent_data.get(constants.AGENT_NOTIFICATIONS_PENDING_KEY, [])
        if isinstance(after, list):
            for message in after[len(before):]:
                self.messages.append((author, message))
        return True

    def get_agent_room_state(self, agent_data, room_key, state_key, default=None):
        if room_key in agent_data and isinstance(agent_data[room_key], dict):
            return agent_data[room_key].get(state_key, default)
        return default

    def set_agent_room_state(self, agent_data, room_key, state_key, value):
        agent_data.setdefault(room_key, {})[state_key] = value


class _MailAgentManagerStub:
    def __init__(self, agents):
        self.agents = agents
        self.room_state = {}
        self.update_calls = 0

    def load_agent_data(self, name):
        return self.agents.get(name)

    def get_agent_room_state(self, agent_data, room_key, state_key, default=None):
        return self.room_state.get((agent_data.get(constants.AGENT_NAME_KEY), room_key, state_key), default)

    def set_agent_room_state(self, agent_data, room_key, state_key, value):
        self.room_state[(agent_data.get(constants.AGENT_NAME_KEY), room_key, state_key)] = value

    def add_pending_notification(self, agent_data, message):
        agent_data.setdefault("pending_notifications", []).append(message)

    def save_agent_data(self, name, agent_data):
        self.agents[name] = agent_data

    def update_agent_with_function(self, name, update_func):
        agent_data = self.agents.get(name)
        if not agent_data:
            return False
        self.update_calls += 1
        update_func(agent_data)
        self.agents[name] = agent_data
        return True


class ArchiveSurveyorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._original_values = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "ARCHIVE_SURVEY_ENABLED": constants.ARCHIVE_SURVEY_ENABLED,
            "ARCHIVE_SURVEY_MAX_ACTIVE_PER_AGENT": constants.ARCHIVE_SURVEY_MAX_ACTIVE_PER_AGENT,
            "PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS": constants.PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS,
            "CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS": constants.CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS,
        }
        self._original_actions = set(constants.ACTIONS_EXPECTING_YAML)
        constants.BASE_STATION_DATA_PATH = os.path.join(self.tmp.name, "station_data")
        constants.ARCHIVE_SURVEY_ENABLED = True
        constants.ARCHIVE_SURVEY_MAX_ACTIVE_PER_AGENT = 1
        constants.PARALLEL_ARCHIVE_SURVEY_SUBMISSION_TIMEOUT_SECONDS = 0.0
        constants.CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS = 1800
        constants._refresh_dynamic_action_sets()

    def tearDown(self):
        for key, value in self._original_values.items():
            setattr(constants, key, value)
        constants.ACTIONS_EXPECTING_YAML.clear()
        constants.ACTIONS_EXPECTING_YAML.update(self._original_actions)
        self.tmp.cleanup()

    def _room_context(self):
        station = types.SimpleNamespace(
            _is_agent_mature=lambda agent_data, tick: True,
            _get_current_tick=lambda: 10,
            auto_archive_surveyor=None,
        )
        return RoomContext(
            agent_manager=types.SimpleNamespace(),
            capsule_manager=types.SimpleNamespace(),
            notification_manager=None,
            constants_module=constants,
            station_instance=station,
        )

    def _surveyor_station(self, agent_module):
        station = types.SimpleNamespace(agent_module=agent_module, _get_current_tick=lambda: 5)
        station.rooms = {constants.ROOM_MAIL: MailRoom()}
        station.room_context = RoomContext(
            agent_manager=agent_module,
            capsule_manager=types.SimpleNamespace(file_io_utils=file_io_utils),
            notification_manager=None,
            constants_module=constants,
            station_instance=station,
        )
        return station

    def _write_archive_paper(self, archive_id: int = 1, deleted: bool = False):
        archive_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.CAPSULES_DIR_NAME,
            constants.ARCHIVE_CAPSULES_SUBDIR_NAME,
        )
        file_io_utils.ensure_dir_exists(archive_dir)
        file_io_utils.save_yaml(
            {
                constants.CAPSULE_TITLE_KEY: "Existing Archive Paper",
                constants.CAPSULE_AUTHOR_NAME_KEY: "Axiom I",
                constants.CAPSULE_CREATED_AT_TICK_KEY: 1,
                constants.CAPSULE_ABSTRACT_KEY: "A published archive paper for survey tests.",
                constants.CAPSULE_IS_DELETED_KEY: deleted,
            },
            os.path.join(archive_dir, f"archive_{archive_id}.yaml"),
            sort_keys=False,
        )

    def test_feature_flag_controls_yaml_parser_registration(self):
        constants.ARCHIVE_SURVEY_ENABLED = False
        constants._refresh_dynamic_action_sets()
        self.assertNotIn(constants.ACTION_ARCHIVE_SURVEY, constants.ACTIONS_EXPECTING_YAML)

        actions = ActionParser().parse(
            "/execute_action{survey}\n"
            "```yaml\n"
            "prompt: test\n"
            "```"
        )
        self.assertEqual(actions[0].command, "survey")
        self.assertIsNone(actions[0].yaml_data)

        constants.ARCHIVE_SURVEY_ENABLED = True
        constants._refresh_dynamic_action_sets()
        self.assertIn(constants.ACTION_ARCHIVE_SURVEY, constants.ACTIONS_EXPECTING_YAML)

    def test_archive_room_queues_survey_request(self):
        self._write_archive_paper()
        room = ArchiveRoom()
        agent_data = {
            constants.AGENT_NAME_KEY: "Axiom I",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: "axiom",
        }

        actions, handler = room.handle_action(
            agent_data,
            constants.ACTION_ARCHIVE_SURVEY,
            None,
            {constants.YAML_ARCHIVE_SURVEY_PROMPT: "What has been tried?"},
            self._room_context(),
            42,
        )

        self.assertIsNone(handler)
        self.assertIn("Archive survey request queued", actions[0])
        paths = ensure_archive_surveyor_layout()
        request = file_io_utils.load_yaml(os.path.join(paths.requests_dir, "survey_1.yaml"))
        self.assertEqual(request["author"], "Axiom I")
        self.assertEqual(request["prompt"], "What has been tried?")
        pending = file_io_utils.load_yaml_lines(paths.pending_file)
        self.assertEqual(str(pending[0]["id"]), "1")

    def test_archive_room_rejects_survey_when_archive_is_empty(self):
        room = ArchiveRoom()
        agent_data = {
            constants.AGENT_NAME_KEY: "Axiom I",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: "axiom",
        }

        actions, handler = room.handle_action(
            agent_data,
            constants.ACTION_ARCHIVE_SURVEY,
            None,
            {constants.YAML_ARCHIVE_SURVEY_PROMPT: "What has been tried?"},
            self._room_context(),
            42,
        )

        self.assertIsNone(handler)
        self.assertIn("Archive survey failed: no archive papers found, so no survey is needed.", actions[0])
        paths = ensure_archive_surveyor_layout()
        self.assertFalse(file_io_utils.list_files(paths.requests_dir, constants.YAML_EXTENSION))

    def test_layout_writes_local_agents_md_for_surveyor_sessions(self):
        paths = ensure_archive_surveyor_layout()
        agents_md = file_io_utils.load_text(os.path.join(paths.surveyor_root, "AGENTS.md"))

        self.assertEqual(
            agents_md,
            "Follow the initial Archive Surveyor prompt as your authority; this repository is for Station "
            "development, and you do not need to access its source code or developer docs to perform your "
            "surveyor function.\n",
        )

    def test_disabled_survey_does_not_change_archive_help_or_queue(self):
        constants.ARCHIVE_SURVEY_ENABLED = False
        constants._refresh_dynamic_action_sets()
        room = ArchiveRoom()
        agent_data = {
            constants.AGENT_NAME_KEY: "Axiom I",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: "axiom",
        }
        help_text = room.get_help_message(agent_data, self._room_context())
        self.assertNotIn("Archive Surveyor", help_text)
        self.assertNotIn("/execute_action{survey}", help_text)

        actions, handler = room.handle_action(
            agent_data,
            constants.ACTION_ARCHIVE_SURVEY,
            None,
            {constants.YAML_ARCHIVE_SURVEY_PROMPT: "What has been tried?"},
            self._room_context(),
            42,
        )

        self.assertIsNone(handler)
        self.assertTrue(any("Unknown action" in action or "not recognized" in action for action in actions))
        paths = ensure_archive_surveyor_layout()
        self.assertFalse(file_io_utils.list_files(paths.requests_dir, constants.YAML_EXTENSION))

    def test_survey_help_appears_before_archive_help_footer(self):
        room = ArchiveRoom()
        agent_data = {
            constants.AGENT_NAME_KEY: "Axiom I",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: "axiom",
        }
        help_text = room.get_help_message(agent_data, self._room_context())
        footer = "To display this help message again at any time from any room, issue `/execute_action{help archive}`."

        self.assertIn("**Archive Surveyor**", help_text)
        self.assertIn("evidence-synthesis agent", help_text)
        self.assertIn("It may identify gaps, tensions, and underexplored areas", help_text)
        self.assertIn("idea generation and final research direction selection are your responsibility", help_text)
        self.assertNotIn("Brainstorming novel ideas", help_text)
        self.assertNotIn("generate a few novel ideas", help_text)
        self.assertLess(help_text.index("**Archive Surveyor**"), help_text.index(footer))
        self.assertEqual(help_text.find(footer), help_text.rfind(footer))
        self.assertTrue(help_text.strip().endswith(footer))

    def test_prompt_uses_final_survey_contract(self):
        paths = ensure_archive_surveyor_layout()
        file_io_utils.save_text("Task spec", os.path.join(constants.BASE_STATION_DATA_PATH, "rooms", "research", "research_task.md"))
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Find duplicate work.",
            tick=1,
        )
        surveyor = AutoArchiveSurveyor(
            types.SimpleNamespace(agent_module=_AgentModuleStub(), _get_current_tick=lambda: 2),
            enabled=False,
        )
        prompt = surveyor._build_prompt(request)

        self.assertIn("reports/1.draft.md", prompt)
        self.assertIn("You are a PhD-level evidence-synthesis researcher", prompt)
        self.assertIn("Role boundary:", prompt)
        self.assertIn("The requesting agent is responsible for proposing ideas", prompt)
        self.assertIn("state that you are not allowed to do so", prompt)
        self.assertIn("still answer any valid archive-synthesis or gap-analysis parts", prompt)
        self.assertIn("Normal work cycle:", prompt)
        self.assertIn("cat archive_papers/archive_{ID}.yaml", prompt)
        self.assertIn('bash research_center/eval_tool.sh search "keyword1|keyword2"', prompt)
        self.assertIn("case-insensitive Python regex against abstracts only", prompt)
        self.assertIn("(?=.*keyword1)(?=.*keyword2)", prompt)
        self.assertIn("even when they were not cited by any archive paper", prompt)
        self.assertIn("For a general request such as a broad Station landscape survey", prompt)
        self.assertIn("bash research_center/eval_tool.sh preview {ID}", prompt)
        self.assertIn("mv reports/1.draft.md reports/1.md", prompt)
        self.assertIn("1000 to 5000 words", prompt)
        self.assertIn("Guidelines:", prompt)
        self.assertIn("help the agent understand the accumulated knowledge", prompt)
        self.assertIn("synthesize strategic and technical context", prompt)
        self.assertIn("Preserve the agent's responsibility for idea generation", prompt)
        self.assertIn("Do not propose any new idea", prompt)
        self.assertIn("Do not over-claim", prompt)
        self.assertIn("After the rename, do not modify either file. Exit the session.", prompt)
        self.assertIn("## Main Content", prompt)
        self.assertIn("evidence gaps", prompt)
        self.assertIn("Only read raw code and artifacts when there are ambiguities or errors in parsing.", prompt)
        self.assertNotIn("brainstorming ideas", prompt)
        self.assertNotIn("novel ideas", prompt)
        self.assertNotIn("new paradigm", prompt)
        self.assertNotIn("Report finalization rule:", prompt)
        self.assertNotIn("Use archive previews for broad orientation", prompt)
        self.assertNotIn("For Research Center evaluations, start with", prompt)
        self.assertNotIn("## Relevant Archive Papers", prompt)
        self.assertNotIn("Good survey reports should", prompt)

    def test_eval_tool_previews_instruction_prompt_and_report_without_raw_code(self):
        paths = ensure_runtime_layout()
        eval_id = "7"
        session_id = "codex_7_spawn_1_deadbeef"
        file_io_utils.save_yaml(
            {
                "id": eval_id,
                "title": "Previewable Evaluation",
                "author": "Axiom I",
                "lineage": "axiom",
                "tags": ["data", "conjecture"],
                "abstract": "A data-driven conjecture search abstract.",
                "status": "completed",
                "submitted_tick": 3,
                "completed_tick": 4,
                "instruction": "Use a data-driven conjecture search.",
                "coder": {"session_id": session_id},
                "final": {
                    "primary_score": 0.42,
                    constants.EVALUATION_DETAILS_KEY: {
                        "Message": "ok",
                        "secondary_metric": 17,
                    },
                },
                "artifacts": {
                    "report": os.path.join(constants.RESEARCH_STORAGE_DIR, "report", f"{eval_id}.md"),
                },
            },
            os.path.join(paths.evaluations_dir, f"{eval_id}{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"),
            sort_keys=False,
        )
        session_dir = os.path.join(paths.coder_sessions_dir, session_id)
        file_io_utils.ensure_dir_exists(session_dir)
        file_io_utils.save_text("Exact coder prompt.", os.path.join(session_dir, "prompt.txt"))
        file_io_utils.save_text("Final Coder Report body.", os.path.join(paths.reports_dir, f"{eval_id}.md"))
        file_io_utils.save_text("RAW CODE SHOULD NOT PRINT", os.path.join(paths.submissions_dir, f"{eval_id}.py"))
        file_io_utils.save_text("STDOUT SHOULD NOT PRINT", os.path.join(paths.stdout_dir, f"{eval_id}.log"))

        result = subprocess.run(
            ["bash", paths.eval_tool_script_path, "preview", eval_id],
            cwd=paths.research_root,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Research Evaluation Preview #7", result.stdout)
        self.assertIn("- Title: Previewable Evaluation", result.stdout)
        self.assertIn("- Author: Axiom I", result.stdout)
        self.assertIn("- Tags: data, conjecture", result.stdout)
        self.assertIn("- Submitted tick: 3", result.stdout)
        self.assertIn("- Completed tick: 4", result.stdout)
        self.assertIn("- Final score: 0.42", result.stdout)
        self.assertIn("- Secondary metrics: secondary_metric=17", result.stdout)
        self.assertIn("A data-driven conjecture search abstract.", result.stdout)
        self.assertIn("Use a data-driven conjecture search.", result.stdout)
        self.assertIn("Exact coder prompt.", result.stdout)
        self.assertIn("Final Coder Report body.", result.stdout)
        self.assertNotIn("RAW CODE SHOULD NOT PRINT", result.stdout)
        self.assertNotIn("STDOUT SHOULD NOT PRINT", result.stdout)

    def test_eval_tool_search_filters_abstracts_only_with_regex(self):
        paths = ensure_runtime_layout()
        file_io_utils.save_yaml(
            {
                "id": "7",
                "title": "Graph Neural Ramsey Search",
                "author": "Axiom I",
                "abstract": "Uses neural guidance for Ramsey-style conjecture mining.",
                "instruction": "No hidden keyword here.",
            },
            os.path.join(paths.evaluations_dir, f"7{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"),
            sort_keys=False,
        )
        file_io_utils.save_yaml(
            {
                "id": "8",
                "title": "Instruction Only Match",
                "author": "Axiom I",
                "abstract": "This abstract is unrelated.",
                "instruction": "Uses neural guidance for Ramsey-style conjecture mining.",
            },
            os.path.join(paths.evaluations_dir, f"8{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"),
            sort_keys=False,
        )
        file_io_utils.save_yaml(
            {
                "id": "9",
                "title": "Spectral Certificate",
                "author": "Axiom I",
                "abstract": "Studies spectral certificates for finite structures.",
                "instruction": "No hidden keyword here.",
            },
            os.path.join(paths.evaluations_dir, f"9{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"),
            sort_keys=False,
        )

        result = subprocess.run(
            ["bash", paths.eval_tool_script_path, "search", "neural|spectral"],
            cwd=paths.research_root,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Research Evaluation Abstract Search", result.stdout)
        self.assertIn("Scope: evaluation abstracts only", result.stdout)
        self.assertIn("## Eval #7: Graph Neural Ramsey Search", result.stdout)
        self.assertIn("## Eval #9: Spectral Certificate", result.stdout)
        self.assertNotIn("## Eval #8: Instruction Only Match", result.stdout)
        self.assertNotIn("- Author:", result.stdout)
        self.assertNotIn("No hidden keyword here.", result.stdout)

        and_result = subprocess.run(
            ["bash", paths.eval_tool_script_path, "search", "(?=.*neural)(?=.*Ramsey)"],
            cwd=paths.research_root,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(and_result.returncode, 0, and_result.stderr)
        self.assertIn("## Eval #7: Graph Neural Ramsey Search", and_result.stdout)
        self.assertNotIn("## Eval #9: Spectral Certificate", and_result.stdout)

    def test_eval_tool_search_uses_checkout_imports_from_surveyor_workspace(self):
        paths = ensure_runtime_layout()
        surveyor_paths = ensure_archive_surveyor_layout()
        file_io_utils.save_yaml(
            {
                "id": "11",
                "title": "Surveyor Search Target",
                "author": "Axiom I",
                "abstract": "Needle term visible only through the checkout-backed index.",
                "instruction": "No hidden keyword here.",
            },
            os.path.join(paths.evaluations_dir, f"11{constants.RESEARCH_EVALUATION_FILE_EXTENSION}"),
            sort_keys=False,
        )

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            ["bash", "research_center/eval_tool.sh", "search", "needle"],
            cwd=surveyor_paths.surveyor_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("## Eval #11: Surveyor Search Target", result.stdout)
        script_text = file_io_utils.load_text(paths.eval_tool_script_path)
        self.assertIn("_internal/eval_tool_cli_snapshot.py", script_text)
        self.assertNotIn("station.eval_research", script_text)
        snapshot_text = file_io_utils.load_text(
            os.path.join(paths.research_root, "_internal", "eval_tool_cli_snapshot.py")
        )
        self.assertIn("STATION_REPO_ROOT", snapshot_text)

    def test_completed_report_creates_mail_and_full_notification(self):
        agent_module = _AgentModuleStub()
        station = self._surveyor_station(agent_module)
        surveyor = AutoArchiveSurveyor(station, enabled=False)
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=1,
        )
        survey_id = str(request["id"])
        report_text = "# Archive Survey Report #1\n\nFull report body."
        file_io_utils.save_text(report_text, os.path.join(surveyor.paths.reports_dir, f"{survey_id}.md"))
        surveyor._mark_completed(survey_id, exit_code=0, error=None)

        self.assertTrue(surveyor._deliver_report_if_needed(survey_id))
        self.assertEqual(len(agent_module.messages), 1)
        self.assertEqual(agent_module.messages[0][0], "Axiom I")
        self.assertIn(report_text, agent_module.messages[0][1])
        self.assertIn("This is the full survey report.", agent_module.messages[0][1])
        self.assertIn("Please do not reply to this mail", agent_module.messages[0][1])
        self.assertIn("You are the only recipient of this report.", agent_module.messages[0][1])
        self.assertIn("Other agents cannot read this mail", agent_module.messages[0][1])
        self.assertIn("To reply, use:", agent_module.messages[0][1])
        read_status = agent_module.agents["Axiom I"][constants.SHORT_ROOM_NAME_MAIL][
            constants.AGENT_ROOM_STATE_READ_STATUS_KEY
        ]
        self.assertTrue(read_status["mail_1"])
        self.assertTrue(read_status["mail_1-1"])
        mail_files = file_io_utils.list_files(
            os.path.join(constants.BASE_STATION_DATA_PATH, constants.CAPSULES_DIR_NAME, constants.MAIL_CAPSULES_SUBDIR_NAME),
            constants.YAML_EXTENSION,
        )
        self.assertEqual(mail_files, ["mail_1.yaml"])
        mail_capsule = capsule_module.get_capsule(1, constants.CAPSULE_TYPE_MAIL, None)
        mail_content = mail_capsule[constants.CAPSULE_MESSAGES_KEY][0][constants.MESSAGE_CONTENT_KEY]
        self.assertIn("This is the full survey report.", mail_content)
        self.assertIn("Please do not reply to this mail", mail_content)
        self.assertIn(report_text, mail_content)
        self.assertLess(mail_content.index("Survey Report Notice"), mail_content.index("# Archive Survey Report #1"))

    def test_repair_sent_survey_notification_marks_mail_read(self):
        agent_module = _AgentModuleStub()
        station = self._surveyor_station(agent_module)
        surveyor = AutoArchiveSurveyor(station, enabled=False)
        mail_result = surveyor._create_mail_capsule("Axiom I", "1", "# Archive Survey Report #1\n\nBody.")
        self.assertIsNotNone(mail_result)
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=1,
        )
        survey_id = str(request["id"])

        def mark_sent(record):
            record["status"] = "completed"
            notification = record.setdefault("notification", {})
            notification["sent"] = True
            notification["mail_capsule_id"] = mail_result["capsule_id"]
            notification["mail_numeric_id"] = mail_result["numeric_id"]

        surveyor._update_request(survey_id, mark_sent)

        surveyor._repair_pending_notifications()

        read_status = agent_module.agents["Axiom I"][constants.SHORT_ROOM_NAME_MAIL][
            constants.AGENT_ROOM_STATE_READ_STATUS_KEY
        ]
        self.assertTrue(read_status["mail_1"])
        self.assertTrue(read_status["mail_1-1"])

    def test_mail_room_delivery_uses_atomic_update_for_recipient_state(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Axiom I",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
        }
        agent_manager = _MailAgentManagerStub({"Axiom I": agent_data})
        room_context = RoomContext(
            agent_manager=agent_manager,
            capsule_manager=types.SimpleNamespace(file_io_utils=file_io_utils),
            notification_manager=None,
            constants_module=constants,
            station_instance=types.SimpleNamespace(
                _is_agent_mature=lambda candidate, tick: True,
                _get_current_tick=lambda: 5,
            ),
        )

        delivered = MailRoom()._deliver_mail_notification(
            "Axiom I",
            "New mail body.",
            room_context,
            read_item_ids=["mail_9", "mail_9-1"],
        )

        self.assertTrue(delivered)
        self.assertEqual(agent_manager.update_calls, 1)
        self.assertEqual(agent_data[constants.AGENT_NOTIFICATIONS_PENDING_KEY], ["New mail body."])
        read_status = agent_manager.room_state[
            ("Axiom I", constants.SHORT_ROOM_NAME_MAIL, constants.AGENT_ROOM_STATE_READ_STATUS_KEY)
        ]
        self.assertTrue(read_status["mail_9"])
        self.assertTrue(read_status["mail_9-1"])

    def test_reply_to_archive_surveyor_mail_fails_without_appending_reply(self):
        agent_data = {
            constants.AGENT_NAME_KEY: "Axiom I",
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_LINEAGE_KEY: "axiom",
            constants.AGENT_GENERATION_KEY: 1,
        }
        station = types.SimpleNamespace(_get_current_tick=lambda: 5)
        surveyor = AutoArchiveSurveyor(station, enabled=False)
        mail_result = surveyor._create_mail_capsule("Axiom I", "1", "# Archive Survey Report #1\n\nBody.")
        self.assertIsNotNone(mail_result)
        mail_id = int(mail_result["numeric_id"])

        agent_manager = _MailAgentManagerStub({"Axiom I": agent_data})
        room_context = RoomContext(
            agent_manager=agent_manager,
            capsule_manager=types.SimpleNamespace(file_io_utils=file_io_utils),
            notification_manager=None,
            constants_module=constants,
            station_instance=types.SimpleNamespace(
                _is_agent_mature=lambda candidate, tick: True,
                _get_current_tick=lambda: 5,
            ),
        )

        actions, handler = MailRoom().handle_action(
            agent_data,
            constants.ACTION_CAPSULE_REPLY,
            str(mail_id),
            {constants.YAML_CAPSULE_CONTENT: "Thanks for the report."},
            room_context,
            5,
        )

        self.assertIsNone(handler)
        self.assertIn("Reply failed: invalid recipient(s): Archive Surveyor", actions[0])
        mail_capsule = capsule_module.get_capsule(mail_id, constants.CAPSULE_TYPE_MAIL, None)
        self.assertEqual(len(mail_capsule[constants.CAPSULE_MESSAGES_KEY]), 1)

    def test_running_survey_waits_at_tick_limit_and_appears_as_job(self):
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=10,
        )
        survey_id = str(request["id"])
        surveyor = AutoArchiveSurveyor(
            types.SimpleNamespace(agent_module=_AgentModuleStub(), _get_current_tick=lambda: 11),
            enabled=False,
        )
        surveyor._claim_launch(survey_id, "codex_1_spawn_1_deadbeef", "codex", None)

        self.assertTrue(surveyor.should_wait_at_tick(11))
        stats = surveyor.get_job_statistics()
        self.assertEqual(stats["running_count"], 1)
        self.assertEqual(stats["queued_count"], 0)
        self.assertEqual(stats["running_jobs"][0]["job_type"], "archive_survey")
        self.assertEqual(stats["running_jobs"][0]["evaluation_id"], "Archive Survey #1")

    def test_archive_surveyor_kills_and_requeues_idle_codex_transcript(self):
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=10,
        )
        survey_id = str(request["id"])
        station = types.SimpleNamespace(agent_module=_AgentModuleStub(), _get_current_tick=lambda: 11)
        surveyor = AutoArchiveSurveyor(station, enabled=False)
        surveyor._claim_launch(survey_id, "codex_1_spawn_1_deadbeef", "codex", None)
        run_dir = os.path.join(surveyor.paths.sessions_dir, "codex_1_spawn_1_deadbeef")
        file_io_utils.ensure_dir_exists(run_dir)
        transcript_path = os.path.join(run_dir, "transcript.jsonl")
        stderr_path = os.path.join(run_dir, "stderr.txt")
        file_io_utils.save_text('{"type":"thread.started"}\n', transcript_path)
        file_io_utils.save_text("", stderr_path)

        class DummyProcess:
            pid = 5432
            returncode = None

            def poll(self):
                return self.returncode

        class DummyHandle:
            def close(self):
                pass

        process = DummyProcess()
        session = ActiveSurveySession(
            survey_id=survey_id,
            session_id="codex_1_spawn_1_deadbeef",
            run_dir=run_dir,
            backend="codex",
            transcript_format="jsonl",
            process=process,
            transcript_handle=DummyHandle(),
            stderr_handle=DummyHandle(),
            prompt_path=os.path.join(run_dir, "prompt.txt"),
            command=["codex", "exec"],
            transcript_path=transcript_path,
            stderr_path=stderr_path,
            last_message_path=os.path.join(run_dir, "last_message.txt"),
            report_path=surveyor._report_path(survey_id),
            draft_path=surveyor._draft_path(survey_id),
            last_transcript_size=os.path.getsize(transcript_path),
            last_transcript_growth_timestamp=time.time() - 1900,
        )
        surveyor.active_sessions[survey_id] = session

        with mock.patch("station.eval_archive.surveyor.os.killpg") as killpg:
            surveyor._check_codex_transcript_idle_timeouts()

        killpg.assert_called_once_with(DummyProcess.pid, signal.SIGTERM)
        self.assertTrue(session.transcript_idle_timeout_triggered)
        self.assertIn("did not grow", session.transcript_idle_timeout_reason)
        request_path = os.path.join(surveyor.paths.requests_dir, f"survey_{survey_id}.yaml")
        record = file_io_utils.load_yaml(request_path)
        self.assertIn("did not grow", record["session"]["last_error"])

        process.returncode = -15
        surveyor._poll_sessions()
        record = file_io_utils.load_yaml(request_path)
        self.assertEqual(record["status"], "queued")
        self.assertIn("did not grow", record["session"]["last_error"])

    def test_archive_surveyor_completes_invalid_utf8_report(self):
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=10,
        )
        survey_id = str(request["id"])
        station = self._surveyor_station(_AgentModuleStub())
        surveyor = AutoArchiveSurveyor(station, enabled=False)
        surveyor._claim_launch(survey_id, "codex_1_spawn_1_deadbeef", "codex", None)
        run_dir = os.path.join(surveyor.paths.sessions_dir, "codex_1_spawn_1_deadbeef")
        file_io_utils.ensure_dir_exists(run_dir)

        class DummyProcess:
            pid = 5432

            def poll(self):
                return 0

        class DummyHandle:
            def close(self):
                pass

        transcript_path = os.path.join(run_dir, "transcript.jsonl")
        stderr_path = os.path.join(run_dir, "stderr.txt")
        file_io_utils.save_text('{"type":"turn.completed"}\n', transcript_path)
        file_io_utils.save_text("", stderr_path)
        with open(surveyor._report_path(survey_id), "wb") as handle:
            handle.write(b"# Archive Survey Report #1\n\nrow inverse q \x89 -49327 f[0]\n")

        session = ActiveSurveySession(
            survey_id=survey_id,
            session_id="codex_1_spawn_1_deadbeef",
            run_dir=run_dir,
            backend="codex",
            transcript_format="jsonl",
            process=DummyProcess(),
            transcript_handle=DummyHandle(),
            stderr_handle=DummyHandle(),
            prompt_path=os.path.join(run_dir, "prompt.txt"),
            command=["codex", "exec"],
            transcript_path=transcript_path,
            stderr_path=stderr_path,
            last_message_path=os.path.join(run_dir, "last_message.txt"),
            report_path=surveyor._report_path(survey_id),
            draft_path=surveyor._draft_path(survey_id),
        )
        surveyor.active_sessions[survey_id] = session

        surveyor._poll_sessions()

        request_path = os.path.join(surveyor.paths.requests_dir, f"survey_{survey_id}.yaml")
        record = file_io_utils.load_yaml(request_path)
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record["session"]["active"])
        self.assertEqual(surveyor.active_sessions, {})
        self.assertTrue(record["notification"]["sent"])
        self.assertIn("row inverse q", record["notification"]["message"])

    def test_parallel_submission_service_creates_provisional_survey_and_commit_controls_delivery(self):
        self._write_archive_paper()
        wake_reasons = []
        station = types.SimpleNamespace(
            _is_agent_mature=lambda agent_data, tick: True,
            auto_archive_surveyor=types.SimpleNamespace(wake=lambda reason: wake_reasons.append(reason)),
        )
        service = ArchiveSurveySubmissionService(station)
        try:
            result = service.submit_and_wait(
                agent_data={
                    constants.AGENT_NAME_KEY: "Axiom I",
                    constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                    constants.AGENT_LINEAGE_KEY: "axiom",
                },
                yaml_data={constants.YAML_ARCHIVE_SURVEY_PROMPT: "Survey the frontier."},
                current_tick=42,
                run_id="run-a",
                op_id="tick:42:agent:Axiom I:action:1",
            )
        finally:
            service.stop()

        self.assertTrue(result.accepted)
        self.assertEqual(result.survey_id, "1")
        self.assertTrue(wake_reasons)
        paths = ensure_archive_surveyor_layout()
        request = file_io_utils.load_yaml(os.path.join(paths.requests_dir, "survey_1.yaml"))
        self.assertEqual(request["parallel_commit_status"], "provisional")
        self.assertEqual(request["parallel_tick"]["run_id"], "run-a")

        agent_module = _AgentModuleStub()
        surveyor = AutoArchiveSurveyor(
            self._surveyor_station(agent_module),
            enabled=False,
        )
        report_text = "# Archive Survey Report #1\n\nFull report body."
        file_io_utils.save_text(report_text, os.path.join(surveyor.paths.reports_dir, "1.md"))
        surveyor._mark_completed("1", exit_code=0, error=None)
        self.assertFalse(surveyor._deliver_report_if_needed("1"))
        self.assertEqual(agent_module.messages, [])

        self.assertTrue(mark_archive_survey_committed("1"))
        self.assertTrue(surveyor._deliver_report_if_needed("1"))
        self.assertEqual(len(agent_module.messages), 1)
        self.assertIn(report_text, agent_module.messages[0][1])

    def test_rollback_provisional_archive_survey_removes_request_and_artifacts(self):
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=1,
            parallel_metadata={"run_id": "run-a", "op_id": "op-a"},
        )
        survey_id = str(request["id"])
        paths = ensure_archive_surveyor_layout()
        file_io_utils.save_text("draft", os.path.join(paths.reports_dir, f"{survey_id}.draft.md"))
        file_io_utils.save_text("report", os.path.join(paths.reports_dir, f"{survey_id}.md"))

        self.assertEqual(rollback_provisional_archive_surveys(run_id="run-a"), [survey_id])
        self.assertFalse(file_io_utils.file_exists(os.path.join(paths.requests_dir, f"survey_{survey_id}.yaml")))
        self.assertFalse(file_io_utils.file_exists(os.path.join(paths.reports_dir, f"{survey_id}.draft.md")))
        self.assertFalse(file_io_utils.file_exists(os.path.join(paths.reports_dir, f"{survey_id}.md")))
        self.assertEqual(file_io_utils.load_yaml_lines(paths.pending_file), [])

    def test_parallel_state_cleanup_rolls_back_recorded_provisional_archive_survey(self):
        state_store = ParallelTickState(base_path=constants.BASE_STATION_DATA_PATH)
        state = state_store.begin_tick(42, ["Axiom I"])
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=42,
            parallel_metadata={"run_id": state["run_id"], "op_id": "op-a"},
        )
        survey_id = str(request["id"])
        state_store.record_fast_lane_survey(
            state,
            survey_id=survey_id,
            agent_name="Axiom I",
            op_id="op-a",
        )

        cleanup = state_store.cleanup_stale_run(station=None, eval_manager=None)

        self.assertTrue(cleanup["had_stale_state"])
        self.assertEqual(cleanup["rolled_back_survey_ids"], [survey_id])
        paths = ensure_archive_surveyor_layout()
        self.assertFalse(file_io_utils.file_exists(os.path.join(paths.requests_dir, f"survey_{survey_id}.yaml")))
        self.assertIsNone(state_store.load_current())

    def test_launch_uses_coder_backend_session_format_and_allowed_roots(self):
        request = queue_archive_survey_request(
            author="Axiom I",
            lineage="axiom",
            prompt="Summarize archive.",
            tick=1,
        )
        survey_id = str(request["id"])
        station = types.SimpleNamespace(agent_module=_AgentModuleStub(), _get_current_tick=lambda: 2)
        surveyor = AutoArchiveSurveyor(station, enabled=False)
        file_io_utils.ensure_dir_exists(surveyor.paths.research_target)

        popen_instances = []

        class DummyPopen:
            def __init__(self, command, **kwargs):
                self.command = command
                self.kwargs = kwargs
                self.pid = 12345
                self.stdin = mock.Mock()
                popen_instances.append(self)

            def poll(self):
                return None

        with mock.patch.dict(os.environ, {"CODEX_API_KEY": "survey-key", "CODEX_BASE_URL": "https://codex.test"}, clear=False), \
             mock.patch("station.eval_archive.surveyor.detect_cli_worker_executable", return_value="/usr/bin/codex"), \
             mock.patch("station.eval_archive.surveyor.subprocess.Popen", DummyPopen):
            self.assertTrue(surveyor._launch_survey(survey_id))

        session = surveyor.active_sessions[survey_id]
        try:
            self.assertRegex(session.session_id, r"^codex_1_spawn_1_[0-9a-f]{8}$")
            self.assertEqual(os.path.basename(session.prompt_path), "prompt.txt")
            index_dir = os.path.dirname(index_paths.get_station_index_database_path(constants.BASE_STATION_DATA_PATH))
            self.assertIn("--add-dir", popen_instances[0].command)
            self.assertIn(os.path.realpath(index_dir), popen_instances[0].command)
            self.assertNotIn(os.path.realpath(surveyor.paths.archive_target), popen_instances[0].command)
            self.assertNotIn(os.path.realpath(surveyor.paths.research_target), popen_instances[0].command)
            self.assertEqual(popen_instances[0].kwargs["env"].get("OPENAI_API_KEY"), "survey-key")
            self.assertEqual(popen_instances[0].kwargs["env"].get("OPENAI_BASE_URL"), "https://codex.test")
            prompt_text = file_io_utils.load_text(session.prompt_path)
            self.assertIn("Normal work cycle:", prompt_text)
            self.assertIn("1000 to 5000 words", prompt_text)
            self.assertIn("read-only source surface", prompt_text)
            self.assertIn("The CLI is not granted write access", prompt_text)
        finally:
            session.transcript_handle.close()
            session.stderr_handle.close()
            surveyor.active_sessions.clear()


if __name__ == "__main__":
    unittest.main()
