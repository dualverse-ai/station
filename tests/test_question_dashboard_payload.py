import unittest
from pathlib import Path

from station import constants
from web_interface.question_utils import (
    build_question_detail_payload_from_capsule,
    build_question_list_payload,
    build_question_survey_preview,
)


class _FakeCapsuleModule:
    def __init__(self, rows, total):
        self.rows = rows
        self.total = total
        self.calls = []

    def list_capsules_page(self, capsule_type, lineage_name, **kwargs):
        self.calls.append((capsule_type, lineage_name, kwargs))
        return self.rows, self.total


class QuestionDashboardPayloadTests(unittest.TestCase):
    def test_question_list_defaults_to_newest_first_with_one_hundred_rows(self):
        module = _FakeCapsuleModule([], total=0)

        payload = build_question_list_payload(module)

        self.assertEqual(100, payload["page_size"])
        _capsule_type, _lineage, kwargs = module.calls[0]
        self.assertEqual("created_at_tick", kwargs["sort_by"])
        self.assertEqual("desc", kwargs["sort_direction"])

    def test_question_thread_uses_theme_surfaces_and_forum_byline(self):
        styles = Path("web_interface/static/css/style.css").read_text(encoding="utf-8")
        javascript = Path("web_interface/static/js/dashboard.js").read_text(encoding="utf-8")

        self.assertIn("--question-thread-bg: #f8fafc", styles)
        self.assertIn("--question-accepted-bg: #ecfdf5", styles)
        self.assertIn("background: var(--question-thread-bg)", styles)
        self.assertIn("background: var(--question-abstract-bg)", styles)
        self.assertIn("background: var(--question-accepted-bg)", styles)
        self.assertIn("question-thread-byline", javascript)
        self.assertIn("posted the question", javascript)
        self.assertIn("posted a reply", javascript)

    def test_station_tool_buttons_have_the_requested_placement(self):
        template = Path("web_interface/templates/dashboard.html").read_text(encoding="utf-8")
        primary_start = template.index('<div class="sidebar-primary-tools">')
        popover_start = template.index('<div id="more-tools-popover"', primary_start)
        primary_markup = template[primary_start:popover_start]
        popover_end = template.index('<div id="dashboard-status-messages"', popover_start)
        popover_markup = template[popover_start:popover_end]

        self.assertLess(primary_markup.index("open-archive-papers-modal-button"), primary_markup.index("open-question-room-modal-button"))
        self.assertNotIn("create-api-agent-modal-button", primary_markup)
        self.assertIn("create-api-agent-modal-button", popover_markup)

    def test_question_list_is_bounded_and_uses_index_page_arguments(self):
        module = _FakeCapsuleModule(
            [{
                constants.CAPSULE_ID_KEY: "question_12",
                constants.CAPSULE_TITLE_KEY: "Indexed question",
                constants.CAPSULE_AUTHOR_NAME_KEY: "Ada I",
                constants.CAPSULE_CREATED_AT_TICK_KEY: 41,
                constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY: 47,
                constants.CAPSULE_ABSTRACT_KEY: "A short question abstract.",
                constants.QUESTION_STATUS_KEY: constants.QUESTION_STATUS_SOLVED,
                constants.QUESTION_NET_UPVOTE_KEY: 8,
                constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY: "question_12-2",
                "total_message_count": 3,
            }],
            total=151,
        )

        payload = build_question_list_payload(
            module,
            page=2,
            page_size=500,
            sort_by="net_upvote",
            sort_direction="asc",
        )

        self.assertEqual(100, payload["page_size"])
        self.assertEqual(2, payload["total_pages"])
        self.assertEqual(2, payload["questions"][0]["reply_count"])
        _capsule_type, _lineage, kwargs = module.calls[0]
        self.assertEqual(constants.CAPSULE_TYPE_QUESTION, _capsule_type)
        self.assertEqual("question_net_upvote", kwargs["sort_by"])
        self.assertEqual("asc", kwargs["sort_direction"])
        self.assertEqual(100, kwargs["page_size"])

    def test_question_survey_preview_is_bounded_and_index_backed(self):
        module = _FakeCapsuleModule(
            [{
                constants.CAPSULE_ID_KEY: "question_12",
                constants.CAPSULE_TITLE_KEY: "Indexed question",
                constants.CAPSULE_AUTHOR_NAME_KEY: "Ada I",
                constants.CAPSULE_ABSTRACT_KEY: "A short question abstract.",
                constants.QUESTION_STATUS_KEY: constants.QUESTION_STATUS_SOLVED,
                constants.QUESTION_NET_UPVOTE_KEY: 8,
                constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY: "question_12-3",
            }],
            total=1200,
        )

        preview = build_question_survey_preview(module, limit=5000)

        self.assertIn("Showing 1 of 1200", preview)
        self.assertIn("Question #12: Indexed question", preview)
        self.assertIn("Accepted solution message: question_12-3", preview)
        _capsule_type, _lineage, kwargs = module.calls[0]
        self.assertEqual(constants.CAPSULE_TYPE_QUESTION, _capsule_type)
        self.assertEqual(1000, kwargs["page_size"])
        self.assertEqual("numeric_id", kwargs["sort_by"])

    def test_question_detail_marks_the_accepted_solution(self):
        capsule_data = {
            constants.CAPSULE_ID_KEY: "question_7",
            constants.CAPSULE_TITLE_KEY: "Question title",
            constants.CAPSULE_AUTHOR_NAME_KEY: "Noether I",
            constants.CAPSULE_CREATED_AT_TICK_KEY: 20,
            constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY: 25,
            constants.CAPSULE_ABSTRACT_KEY: "Question abstract.",
            constants.QUESTION_STATUS_KEY: constants.QUESTION_STATUS_SOLVED,
            constants.QUESTION_NET_UPVOTE_KEY: 4,
            constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY: "question_7-3",
            constants.CAPSULE_MESSAGES_KEY: [
                {
                    constants.MESSAGE_ID_KEY: "question_7-1",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Noether I",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 20,
                    constants.MESSAGE_CONTENT_KEY: "Question body.",
                },
                {
                    constants.MESSAGE_ID_KEY: "question_7-2",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Deleted Agent",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 22,
                    constants.MESSAGE_CONTENT_KEY: "Deleted reply.",
                    constants.MESSAGE_IS_DELETED_KEY: True,
                },
                {
                    constants.MESSAGE_ID_KEY: "question_7-3",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Euler I",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 25,
                    constants.MESSAGE_CONTENT_KEY: "Accepted answer.",
                    constants.QUESTION_SOLUTION_NET_UPVOTE_KEY: 5,
                },
            ],
        }

        payload = build_question_detail_payload_from_capsule(capsule_data)

        self.assertEqual(2, len(payload["messages"]))
        self.assertTrue(payload["messages"][0]["is_question"])
        self.assertFalse(payload["messages"][0]["is_accepted"])
        self.assertTrue(payload["messages"][1]["is_accepted"])
        self.assertEqual(5, payload["messages"][1]["solution_net_upvote"])
        self.assertNotIn("Deleted reply.", [message["content"] for message in payload["messages"]])


if __name__ == "__main__":
    unittest.main()
