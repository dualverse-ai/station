import unittest

from station import constants
from web_interface.archive_utils import build_archive_detail_payload_from_capsule


class ArchiveDashboardPayloadTests(unittest.TestCase):
    def test_archive_detail_payload_includes_reviewer_and_other_replies(self):
        capsule_data = {
            constants.CAPSULE_ID_KEY: "archive_7",
            constants.CAPSULE_TITLE_KEY: "Paper Title",
            constants.CAPSULE_AUTHOR_NAME_KEY: "Axiom I",
            constants.CAPSULE_CREATED_AT_TICK_KEY: 42,
            constants.CAPSULE_ABSTRACT_KEY: "A concise abstract.",
            constants.CAPSULE_MESSAGES_KEY: [
                {
                    constants.MESSAGE_ID_KEY: "archive_7-1",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Axiom I",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 42,
                    constants.MESSAGE_TITLE_KEY: "Paper Title",
                    constants.MESSAGE_CONTENT_KEY: "Manuscript body.",
                    constants.MESSAGE_IS_DELETED_KEY: False,
                },
                {
                    constants.MESSAGE_ID_KEY: "archive_7-2",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Archive Review System",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 43,
                    constants.MESSAGE_CONTENT_KEY: (
                        "**Reviewer Evaluation**\n\n"
                        "**Score:** 8/10\n\n"
                        "**Reviewer Comments:**\nUseful result."
                    ),
                    constants.MESSAGE_IS_DELETED_KEY: False,
                },
                {
                    constants.MESSAGE_ID_KEY: "archive_7-3",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Peer Agent",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 44,
                    constants.MESSAGE_CONTENT_KEY: "Additional capsule discussion.",
                    constants.MESSAGE_IS_DELETED_KEY: False,
                },
                {
                    constants.MESSAGE_ID_KEY: "archive_7-4",
                    constants.MESSAGE_AUTHOR_NAME_KEY: "Archive Review System",
                    constants.MESSAGE_POSTED_AT_TICK_KEY: 45,
                    constants.MESSAGE_CONTENT_KEY: "Deleted reply.",
                    constants.MESSAGE_IS_DELETED_KEY: True,
                },
            ],
        }

        payload = build_archive_detail_payload_from_capsule(capsule_data)

        self.assertEqual(8.0, payload["reviewer_score"])
        self.assertEqual(2, payload["reply_count"])
        paper_markdown = payload["paper_markdown"]
        self.assertIn("Manuscript body.", paper_markdown)
        self.assertIn("## Capsule Replies", paper_markdown)
        self.assertIn("### Reply 1 by Archive Review System at Tick 43", paper_markdown)
        self.assertIn("**Reviewer Evaluation**", paper_markdown)
        self.assertIn("### Reply 2 by Peer Agent at Tick 44", paper_markdown)
        self.assertIn("Additional capsule discussion.", paper_markdown)
        self.assertNotIn("Deleted reply.", paper_markdown)


if __name__ == "__main__":
    unittest.main()
