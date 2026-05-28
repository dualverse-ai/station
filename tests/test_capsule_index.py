import os
import shutil
import tempfile
import unittest

from station import capsule
from station import capsule_index
from station import constants
from station import file_io_utils
from web_interface.stream_utils import sanitize_stream_event_payload


class CapsuleIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_capsule_index_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        self.author = {
            constants.AGENT_NAME_KEY: "Alice",
            constants.AGENT_LINEAGE_KEY: "Test",
            constants.AGENT_GENERATION_KEY: 1,
        }

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capsule_index_tracks_metadata_unread_search_and_pages(self):
        numeric_id, capsule_data = capsule.create_capsule(
            {
                "title": "Indexed capsule",
                "content": "hello world",
                "abstract": "short abstract",
                "tags": "fast, math",
            },
            constants.CAPSULE_TYPE_PUBLIC,
            self.author,
            1,
        )
        self.assertEqual(1, numeric_id)
        self.assertTrue(
            capsule.add_message_to_capsule(
                numeric_id,
                constants.CAPSULE_TYPE_PUBLIC,
                {"content": "second message"},
                self.author,
                2,
            )
        )

        first_message_id = capsule_data[constants.CAPSULE_MESSAGES_KEY][0][constants.MESSAGE_ID_KEY]
        rows = capsule.list_capsules(
            constants.CAPSULE_TYPE_PUBLIC,
            None,
            agent_read_status={first_message_id: True},
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("Indexed capsule", rows[0][constants.CAPSULE_TITLE_KEY])
        self.assertEqual(2, rows[0]["total_message_count"])
        self.assertEqual(1, rows[0][constants.CAPSULE_UNREAD_MESSAGE_COUNT_KEY])

        tagged = capsule.list_capsules(constants.CAPSULE_TYPE_PUBLIC, None, tag_filter="math")
        self.assertEqual([numeric_id], [int(item[constants.CAPSULE_ID_KEY].split("_")[-1]) for item in tagged])

        page, total = capsule.list_capsules_page(
            constants.CAPSULE_TYPE_PUBLIC,
            None,
            page=1,
            page_size=1,
        )
        self.assertEqual(1, total)
        self.assertEqual(1, len(page))

        db_path = os.path.join(
            self.tmpdir,
            constants.STATION_INDEX_DIR_NAME,
            constants.STATION_INDEX_DB_FILENAME,
        )
        self.assertTrue(os.path.exists(db_path))

    def test_mail_visibility_uses_author_or_recipient(self):
        capsule.create_capsule(
            {
                "title": "Mail",
                "content": "secret",
                "recipients": ["Bob"],
            },
            constants.CAPSULE_TYPE_MAIL,
            self.author,
            1,
        )

        alice_page, alice_total = capsule.list_capsules_page(
            constants.CAPSULE_TYPE_MAIL,
            None,
            visible_agent_name="Alice",
        )
        bob_page, bob_total = capsule.list_capsules_page(
            constants.CAPSULE_TYPE_MAIL,
            None,
            visible_agent_name="Bob",
        )
        eve_page, eve_total = capsule.list_capsules_page(
            constants.CAPSULE_TYPE_MAIL,
            None,
            visible_agent_name="Eve",
        )

        self.assertEqual(1, alice_total)
        self.assertEqual(1, bob_total)
        self.assertEqual(0, eve_total)
        self.assertEqual("Mail", alice_page[0][constants.CAPSULE_TITLE_KEY])
        self.assertEqual("Mail", bob_page[0][constants.CAPSULE_TITLE_KEY])
        self.assertEqual([], eve_page)

    def test_external_yaml_edits_require_explicit_rebuild(self):
        numeric_id, _capsule_data = capsule.create_capsule(
            {
                "title": "Before",
                "content": "hello",
                "abstract": "short",
                "tags": "old",
            },
            constants.CAPSULE_TYPE_PUBLIC,
            self.author,
            1,
        )
        path = os.path.join(
            self.tmpdir,
            constants.CAPSULES_DIR_NAME,
            constants.PUBLIC_CAPSULES_SUBDIR_NAME,
            f"public_{numeric_id}.yaml",
        )
        data = file_io_utils.load_yaml(path)
        data[constants.CAPSULE_TITLE_KEY] = "After"
        data[constants.CAPSULE_TAGS_KEY] = ["new"]
        file_io_utils.save_yaml(data, path)

        rows = capsule.list_capsules(constants.CAPSULE_TYPE_PUBLIC, None, tag_filter="new")
        self.assertEqual([], rows)

        capsule_index.rebuild_capsule_index()
        rows = capsule.list_capsules(constants.CAPSULE_TYPE_PUBLIC, None, tag_filter="new")
        self.assertEqual(1, len(rows))
        self.assertEqual("After", rows[0][constants.CAPSULE_TITLE_KEY])

    def test_stream_sanitizer_sends_full_text_only_for_selected_agent(self):
        payload = {
            "event": "llm_event",
            "data": {
                "agent_name": "Ada I",
                "type": "observation",
                "text_content": "x" * 1000,
                "thinking_text": "hidden",
            },
            "timestamp": 1.0,
        }

        sanitized = sanitize_stream_event_payload(payload, selected_agent_name="Noether I")

        self.assertNotIn("text_content", sanitized["data"])
        self.assertNotIn("thinking_text", sanitized["data"])
        self.assertNotIn("text_preview", sanitized["data"])
        self.assertEqual(1000, sanitized["data"]["full_length"])
        self.assertTrue(sanitized["data"]["content_omitted"])

        selected = sanitize_stream_event_payload(payload, selected_agent_name="Ada I")
        self.assertEqual("x" * 1000, selected["data"]["text_content"])
        self.assertEqual("hidden", selected["data"]["thinking_text"])
        self.assertNotIn("text_preview", selected["data"])
        self.assertNotIn("content_omitted", selected["data"])

    def test_rebuild_flag_can_come_from_environment(self):
        old_value = os.environ.get("STATION_REBUILD_DB")
        try:
            os.environ["STATION_REBUILD_DB"] = "1"
            self.assertTrue(capsule_index.should_rebuild_from_process_args())
        finally:
            if old_value is None:
                os.environ.pop("STATION_REBUILD_DB", None)
            else:
                os.environ["STATION_REBUILD_DB"] = old_value

    def test_tmp_database_override_is_station_specific(self):
        old_override = os.environ.get("STATION_INDEX_DB_PATH")
        try:
            file_io_utils.save_yaml(
                {constants.STATION_ID_KEY: "station alpha"},
                os.path.join(self.tmpdir, constants.STATION_CONFIG_FILENAME),
            )
            os.environ["STATION_INDEX_DB_PATH"] = os.path.join("/tmp", constants.STATION_INDEX_DB_FILENAME)
            db_path = capsule_index.get_database_path()
            self.assertEqual(os.path.join("/tmp", "station_alpha_station_index.sqlite3"), db_path)
        finally:
            if old_override is None:
                os.environ.pop("STATION_INDEX_DB_PATH", None)
            else:
                os.environ["STATION_INDEX_DB_PATH"] = old_override


if __name__ == "__main__":
    unittest.main()
