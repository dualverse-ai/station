import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from station import capsule
from station import capsule_index
from station import constants
from station import file_io_utils
from web_interface.live_event_broker import DashboardEventBroker
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

    def test_question_capsule_index_tracks_status_and_upvote_metadata(self):
        numeric_id, capsule_data = capsule.create_capsule(
            {
                "title": "Question",
                "content": "question body",
                "abstract": "question abstract",
                "tags": "problem",
            },
            constants.CAPSULE_TYPE_QUESTION,
            self.author,
            1,
        )
        self.assertEqual(1, numeric_id)
        self.assertEqual(constants.QUESTION_STATUS_PENDING, capsule_data[constants.QUESTION_STATUS_KEY])

        path = os.path.join(
            self.tmpdir,
            constants.CAPSULES_DIR_NAME,
            constants.QUESTION_CAPSULES_SUBDIR_NAME,
            f"question_{numeric_id}.yaml",
        )
        capsule_data[constants.QUESTION_STATUS_KEY] = constants.QUESTION_STATUS_OPEN
        capsule_data[constants.QUESTION_NET_UPVOTE_KEY] = 3
        capsule_data[constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY] = "question_1-2"
        file_io_utils.save_yaml(capsule_data, path)
        capsule_index.rebuild_capsule_index()

        rows = capsule.list_capsules(constants.CAPSULE_TYPE_QUESTION, None, tag_filter="problem")
        self.assertEqual(1, len(rows))
        self.assertEqual(constants.QUESTION_STATUS_OPEN, rows[0][constants.QUESTION_STATUS_KEY])
        self.assertEqual(3, rows[0][constants.QUESTION_NET_UPVOTE_KEY])
        self.assertEqual("question_1-2", rows[0][constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY])

    def test_question_capsule_pages_sort_in_sqlite(self):
        first_id, first = capsule.create_capsule(
            {"title": "Lower vote", "content": "body"},
            constants.CAPSULE_TYPE_QUESTION,
            self.author,
            2,
        )
        second_id, second = capsule.create_capsule(
            {"title": "Higher vote", "content": "body"},
            constants.CAPSULE_TYPE_QUESTION,
            self.author,
            5,
        )
        for numeric_id, data, votes in ((first_id, first, -1), (second_id, second, 7)):
            path = os.path.join(
                self.tmpdir,
                constants.CAPSULES_DIR_NAME,
                constants.QUESTION_CAPSULES_SUBDIR_NAME,
                f"question_{numeric_id}.yaml",
            )
            data[constants.QUESTION_NET_UPVOTE_KEY] = votes
            file_io_utils.save_yaml(data, path)
        capsule_index.rebuild_capsule_index()

        page, total = capsule.list_capsules_page(
            constants.CAPSULE_TYPE_QUESTION,
            None,
            page=1,
            page_size=1,
            sort_by="question_net_upvote",
            sort_direction="desc",
        )

        self.assertEqual(2, total)
        self.assertEqual("Higher vote", page[0][constants.CAPSULE_TITLE_KEY])

    def test_sort_indexes_upgrade_existing_database_without_yaml_rebuild(self):
        capsule.create_capsule(
            {"title": "Existing question", "content": "body"},
            constants.CAPSULE_TYPE_QUESTION,
            self.author,
            2,
        )
        db_path = capsule_index.get_database_path()
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP INDEX IF EXISTS idx_capsule_question_vote_sort")
        capsule_index._SORT_INDEX_READY_PATHS.discard(db_path)

        with mock.patch.object(capsule_index, "rebuild_capsule_index") as rebuild:
            capsule_index.ensure_capsule_index()

        rebuild.assert_not_called()
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
                ("idx_capsule_question_vote_sort",),
            ).fetchone()
        self.assertEqual(("idx_capsule_question_vote_sort",), row)

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

    def test_dashboard_event_broker_is_bounded_broadcast_and_starts_new_clients_live(self):
        broker = DashboardEventBroker(max_events=4, max_replay=2)
        for number in range(1, 1001):
            broker.put_nowait({"event": "test", "data": {"number": number}})

        new_client = broker.open_cursor()
        self.assertEqual(1000, new_client.cursor)
        self.assertEqual([], broker.read_after(new_client.cursor).events)

        first_reader = broker.read_after(998)
        second_reader = broker.read_after(998)
        self.assertEqual([999, 1000], [sequence for sequence, _event in first_reader.events])
        self.assertEqual(first_reader.events, second_reader.events)
        self.assertEqual(4, broker.buffered_count)

    def test_dashboard_event_broker_bounds_reconnect_backlog_and_recovers_after_restart(self):
        broker = DashboardEventBroker(max_events=10, max_replay=3)
        for number in range(1, 11):
            broker.put({"event": "test", "data": {"number": number}})

        batch = broker.read_after(1, limit=10)
        self.assertTrue(batch.reset)
        self.assertEqual(6, batch.dropped_count)
        self.assertEqual([8, 9, 10], [sequence for sequence, _event in batch.events])

        restarted_broker = DashboardEventBroker()
        reset_state = restarted_broker.open_cursor(999)
        self.assertTrue(reset_state.reset)
        self.assertEqual(0, reset_state.cursor)

    def test_recent_yaml_tick_window_reads_only_requested_ticks(self):
        path = os.path.join(self.tmpdir, "dialogue.yamll")
        for tick in range(1, 7):
            file_io_utils.append_yaml_line(
                {"tick": tick, "type": "observation", "content": f"first {tick}\n---\ninside content"},
                path,
            )
            file_io_utils.append_yaml_line(
                {"tick": tick, "type": "submission", "content": f"second {tick}"},
                path,
            )

        entries, metadata = file_io_utils.load_yaml_lines_tick_window(
            path,
            window="recent",
            tick_limit=2,
        )

        self.assertEqual([5, 5, 6, 6], [entry["tick"] for entry in entries])
        self.assertEqual("first 5\n---\ninside content", entries[0]["content"])
        self.assertTrue(metadata["is_partial"])
        self.assertTrue(metadata["has_older"])

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
