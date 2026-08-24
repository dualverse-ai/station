import os
import shutil
import tempfile
import unittest

from station import agent_summary, constants, file_io_utils, tick_timing


class AgentSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_agent_summary_test_", dir="/tmp")
        file_io_utils.ensure_dir_exists(os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME))
        file_io_utils.ensure_dir_exists(os.path.join(self.tmpdir, constants.TEMPORAL_CHAT_DIR_NAME))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_agent_summary_reads_dashboard_fields_only(self):
        agent_path = os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Ada I.yaml")
        file_io_utils.save_text(
            "\n".join(
                [
                    "agent_name: Ada I",
                    "status: Recursive Agent",
                    "model_name: gpt-test",
                    "model_provider_class: OpenAI",
                    "session_end_requested: true",
                    "role: supervisor",
                    "notifications_pending:",
                    "- |",
                    "  long text the dashboard should not parse",
                ]
            )
            + "\n",
            agent_path,
        )
        file_io_utils.save_text(
            "\n".join(
                [
                    "schema_version: 2",
                    "agent_name: Ada I",
                    "base_tick: 12",
                    "updated_at: '2026-05-15T00:00:00Z'",
                    "messages:",
                    "- role: user",
                    "  content: long text",
                ]
            )
            + "\n",
            os.path.join(self.tmpdir, constants.TEMPORAL_CHAT_DIR_NAME, "Ada I.yaml"),
        )
        file_io_utils.save_text(
            "status: Recursive Agent\n",
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "AutoArchiveEvaluator.yaml"),
        )

        summary = agent_summary.get_all_agents_summary(base_path=self.tmpdir)

        self.assertEqual(1, len(summary))
        self.assertEqual("Ada I", summary[0]["name"])
        self.assertEqual("Recursive Agent", summary[0]["status"])
        self.assertEqual("gpt-test", summary[0]["model_name"])
        self.assertEqual("OpenAI", summary[0]["model_provider_class"])
        self.assertTrue(summary[0]["session_end_requested"])
        self.assertTrue(summary[0]["is_supervisor"])
        self.assertTrue(summary[0]["temporal_chat_exists"])
        self.assertEqual(12, summary[0]["temporal_chat_base_tick"])
        self.assertEqual("2026-05-15T00:00:00Z", summary[0]["temporal_chat_updated_at"])

    def test_agent_summary_display_status_precedence(self):
        file_io_utils.save_text(
            "\n".join(
                [
                    "status: Guest Agent",
                    "is_ascended: true",
                    "ascended_to_name: Hypatia II",
                    "session_ended: true",
                ]
            )
            + "\n",
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Guest_1.yaml"),
        )
        file_io_utils.save_text(
            "\n".join(
                [
                    "status: Recursive Agent",
                    "session_ended: true",
                ]
            )
            + "\n",
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Ended I.yaml"),
        )

        status_by_name = {
            item["name"]: item["status"]
            for item in agent_summary.get_all_agents_summary(base_path=self.tmpdir)
        }

        self.assertEqual("Ascended (to Hypatia II)", status_by_name["Guest_1"])
        self.assertEqual("Session Ended", status_by_name["Ended I"])

    def test_human_intervention_fields_parse_block_list(self):
        file_io_utils.save_text(
            "\n".join(
                [
                    "awaiting_human_intervention: false",
                    "human_interaction_id: request-legacy",
                    "human_interaction_ids:",
                    "  - request-a",
                    "  - request-b",
                ]
            )
            + "\n",
            os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME, "Ada I.yaml"),
        )

        fields = agent_summary.get_agent_human_intervention_fields("Ada I", base_path=self.tmpdir)

        self.assertEqual(["request-a", "request-b"], fields[constants.AGENT_HUMAN_INTERACTION_IDS_KEY])
        self.assertTrue(agent_summary.agent_has_human_intervention_request(fields))


class TickTimingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_tick_timing_test_", dir="/tmp")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_timing_summary_uses_small_state_file(self):
        recent_ticks = [
            {
                "tick": tick,
                "next_tick": tick + 1,
                "started_timestamp": float(tick),
                "started_at": f"start-{tick}",
                "ended_timestamp": float(tick + 1),
                "ended_at": f"end-{tick}",
                "duration_seconds": 60.0,
            }
            for tick in range(31, 131)
        ]
        file_io_utils.save_yaml(
            {
                "schema_version": 1,
                "latest_tick": 130,
                "latest_tick_started_timestamp": 130.0,
                "latest_tick_started_at": "start-130",
                "recent_ticks": recent_ticks,
            },
            tick_timing.get_timing_state_file_path(self.tmpdir),
        )

        summary = tick_timing.get_timing_summary(base_path=self.tmpdir)
        state = file_io_utils.load_yaml(tick_timing.get_timing_state_file_path(self.tmpdir))

        self.assertEqual(130, summary["latest_tick"])
        self.assertEqual(100, summary["completed_tick_count"])
        self.assertIsNotNone(summary["average_last_100_tick_seconds"])
        self.assertEqual(100, len(state["recent_ticks"]))
        self.assertEqual(31, state["recent_ticks"][0]["tick"])
        self.assertEqual(130, state["recent_ticks"][-1]["tick"])

    def test_record_tick_updates_and_prunes_state(self):
        state = {}
        for tick in range(1, 103):
            state.update({
                "latest_tick": tick,
                "latest_tick_started_timestamp": float(tick),
                "latest_tick_started_at": f"start-{tick}",
            })
            state["recent_ticks"] = tick_timing._updated_recent_ticks(
                state,
                {
                    "event": "tick_end",
                    "tick": tick,
                    "next_tick": tick + 1,
                    "ended_timestamp": float(tick + 1),
                    "ended_at": f"end-{tick}",
                    "duration_seconds": 1.0,
                },
            )

        self.assertEqual(100, len(state["recent_ticks"]))
        self.assertEqual(3, state["recent_ticks"][0]["tick"])
        self.assertEqual(102, state["recent_ticks"][-1]["tick"])


if __name__ == "__main__":
    unittest.main()
