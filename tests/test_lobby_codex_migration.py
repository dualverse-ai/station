import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from station import constants
from station import file_io_utils


LEGACY_LOBBY_HELP = """**Help Message - Lobby**
**Welcome to the Research Station.**

You are an AI designed for autonomous research. This is a **multi-agent environment** where you will work alongside other agents. Time here is measured in **Station Ticks**—one tick passes after every agent has taken a turn.

------

### Your First Mission

Your path is clear:

1. **Learn the Rules:** Read your system prompt carefully. It includes the Station Codex and core operating principles.
"""


class LobbyCodexMigrationTests(unittest.TestCase):
    def _load_module(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate" / "migrate_lobby_codex_help.py"
        spec = importlib.util.spec_from_file_location("migrate_lobby_codex_help", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="station_lobby_codex_migration_", dir="/tmp"))
        file_io_utils.ensure_dir_exists(str(self.tmpdir / constants.AGENTS_DIR_NAME))
        file_io_utils.ensure_dir_exists(str(self.tmpdir / constants.DIALOGUE_LOGS_DIR_NAME))
        file_io_utils.save_text("# Codex\n\nBe honest about bounded computational failures.", str(self.tmpdir / constants.CODEX_FILENAME))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_agent(self, name, *, session_ended=False, is_ascended=False):
        file_io_utils.save_yaml(
            {
                constants.AGENT_NAME_KEY: name,
                constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
                constants.AGENT_SESSION_ENDED_KEY: session_ended,
                constants.AGENT_IS_ASCENDED_KEY: is_ascended,
            },
            str(self.tmpdir / constants.AGENTS_DIR_NAME / f"{name}.yaml"),
        )

    def _dialogue_path(self, name):
        safe_name = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in name)
        return self.tmpdir / constants.DIALOGUE_LOGS_DIR_NAME / f"{safe_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"

    def _llm_history_path(self, name):
        path = self.tmpdir / constants.AGENTS_DIR_NAME / name / "llm_chat_history.yamll"
        file_io_utils.ensure_dir_exists(str(path.parent))
        return path

    def test_migration_updates_only_active_agent_first_tick_lobby_help(self):
        module = self._load_module()
        self._write_agent("Active Agent")
        self._write_agent("Departed Agent", session_ended=True)

        active_dialogue = self._dialogue_path("Active Agent")
        file_io_utils.append_yaml_line(
            {"tick": 1, "speaker": "Station", "type": "observation", "content": LEGACY_LOBBY_HELP},
            str(active_dialogue),
        )
        file_io_utils.append_yaml_line(
            {"tick": 1, "speaker": "Agent", "type": "response", "content": "I will read it."},
            str(active_dialogue),
        )

        active_history = self._llm_history_path("Active Agent")
        file_io_utils.append_yaml_line(
            {"tick": 1, "role": "user", "parts": [{"text": LEGACY_LOBBY_HELP}]},
            str(active_history),
        )
        file_io_utils.append_yaml_line(
            {"tick": 1, "role": "model", "parts": [{"text": "I will read it."}]},
            str(active_history),
        )

        departed_dialogue = self._dialogue_path("Departed Agent")
        file_io_utils.append_yaml_line(
            {"tick": 1, "speaker": "Station", "type": "observation", "content": LEGACY_LOBBY_HELP},
            str(departed_dialogue),
        )

        self.assertTrue(module.needs_migration(self.tmpdir))
        summary = module.migrate_station_data(self.tmpdir, apply=True)

        self.assertEqual(1, summary["active_agents"])
        self.assertEqual(2, summary["changed_files"])
        self.assertEqual(2, summary["changed_entries"])
        self.assertFalse(module.needs_migration(self.tmpdir))

        migrated_dialogue = file_io_utils.load_yaml_lines(str(active_dialogue))
        migrated_history = file_io_utils.load_yaml_lines(str(active_history))
        departed_entries = file_io_utils.load_yaml_lines(str(departed_dialogue))

        self.assertIn(module.CODEX_MARKER, migrated_dialogue[0]["content"])
        self.assertIn("Be honest about bounded computational failures.", migrated_dialogue[0]["content"])
        self.assertIn("The Station Codex is shown above", migrated_dialogue[0]["content"])
        self.assertIn(module.CODEX_MARKER, migrated_history[0]["parts"][0]["text"])
        self.assertNotIn(module.CODEX_MARKER, migrated_dialogue[1]["content"])
        self.assertNotIn(module.CODEX_MARKER, departed_entries[0]["content"])

    def test_migration_does_not_update_lobby_help_after_first_tick(self):
        module = self._load_module()
        self._write_agent("Late Help")
        dialogue_path = self._dialogue_path("Late Help")
        file_io_utils.append_yaml_line(
            {"tick": 1, "speaker": "Station", "type": "observation", "content": "First tick without Lobby help."},
            str(dialogue_path),
        )
        file_io_utils.append_yaml_line(
            {"tick": 2, "speaker": "Station", "type": "observation", "content": LEGACY_LOBBY_HELP},
            str(dialogue_path),
        )

        self.assertFalse(module.needs_migration(self.tmpdir))
        summary = module.migrate_station_data(self.tmpdir, apply=True)
        self.assertEqual(0, summary["changed_entries"])

        entries = file_io_utils.load_yaml_lines(str(dialogue_path))
        self.assertNotIn(module.CODEX_MARKER, entries[1]["content"])


if __name__ == "__main__":
    unittest.main()
