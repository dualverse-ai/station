import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from station import backup_utils, research_storage
from station_tools.multistart_backup import (
    active_multistart_info,
    create_active_multistart_archive,
    latest_active_multistart_manifest,
    remove_active_multistart_research_storage_allocations,
    restore_active_multistart_archive,
)
from station_tools.commands import archive as archive_command
from station_tools.config import ToolsConfig


class MultistartBackupTests(unittest.TestCase):
    def test_ordinary_automatic_backups_remain_disabled_inside_multistart_branches(self):
        with mock.patch.dict(os.environ, {"STATION_MULTISTART_BRANCH": "1"}):
            self.assertFalse(backup_utils.should_create_automatic_backup(100))

    def _write_active_job(self, repo: Path) -> tuple[Path, Path, Path]:
        multistart = repo / "station_multistart"
        job = multistart / "10_jobabc"
        origin = job / "origin_station_data"
        seed1 = job / "station_data_s1"
        seed2 = job / "station_data_s2"
        for data_root, tick in ((origin, 10), (seed1, 15), (seed2, 17)):
            (data_root / "agents").mkdir(parents=True)
            (data_root / "station_config.yaml").write_text(
                yaml.safe_dump({"station_id": "station-id", "current_tick": tick}),
                encoding="utf-8",
            )
            (data_root / "agents" / "common.yaml").write_text("same content\n", encoding="utf-8")
            (data_root / "index").mkdir()
            (data_root / "index" / "station_index.sqlite3").write_text("transient", encoding="utf-8")
        (multistart / "controller.log").write_text("controller\n", encoding="utf-8")
        (multistart / "controller.pid").write_text("12345\n", encoding="utf-8")
        (job / "admin").mkdir()
        (job / "admin" / "seed1").symlink_to(seed1, target_is_directory=True)
        (multistart / "current_job.yaml").write_text(
            yaml.safe_dump({"job_id": "jobabc", "job_dir": str(job), "status": "running"}),
            encoding="utf-8",
        )
        (job / "state.yaml").write_text(
            yaml.safe_dump(
                {
                    "job_id": "jobabc",
                    "origin_station_id": "station-id",
                    "station_name": "Test",
                    "branch_tick": 10,
                    "status": "running",
                    "branches": [
                        {"seed": 1, "data_root": str(seed1), "status": "running", "pid": 111},
                        {"seed": 2, "data_root": str(seed2), "status": "running", "pid": 222},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return multistart, job, seed1

    def test_active_archive_deduplicates_and_restores_without_pids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "source"
            self._write_active_job(repo)

            info = active_multistart_info(repo)
            self.assertIsNotNone(info)
            self.assertEqual(17, info["station_tick"])
            backup_dir = repo / "backup" / "station-id"
            manifest_path = create_active_multistart_archive(repo, info, backup_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected = latest_active_multistart_manifest(backup_dir)
            self.assertIsNotNone(selected)
            self.assertEqual(manifest_path, selected[0])
            second_manifest_path = create_active_multistart_archive(repo, info, backup_dir)
            self.assertNotEqual(manifest_path, second_manifest_path)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(second_manifest_path.is_file())

            common_hashes = {
                item["hash"]
                for item in manifest["files"]
                if item["path"].endswith("agents/common.yaml")
            }
            self.assertEqual(1, len(common_hashes))
            common_hash = next(iter(common_hashes))
            self.assertTrue((backup_dir / "objects" / common_hash[:2] / common_hash[2:]).is_file())
            self.assertTrue((backup_dir / "multistart_archives" / "active_jobabc.json").is_file())

            restored = root / "restored_multistart"
            restore_active_multistart_archive(manifest_path, restored)
            self.assertTrue((restored / "current_job.yaml").is_file())
            self.assertFalse((restored / "controller.pid").exists())
            self.assertFalse((restored / "10_jobabc" / "station_data_s1" / "index").exists())
            restored_link = restored / "10_jobabc" / "admin" / "seed1"
            self.assertTrue(restored_link.is_symlink())
            self.assertEqual(
                str(restored / "10_jobabc" / "station_data_s1"),
                str(restored_link.readlink()),
            )
            restored_state = yaml.safe_load((restored / "10_jobabc" / "state.yaml").read_text(encoding="utf-8"))
            self.assertEqual([None, None], [branch["pid"] for branch in restored_state["branches"]])
            self.assertEqual(
                str(restored / "10_jobabc" / "station_data_s1"),
                restored_state["branches"][0]["data_root"],
            )

            snapshots = backup_dir / "snapshots"
            snapshots.mkdir()
            (snapshots / "tick_10.json").write_text(
                json.dumps({"station_id": "station-id", "tick": 10, "files": [], "symlinks": []}),
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "restore.sh"
            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = project_root if not env.get("PYTHONPATH") else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"

            active_result = subprocess.run(
                ["bash", str(script), "--output", "restored_active", "station-id"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, active_result.returncode, active_result.stdout + active_result.stderr)
            self.assertIn("Using active multistart archive at tick 17", active_result.stdout)
            self.assertTrue((repo / "restored_active" / "current_job.yaml").is_file())

            (snapshots / "tick_17.json").write_text(
                json.dumps({"station_id": "station-id", "tick": 17, "files": [], "symlinks": []}),
                encoding="utf-8",
            )
            equal_result = subprocess.run(
                ["bash", str(script), "--output", "restored_equal", "station-id"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, equal_result.returncode, equal_result.stdout + equal_result.stderr)
            self.assertIn("Using ordinary tick 17; archived multistart tick 17 is not newer", equal_result.stdout)

            (snapshots / "tick_20.json").write_text(
                json.dumps({"station_id": "station-id", "tick": 20, "files": [], "symlinks": []}),
                encoding="utf-8",
            )
            normal_result = subprocess.run(
                ["bash", str(script), "--output", "restored_normal", "station-id"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, normal_result.returncode, normal_result.stdout + normal_result.stderr)
            self.assertIn("Using ordinary tick 20; archived multistart tick 17 is not newer", normal_result.stdout)
            self.assertTrue((repo / "restored_normal").is_dir())

    def test_active_archive_materializes_managed_branch_and_origin_research_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "source"
            _multistart, job, seed1 = self._write_active_job(repo)
            seed2 = job / "station_data_s2"
            storage_base = root / "research_storage"
            seed_storage = storage_base / "11111111-1111-1111-1111-111111111111"
            origin_storage = storage_base / "00000000-0000-0000-0000-000000000000"
            artifact = seed_storage / "lineages" / "alpha" / "artifact.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("remote artifact", encoding="utf-8")
            origin_artifact = origin_storage / "lineages" / "alpha" / "origin.txt"
            origin_artifact.parent.mkdir(parents=True)
            origin_artifact.write_text("origin artifact", encoding="utf-8")
            research_root = seed1 / "rooms" / "research"
            research_root.mkdir(parents=True)
            (research_root / "storage").symlink_to(seed_storage, target_is_directory=True)
            seed2_research = seed2 / "rooms" / "research"
            seed2_research.mkdir(parents=True)
            (seed2_research / "storage").symlink_to(seed_storage, target_is_directory=True)
            origin_research = job / "origin_station_data" / "rooms" / "research"
            origin_research.mkdir(parents=True, exist_ok=True)
            (origin_research / "storage").symlink_to(origin_storage, target_is_directory=True)
            research_storage.write_allocation_marker(
                seed_storage,
                {
                    "kind": "multistart_seed",
                    "station_id": "station-id",
                    "job_id": "jobabc",
                    "seed": 1,
                },
            )
            research_storage.write_allocation_marker(
                origin_storage,
                {"kind": "live", "station_id": "station-id"},
            )
            (job / research_storage.JOB_MANIFEST_FILENAME).write_text(
                yaml.safe_dump({
                    "base_path": str(storage_base),
                    "station_id": "station-id",
                    "job_id": "jobabc",
                    "origin": {"target": str(origin_storage), "owned": True},
                    "seeds": {"1": {"target": str(seed_storage)}},
                }),
                encoding="utf-8",
            )

            info = active_multistart_info(repo)
            self.assertIsNotNone(info)
            archive_dir = repo / "backup" / "station-id"
            manifest_path = create_active_multistart_archive(repo, info, archive_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = (
                "10_jobabc/station_data_s1/rooms/research/storage/"
                "lineages/alpha/artifact.txt"
            )
            self.assertIn(expected, {item["path"] for item in manifest["files"]})
            self.assertNotIn(
                "10_jobabc/station_data_s1/rooms/research/storage",
                {item["path"] for item in manifest.get("symlinks", [])},
            )
            self.assertIn(
                "10_jobabc/station_data_s2/rooms/research/storage/lineages/alpha/artifact.txt",
                {item["path"] for item in manifest["files"]},
            )
            self.assertIn(
                "10_jobabc/origin_station_data/rooms/research/storage/lineages/alpha/origin.txt",
                {item["path"] for item in manifest["files"]},
            )

            restored = root / "restored"
            restore_active_multistart_archive(manifest_path, restored)
            restored_storage = restored / "10_jobabc" / "station_data_s1" / "rooms" / "research" / "storage"
            self.assertFalse(restored_storage.is_symlink())
            self.assertEqual(
                "remote artifact",
                (restored_storage / "lineages" / "alpha" / "artifact.txt").read_text(encoding="utf-8"),
            )
            cleanup = remove_active_multistart_research_storage_allocations(info)
            self.assertTrue(cleanup["success"], cleanup)
            self.assertFalse(seed_storage.exists())
            self.assertFalse(origin_storage.exists())
            repeated_cleanup = remove_active_multistart_research_storage_allocations(info)
            self.assertTrue(repeated_cleanup["success"], repeated_cleanup)
            self.assertTrue(repeated_cleanup["already_missing"])

    def test_station_archive_accepts_active_multistart_without_live_station_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_9"
            self._write_active_job(repo)
            (repo / ".git").mkdir()
            (repo / "start.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            args = Namespace(targets=[str(repo)], yes=True, keep_backup_dir=True)
            config = ToolsConfig(station_patterns=(str(repo),))

            with (
                mock.patch.object(archive_command.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(archive_command, "_stop_station", return_value=True),
                mock.patch.object(archive_command, "_sudo_rm_rf", return_value=True) as remove_tree,
                mock.patch.object(
                    archive_command.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ) as run_command,
            ):
                result = archive_command.run(args, config)

            self.assertEqual(0, result)
            manifest = repo / "backup" / "station-id" / "multistart_archives" / "active_jobabc.json"
            self.assertTrue(manifest.is_file())
            zip_calls = [call.args[0] for call in run_command.call_args_list]
            self.assertTrue(any(command[:2] == ["zip", "-r"] for command in zip_calls))
            self.assertTrue(any(command[:2] == ["zip", "-T"] for command in zip_calls))
            self.assertTrue(
                any(
                    command[:3] == ["zip", "-r", "Test_tick_17_score_0-000000_ms.zip"]
                    for command in zip_calls
                )
            )
            removed_paths = [call.args[0] for call in remove_tree.call_args_list]
            self.assertIn(repo / "station_multistart", removed_paths)
            self.assertIn(repo / "station_data", removed_paths)

    def test_station_archive_retains_multistart_metadata_when_storage_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_9"
            self._write_active_job(repo)
            (repo / ".git").mkdir()
            (repo / "start.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            args = Namespace(targets=[str(repo)], yes=True, keep_backup_dir=True)
            config = ToolsConfig(station_patterns=(str(repo),))

            with (
                mock.patch.object(archive_command.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(archive_command, "_stop_station", return_value=True),
                mock.patch.object(archive_command, "_sudo_rm_rf", return_value=True) as remove_tree,
                mock.patch.object(
                    archive_command,
                    "remove_active_multistart_research_storage_allocations",
                    return_value={"success": False, "failures": ["mount unavailable"]},
                ),
                mock.patch.object(
                    archive_command.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
            ):
                result = archive_command.run(args, config)

            self.assertEqual(1, result)
            removed_paths = [call.args[0] for call in remove_tree.call_args_list]
            self.assertNotIn(repo / "station_multistart", removed_paths)
            self.assertNotIn(repo / "station_data", removed_paths)

    def test_station_archive_preserves_normal_station_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_4"
            station_data = repo / "station_data"
            station_data.mkdir(parents=True)
            (station_data / "station_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "station_id": "normal-id",
                        "station_name": "Normal Station",
                        "current_tick": 8,
                        "top_score": 1.23,
                    }
                ),
                encoding="utf-8",
            )
            snapshots = repo / "backup" / "normal-id" / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_8.json").write_text("{}", encoding="utf-8")
            (repo / ".git").mkdir()
            (repo / "start.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            args = Namespace(targets=[str(repo)], yes=True, keep_backup_dir=True)
            config = ToolsConfig(station_patterns=(str(repo),))

            with (
                mock.patch.object(archive_command.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(archive_command, "_stop_station", return_value=True),
                mock.patch.object(archive_command, "_sudo_rm_rf", return_value=True) as remove_tree,
                mock.patch.object(
                    archive_command.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ) as run_command,
                mock.patch.object(archive_command, "create_active_multistart_archive") as active_archive,
            ):
                result = archive_command.run(args, config)

            self.assertEqual(0, result)
            active_archive.assert_not_called()
            removed_paths = [call.args[0] for call in remove_tree.call_args_list]
            self.assertIn(repo / "station_data", removed_paths)
            self.assertNotIn(repo / "station_multistart", removed_paths)
            commands = [call.args[0] for call in run_command.call_args_list]
            self.assertIn(["zip", "-r", "Normal_Station_tick_8_score_1-230000.zip", "normal-id"], commands)
            self.assertIn(["zip", "-T", "Normal_Station_tick_8_score_1-230000.zip"], commands)

    def test_station_archive_removes_managed_normal_research_storage_after_zip_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_4"
            station_data = repo / "station_data"
            station_data.mkdir(parents=True)
            (station_data / "station_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "station_id": "normal-id",
                        "station_name": "Normal Station",
                        "current_tick": 8,
                    }
                ),
                encoding="utf-8",
            )
            snapshots = repo / "backup" / "normal-id" / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_8.json").write_text("{}", encoding="utf-8")
            (repo / ".git").mkdir()
            (repo / "start.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            allocation = Path(tmp) / "remote_storage" / "11111111-1111-1111-1111-111111111111"
            allocation.mkdir(parents=True)
            (allocation / "artifact.txt").write_text("remote artifact", encoding="utf-8")
            research_storage.write_allocation_marker(
                allocation,
                {"kind": "live", "station_id": "normal-id"},
            )
            research_root = station_data / "rooms" / "research"
            research_root.mkdir(parents=True)
            (research_root / "storage").symlink_to(allocation, target_is_directory=True)

            args = Namespace(targets=[str(repo)], yes=True, keep_backup_dir=True)
            config = ToolsConfig(station_patterns=(str(repo),))

            def remove_local(path):
                research_storage.remove_tree_allow_read_only(path)
                return True

            def run_zip(command, **_kwargs):
                if command[:2] == ["zip", "-T"]:
                    self.assertTrue(allocation.is_dir())
                return SimpleNamespace(returncode=0)

            with (
                mock.patch.object(archive_command.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(archive_command, "_stop_station", return_value=True),
                mock.patch.object(archive_command, "_sudo_rm_rf", side_effect=remove_local),
                mock.patch.object(archive_command.subprocess, "run", side_effect=run_zip),
            ):
                result = archive_command.run(args, config)

            self.assertEqual(0, result)
            self.assertFalse(allocation.exists())
            self.assertFalse(research_storage.allocation_marker_path(allocation).exists())

    def test_station_archive_keeps_managed_normal_research_storage_when_zip_verification_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_4"
            station_data = repo / "station_data"
            station_data.mkdir(parents=True)
            (station_data / "station_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "station_id": "normal-id",
                        "station_name": "Normal Station",
                        "current_tick": 8,
                    }
                ),
                encoding="utf-8",
            )
            snapshots = repo / "backup" / "normal-id" / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_8.json").write_text("{}", encoding="utf-8")
            (repo / ".git").mkdir()
            (repo / "start.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            allocation = Path(tmp) / "remote_storage" / "11111111-1111-1111-1111-111111111111"
            allocation.mkdir(parents=True)
            research_storage.write_allocation_marker(
                allocation,
                {"kind": "live", "station_id": "normal-id"},
            )
            research_root = station_data / "rooms" / "research"
            research_root.mkdir(parents=True)
            (research_root / "storage").symlink_to(allocation, target_is_directory=True)

            args = Namespace(targets=[str(repo)], yes=True, keep_backup_dir=True)
            config = ToolsConfig(station_patterns=(str(repo),))

            def remove_local(path):
                research_storage.remove_tree_allow_read_only(path)
                return True

            def run_zip(command, **_kwargs):
                return SimpleNamespace(returncode=1 if command[:2] == ["zip", "-T"] else 0)

            with (
                mock.patch.object(archive_command.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(archive_command, "_stop_station", return_value=True),
                mock.patch.object(archive_command, "_sudo_rm_rf", side_effect=remove_local),
                mock.patch.object(archive_command.subprocess, "run", side_effect=run_zip),
            ):
                result = archive_command.run(args, config)

            self.assertEqual(1, result)
            self.assertTrue(allocation.is_dir())
            self.assertTrue(research_storage.allocation_marker_path(allocation).is_file())

    def test_station_archive_refuses_unowned_normal_research_storage_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_4"
            station_data = repo / "station_data"
            station_data.mkdir(parents=True)
            (station_data / "station_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "station_id": "normal-id",
                        "station_name": "Normal Station",
                        "current_tick": 8,
                    }
                ),
                encoding="utf-8",
            )
            snapshots = repo / "backup" / "normal-id" / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_8.json").write_text("{}", encoding="utf-8")
            (repo / ".git").mkdir()
            (repo / "start.sh").write_text("#!/bin/bash\n", encoding="utf-8")

            allocation = Path(tmp) / "unowned_storage"
            allocation.mkdir()
            research_root = station_data / "rooms" / "research"
            research_root.mkdir(parents=True)
            (research_root / "storage").symlink_to(allocation, target_is_directory=True)

            args = Namespace(targets=[str(repo)], yes=True, keep_backup_dir=True)
            config = ToolsConfig(station_patterns=(str(repo),))

            with (
                mock.patch.object(archive_command.shutil, "which", return_value="/usr/bin/tool"),
                mock.patch.object(archive_command, "_stop_station", return_value=True),
                mock.patch.object(archive_command, "_sudo_rm_rf") as remove_tree,
                mock.patch.object(archive_command.subprocess, "run") as run_command,
            ):
                result = archive_command.run(args, config)

            self.assertEqual(1, result)
            remove_tree.assert_not_called()
            run_command.assert_not_called()
            self.assertTrue(station_data.is_dir())
            self.assertTrue(allocation.is_dir())

    def test_restore_accepts_zip_path_after_backup_directory_was_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_9"
            self._write_active_job(repo)
            info = active_multistart_info(repo)
            backup_dir = repo / "backup" / "station-id"
            create_active_multistart_archive(repo, info, backup_dir)
            zip_path = repo / "backup" / "Test_tick_17_score_0-000000_ms.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in backup_dir.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(repo / "backup"))
            shutil.rmtree(backup_dir)
            root_zip = repo / zip_path.name
            shutil.move(zip_path, root_zip)
            shutil.rmtree(repo / "backup")

            script = Path(__file__).resolve().parents[1] / "scripts" / "restore.sh"
            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = project_root if not env.get("PYTHONPATH") else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            result = subprocess.run(
                ["bash", str(script), "--output", "restored", "station-id"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("Extracting archived backup", result.stdout)
            self.assertIn("Using active multistart archive at tick 17", result.stdout)
            self.assertTrue((repo / "restored" / "current_job.yaml").is_file())

    def test_restore_accepts_general_normal_station_zip_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            repo.mkdir()
            source_backup = Path(tmp) / "normal-id"
            snapshots = source_backup / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_3.json").write_text(
                json.dumps({"station_id": "normal-id", "tick": 3, "files": [], "symlinks": []}),
                encoding="utf-8",
            )
            zip_path = Path(tmp) / "completely_custom_name.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in source_backup.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(Path(tmp)))

            script = Path(__file__).resolve().parents[1] / "scripts" / "restore.sh"
            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = project_root if not env.get("PYTHONPATH") else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            result = subprocess.run(
                ["bash", str(script), "--output", "restored", str(zip_path)],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("Using latest ordinary tick: 3", result.stdout)
            self.assertTrue((repo / "restored").is_dir())
            self.assertFalse(zip_path.exists())
            self.assertIn("Removed station archive", result.stdout)

    def test_restore_keeps_zip_when_restore_is_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            repo.mkdir()
            source_backup = Path(tmp) / "normal-id"
            snapshots = source_backup / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_3.json").write_text(
                json.dumps({"station_id": "normal-id", "tick": 3, "files": [], "symlinks": []}),
                encoding="utf-8",
            )
            zip_path = Path(tmp) / "cancelled_restore.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in source_backup.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(Path(tmp)))
            (repo / "restored").mkdir()

            script = Path(__file__).resolve().parents[1] / "scripts" / "restore.sh"
            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = project_root if not env.get("PYTHONPATH") else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            result = subprocess.run(
                ["bash", str(script), "--output", "restored", str(zip_path)],
                cwd=repo,
                env=env,
                input="n\n",
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertTrue(zip_path.is_file())
            self.assertIn("Restore cancelled by user", result.stdout)

    def test_restore_replaces_read_only_target_without_following_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            backup = repo / "backup" / "normal-id"
            snapshots = backup / "snapshots"
            snapshots.mkdir(parents=True)
            (snapshots / "tick_3.json").write_text(
                json.dumps({"station_id": "normal-id", "tick": 3, "files": [], "symlinks": []}),
                encoding="utf-8",
            )
            restored = repo / "restored"
            read_only = restored / "rooms" / "research" / "storage" / "system"
            read_only.mkdir(parents=True)
            (read_only / "old.txt").write_text("old", encoding="utf-8")
            external = repo / "external"
            external.mkdir()
            external_file = external / "keep.txt"
            external_file.write_text("keep", encoding="utf-8")
            (read_only / "external-link").symlink_to(external, target_is_directory=True)
            os.chmod(read_only, 0o555)

            script = Path(__file__).resolve().parents[1] / "scripts" / "restore.sh"
            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = project_root if not env.get("PYTHONPATH") else f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
            result = subprocess.run(
                ["bash", str(script), "--output", "restored", "normal-id"],
                cwd=repo,
                env=env,
                input="y\n",
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertTrue(restored.is_dir())
            self.assertFalse((restored / "rooms").exists())
            self.assertEqual("keep", external_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
