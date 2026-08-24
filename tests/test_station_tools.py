from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import patch
import io
import json
import os
import subprocess
import sqlite3
import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path

import yaml

from station import startup_overrides
from station_tools.cli import build_parser
from station_tools.commands import init as init_command
from station_tools.commands import resume as resume_command
from station_tools.commands import restore as restore_command
from station_tools.commands import update as update_command
from station_tools.frontend_api import ApiEndpoint
from station_tools.config import ToolsConfig, default_config_path, load_config
from station_tools.selectors import (
    discover_repos,
    select_repos,
    split_target_tokens,
    suffix_for_repo,
    targets_or_current,
)
from station_tools.station_templates import (
    DEFAULT_STATION_TEMPLATE_SOURCE,
    STATION_TEMPLATE_SOURCE_KEY,
    configured_station_template_source,
    refresh_station_template_files,
    resolve_station_template,
)


def load_monitor_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "monitor_station.py"
    spec = importlib.util.spec_from_file_location("test_monitor_station", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StationToolsTest(unittest.TestCase):
    def test_restore_parser_accepts_uuid_and_general_zip(self):
        uuid_args = build_parser().parse_args(["restore", "abc123"])
        self.assertEqual("abc123", uuid_args.source)
        self.assertIsNone(uuid_args.tick)

        zip_args = build_parser().parse_args(
            ["restore", "/tmp/station_archive.zip", "42", "--output", "restored"]
        )
        self.assertEqual("/tmp/station_archive.zip", zip_args.source)
        self.assertEqual("42", zip_args.tick)
        self.assertEqual("restored", zip_args.output)

    def test_restore_command_runs_checkout_restore_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_3"
            script = repo / "scripts" / "restore.sh"
            script.parent.mkdir(parents=True)
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            archive_path = Path(tmp) / "portable.zip"
            archive_path.write_bytes(b"zip")
            args = Namespace(
                source=str(archive_path),
                tick=None,
                output=None,
            )

            with (
                patch.object(restore_command, "_resolve_repo", return_value=repo.resolve()),
                patch.object(
                    restore_command.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as run_command,
            ):
                result = restore_command.run(args, ToolsConfig())

            self.assertEqual(0, result)
            run_command.assert_called_once_with(
                ["bash", str(script), str(archive_path.resolve())],
                cwd=repo.resolve(),
                check=False,
            )

    def make_station(self, root: Path, name: str, station_name: str = "Test") -> Path:
        repo = root / name
        (repo / "station_data").mkdir(parents=True)
        (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (repo / ".git").mkdir()
        (repo / "station_data" / "station_config.yaml").write_text(
            yaml.safe_dump({"station_name": station_name, "station_id": f"id-{name}", "current_tick": 7}),
            encoding="utf-8",
        )
        return repo

    def write_research_index(self, data_root: Path, active_coders: int) -> None:
        evaluations_dir = data_root / "rooms" / "research" / "evaluations"
        index_dir = data_root / "index"
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "station_index.sqlite3"
        scope = str(evaluations_dir.resolve())
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE research_evaluations (
                    evaluations_dir TEXT NOT NULL,
                    eval_id TEXT NOT NULL,
                    active_coder INTEGER NOT NULL DEFAULT 0,
                    coder_active INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            for index in range(active_coders):
                conn.execute(
                    "INSERT INTO research_evaluations(evaluations_dir, eval_id, active_coder, coder_active) VALUES (?, ?, 1, 1)",
                    (scope, str(index + 1)),
                )
            conn.execute(
                "INSERT INTO research_evaluations(evaluations_dir, eval_id, active_coder, coder_active) VALUES (?, ?, 0, 0)",
                (scope, str(active_coders + 1)),
            )

    def test_suffix_for_repo(self):
        self.assertEqual(suffix_for_repo(Path("/x/station")), "1")
        self.assertEqual(suffix_for_repo(Path("/x/station_3")), "3")
        self.assertEqual(suffix_for_repo(Path("/x/station_abc")), "abc")

    def test_split_target_tokens_accepts_commas_and_spaces(self):
        self.assertEqual(split_target_tokens(["1,2", "abc", "4, 5"]), ["1", "2", "abc", "4", "5"])

    def test_discover_and_select_by_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            station = self.make_station(root, "station")
            station_abc = self.make_station(root, "station_abc")
            patterns = (str(root / "station"), str(root / "station_*"))

            self.assertEqual(discover_repos(patterns), (station, station_abc))
            selection = select_repos(["abc"], patterns)
            self.assertEqual(selection.repos, (station_abc,))

    def test_monitor_discovers_active_multistart_without_live_station_data(self):
        monitor = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "station_2"
            job = repo / "station_multistart" / "502_jobabc"
            branch = job / "station_data_s1"
            branch_2 = job / "station_data_s2"
            branch.mkdir(parents=True)
            branch_2.mkdir(parents=True)
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / ".git").mkdir()
            (repo / "station_multistart" / "current_job.yaml").write_text(
                yaml.safe_dump(
                    {
                        "job_id": "jobabc",
                        "mode": "stagnation",
                        "status": "running",
                        "branch_tick": 502,
                        "seed_count": 2,
                        "roll_ticks": 40,
                        "job_dir": str(job),
                    }
                ),
                encoding="utf-8",
            )
            (job / "state.yaml").write_text(
                yaml.safe_dump(
                    {
                        "job_id": "jobabc",
                        "mode": "stagnation",
                        "status": "running",
                        "station_name": "Multistart Station",
                        "origin_station_id": "origin-id",
                        "branch_tick": 502,
                        "seed_count": 2,
                        "roll_ticks": 40,
                        "branches": [
                            {
                                "seed": 1,
                                "data_root": str(branch),
                                "status": "running",
                                "start_tick": 502,
                                "target_tick": 542,
                                "current_tick": 510,
                            },
                            {
                                "seed": 2,
                                "data_root": str(branch_2),
                                "status": "completed",
                                "start_tick": 502,
                                "target_tick": 542,
                                "current_tick": 542,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (branch / "station_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "station_name": "Seed 1",
                        "station_id": "origin-id",
                        "current_tick": 511,
                        "top_tick": 502,
                        "top_evaluation_id": "20",
                        "top_score": 12.0,
                        "top_sort_key": [1.0, 0.0],
                        "stagnation_counter": 329,
                        "station_status": "Stagnation I",
                        "status_history": [
                            {"status": "Healthy", "start_tick": 1},
                            {"status": "Stagnation I", "start_tick": 502},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (branch_2 / "station_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "station_name": "Seed 2",
                        "station_id": "origin-id",
                        "current_tick": 542,
                        "top_tick": 530,
                        "top_evaluation_id": "21",
                        "top_score": 1.0,
                        "top_sort_key": [1.0, 1.0],
                        "stagnation_counter": 12,
                        "station_status": "Healthy",
                    }
                ),
                encoding="utf-8",
            )
            self.write_research_index(branch, active_coders=2)

            paths = monitor.discover_station_paths((str(root / "station_*"),))
            self.assertEqual(paths, (repo,))

            config = monitor.MonitorConfig(
                cpu_path=Path("/tmp/missing_cpu.json"),
                gpu_path=Path("/tmp/missing_gpu.json"),
                station_patterns=(str(root / "station_*"),),
                cpu_ids=None,
                gpu_ids=None,
                cpu_total=None,
                gpu_total=None,
                interval=5,
                history_size=60,
                chart_window_seconds=3600,
                chart_buckets=60,
                chart_height=8,
                show_chart=False,
                lock_timeout=1,
                external_scheme="https",
                external_host="",
                external_port_base=None,
                external_port_step=1,
                show_external_links=False,
                use_local_api_status=False,
                local_api_timeout=0.1,
                once=True,
                clear=False,
                color=False,
            )
            snapshots = monitor.collect_station_snapshots(config, 1000.0)
            self.assertEqual(1, len(snapshots))
            self.assertEqual("Multistart Station", snapshots[0].name)
            self.assertIn("multistart rolling branches 1/2", snapshots[0].status_summary)
            self.assertIn("best=s2", snapshots[0].status_summary)
            self.assertEqual(682, snapshots[0].next_stagnation_tick)
            self.assertEqual(2, snapshots[0].active_coders)
            self.assertEqual(1.0, snapshots[0].top_score)
            self.assertEqual(530, snapshots[0].top_tick)
            self.assertEqual(329, snapshots[0].ticks_since_last_breakthrough)

    def test_monitor_multistart_missing_sort_key_falls_back_to_score(self):
        monitor = load_monitor_module()
        lower = monitor.MultistartTopRecord(
            sort_key=monitor.normalize_top_sort_key(None, 1.0),
            score=1.0,
            top_tick=10,
            evaluation_id="1",
            seed=1,
        )
        higher = monitor.MultistartTopRecord(
            sort_key=monitor.normalize_top_sort_key(None, 12.0),
            score=12.0,
            top_tick=20,
            evaluation_id="2",
            seed=2,
        )

        self.assertTrue(monitor.multistart_top_record_is_better(higher, lower))
        self.assertFalse(monitor.multistart_top_record_is_better(lower, higher))

    def test_monitor_exact_top_comparison_does_not_apply_breakthrough_eps(self):
        monitor = load_monitor_module()
        current = monitor.MultistartTopRecord(
            sort_key=(1.0,),
            score=1.0,
            top_tick=10,
            evaluation_id="1",
            seed=1,
        )
        candidate = monitor.MultistartTopRecord(
            sort_key=(1.005,),
            score=1.005,
            top_tick=20,
            evaluation_id="2",
            seed=2,
        )

        self.assertTrue(monitor.multistart_top_record_is_better(candidate, current))

    def test_monitor_counts_normal_station_active_coders(self):
        monitor = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self.make_station(root, "station", station_name="Normal Station")
            station_config_path = repo / "station_data" / "station_config.yaml"
            station_config = yaml.safe_load(station_config_path.read_text(encoding="utf-8"))
            station_config["stagnation_counter"] = 186
            station_config_path.write_text(yaml.safe_dump(station_config), encoding="utf-8")
            self.write_research_index(repo / "station_data", active_coders=3)
            config = monitor.MonitorConfig(
                cpu_path=Path("/tmp/missing_cpu.json"),
                gpu_path=Path("/tmp/missing_gpu.json"),
                station_patterns=(str(root / "station"),),
                cpu_ids=None,
                gpu_ids=None,
                cpu_total=None,
                gpu_total=None,
                interval=5,
                history_size=60,
                chart_window_seconds=3600,
                chart_buckets=60,
                chart_height=8,
                show_chart=False,
                lock_timeout=1,
                external_scheme="https",
                external_host="",
                external_port_base=None,
                external_port_step=1,
                show_external_links=False,
                use_local_api_status=False,
                local_api_timeout=0.1,
                once=True,
                clear=False,
                color=False,
            )
            snapshots = monitor.collect_station_snapshots(config, 1000.0)
            self.assertEqual(1, len(snapshots))
            self.assertEqual(3, snapshots[0].active_coders)
            self.assertEqual(186, snapshots[0].ticks_since_last_breakthrough)

            table = "\n".join(monitor.render_status_table(snapshots, 160, monitor.Palette(False)))
            self.assertIn("NEXT", table)
            self.assertIn("COD", table)
            self.assertIn("ERS", table)
            self.assertIn("STATUS", table)
            self.assertIn("RUNNING", table)
            self.assertIn("FOLDER", table)
            self.assertIn("Normal Station", table)
            self.assertIn("CODERS", table)
            self.assertIn("BT", table)

    def test_monitor_running_status_prefixes_human_intervention_count(self):
        monitor = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self.make_station(root, "station", station_name="Human Station")
            agents_dir = repo / "station_data" / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / "Ada I.yaml").write_text(
                "\n".join(
                    [
                        "awaiting_human_intervention: true",
                        "human_interaction_ids:",
                        "  - request-a",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = monitor.MonitorConfig(
                cpu_path=Path("/tmp/missing_cpu.json"),
                gpu_path=Path("/tmp/missing_gpu.json"),
                station_patterns=(str(root / "station"),),
                cpu_ids=None,
                gpu_ids=None,
                cpu_total=None,
                gpu_total=None,
                interval=5,
                history_size=60,
                chart_window_seconds=3600,
                chart_buckets=60,
                chart_height=8,
                show_chart=False,
                lock_timeout=1,
                external_scheme="https",
                external_host="",
                external_port_base=None,
                external_port_step=1,
                show_external_links=False,
                use_local_api_status=False,
                local_api_timeout=0.1,
                once=True,
                clear=False,
                color=False,
            )
            snapshots = monitor.collect_station_snapshots(config, 1000.0)
            self.assertEqual(1, snapshots[0].human_intervention_count)
            table = "\n".join(monitor.render_status_table(snapshots, 160, monitor.Palette(False)))
            self.assertIn("1 human intervention needed; Status API unavailable", table)

    def test_monitor_colors_disk_usage_by_pressure_threshold(self):
        monitor = load_monitor_module()
        palette = monitor.Palette(True)

        self.assertEqual(
            f"{palette.green}69.9%{palette.reset}",
            monitor.color_disk_usage(69.9, "69.9%", palette),
        )
        self.assertEqual(
            f"{palette.yellow}70.0%{palette.reset}",
            monitor.color_disk_usage(70.0, "70.0%", palette),
        )
        self.assertEqual(
            f"{palette.red}90.0%{palette.reset}",
            monitor.color_disk_usage(90.0, "90.0%", palette),
        )
        self.assertEqual(
            "95.0%",
            monitor.color_disk_usage(95.0, "95.0%", monitor.Palette(False)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            usage = Namespace(total=100, used=91, free=9)
            station = Namespace(path=Path(tmp))
            with patch.object(monitor.shutil, "disk_usage", return_value=usage):
                summary = monitor.disk_usage_summary((station,), palette)
        self.assertIn(f"{palette.red}91.0%{palette.reset}", summary)

    def test_monitor_human_count_prefers_station_statistics_endpoint(self):
        monitor = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self.make_station(root, "station", station_name="Human Station")
            config = monitor.MonitorConfig(
                cpu_path=Path("/tmp/missing_cpu.json"),
                gpu_path=Path("/tmp/missing_gpu.json"),
                station_patterns=(str(root / "station"),),
                cpu_ids=None,
                gpu_ids=None,
                cpu_total=None,
                gpu_total=None,
                interval=5,
                history_size=60,
                chart_window_seconds=3600,
                chart_buckets=60,
                chart_height=8,
                show_chart=False,
                lock_timeout=1,
                external_scheme="https",
                external_host="",
                external_port_base=None,
                external_port_step=1,
                show_external_links=False,
                use_local_api_status=True,
                local_api_timeout=0.1,
                once=True,
                clear=False,
                color=False,
            )
            with patch.object(
                monitor,
                "fetch_local_station_statistics",
                return_value={"pending_human_requests": {"agents": ["Ada I", "Babbage II"]}},
            ), patch.object(
                monitor,
                "fetch_local_orchestrator_status",
                return_value={"is_running": True, "agents_awaiting_human": ["Ada I"]},
            ):
                running_status, human_count = monitor.build_runtime_status_summary(
                    repo,
                    repo / "station_data",
                    {"station_status": "Healthy"},
                    config,
                )
            self.assertEqual("Running", running_status)
            self.assertEqual(2, human_count)

    def test_monitor_reports_pending_multistart_disk_space_instead_of_generic_pause(self):
        monitor = load_monitor_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self.make_station(root, "station", station_name="Blocked Station")
            pending_dir = repo / "station_multistart"
            pending_dir.mkdir()
            (pending_dir / "pending_stagnation.yaml").write_text(
                yaml.safe_dump(
                    {
                        "status": "blocked_disk_space",
                        "branch_tick": 7,
                        "disk_space": {"must_free_bytes": 13_636_237_188},
                    }
                ),
                encoding="utf-8",
            )
            config = monitor.MonitorConfig(
                cpu_path=Path("/tmp/missing_cpu.json"),
                gpu_path=Path("/tmp/missing_gpu.json"),
                station_patterns=(str(repo),),
                cpu_ids=None,
                gpu_ids=None,
                cpu_total=None,
                gpu_total=None,
                interval=5,
                history_size=60,
                chart_window_seconds=3600,
                chart_buckets=60,
                chart_height=8,
                show_chart=False,
                lock_timeout=1,
                external_scheme="https",
                external_host="",
                external_port_base=None,
                external_port_step=1,
                show_external_links=False,
                use_local_api_status=True,
                local_api_timeout=0.1,
                once=True,
                clear=False,
                color=False,
            )
            with patch.object(
                monitor,
                "fetch_local_station_statistics",
                return_value={"pending_human_requests": {"agents": []}},
            ), patch.object(
                monitor,
                "fetch_local_orchestrator_status",
                return_value={
                    "is_paused": True,
                    "pause_reason": (
                        "Stagnation multistart requested at tick 7; "
                        "waiting for controller branch selection."
                    ),
                },
            ):
                running_status, human_count = monitor.build_runtime_status_summary(
                    repo,
                    repo / "station_data",
                    {"current_tick": 7},
                    config,
                )

            self.assertEqual(
                "Stagnation multistart blocked: insufficient disk space; est. extra 12.7 GiB needed",
                running_status,
            )
            self.assertEqual(0, human_count)

    def test_load_config_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp) / "station_tools.toml")
            self.assertEqual(config.station_patterns, ("~/station", "~/station_*"))

    def test_resume_uses_multistart_endpoint_in_wait_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self.make_station(root, "station", station_name="Multistart Station")
            args = Namespace(targets=[], status_only=False, timeout=3.0)
            config = ToolsConfig(station_patterns=(str(repo),))
            endpoint = ApiEndpoint(base_url="https://127.0.0.1:8443", auth_header=None)

            with patch(
                "station_tools.commands.resume.find_endpoint",
                return_value=(
                    endpoint,
                    {
                        "success": True,
                        "status": {
                            "is_running": False,
                            "is_paused": True,
                            "current_tick": -1,
                            "multistart": {"stage": "halted"},
                        },
                    },
                ),
            ), patch(
                "station_tools.commands.resume.request_json",
                return_value={"success": True, "message": "Resume requested for paused/pending branches."},
            ) as request:
                with redirect_stdout(io.StringIO()) as output:
                    rc = resume_command.run(args, config)

            self.assertEqual(0, rc)
            request.assert_called_once_with(endpoint, "/api/multistart/resume", method="POST", timeout=3.0)
            self.assertIn("multistart stage=halted", output.getvalue())

    def test_default_config_path_is_user_config(self):
        self.assertEqual(default_config_path(), Path.home() / ".config" / "station-tools" / "station_tools.toml")

    def test_load_config_reads_hooks_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "station_tools.toml"
            path.write_text(
                """
station_patterns = ["~/station_*"]

[env]
PATH = "/mnt/stephen/bin:${PATH}"

[hooks.update]
before_start = "api 5"
""",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.station_patterns, ("~/station_*",))
            self.assertEqual(config.env["PATH"], "/mnt/stephen/bin:${PATH}")
            self.assertEqual(config.hooks["update"]["before_start"], "api 5")

    def test_multi_init_is_not_registered_as_station_tools_command(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["multi_init", "3", "8", "kissing_free", "Station Name"])

    def test_update_runner_skips_start_for_inactive_station(self):
        config = ToolsConfig(hooks={"update": {"before_start": "export TEST_API_KEY=present"}})
        args = Namespace(no_hooks=False, git_pull_timeout=300, force=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "station_2"
            repo.mkdir()
            runner = tmp_path / "runner.sh"
            status = tmp_path / "status"
            log = tmp_path / "log"

            update_command._make_runner(
                repo,
                repo.name,
                runner,
                status,
                log,
                config,
                args,
                start_after_update=False,
            )

            script = runner.read_text(encoding="utf-8")

        self.assertIn("start_after_update=0", script)
        self.assertIn('write_status "UPDATED:$(date -Is)"', script)
        self.assertIn("skipping start.sh", script)
        self.assertLess(script.index('exit 0\nfi'), script.index("python -m station_tools.station_templates"))

    def test_update_runner_starts_active_station(self):
        config = ToolsConfig()
        args = Namespace(no_hooks=False, git_pull_timeout=300, force=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "station_2"
            repo.mkdir()
            runner = tmp_path / "runner.sh"
            status = tmp_path / "status"
            log = tmp_path / "log"

            update_command._make_runner(
                repo,
                repo.name,
                runner,
                status,
                log,
                config,
                args,
                start_after_update=True,
            )

            script = runner.read_text(encoding="utf-8")

        self.assertIn("start_after_update=1", script)
        self.assertIn('start_args=(-s)', script)
        self.assertIn('./stop.sh "${stop_args[@]}"', script)
        self.assertIn('./start.sh "${start_args[@]}"', script)
        self.assertIn("force_start=0", script)
        self.assertLess(
            script.index('./stop.sh "${stop_args[@]}"'),
            script.index("git_pull_with_retry\nif"),
        )
        self.assertIn(f"station_template_source={DEFAULT_STATION_TEMPLATE_SOURCE}", script)
        self.assertEqual("git-updated", update_command._status_label("UPDATED:2026-01-01T00:00:00"))

    def test_update_runner_uses_persisted_station_template_source(self):
        config = ToolsConfig()
        args = Namespace(no_hooks=False, git_pull_timeout=300, force=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "station_2"
            (repo / "station_data").mkdir(parents=True)
            (repo / "station_data" / "station_config.yaml").write_text(
                yaml.safe_dump({STATION_TEMPLATE_SOURCE_KEY: "example_private/station/custom"}),
                encoding="utf-8",
            )
            runner = tmp_path / "runner.sh"

            update_command._make_runner(
                repo,
                repo.name,
                runner,
                tmp_path / "status",
                tmp_path / "log",
                config,
                args,
                start_after_update=True,
            )

            script = runner.read_text(encoding="utf-8")

        self.assertIn("station_template_source=example_private/station/custom", script)
        self.assertIn('template_dir="$repo/$station_template_source"', script)
        self.assertIn("python -m station_tools.station_templates", script)
        self.assertIn('"$repo" "$station_template_source"', script)
        self.assertNotIn("mkdir -p station_data", script)

    def test_station_template_source_uses_active_multistart_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_2"
            job_dir = repo / "station_multistart" / "20_job"
            origin = job_dir / "origin_station_data"
            origin.mkdir(parents=True)
            (repo / "station_multistart" / "current_job.yaml").write_text(
                yaml.safe_dump({"job_dir": str(job_dir)}),
                encoding="utf-8",
            )
            (origin / "station_config.yaml").write_text(
                yaml.safe_dump({STATION_TEMPLATE_SOURCE_KEY: "example_private/station/custom"}),
                encoding="utf-8",
            )

            source = configured_station_template_source(repo)

        self.assertEqual("example_private/station/custom", source)

    def test_station_template_source_normalizes_legacy_multistart_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_5"
            job_dir = repo / "station_multistart" / "20_job"
            origin = job_dir / "origin_station_data"
            origin.mkdir(parents=True)
            (repo / "station_multistart" / "current_job.yaml").write_text(
                yaml.safe_dump({"job_dir": str(job_dir)}),
                encoding="utf-8",
            )
            (origin / "station_config.yaml").write_text(
                yaml.safe_dump({STATION_TEMPLATE_SOURCE_KEY: "example/station_default"}),
                encoding="utf-8",
            )

            source = configured_station_template_source(repo)

        self.assertEqual("example/station/default", source)

    def test_template_refresh_updates_every_active_multistart_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_2"
            template = repo / DEFAULT_STATION_TEMPLATE_SOURCE
            template.mkdir(parents=True)
            (template / "meta_prompts.yaml").write_text("new-meta\n", encoding="utf-8")
            (template / "random_prompts.yaml").write_text("new-random\n", encoding="utf-8")
            (template / "codex.md").write_text("new-codex\n", encoding="utf-8")

            job_dir = repo / "station_multistart" / "20_job"
            origin = job_dir / "origin_station_data"
            branch_1 = job_dir / "station_data_s1"
            branch_2 = job_dir / "station_data_s2"
            for data_root in (origin, branch_1, branch_2):
                data_root.mkdir(parents=True)
                (data_root / "meta_prompts.yaml").write_text("old\n", encoding="utf-8")
            (repo / "station_multistart" / "current_job.yaml").write_text(
                yaml.safe_dump({"job_dir": str(job_dir)}),
                encoding="utf-8",
            )
            (job_dir / "state.yaml").write_text(
                yaml.safe_dump(
                    {
                        "branches": [
                            {"seed": 1, "data_root": str(branch_1)},
                            {"seed": 2, "data_root": str(branch_2)},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stray_live_root = repo / "station_data"
            stray_live_root.mkdir()
            (stray_live_root / "meta_prompts.yaml").write_text("stray\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "station_tools.station_templates",
                    str(repo),
                    DEFAULT_STATION_TEMPLATE_SOURCE,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            refreshed = (origin, branch_1, branch_2)
            for data_root in refreshed:
                self.assertIn(f"refreshed station template files in {data_root}", result.stdout)
                self.assertEqual("new-meta\n", (data_root / "meta_prompts.yaml").read_text(encoding="utf-8"))
                self.assertEqual("new-random\n", (data_root / "random_prompts.yaml").read_text(encoding="utf-8"))
                self.assertEqual("new-codex\n", (data_root / "codex.md").read_text(encoding="utf-8"))
            self.assertEqual("stray\n", (stray_live_root / "meta_prompts.yaml").read_text(encoding="utf-8"))

    def test_template_refresh_preserves_normal_station_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station_2"
            template = repo / DEFAULT_STATION_TEMPLATE_SOURCE
            template.mkdir(parents=True)
            (template / "meta_prompts.yaml").write_text("new-meta\n", encoding="utf-8")

            refreshed = refresh_station_template_files(repo, DEFAULT_STATION_TEMPLATE_SOURCE)

            self.assertEqual((repo / "station_data",), refreshed)
            self.assertEqual(
                "new-meta\n",
                (repo / "station_data" / "meta_prompts.yaml").read_text(encoding="utf-8"),
            )

    def test_update_runner_forwards_force_to_start(self):
        config = ToolsConfig()
        args = Namespace(no_hooks=False, git_pull_timeout=300, force=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "station_2"
            repo.mkdir()
            runner = tmp_path / "runner.sh"
            status = tmp_path / "status"
            log = tmp_path / "log"

            update_command._make_runner(
                repo,
                repo.name,
                runner,
                status,
                log,
                config,
                args,
                start_after_update=True,
            )

            script = runner.read_text(encoding="utf-8")

        self.assertIn("force_start=1", script)
        self.assertIn("start_args+=(--force)", script)

    def test_update_parser_accepts_force(self):
        args = build_parser().parse_args(["update", "3", "--force"])
        self.assertEqual("update", args.command)
        self.assertTrue(args.force)

    def test_init_parser_accepts_post_copy_cmd(self):
        init_args = build_parser().parse_args(
            ["init", "3", "kissing_margin", "Station Name", "--post-copy-cmd", "bash replace_n.sh 4"]
        )
        self.assertEqual("bash replace_n.sh 4", init_args.post_copy_cmd)

    def test_init_parser_accepts_test_mode(self):
        init_args = build_parser().parse_args(["init", "3", "kissing_margin", "Station Name", "--test"])
        self.assertTrue(init_args.test)

    def test_init_parser_accepts_no_spawn(self):
        init_args = build_parser().parse_args(["init", "book", "--no-spawn"])
        self.assertTrue(init_args.no_spawn)

    def test_init_parser_accepts_multistart(self):
        init_args = build_parser().parse_args(["init", "book", "--multistart"])
        self.assertTrue(init_args.multistart)

    def test_init_parser_accepts_no_multistart(self):
        init_args = build_parser().parse_args(["init", "3", "kissing_margin", "Station Name", "--no-multistart"])
        self.assertTrue(init_args.no_multistart)

    def test_init_parser_accepts_no_start(self):
        init_args = build_parser().parse_args(["init", "book", "--no-start"])
        self.assertTrue(init_args.no_start)

    def test_init_parser_accepts_station_template(self):
        init_args = build_parser().parse_args(
            ["init", "3", "kissing_margin", "Station Name", "--station-template", "custom"]
        )
        self.assertEqual("custom", init_args.station_template)

    def test_init_defaults_to_current_checkout_and_task_derived_name(self):
        args = build_parser().parse_args(["init", "book"])
        repo = Path("/tmp/current-station")

        with patch.object(init_command.Path, "cwd", return_value=repo):
            request = init_command._parse_init_request(args)

        self.assertEqual(repo, request.repo)
        self.assertEqual("book", request.task_name)
        self.assertEqual("Book", request.station_name)

    def test_init_accepts_optional_name_and_explicit_station_id(self):
        args = build_parser().parse_args(
            ["init", "--station-id", "3", "epoch/book", "Book Problem"]
        )

        request = init_command._parse_init_request(args)

        self.assertEqual(Path.home() / "station_3", request.repo)
        self.assertEqual("epoch/book", request.task_name)
        self.assertEqual("Book Problem", request.station_name)

    def test_init_preserves_legacy_positional_station_id(self):
        args = build_parser().parse_args(["init", "3", "book", "Book Problem"])

        request = init_command._parse_init_request(args)

        self.assertEqual(Path.home() / "station_3", request.repo)
        self.assertEqual("book", request.task_name)
        self.assertEqual("Book Problem", request.station_name)

    def test_omitted_command_targets_default_to_current_station_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            with patch("station_tools.selectors.Path.cwd", return_value=repo):
                targets = targets_or_current([])

        self.assertEqual((str(repo),), targets)

    def test_research_task_resolution_scans_every_group_and_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            public = repo / "example" / "research_epoch" / "book"
            private = repo / "example_private" / "research_misc" / "book"
            public.mkdir(parents=True)
            private.mkdir(parents=True)

            candidates = init_command._find_task_candidates(repo, "book")

        self.assertEqual(
            ["example/research_epoch/book", "example_private/research_misc/book"],
            [candidate.source for candidate in candidates],
        )

    def test_research_task_resolution_accepts_group_and_canonical_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            public = repo / "example" / "research_epoch" / "book"
            private = repo / "example_private" / "research_misc" / "book"
            public.mkdir(parents=True)
            private.mkdir(parents=True)

            grouped = init_command._find_task_candidates(repo, "epoch/book")
            canonical = init_command._find_task_candidates(
                repo,
                "example_private/research_misc/book",
            )
            legacy = init_command._find_task_candidates(repo, "epoch_book")

        self.assertEqual([public], [candidate.path for candidate in grouped])
        self.assertEqual([private], [candidate.path for candidate in canonical])
        self.assertEqual([public], [candidate.path for candidate in legacy])

    def test_research_task_ambiguity_prompts_interactive_user(self):
        candidates = [
            init_command.ResearchTaskCandidate(Path("/public"), "example/research_epoch/book", "epoch", "book"),
            init_command.ResearchTaskCandidate(
                Path("/private"),
                "example_private/research_epoch/book",
                "epoch",
                "book",
            ),
        ]

        with patch.object(init_command.sys, "stdin") as stdin:
            stdin.isatty.return_value = True
            with patch("builtins.input", return_value="2"):
                with redirect_stdout(io.StringIO()):
                    selected = init_command._choose_task_candidate(candidates)

        self.assertEqual(Path("/private"), selected.path)

    def test_research_task_ambiguity_fails_noninteractive_use(self):
        candidates = [
            init_command.ResearchTaskCandidate(Path("/one"), "example/research_epoch/book", "epoch", "book"),
            init_command.ResearchTaskCandidate(Path("/two"), "example/research_misc/book", "misc", "book"),
        ]

        with patch.object(init_command.sys, "stdin") as stdin:
            stdin.isatty.return_value = False
            output = io.StringIO()
            with redirect_stdout(output):
                selected = init_command._choose_task_candidate(candidates)

        self.assertIsNone(selected)
        self.assertIn("task name is ambiguous", output.getvalue())

    def test_station_template_resolution_prefers_example_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            private = repo / "example_private" / "station" / "custom"
            public = repo / "example" / "station" / "custom"
            private.mkdir(parents=True)
            public.mkdir(parents=True)

            path, source = resolve_station_template(repo, "custom")

        self.assertEqual(private, path)
        self.assertEqual("example_private/station/custom", source)

    def test_station_template_resolution_accepts_explicit_public_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            public = repo / "example" / "station" / "custom"
            public.mkdir(parents=True)

            path, source = resolve_station_template(repo, "example/station/custom")

        self.assertEqual(public, path)
        self.assertEqual("example/station/custom", source)

    def test_station_template_resolution_normalizes_old_flat_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            public = repo / "example" / "station" / "default"
            private = repo / "example_private" / "station" / "gpt-5-5"
            public.mkdir(parents=True)
            private.mkdir(parents=True)

            public_path, public_source = resolve_station_template(repo, "example/station_default")
            private_path, private_source = resolve_station_template(
                repo,
                "example_private/station_gpt-5-5",
            )

        self.assertEqual(public, public_path)
        self.assertEqual("example/station/default", public_source)
        self.assertEqual(private, private_path)
        self.assertEqual("example_private/station/gpt-5-5", private_source)

    def test_station_template_resolution_rejects_paths_outside_template_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                resolve_station_template(Path(tmp), "../station_custom")

    def test_gpt_5_5_station_template_configuration(self):
        repo = Path(__file__).resolve().parents[1]
        template = repo / "example" / "station" / "gpt-5-5"
        config = yaml.safe_load((template / "constant_config.yaml").read_text(encoding="utf-8"))
        init_agents = yaml.safe_load((template / "init_agents.yaml").read_text(encoding="utf-8"))

        self.assertEqual(200, config["SUPERVISOR_ASSIGNMENT_COOLDOWN_TICKS"])
        self.assertEqual("gpt-5.5", config["MULTISTART_ADMIN_MODEL_NAME"])
        self.assertEqual("OpenAI", config["AUTO_EVAL_ARCHIVE_MODEL_CLASS"])
        self.assertEqual("gpt-5.5", config["AUTO_EVAL_ARCHIVE_MODEL_NAME"])
        self.assertEqual("codex", config["RESEARCH_CODER_BACKEND"])
        self.assertEqual("gpt-5.5", config["RESEARCH_CODER_MODEL_NAME"])
        self.assertFalse(config["RESEARCH_CODER_AUDIT_ENABLED"])
        self.assertEqual("codex", config["ARCHIVE_SURVEY_BACKEND"])
        self.assertEqual("gpt-5.5", config["ARCHIVE_SURVEY_MODEL_NAME"])
        self.assertEqual("OpenAI", config["REFLECTION_META_MODEL_PROVIDER_CLASS"])
        self.assertEqual("gpt-5.5", config["REFLECTION_META_MODEL_NAME"])
        self.assertEqual(25, config["REFLECTION_META_INTERVAL"])
        self.assertEqual(2, init_agents.count("GPT-5.5"))
        self.assertNotIn("GPT-5.6 Sol", init_agents)

    def test_startup_test_overrides_station_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_data = Path(tmp) / "station_data"
            station_data.mkdir()
            (station_data / "constant_config.yaml").write_text("RESEARCH_EVAL_TIMEOUT: 123\n", encoding="utf-8")
            (station_data / "init_agents.yaml").write_text("- Old Model\n", encoding="utf-8")

            startup_overrides.apply_test_config(station_data)

            constants_config = yaml.safe_load((station_data / "constant_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(123, constants_config["RESEARCH_EVAL_TIMEOUT"])
            self.assertEqual(0, constants_config["MULTISTART_INIT_SEEDS"])
            self.assertEqual(0, constants_config["MULTISTART_STAGNATION_SEEDS"])
            self.assertEqual(1, constants_config["MULTISTART_INIT_MAX_PARALLEL"])
            self.assertEqual(20, constants_config["PAUSE_AFTER_TICK_END"])
            self.assertEqual(["GPT-5.5", "Gemini 3.1 Pro"], yaml.safe_load((station_data / "init_agents.yaml").read_text(encoding="utf-8")))
            self.assertFalse((station_data / "station_config.yaml").exists())

    def test_startup_no_multistart_overrides_station_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_data = Path(tmp) / "station_data"
            station_data.mkdir()
            (station_data / "constant_config.yaml").write_text("RESEARCH_EVAL_TIMEOUT: 123\n", encoding="utf-8")
            (station_data / "init_agents.yaml").write_text("- Existing Model\n", encoding="utf-8")

            startup_overrides.apply_no_multistart(station_data)

            constants_config = yaml.safe_load((station_data / "constant_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(123, constants_config["RESEARCH_EVAL_TIMEOUT"])
            self.assertEqual(0, constants_config["MULTISTART_INIT_SEEDS"])
            self.assertEqual(0, constants_config["MULTISTART_STAGNATION_SEEDS"])
            self.assertEqual(1, constants_config["MULTISTART_INIT_MAX_PARALLEL"])
            self.assertEqual(["Existing Model"], yaml.safe_load((station_data / "init_agents.yaml").read_text(encoding="utf-8")))
            self.assertNotIn("PAUSE_AFTER_TICK_END", constants_config)

    def test_init_writes_initial_station_config_name_and_template_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            station_data = repo / "station_data"
            station_data.mkdir()

            with patch.object(init_command.station_config, "current_git_commit", return_value="abc123"):
                init_command.write_initial_station_config(
                    station_data,
                    "Named Station",
                    repo,
                    "example_private/station/custom",
                )

            config = yaml.safe_load((station_data / "station_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual("Named Station", config["station_name"])
            self.assertEqual(0, config["current_tick"])
            self.assertEqual("Healthy", config["station_status"])
            self.assertEqual([], config["agent_turn_order"])
            self.assertEqual("example_private/station/custom", config[STATION_TEMPLATE_SOURCE_KEY])
            self.assertEqual("abc123", config["git_commit"])
            self.assertTrue(config["station_id"])
            self.assertEqual([{"status": "Healthy", "start_tick": 0}], config["status_history"])

    def test_disable_initial_agent_spawn_only_removes_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_data = Path(tmp) / "station_data"
            station_data.mkdir()
            (station_data / "init_agents.yaml").write_text("- GPT-5.6 Sol\n", encoding="utf-8")
            (station_data / "constant_config.yaml").write_text(
                "RESEARCH_EVAL_TIMEOUT: 123\nMULTISTART_STAGNATION_SEEDS: 4\n",
                encoding="utf-8",
            )

            init_command.disable_initial_agent_spawn(station_data)

            config = yaml.safe_load(
                (station_data / "constant_config.yaml").read_text(encoding="utf-8")
            )
            self.assertFalse((station_data / "init_agents.yaml").exists())

        self.assertEqual(123, config["RESEARCH_EVAL_TIMEOUT"])
        self.assertEqual(4, config["MULTISTART_STAGNATION_SEEDS"])

    def test_configure_multistart_for_init_disables_both_modes_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_data = Path(tmp) / "station_data"
            station_data.mkdir()
            (station_data / "constant_config.yaml").write_text(
                "MULTISTART_INIT_SEEDS: 8\nMULTISTART_STAGNATION_SEEDS: 8\n",
                encoding="utf-8",
            )

            init_command.configure_multistart_for_init(station_data, enabled=False)

            config = yaml.safe_load(
                (station_data / "constant_config.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(0, config["MULTISTART_INIT_SEEDS"])
        self.assertEqual(0, config["MULTISTART_STAGNATION_SEEDS"])
        self.assertEqual(1, config["MULTISTART_INIT_MAX_PARALLEL"])
        self.assertEqual(1, config["MULTISTART_STAGNATION_MAX_PARALLEL"])

    def test_configure_multistart_for_init_preserves_settings_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_data = Path(tmp) / "station_data"
            station_data.mkdir()
            config_path = station_data / "constant_config.yaml"
            config_path.write_text(
                "MULTISTART_INIT_SEEDS: 8\nMULTISTART_STAGNATION_SEEDS: 6\n",
                encoding="utf-8",
            )

            init_command.configure_multistart_for_init(station_data, enabled=True)

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(8, config["MULTISTART_INIT_SEEDS"])
        self.assertEqual(6, config["MULTISTART_STAGNATION_SEEDS"])

    def test_pause_after_tick_end_constant_pauses_orchestrator(self):
        from station import constants
        from station.station_runner import Orchestrator

        class FakeStation:
            @property
            def config(self):
                raise AssertionError("PAUSE_AFTER_TICK_END must not read station_config.yaml")

            def __init__(self):
                self.saved_index = None

            def save_next_agent_index_to_config(self, index):
                self.saved_index = index

            def _get_current_tick(self):
                return 21

        orchestrator = object.__new__(Orchestrator)
        orchestrator.station = FakeStation()
        orchestrator.is_paused = False
        orchestrator.pause_requested = False
        orchestrator.pause_condition_met = False
        orchestrator.pause_reason_message = ""
        orchestrator.current_agent_index_in_turn_order = 3
        orchestrator.events = []
        orchestrator._push_log_event = lambda event_type, payload: orchestrator.events.append((event_type, payload))

        with patch.object(constants, "PAUSE_AFTER_TICK_END", 20):
            self.assertTrue(Orchestrator.maybe_pause_after_configured_tick_end(orchestrator, 20))

        self.assertTrue(orchestrator.is_paused)
        self.assertEqual(0, orchestrator.current_agent_index_in_turn_order)
        self.assertEqual(0, orchestrator.station.saved_index)

    def test_restore_script_finds_nested_unchosen_seed_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            backup = repo / "backup" / "chosen-id" / "unchosen_seeds" / "seed-id"
            snapshots = backup / "snapshots"
            (backup / "objects").mkdir(parents=True)
            snapshots.mkdir(parents=True)
            (repo / "backup" / "chosen-id" / "unchosen_seeds" / "seed-data").mkdir(parents=True)
            (snapshots / "tick_3.json").write_text(
                json.dumps({"station_id": "seed-id", "tick": 3, "backup_type": "manual", "files": []}),
                encoding="utf-8",
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "restore.sh"
            env = os.environ.copy()
            repo_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = repo_root if not env.get("PYTHONPATH") else f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"

            result = subprocess.run(
                ["bash", str(script), "--output", "restored", "seed", "3"],
                cwd=repo,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertTrue((repo / "restored").is_dir())
            self.assertIn("backup/chosen-id/unchosen_seeds/seed-id", result.stdout)

    def test_init_replaces_template_research_dir_and_merges_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            template_dir = repo / "example" / "station" / "default"
            default_research = template_dir / "rooms" / "research"
            task_dir = repo / "example" / "research_misc" / "task"
            default_research.mkdir(parents=True)
            (default_research / "template_only.txt").write_text("template", encoding="utf-8")
            (template_dir / "constant_config.yaml").write_text(
                "TEMPLATE_ONLY: true\nSHARED: template\n",
                encoding="utf-8",
            )
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "research" / "task_only.txt").write_text("task", encoding="utf-8")
            (task_dir / "constant_config.yaml").write_text(
                "TASK_ONLY: true\nSHARED: task\n",
                encoding="utf-8",
            )

            station_data = repo / "station_data"
            init_command.copy_station_template_and_task(station_data, template_dir, task_dir)

            self.assertFalse((station_data / "rooms" / "research" / "template_only.txt").exists())
            self.assertEqual("task", (station_data / "rooms" / "research" / "task_only.txt").read_text(encoding="utf-8"))
            merged = yaml.safe_load((station_data / "constant_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual({"TEMPLATE_ONLY": True, "SHARED": "task", "TASK_ONLY": True}, merged)

    def test_init_task_constant_config_is_optional_and_adds_no_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            template_dir = repo / "example" / "station" / "default"
            task_dir = repo / "example" / "research_misc" / "task"
            template_dir.mkdir(parents=True)
            (template_dir / "constant_config.yaml").write_text(
                "TEMPLATE_ONLY: true\nSHARED: template\n",
                encoding="utf-8",
            )
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "research" / "research_task.md").write_text(
                "# Task\n",
                encoding="utf-8",
            )

            station_data = repo / "station_data"
            init_command.copy_station_template_and_task(station_data, template_dir, task_dir)

            merged = yaml.safe_load(
                (station_data / "constant_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual({"TEMPLATE_ONLY": True, "SHARED": "template"}, merged)

    def test_init_rejects_present_task_constant_config_that_is_not_a_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            template_dir = repo / "example" / "station" / "default"
            task_dir = repo / "example" / "research_misc" / "task"
            template_dir.mkdir(parents=True)
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "constant_config.yaml").write_text("- invalid\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                init_command.copy_station_template_and_task(
                    repo / "station_data",
                    template_dir,
                    task_dir,
                )

    def test_init_run_accepts_task_without_constant_config(self):
        args = Namespace(
            dry_run=False,
            no_hooks=True,
            no_start=True,
            post_copy_cmd="",
            station_id="1",
            task_name="task",
            station_name="Station",
        )
        config = ToolsConfig()

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            task_dir = repo / "example" / "research_misc" / "task"
            (repo / "start.sh").parent.mkdir(parents=True, exist_ok=True)
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / DEFAULT_STATION_TEMPLATE_SOURCE).mkdir(parents=True)
            (task_dir / "research").mkdir(parents=True)

            with patch("station_tools.commands.init.token_to_path", return_value=repo):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(0, init_command.run(args, config))

            self.assertTrue((repo / "station_data" / "constant_config.yaml").is_file())

    def test_init_runs_hook_and_start_in_same_shell(self):
        config = ToolsConfig(hooks={"init": {"before_start": "export TEST_API_KEY=present"}})

        with patch("station_tools.commands.init.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            rc = init_command._run_start_with_hook(Path("/repo"), "Station Name", config, no_hooks=False)

        self.assertEqual(0, rc)
        command = run_mock.call_args.args[0]
        self.assertEqual(["bash", "-lc"], command[:2])
        script = command[2]
        self.assertIn("export TEST_API_KEY=present", script)
        self.assertIn("exec ./start.sh --name 'Station Name'", script)

    def test_init_no_hooks_runs_start_without_hook_body(self):
        config = ToolsConfig(hooks={"init": {"before_start": "export TEST_API_KEY=present"}})

        with patch("station_tools.commands.init.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            init_command._run_start_with_hook(Path("/repo"), "Station Name", config, no_hooks=True)

        script = run_mock.call_args.args[0][2]
        self.assertNotIn("export TEST_API_KEY=present", script)
        self.assertIn("exec ./start.sh --name 'Station Name'", script)

    def test_init_start_forwards_startup_flags(self):
        config = ToolsConfig()

        with patch("station_tools.commands.init.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            init_command._run_start_with_hook(
                Path("/repo"),
                "Station Name",
                config,
                no_hooks=True,
                test=True,
                no_multistart=True,
            )

        script = run_mock.call_args.args[0][2]
        self.assertIn("exec ./start.sh --name 'Station Name' --test --no-multistart", script)

    def test_init_post_copy_cmd_uses_noninteractive_shell(self):
        with patch("station_tools.commands.init.subprocess.run") as run_mock:
            run_mock.return_value.returncode = 0
            with redirect_stdout(io.StringIO()):
                rc = init_command.run_post_copy_command(Path("/repo"), "python replace_d.py 13")

        self.assertEqual(0, rc)
        self.assertEqual(["bash", "-lc", "python replace_d.py 13"], run_mock.call_args.args[0])

    def test_init_run_invokes_post_copy_before_start(self):
        args = Namespace(
            dry_run=False,
            no_hooks=True,
            post_copy_cmd="bash post.sh",
            station_id="1",
            task_name="task",
            station_name="Station",
        )
        config = ToolsConfig()
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            task_dir = repo / "example" / "research_misc" / "task"
            (repo / "start.sh").parent.mkdir(parents=True, exist_ok=True)
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / DEFAULT_STATION_TEMPLATE_SOURCE).mkdir(parents=True)
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "constant_config.yaml").write_text("A: 1\n", encoding="utf-8")

            with patch("station_tools.commands.init.token_to_path", return_value=repo):
                with patch.object(init_command, "copy_station_template_and_task", side_effect=lambda *_: calls.append("copy")):
                    with patch.object(init_command, "write_initial_station_config", side_effect=lambda *_: calls.append("name")):
                        with patch.object(init_command, "run_post_copy_command", side_effect=lambda *_: calls.append("post") or 0):
                            with patch.object(init_command, "_run_start_with_hook", side_effect=lambda *_, **__: calls.append("start") or 0):
                                with redirect_stdout(io.StringIO()):
                                    self.assertEqual(0, init_command.run(args, config))

        self.assertEqual(["copy", "name", "post", "start"], calls)

    def test_init_no_start_completes_setup_without_hook_or_start(self):
        args = Namespace(
            dry_run=False,
            no_hooks=False,
            no_start=True,
            post_copy_cmd="bash post.sh",
            station_id="1",
            task_name="task",
            station_name="Station",
        )
        config = ToolsConfig(hooks={"init": {"before_start": "api-profile"}})
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            task_dir = repo / "example" / "research_misc" / "task"
            (repo / "start.sh").parent.mkdir(parents=True, exist_ok=True)
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / DEFAULT_STATION_TEMPLATE_SOURCE).mkdir(parents=True)
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "constant_config.yaml").write_text("A: 1\n", encoding="utf-8")

            with patch("station_tools.commands.init.token_to_path", return_value=repo):
                with patch.object(init_command, "copy_station_template_and_task", side_effect=lambda *_: calls.append("copy")):
                    with patch.object(init_command, "write_initial_station_config", side_effect=lambda *_: calls.append("name")):
                        with patch.object(init_command, "run_post_copy_command", side_effect=lambda *_: calls.append("post") or 0):
                            with patch.object(init_command, "_run_start_with_hook", side_effect=AssertionError("start called")):
                                output = io.StringIO()
                                with redirect_stdout(output):
                                    self.assertEqual(0, init_command.run(args, config))

        self.assertEqual(["copy", "name", "post"], calls)
        self.assertIn("skipped init.before_start hook and start.sh", output.getvalue())

    def test_init_run_writes_station_name_to_copied_data(self):
        args = Namespace(
            dry_run=False,
            no_hooks=True,
            post_copy_cmd="",
            station_id="1",
            task_name="task",
            station_name="Station",
        )
        config = ToolsConfig()
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            task_dir = repo / "example" / "research_misc" / "task"
            (repo / "start.sh").parent.mkdir(parents=True, exist_ok=True)
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (repo / DEFAULT_STATION_TEMPLATE_SOURCE).mkdir(parents=True)
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "constant_config.yaml").write_text("A: 1\n", encoding="utf-8")

            with patch("station_tools.commands.init.token_to_path", return_value=repo):
                with patch.object(init_command.station_config, "current_git_commit", return_value="abc123"):
                    with patch.object(init_command, "run_post_copy_command", side_effect=lambda *_: calls.append("post") or 0):
                        with patch.object(init_command, "_run_start_with_hook", return_value=0):
                            with redirect_stdout(io.StringIO()):
                                self.assertEqual(0, init_command.run(args, config))

            station_config = yaml.safe_load((repo / "station_data" / "station_config.yaml").read_text(encoding="utf-8"))
            self.assertEqual("Station", station_config["station_name"])
            self.assertEqual("abc123", station_config["git_commit"])
            self.assertEqual(DEFAULT_STATION_TEMPLATE_SOURCE, station_config[STATION_TEMPLATE_SOURCE_KEY])

    def test_init_run_merges_selected_template_and_task_constant_configs(self):
        args = Namespace(
            dry_run=False,
            no_hooks=True,
            post_copy_cmd="",
            station_template="custom",
            station_id="1",
            task_name="task",
            station_name="Station",
        )
        config = ToolsConfig()

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "station"
            template = repo / "example" / "station" / "custom"
            task_dir = repo / "example" / "research_misc" / "task"
            template.mkdir(parents=True)
            (template / "constant_config.yaml").write_text(
                "TEMPLATE_ONLY: true\nSHARED: template\n",
                encoding="utf-8",
            )
            (task_dir / "research").mkdir(parents=True)
            (task_dir / "constant_config.yaml").write_text(
                "TASK_ONLY: true\nSHARED: task\n",
                encoding="utf-8",
            )
            (repo / "start.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            with patch("station_tools.commands.init.token_to_path", return_value=repo):
                with patch.object(init_command.station_config, "current_git_commit", return_value="abc123"):
                    with patch.object(init_command, "_run_start_with_hook", return_value=0):
                        with redirect_stdout(io.StringIO()):
                            self.assertEqual(0, init_command.run(args, config))

            merged = yaml.safe_load((repo / "station_data" / "constant_config.yaml").read_text(encoding="utf-8"))
            station_config = yaml.safe_load(
                (repo / "station_data" / "station_config.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(True, merged["TEMPLATE_ONLY"])
        self.assertEqual("task", merged["SHARED"])
        self.assertEqual(True, merged["TASK_ONLY"])
        self.assertEqual(0, merged["MULTISTART_INIT_SEEDS"])
        self.assertEqual(0, merged["MULTISTART_STAGNATION_SEEDS"])
        self.assertEqual("example/station/custom", station_config[STATION_TEMPLATE_SOURCE_KEY])


if __name__ == "__main__":
    unittest.main()
