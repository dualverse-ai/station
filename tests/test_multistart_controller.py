import os
import contextlib
import io
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from station import research_storage
from station.multistart import controller, paths, start_hook, state, waiting


class MultistartControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="station_multistart_test_", dir="/tmp"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_station_config(self, current_tick=1):
        station_data = self.tmpdir / "station_data"
        station_data.mkdir(parents=True, exist_ok=True)
        (station_data / "station_config.yaml").write_text(
            yaml.safe_dump({"station_name": "Test", "station_id": "station-id", "current_tick": current_tick}),
            encoding="utf-8",
        )
        return station_data

    def test_waiting_mode_requires_active_job(self):
        self.assertFalse(waiting.waiting_mode_active(self.tmpdir))
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "1_abc"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir)},
        )
        self._write_station_config()
        self.assertTrue(waiting.waiting_mode_active(self.tmpdir))
        shutil.rmtree(self.tmpdir / "station_data")
        self.assertTrue(waiting.waiting_mode_active(self.tmpdir))

    def test_waiting_mode_includes_pending_init_request(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        state.save_yaml_mapping(paths.pending_init_path(self.tmpdir), {
            "mode": "init",
            "status": "blocked_disk_space",
            "message": "free disk space",
        })

        self.assertTrue(waiting.waiting_mode_active(self.tmpdir))
        status = waiting.public_status(self.tmpdir)
        self.assertTrue(status["active"])
        self.assertEqual("waiting for disk space", status["stage"])
        self.assertEqual("free disk space", status["stage_note"])

    def test_ipc_response_disconnect_does_not_raise(self):
        ctrl = controller.Controller(self.tmpdir)
        conn = mock.Mock()
        conn.sendall.side_effect = BrokenPipeError(32, "Broken pipe")

        with mock.patch.object(ctrl, "log") as log:
            sent = ctrl._send_ipc_response(conn, {"success": True})

        self.assertFalse(sent)
        log.assert_called_once()

    def test_resume_recovers_missing_controller_and_retries_ipc(self):
        unavailable = {
            "success": False,
            "error": "controller socket not found",
            "controller_unavailable": True,
        }
        resumed = {"success": True, "message": "Resume requested."}
        with mock.patch.object(
            controller.ipc,
            "send_message",
            side_effect=[unavailable, resumed],
        ) as send_message, mock.patch(
            "station.multistart.controller.recover_controller",
            return_value={"success": True, "pid": 12345},
        ) as recover:
            response = controller.ipc.request_resume_branches(repo=self.tmpdir)

        self.assertTrue(response["success"])
        self.assertTrue(response["controller_recovered"])
        self.assertEqual(2, send_message.call_count)
        recover.assert_called_once_with(self.tmpdir.resolve())

    def test_resume_reports_controller_recovery_failure(self):
        unavailable = {
            "success": False,
            "error": "controller socket not found",
            "controller_unavailable": True,
        }
        with mock.patch.object(
            controller.ipc,
            "send_message",
            return_value=unavailable,
        ), mock.patch(
            "station.multistart.controller.recover_controller",
            return_value={"success": False, "error": "could not restart"},
        ):
            response = controller.ipc.request_resume_branches(repo=self.tmpdir)

        self.assertFalse(response["success"])
        self.assertTrue(response["controller_unavailable"])
        self.assertIn("automatic controller recovery failed", response["error"])

    def test_resume_does_not_replace_controller_after_connected_response_timeout(self):
        timed_out = {
            "success": False,
            "error": "controller IPC timed out",
            "controller_unavailable": False,
        }
        with mock.patch.object(
            controller.ipc,
            "send_message",
            return_value=timed_out,
        ), mock.patch("station.multistart.controller.recover_controller") as recover:
            response = controller.ipc.request_resume_branches(repo=self.tmpdir)

        self.assertEqual(timed_out, response)
        recover.assert_not_called()

    def test_recover_controller_retains_process_that_self_healed_ipc(self):
        with mock.patch.object(
            controller.ipc,
            "request_status",
            return_value={"success": True},
        ), mock.patch.object(
            controller,
            "find_running_controller_pid",
            return_value=4321,
        ), mock.patch.object(
            controller,
            "_terminate_unresponsive_controllers",
        ) as terminate, mock.patch.object(controller, "start_detached") as start:
            response = controller.recover_controller(self.tmpdir)

        self.assertTrue(response["success"])
        self.assertTrue(response["already_running"])
        self.assertEqual(4321, response["pid"])
        terminate.assert_not_called()
        start.assert_not_called()

    def test_recover_controller_restarts_missing_process_and_waits_for_ipc(self):
        unavailable = {
            "success": False,
            "error": "controller socket not found",
            "controller_unavailable": True,
        }
        with mock.patch.object(
            controller.ipc,
            "request_status",
            side_effect=[unavailable, {"success": True}],
        ), mock.patch.object(
            controller,
            "find_running_controller_pid",
            return_value=None,
        ), mock.patch.object(
            controller,
            "_terminate_unresponsive_controllers",
            return_value=[],
        ) as terminate, mock.patch.object(
            controller,
            "start_detached",
            return_value=12345,
        ) as start:
            response = controller.recover_controller(self.tmpdir)

        self.assertTrue(response["success"])
        self.assertEqual(12345, response["pid"])
        terminate.assert_called_once_with(self.tmpdir.resolve())
        start.assert_called_once_with(self.tmpdir.resolve(), init=False)

    def test_public_status_labels_unnamed_branches_and_reports_tick_age(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        branch_dir = job_dir / "station_data_s2"
        branch_dir.mkdir(parents=True)
        (branch_dir / "station_config.yaml").write_text(
            yaml.safe_dump({"station_name": "", "station_id": "station-id", "current_tick": 4}),
            encoding="utf-8",
        )
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir)},
        )
        state.save_job_state(
            job_dir,
            {
                "branches": [
                    {
                        "seed": 2,
                        "status": "running",
                        "data_root": str(branch_dir),
                        "target_tick": 8,
                    }
                ]
            },
        )

        branch = waiting.public_status(self.tmpdir)["branches"][0]
        self.assertEqual("", branch["station_name"])
        self.assertEqual("Seed 2", branch["station_label"])
        self.assertEqual(4, branch["current_tick"])
        self.assertIsNotNone(branch["last_tick_timestamp"])
        self.assertIsInstance(branch["last_tick_age_seconds"], int)

    def test_public_status_progress_uses_original_branch_span_after_resume(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "461_abc"
        branch_dir = job_dir / "station_data_s2"
        branch_dir.mkdir(parents=True)
        (branch_dir / "station_config.yaml").write_text(
            yaml.safe_dump({"station_name": "Test", "station_id": "station-id", "current_tick": 483}),
            encoding="utf-8",
        )
        state.save_current_job(
            self.tmpdir,
            {
                "job_id": "abc",
                "mode": "stagnation",
                "status": "running",
                "job_dir": str(job_dir),
                "branch_tick": 461,
                "roll_ticks": 40,
            },
        )
        state.save_job_state(
            job_dir,
            {
                "branch_tick": 461,
                "roll_ticks": 40,
                "branches": [
                    {
                        "seed": 2,
                        "status": "running",
                        "data_root": str(branch_dir),
                        "start_tick": 483,
                        "target_tick": 501,
                        "current_tick": 483,
                    }
                ],
            },
        )

        branch = waiting.public_status(self.tmpdir)["branches"][0]
        self.assertEqual(22, branch["progress_done_ticks"])
        self.assertEqual(40, branch["progress_total_ticks"])
        self.assertEqual(55, branch["progress_percent"])

    def test_public_status_reports_pending_admin_stage_when_all_branches_complete(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        for seed in (1, 2):
            branch_dir = job_dir / f"station_data_s{seed}"
            branch_dir.mkdir(parents=True)
            (branch_dir / "station_config.yaml").write_text(
                yaml.safe_dump({"station_name": "", "station_id": f"station-{seed}", "current_tick": 16}),
                encoding="utf-8",
            )
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 2},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "running",
                "seed_count": 2,
                "branches": [
                    {"seed": 1, "status": "completed"},
                    {"seed": 2, "status": "completed"},
                ],
            },
        )

        payload = waiting.public_status(self.tmpdir)
        self.assertEqual("pending admin", payload["stage"])
        self.assertIn("administrator selection", payload["stage_note"])

    def test_public_status_prefers_detail_status_for_admin_selection(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 1},
        )
        state.save_job_state(
            job_dir,
            {"status": "selecting", "seed_count": 1, "branches": []},
        )

        payload = waiting.public_status(self.tmpdir)
        self.assertEqual("selecting", payload["status"])
        self.assertEqual("admin running", payload["stage"])

    def test_public_status_reports_paused_control_and_branch_note(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        (branch_dir / "station_config.yaml").write_text(
            yaml.safe_dump({"station_name": "Seed One", "station_id": "station-1", "current_tick": 6}),
            encoding="utf-8",
        )
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 1},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "running",
                state.CONTROL_KEY: state.CONTROL_PAUSED,
                "seed_count": 1,
                "branches": [
                    {
                        "seed": 1,
                        "status": "paused",
                        "data_root": str(branch_dir),
                        "current_tick": 6,
                        "target_tick": 16,
                        "pause_reason": "manual multistart pause",
                    }
                ],
            },
        )

        payload = waiting.public_status(self.tmpdir)
        self.assertEqual(state.CONTROL_PAUSED, payload["control"])
        self.assertEqual("paused", payload["stage"])
        self.assertEqual(1, payload["counts"]["paused"])
        self.assertEqual("manual multistart pause", payload["branches"][0]["note"])

    def test_public_status_reports_paused_control_before_branches_launch(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 2},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "running",
                state.CONTROL_KEY: state.CONTROL_PAUSED,
                "seed_count": 2,
                "branches": [
                    {"seed": 1, "status": "pending"},
                    {"seed": 2, "status": "pending"},
                ],
            },
        )

        payload = waiting.public_status(self.tmpdir)
        self.assertEqual("paused", payload["stage"])
        self.assertIn("launch paused", payload["stage_note"])

    def test_controller_pause_resume_sets_control_and_skips_finished_branches(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 4},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "running",
                "seed_count": 4,
                "branches": [
                    {"seed": 1, "status": "running", "current_tick": 6, "target_tick": 16},
                    {"seed": 2, "status": "pending"},
                    {"seed": 3, "status": "completed", "current_tick": 18, "target_tick": 16},
                    {"seed": 4, "status": "paused", "current_tick": 16, "target_tick": 16},
                ],
            },
        )

        ctrl = controller.Controller(self.tmpdir)
        paused = ctrl._set_branch_pause(paused=True)
        self.assertTrue(paused["success"])
        self.assertEqual(state.CONTROL_PAUSED, state.load_job_state(job_dir)[state.CONTROL_KEY])
        self.assertEqual(3, paused["affected"])

        resumed = ctrl._set_branch_pause(paused=False)
        self.assertTrue(resumed["success"])
        self.assertEqual(state.CONTROL_RUNNING, state.load_job_state(job_dir)[state.CONTROL_KEY])
        self.assertEqual(2, resumed["affected"])
        self.assertEqual(2, resumed["skipped"])

    def test_controller_resume_requeues_failed_branch_without_restart(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        branch_dir = job_dir / "station_data_s2"
        branch_dir.mkdir(parents=True)
        (branch_dir / "index").mkdir()
        (branch_dir / "index" / "station_index.sqlite3").write_text("stale", encoding="utf-8")
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "failed", "job_dir": str(job_dir), "seed_count": 2},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "failed",
                "seed_count": 2,
                "branches": [
                    {"seed": 1, "status": "completed", "current_tick": 10, "target_tick": 10},
                    {
                        "seed": 2,
                        "status": "failed",
                        "pid": None,
                        "current_tick": 6,
                        "target_tick": 10,
                        "data_root": str(branch_dir),
                        "error": "LLM API error",
                    },
                ],
            },
        )

        ctrl = controller.Controller(self.tmpdir)
        response = ctrl._set_branch_pause(paused=False)
        payload = state.load_job_state(job_dir)
        branches = {branch["seed"]: branch for branch in payload["branches"]}

        self.assertTrue(response["success"])
        self.assertEqual("running", payload["status"])
        self.assertEqual("running", state.load_current_job(self.tmpdir)["status"])
        self.assertEqual("pending", branches[2]["status"])
        self.assertEqual(1, branches[2]["attempts"])
        self.assertFalse((branch_dir / "index").exists())
        self.assertEqual(1, response["affected"])
        self.assertEqual(1, response["skipped"])

    def test_controller_resume_reopens_failed_admin_selection(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "failed", "job_dir": str(job_dir)},
        )
        state.save_job_state(
            job_dir,
            {
                "job_id": "abc",
                "status": "failed",
                "seed_count": 1,
                "branches": [{"seed": 1, "status": "completed", "current_tick": 10, "target_tick": 10}],
                controller.admin.ADMIN_SELECTION_STATE_KEY: {
                    "status": "blocked",
                    "spawn_count": 3,
                    "resume_count": 14,
                },
            },
        )

        response = controller.Controller(self.tmpdir)._set_branch_pause(paused=False)

        self.assertTrue(response["success"])
        self.assertEqual("running", state.load_job_state(job_dir)["status"])
        self.assertEqual("running", state.load_current_job(self.tmpdir)["status"])
        self.assertEqual(0, response["affected"])
        self.assertEqual(1, response["skipped"])

    def test_run_loop_keeps_controller_alive_after_job_exception_until_resume(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir)},
        )
        state.save_job_state(
            job_dir,
            {
                "job_id": "abc",
                "status": "running",
                "seed_count": 1,
                "branches": [
                    {
                        "seed": 1,
                        "status": "running",
                        "pid": None,
                        "data_root": str(branch_dir),
                        "current_tick": 3,
                        "target_tick": 10,
                    }
                ],
            },
        )

        ctrl = controller.Controller(self.tmpdir)
        first_failure = threading.Event()
        resumed_run = threading.Event()
        run_results = []
        run_calls = []

        def run_job(_job_dir):
            run_calls.append(_job_dir)
            if len(run_calls) == 1:
                first_failure.set()
                raise RuntimeError("simulated branch failure")
            resumed_run.set()
            ctrl.stop_requested.set()

        with mock.patch.object(ctrl, "start_ipc"), \
                mock.patch.object(ctrl, "cleanup_ipc") as cleanup, \
                mock.patch.object(ctrl, "_run_or_resume_job", side_effect=run_job), \
                mock.patch.object(ctrl, "_check_pending_init_request"), \
                mock.patch.object(ctrl, "_check_pending_stagnation_request"), \
                mock.patch.object(ctrl, "log"), \
                mock.patch.object(controller, "POLL_SECONDS", 0.01):
            thread = threading.Thread(target=lambda: run_results.append(ctrl.run_loop()))
            thread.start()
            self.assertTrue(first_failure.wait(timeout=1.0))

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                current = state.load_current_job(self.tmpdir)
                if current.get("status") == "failed" and current.get("controller_halted_at"):
                    break
                time.sleep(0.01)
            self.assertEqual("failed", state.load_current_job(self.tmpdir)["status"])
            self.assertTrue(thread.is_alive())
            cleanup.assert_not_called()

            conn = mock.Mock()
            conn.recv.return_value = b'{"type":"resume_branches"}\n'
            response = ctrl._handle_connection(conn)
            self.assertTrue(response["success"])

            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(resumed_run.is_set())
        self.assertEqual([0], run_results)
        self.assertEqual(2, len(run_calls))
        cleanup.assert_called_once()

    def test_run_loop_cleans_pid_when_ipc_startup_fails(self):
        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.object(ctrl, "start_ipc", side_effect=PermissionError("bind denied")), \
                mock.patch.object(ctrl, "log"):
            result = ctrl.run_loop()

        self.assertEqual(1, result)
        self.assertFalse(paths.controller_pid_path(self.tmpdir).exists())

    def test_graceful_shutdown_sets_pause_and_shutdown_flag(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 2},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "running",
                "seed_count": 2,
                "branches": [
                    {"seed": 1, "status": "running", "current_tick": 5, "target_tick": 10},
                    {"seed": 2, "status": "pending"},
                ],
            },
        )

        ctrl = controller.Controller(self.tmpdir)
        response = ctrl._request_graceful_shutdown()
        payload = state.load_job_state(job_dir)

        self.assertTrue(response["success"])
        self.assertTrue(response["shutdown_requested"])
        self.assertEqual(state.CONTROL_PAUSED, payload[state.CONTROL_KEY])
        self.assertTrue(payload[state.SHUTDOWN_REQUESTED_KEY])
        self.assertIn(state.SHUTDOWN_REQUESTED_AT_KEY, payload)

    def test_resume_clears_shutdown_requested_flag(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir), "seed_count": 1},
        )
        state.save_job_state(
            job_dir,
            {
                "status": "running",
                state.CONTROL_KEY: state.CONTROL_PAUSED,
                state.SHUTDOWN_REQUESTED_KEY: True,
                state.SHUTDOWN_REQUESTED_AT_KEY: "2026-01-01T00:00:00+00:00",
                "branches": [{"seed": 1, "status": "pending", "pid": None}],
            },
        )

        ctrl = controller.Controller(self.tmpdir)
        response = ctrl._set_branch_pause(paused=False)
        payload = state.load_job_state(job_dir)

        self.assertTrue(response["success"])
        self.assertEqual(state.CONTROL_RUNNING, payload[state.CONTROL_KEY])
        self.assertNotIn(state.SHUTDOWN_REQUESTED_KEY, payload)
        self.assertNotIn(state.SHUTDOWN_REQUESTED_AT_KEY, payload)

    def test_resume_reset_clears_branch_shutdown_fields(self):
        job_dir = self.tmpdir / "station_multistart" / "0_abc"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        payload = {
            "status": "running",
            state.CONTROL_KEY: state.CONTROL_RUNNING,
            state.SHUTDOWN_REQUESTED_KEY: True,
            state.SHUTDOWN_REQUESTED_AT_KEY: "2026-01-01T00:00:00+00:00",
            "branches": [
                {
                    "seed": 1,
                    "status": "paused",
                    "pid": None,
                    "shutdown_requested": True,
                    "shutdown_stopped_at": 1.0,
                    "pause_requested": True,
                    "pause_reason": "graceful multistart shutdown",
                    "paused_at": 1.0,
                }
            ],
        }

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, payload)

        self.assertNotIn(state.SHUTDOWN_REQUESTED_KEY, payload)
        self.assertNotIn(state.SHUTDOWN_REQUESTED_AT_KEY, payload)
        branch = payload["branches"][0]
        self.assertEqual("pending", branch["status"])
        self.assertIsNone(branch["pid"])
        self.assertNotIn("shutdown_requested", branch)
        self.assertNotIn("pause_reason", branch)

    def test_shutdown_stop_terminates_quiescent_paused_old_branch_worker(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "0_abc"
        job_dir.mkdir()
        payload = {
            "status": "running",
            state.CONTROL_KEY: state.CONTROL_PAUSED,
            state.SHUTDOWN_REQUESTED_KEY: True,
            "branches": [
                {"seed": 1, "status": "paused", "pid": 12345},
            ],
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.object(ctrl, "_branch_pid_alive", return_value=True):
            with mock.patch.object(ctrl, "_branch_has_background_work", return_value=False):
                with mock.patch.object(ctrl, "_terminate_pid") as terminate:
                    ctrl._stop_quiescent_paused_branch_workers(job_dir, payload, payload["branches"])

        updated = state.load_job_state(job_dir)
        branch = updated["branches"][0]
        terminate.assert_called_once_with(12345)
        self.assertIsNone(branch["pid"])
        self.assertEqual("paused", branch["status"])
        self.assertTrue(branch["shutdown_requested"])
        self.assertEqual("graceful multistart shutdown", branch["pause_reason"])

    def test_branch_background_work_yaml_fallback_detects_active_coder(self):
        data_root = self.tmpdir / "branch_data"
        evaluations_dir = data_root / "rooms" / "research" / "evaluations"
        evaluations_dir.mkdir(parents=True)
        (evaluations_dir / "7.yaml").write_text(
            yaml.safe_dump({"status": "running", "coder": {"status": "coder_running"}}),
            encoding="utf-8",
        )

        ctrl = controller.Controller(self.tmpdir)
        self.assertTrue(ctrl._branch_has_background_work_from_yaml(data_root))

    def test_start_hook_restarts_controller_for_active_job_even_with_waiting_page(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "1_abc"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "abc", "mode": "init", "status": "running", "job_dir": str(job_dir)},
        )
        with mock.patch("station.multistart.start_hook.controller.pid_running", return_value=False):
            with mock.patch("station.multistart.start_hook.controller.start_detached", return_value=123) as start:
                self.assertEqual(20, start_hook.main(["--repo", str(self.tmpdir)]))
        start.assert_called_once_with(self.tmpdir.resolve(), init=False)

    def test_start_hook_does_not_wait_for_existing_station(self):
        self._write_station_config(current_tick=7)
        with mock.patch("station.multistart.start_hook.controller.start_detached") as start:
            with mock.patch("station.multistart.start_hook.controller.pid_running", return_value=False):
                with mock.patch("station.constants.MULTISTART_INIT_SEEDS", 8):
                    with mock.patch("station.constants.MULTISTART_STAGNATION_SEEDS", 0):
                        self.assertEqual(0, start_hook.main(["--repo", str(self.tmpdir)]))
        start.assert_called_once()

    def test_start_hook_bootstraps_pending_stagnation_without_init(self):
        self._write_station_config(current_tick=0)
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        state.save_yaml_mapping(paths.pending_stagnation_path(self.tmpdir), {"type": "stagnation", "branch_tick": 320})

        with mock.patch("station.multistart.start_hook.controller.start_detached") as start:
            with mock.patch("station.multistart.start_hook.controller.pid_running", return_value=False):
                with mock.patch("station.constants.MULTISTART_INIT_SEEDS", 8):
                    with mock.patch("station.constants.MULTISTART_STAGNATION_SEEDS", 8):
                        self.assertEqual(20, start_hook.main(["--repo", str(self.tmpdir)]))

        start.assert_called_once_with(self.tmpdir.resolve(), init=False)

    def test_controller_does_not_start_init_when_stagnation_request_exists(self):
        self._write_station_config(current_tick=0)
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        state.save_yaml_mapping(paths.pending_stagnation_path(self.tmpdir), {"type": "stagnation", "branch_tick": 320})

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch("station.constants.MULTISTART_INIT_SEEDS", 8):
            with mock.patch.object(ctrl, "create_job") as create_job:
                self.assertFalse(ctrl.start_init_job_if_needed())

        create_job.assert_not_called()

    def test_start_hook_uses_explicit_bootstrap_mode_while_stagnation_request_is_pending(self):
        self._write_station_config(current_tick=0)
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        state.save_yaml_mapping(paths.pending_stagnation_path(self.tmpdir), {"type": "stagnation", "branch_tick": 320})

        with mock.patch("station.multistart.start_hook.controller.start_detached") as start:
            with mock.patch("station.multistart.start_hook.controller.pid_running", return_value=False):
                with mock.patch("station.constants.MULTISTART_INIT_SEEDS", 8):
                    with mock.patch("station.constants.MULTISTART_STAGNATION_SEEDS", 8):
                        with mock.patch(
                            "station.multistart.start_hook.waiting.waiting_mode_active",
                            return_value=False,
                        ):
                            self.assertEqual(
                                start_hook.BOOTSTRAP_STAGNATION_STATUS,
                                start_hook.main(["--repo", str(self.tmpdir)]),
                            )

        start.assert_called_once_with(self.tmpdir.resolve(), init=False)

    def test_start_hook_waits_when_fresh_init_job_becomes_active(self):
        self._write_station_config(current_tick=0)

        with mock.patch("station.multistart.start_hook.controller.start_detached") as start:
            with mock.patch("station.multistart.start_hook.controller.pid_running", return_value=False):
                with mock.patch("station.constants.MULTISTART_INIT_SEEDS", 8):
                    with mock.patch("station.constants.MULTISTART_STAGNATION_SEEDS", 0):
                        with mock.patch(
                            "station.multistart.start_hook.waiting.waiting_mode_active",
                            side_effect=[False, True],
                        ):
                            self.assertEqual(20, start_hook.main(["--repo", str(self.tmpdir)]))

        start.assert_called_once_with(self.tmpdir.resolve(), init=True)

    def test_start_hook_restarts_controller_for_pending_init_request(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        state.save_yaml_mapping(paths.pending_init_path(self.tmpdir), {"mode": "init", "status": "pending"})

        with mock.patch("station.multistart.start_hook.controller.pid_running", return_value=False):
            with mock.patch("station.multistart.start_hook.controller.start_detached") as start:
                self.assertEqual(20, start_hook.main(["--repo", str(self.tmpdir)]))

        start.assert_called_once_with(self.tmpdir.resolve(), init=True)

    def test_reset_incomplete_branches_for_resume_requeues_dead_failed_and_running(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        job_dir.mkdir(parents=True)
        payload = {
            "branches": [
                {"seed": 1, "status": "completed"},
                {"seed": 2, "status": "failed", "pid": None},
                {"seed": 3, "status": "running", "pid": 99999999},
            ]
        }
        state.save_job_state(job_dir, payload)
        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))
        updated = state.load_job_state(job_dir)
        statuses = {branch["seed"]: branch["status"] for branch in updated["branches"]}
        self.assertEqual("completed", statuses[1])
        self.assertEqual("pending", statuses[2])
        self.assertEqual("pending", statuses[3])
        attempts = {branch["seed"]: branch.get("attempts") for branch in updated["branches"]}
        self.assertEqual(1, attempts[2])
        self.assertEqual(1, attempts[3])

    def test_reset_incomplete_branch_preserves_existing_progress_without_connector_failure(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        branch_dir = job_dir / "station_data_s2"
        branch_dir.mkdir(parents=True)
        (branch_dir / "marker.txt").write_text("keep", encoding="utf-8")
        (branch_dir / "index").mkdir()
        (branch_dir / "index" / "station_index.sqlite3").write_text("stale", encoding="utf-8")
        payload = {
            "branches": [
                {
                    "seed": 2,
                    "status": "running",
                    "pid": 99999999,
                    "current_tick": 12,
                    "target_tick": 16,
                    "data_root": str(branch_dir),
                }
            ]
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))

        updated = state.load_job_state(job_dir)["branches"][0]
        self.assertEqual("pending", updated["status"])
        self.assertEqual(12, updated["current_tick"])
        self.assertEqual(16, updated["target_tick"])
        self.assertTrue((branch_dir / "marker.txt").is_file())
        self.assertFalse((branch_dir / "index").exists())

    def test_reset_failed_branch_with_connector_failure_preserves_progress(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        origin = job_dir / "origin_station_data"
        branch_dir = job_dir / "station_data_s2"
        origin.mkdir(parents=True)
        branch_dir.mkdir(parents=True)
        (origin / "marker.txt").write_text("origin", encoding="utf-8")
        (branch_dir / "marker.txt").write_text("resume", encoding="utf-8")
        (branch_dir / "index").mkdir()
        (branch_dir / "index" / "station_index.sqlite3").write_text("stale", encoding="utf-8")
        log_path = job_dir / "branch_s2.log"
        log_path.write_text("ValueError: OpenAI API key not provided\n", encoding="utf-8")
        payload = {
            "branches": [
                {
                    "seed": 2,
                    "status": "failed",
                    "pid": None,
                    "current_tick": 16,
                    "target_tick": 16,
                    "data_root": str(branch_dir),
                    "log_path": str(log_path),
                }
            ]
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))

        updated = state.load_job_state(job_dir)["branches"][0]
        self.assertEqual("pending", updated["status"])
        self.assertEqual(16, updated["current_tick"])
        self.assertEqual(16, updated["target_tick"])
        self.assertEqual("resume", (branch_dir / "marker.txt").read_text(encoding="utf-8"))
        self.assertFalse((branch_dir / "index").exists())
        self.assertFalse(updated.get("failed_attempt_dirs"))

    def test_reset_flag_recopies_branch_from_origin_for_clean_retry(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        origin = job_dir / "origin_station_data"
        branch_dir = job_dir / "station_data_s2"
        origin.mkdir(parents=True)
        branch_dir.mkdir(parents=True)
        (origin / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "same-station", "current_tick": 0}),
            encoding="utf-8",
        )
        (origin / "marker.txt").write_text("origin", encoding="utf-8")
        (branch_dir / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "same-station", "current_tick": 40}),
            encoding="utf-8",
        )
        (branch_dir / "marker.txt").write_text("mutated", encoding="utf-8")
        payload = {
            "branches": [
                {
                    "seed": 2,
                    "status": "failed",
                    "pid": None,
                    "current_tick": 40,
                    "target_tick": 40,
                    "reset_data_on_resume": True,
                    "init_agents_spawned": 0,
                    "data_root": str(branch_dir),
                }
            ]
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))

        updated = state.load_job_state(job_dir)["branches"][0]
        branch_config = yaml.safe_load((branch_dir / "station_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual("pending", updated["status"])
        self.assertNotIn("current_tick", updated)
        self.assertNotIn("reset_data_on_resume", updated)
        self.assertNotIn("init_agents_spawned", updated)
        self.assertEqual(0, branch_config["current_tick"])
        self.assertEqual("same-station", branch_config["station_id"])
        self.assertEqual("origin", (branch_dir / "marker.txt").read_text(encoding="utf-8"))

    def test_reset_failed_branch_with_disk_io_error_repairs_index_in_place(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        origin = job_dir / "origin_station_data"
        branch_dir = job_dir / "station_data_s3"
        origin.mkdir(parents=True)
        branch_dir.mkdir(parents=True)
        (origin / "marker.txt").write_text("origin", encoding="utf-8")
        (branch_dir / "marker.txt").write_text("preserve-progress", encoding="utf-8")
        (branch_dir / "index").mkdir()
        (branch_dir / "index" / "station_index.sqlite3").write_text("stale", encoding="utf-8")
        log_path = job_dir / "branch_s3.log"
        log_path.write_text("sqlite3.OperationalError: disk I/O error\n", encoding="utf-8")
        payload = {
            "branches": [
                {
                    "seed": 3,
                    "status": "failed",
                    "pid": None,
                    "current_tick": 482,
                    "target_tick": 501,
                    "data_root": str(branch_dir),
                    "log_path": str(log_path),
                }
            ]
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))

        updated = state.load_job_state(job_dir)["branches"][0]
        self.assertEqual("pending", updated["status"])
        self.assertEqual(482, updated["current_tick"])
        self.assertEqual(501, updated["target_tick"])
        self.assertEqual("preserve-progress", (branch_dir / "marker.txt").read_text(encoding="utf-8"))
        self.assertFalse((branch_dir / "index").exists())
        self.assertFalse(updated.get("failed_attempt_dirs"))

    def test_reset_pending_branch_with_disk_io_error_repairs_index_in_place(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        origin = job_dir / "origin_station_data"
        branch_dir = job_dir / "station_data_s2"
        origin.mkdir(parents=True)
        branch_dir.mkdir(parents=True)
        (origin / "marker.txt").write_text("origin", encoding="utf-8")
        (branch_dir / "marker.txt").write_text("preserve-progress", encoding="utf-8")
        (branch_dir / "index").mkdir()
        (branch_dir / "index" / "station_index.sqlite3").write_text("stale", encoding="utf-8")
        log_path = job_dir / "branch_s2.log"
        log_path.write_text("OSError: [Errno 28] No space left on device\n", encoding="utf-8")
        payload = {
            "branches": [
                {
                    "seed": 2,
                    "status": "pending",
                    "previous_status": "running",
                    "attempts": 1,
                    "pid": None,
                    "current_tick": 483,
                    "target_tick": 501,
                    "data_root": str(branch_dir),
                    "log_path": str(log_path),
                }
            ]
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))

        updated = state.load_job_state(job_dir)["branches"][0]
        self.assertEqual("pending", updated["status"])
        self.assertEqual(2, updated["attempts"])
        self.assertEqual(483, updated["current_tick"])
        self.assertEqual(501, updated["target_tick"])
        self.assertEqual("preserve-progress", (branch_dir / "marker.txt").read_text(encoding="utf-8"))
        self.assertFalse((branch_dir / "index").exists())
        self.assertFalse(updated.get("failed_attempt_dirs"))

    def test_old_connector_failure_log_does_not_force_repeated_data_reset(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        origin = job_dir / "origin_station_data"
        branch_dir = job_dir / "station_data_s2"
        origin.mkdir(parents=True)
        branch_dir.mkdir(parents=True)
        (origin / "marker.txt").write_text("origin", encoding="utf-8")
        (branch_dir / "marker.txt").write_text("resume", encoding="utf-8")
        log_path = job_dir / "branch_s2.log"
        log_path.write_text("ValueError: OpenAI API key not provided\n", encoding="utf-8")
        payload = {
            "branches": [
                {
                    "seed": 2,
                    "status": "failed",
                    "pid": None,
                    "current_tick": 2,
                    "target_tick": 16,
                    "data_root": str(branch_dir),
                    "log_path": str(log_path),
                    "failed_attempt_dirs": [str(job_dir / "_failed_attempts" / "old")],
                }
            ]
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        ctrl._reset_incomplete_branches_for_resume(job_dir, state.load_job_state(job_dir))

        updated = state.load_job_state(job_dir)["branches"][0]
        self.assertEqual("pending", updated["status"])
        self.assertEqual(2, updated["current_tick"])
        self.assertEqual(16, updated["target_tick"])
        self.assertEqual("resume", (branch_dir / "marker.txt").read_text(encoding="utf-8"))

    def test_missing_interview_file_halts_before_admin_selection(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        branch_dir = job_dir / "station_data_s2"
        branch_dir.mkdir(parents=True)
        payload = {
            "branches": [
                {"seed": 2, "status": "completed", "data_root": str(branch_dir)}
            ]
        }
        state.save_job_state(job_dir, payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "running", "job_dir": str(job_dir)})

        ctrl = controller.Controller(self.tmpdir)
        with self.assertRaisesRegex(RuntimeError, "interview.yamll"):
            ctrl._verify_interviews_before_selection(job_dir, state.load_job_state(job_dir), payload["branches"])

        updated = state.load_job_state(job_dir)
        self.assertEqual("failed", updated["status"])
        self.assertEqual("failed", updated["branches"][0]["status"])
        self.assertIn("interview.yamll", updated["branches"][0]["error"])

    def test_admin_workspace_uses_relative_branch_symlinks(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)

        admin_dir = controller.admin.prepare_workspace({"seed_count": 1}, job_dir)

        link_path = admin_dir / "station_data_s1"
        self.assertTrue(link_path.is_symlink())
        self.assertEqual(os.path.join("..", "station_data_s1"), os.readlink(link_path))

        prompt = (admin_dir / "prompt.md").read_text(encoding="utf-8")
        self.assertNotIn("origin_station_data", prompt)
        self.assertNotIn("Pre-Branch Archive Context", prompt)
        self.assertIn("500 to 2000 words", prompt)
        self.assertIn("Important Work From Previous Stations Not Observed in Current Station", prompt)
        self.assertIn("revise them twice", prompt)
        self.assertIn("rather than arbitrarily trusting one source or discarding the result", prompt)

    def test_stagnation_admin_workspace_includes_origin_archive_context(self):
        job_dir = self.tmpdir / "station_multistart" / "320_job"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        origin_dir = job_dir / "origin_station_data"
        archive_dir = origin_dir / "capsules" / "archive"
        archive_dir.mkdir(parents=True)
        (archive_dir / "archive_7.yaml").write_text(
            yaml.safe_dump(
                {
                    "title": "Existing Basin",
                    "author_name": "Archivist",
                    "created_at_tick": 210,
                    "abstract": "A prior result overlapping one candidate lane.",
                }
            ),
            encoding="utf-8",
        )

        admin_dir = controller.admin.prepare_workspace(
            {"seed_count": 1, "mode": "stagnation", "branch_tick": 320},
            job_dir,
        )

        origin_link = admin_dir / "origin_station_data"
        self.assertTrue(origin_link.is_symlink())
        self.assertEqual(os.path.join("..", "origin_station_data"), os.readlink(origin_link))

        prompt = (admin_dir / "prompt.md").read_text(encoding="utf-8")
        self.assertIn("Pre-Branch Archive Context", prompt)
        self.assertIn("Archive #7: Existing Basin", prompt)
        self.assertIn("cat origin_station_data/capsules/archive/archive_{ID}.yaml", prompt)
        self.assertIn("Important Work From Previous Stations Not Observed in Current Station", prompt)
        self.assertIn("500 to 2000 words", prompt)

        validator = (admin_dir / "validate_submission.py").read_text(encoding="utf-8")
        self.assertIn("expected 500 to 2000", validator)

    def test_reports_complete_uses_same_guidance_limit_for_all_modes(self):
        admin_dir = self.tmpdir / "admin"
        reports_dir = admin_dir / "reports"
        reports_dir.mkdir(parents=True)
        (admin_dir / "station_data_s1").mkdir()
        (reports_dir / "selection_report.md").write_text("| seed | note |\n| --- | --- |\n| 1 | ok |\n", encoding="utf-8")
        (reports_dir / "selected.txt").write_text("1\n", encoding="utf-8")
        (reports_dir / "guidance_report.md").write_text(" ".join(["word"] * 1200), encoding="utf-8")

        self.assertTrue(controller.admin.reports_complete(admin_dir, 1, mode="init"))
        self.assertTrue(controller.admin.reports_complete(admin_dir, 1, mode="stagnation"))

    def test_admin_selection_imports_any_legacy_failure_and_resumes_same_session(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        origin_dir = job_dir / state.ORIGIN_DIR_NAME
        origin_dir.mkdir()
        (origin_dir / "constant_config.yaml").write_text(
            "MULTISTART_ADMIN_MODEL_NAME: gpt-5.5\n",
            encoding="utf-8",
        )
        payload = {"seed_count": 1, "mode": "init", "branches": [{"seed": 1, "status": "completed"}]}
        state.save_job_state(job_dir, payload)
        admin_dir = controller.admin.prepare_workspace(payload, job_dir)
        (admin_dir / controller.admin.ADMIN_TRANSCRIPT_FILENAME).write_text(
            '\n'.join([
                '{"type":"thread.started","thread_id":"legacy-thread"}',
                '{"type":"error","message":"model not provided"}',
                '{"type":"turn.failed"}',
            ]),
            encoding="utf-8",
        )

        launch_calls = []

        class CompleteOnResume:
            def __init__(self, command, **kwargs):
                launch_calls.append(command)
                self.pid = 1234
                self.returncode = 0
                self.stdin = mock.Mock()
                reports_dir = admin_dir / "reports"
                (reports_dir / "selection_report.md").write_text(
                    "| seed | note |\n| --- | --- |\n| 1 | best |\n",
                    encoding="utf-8",
                )
                (reports_dir / "guidance_report.md").write_text(" ".join(["word"] * 500), encoding="utf-8")
                (reports_dir / "selected.txt").write_text("1\n", encoding="utf-8")

            def poll(self):
                return self.returncode

        with mock.patch.object(controller.admin, "_detect_codex_executable", return_value="/bin/true"), \
                mock.patch.object(controller.admin.AdminSelectionManager, "_resume_backoff_schedule", return_value=[0]), \
                mock.patch.object(controller.admin.subprocess, "Popen", CompleteOnResume), \
                mock.patch.object(controller.admin.subprocess, "run", return_value=mock.Mock(returncode=0)):
            selected = controller.admin.run_selection(payload, job_dir)

        self.assertEqual(1, selected)
        self.assertEqual(1, len(launch_calls))
        self.assertIn("resume", launch_calls[0])
        self.assertIn("legacy-thread", launch_calls[0])
        self.assertIn("--model", launch_calls[0])
        self.assertEqual("gpt-5.5", launch_calls[0][launch_calls[0].index("--model") + 1])
        self.assertIn('web_search="disabled"', launch_calls[0])
        self.assertIn("sandbox_workspace_write.network_access=true", launch_calls[0])
        self.assertIn("features.network_proxy.enabled=true", launch_calls[0])
        self.assertTrue(any(
            value.startswith("features.network_proxy.domains=")
            and '"api.openai.com" = "allow"' in value
            for value in launch_calls[0]
        ))
        selection = state.load_job_state(job_dir)[controller.admin.ADMIN_SELECTION_STATE_KEY]
        self.assertEqual("completed", selection["status"])
        self.assertEqual(1, selection["spawn_count"])
        self.assertEqual(1, selection["resume_count"])

    def test_admin_selection_uses_resume_then_fresh_spawn_budgets_before_blocking(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        payload = {"seed_count": 1, "mode": "init", "branches": [{"seed": 1, "status": "completed"}]}
        state.save_job_state(job_dir, payload)
        admin_dir = controller.admin.prepare_workspace(payload, job_dir)
        launch_modes = []

        class FailLaunch:
            def __init__(self, command, **kwargs):
                launch_modes.append("resume" if "resume" in command else "fresh")
                self.pid = 5000 + len(launch_modes)
                self.returncode = 1
                self.stdin = mock.Mock()
                transcript_handle = kwargs["stdout"]
                if len(launch_modes) <= 3:
                    transcript_handle.write(
                        '{"type":"thread.started","thread_id":"thread-a"}\n'
                        '{"type":"error","message":"503 Service Unavailable"}\n'
                    )
                else:
                    transcript_handle.write(
                        '{"type":"turn.failed","error":{"message":"invalid report"}}\n'
                    )
                transcript_handle.flush()

            def poll(self):
                return self.returncode

        with mock.patch.object(controller.admin, "_detect_codex_executable", return_value="/bin/true"), \
                mock.patch.object(controller.admin.constants, "RESEARCH_CODER_MAX_SPAWNS", 2), \
                mock.patch.object(controller.admin.constants, "RESEARCH_CODER_MAX_RESUMES", 2), \
                mock.patch.object(controller.admin.AdminSelectionManager, "_resume_backoff_schedule", return_value=[0]), \
                mock.patch.object(controller.admin.subprocess, "Popen", FailLaunch):
            with self.assertRaises(controller.admin.AdminSelectionAttemptsExhausted):
                controller.admin.run_selection(payload, job_dir)

        self.assertEqual(["fresh", "resume", "resume", "fresh"], launch_modes)
        selection = state.load_job_state(job_dir)[controller.admin.ADMIN_SELECTION_STATE_KEY]
        self.assertEqual("blocked", selection["status"])
        self.assertEqual(2, selection["spawn_count"])
        self.assertEqual(4, len(selection["sessions"]))

    def test_controller_marks_job_failed_after_admin_retry_budget_exhaustion(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        branch_dir = job_dir / "station_data_s1"
        branch_dir.mkdir(parents=True)
        payload = {
            "job_id": "job",
            "status": "selecting",
            "seed_count": 1,
            "max_parallel": 1,
            "branches": [{"seed": 1, "status": "completed", "data_root": str(branch_dir)}],
        }
        state.save_job_state(job_dir, payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "selecting", "job_dir": str(job_dir)})
        ctrl = controller.Controller(self.tmpdir)
        error = controller.admin.AdminSelectionAttemptsExhausted("admin retry budget exhausted")

        with mock.patch.object(ctrl, "_reset_incomplete_branches_for_resume"), \
                mock.patch.object(ctrl, "_verify_interviews_before_selection"), \
                mock.patch.object(controller.admin, "run_selection", side_effect=error):
            with self.assertRaises(controller.admin.AdminSelectionAttemptsExhausted):
                ctrl._run_or_resume_job(job_dir)

        self.assertEqual("failed", state.load_job_state(job_dir)["status"])
        current = state.load_current_job(self.tmpdir)
        self.assertEqual("failed", current["status"])
        self.assertIn("admin retry budget exhausted", current["message"])

    def test_finalize_resume_completes_installed_job_without_recopying(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        archive_root = self.tmpdir / "station_data" / "multistart" / "1_job"
        archive_root.mkdir(parents=True)
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 2,
            "branches": [{"seed": 1, "status": "completed"}, {"seed": 2, "status": "completed"}],
        }
        job_dir.mkdir(parents=True)
        state.save_job_state(job_dir, payload)
        state.save_yaml_mapping(archive_root / "state.yaml", payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "finalizing", "job_dir": str(job_dir)})
        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.object(ctrl, "_post_guidance_message") as guidance:
            with mock.patch.object(ctrl, "_create_manual_backup") as backup:
                with mock.patch.object(ctrl, "_restart_normal_station") as restart:
                    ctrl.finalize_job(job_dir, payload, 2)
        guidance.assert_called_once()
        backup.assert_called_once_with("1_job")
        restart.assert_called_once()
        self.assertFalse(state.load_current_job(self.tmpdir))
        self.assertFalse(job_dir.exists())
        archived = state.load_yaml_mapping(archive_root / "state.yaml")
        self.assertEqual("complete", archived["status"])
        self.assertTrue(archived["finalization_steps"]["guidance_posted"])
        self.assertTrue(archived["finalization_steps"]["manual_backup_created"])

    def test_finalize_deletes_disposable_placeholder_live_station_data_without_config(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        selected = job_dir / "station_data_s2"
        selected.mkdir(parents=True)
        (selected / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "selected", "current_tick": 3}),
            encoding="utf-8",
        )
        (selected / "interview.yamll").write_text("agent_name: Test\nanswer: ok\n", encoding="utf-8")
        eval_path = selected / "rooms" / "research" / "evaluations" / "1.yaml"
        eval_path.parent.mkdir(parents=True)
        eval_path.write_text(f"submission_path: {selected}/rooms/research/storage/submission/1.py\n", encoding="utf-8")
        index_dir = selected / "index"
        index_dir.mkdir()
        for suffix in ("", "-wal", "-shm"):
            (index_dir / f"station_index.sqlite3{suffix}").write_text("stale index", encoding="utf-8")
        nested_index_dir = selected / "multistart" / "old_job" / "station_data_s1" / "index"
        nested_index_dir.mkdir(parents=True)
        (nested_index_dir / "station_index.sqlite3").write_text("nested stale index", encoding="utf-8")
        research_tmp = selected / "rooms" / "research" / "storage" / "tmp" / "lineage"
        research_tmp.mkdir(parents=True)
        (research_tmp / "scratch.txt").write_text("tmp", encoding="utf-8")
        shared_tmp = selected / "rooms" / "research" / "storage" / "shared" / "tmp" / "workspace"
        shared_tmp.mkdir(parents=True)
        (shared_tmp / "scratch.txt").write_text("shared tmp", encoding="utf-8")
        sync_dir = selected / "sync"
        sync_dir.mkdir()
        (sync_dir / "state.yaml").write_text("sync", encoding="utf-8")
        admin_dir = job_dir / "admin"
        admin_dir.mkdir()
        (admin_dir / "station_data_s2").symlink_to(selected, target_is_directory=True)
        live = self.tmpdir / "station_data"
        (live / "index").mkdir(parents=True)
        (live / "index" / "station_index.sqlite3").write_text("placeholder", encoding="utf-8")
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 2,
            "branches": [{"seed": 2, "status": "completed"}],
        }
        state.save_job_state(job_dir, payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "finalizing", "job_dir": str(job_dir)})

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.object(ctrl, "_post_guidance_message"):
            with mock.patch.object(ctrl, "_create_manual_backup"):
                with mock.patch.object(ctrl, "_restart_normal_station"):
                    ctrl.finalize_job(job_dir, payload, 2)

        self.assertTrue((self.tmpdir / "station_data" / "station_config.yaml").is_file())
        self.assertFalse((self.tmpdir / "station_data" / "interview.yamll").exists())
        installed_eval = self.tmpdir / "station_data" / "rooms" / "research" / "evaluations" / "1.yaml"
        self.assertIn(str(self.tmpdir / "station_data"), installed_eval.read_text(encoding="utf-8"))
        self.assertNotIn(str(selected), installed_eval.read_text(encoding="utf-8"))
        self.assertFalse((self.tmpdir / "station_data" / "index" / "station_index.sqlite3").exists())
        self.assertFalse((self.tmpdir / "station_data" / "index" / "station_index.sqlite3-wal").exists())
        self.assertFalse((self.tmpdir / "station_data" / "index" / "station_index.sqlite3-shm").exists())
        self.assertFalse((self.tmpdir / "station_data" / "index").exists())
        archived_root = self.tmpdir / "station_data" / "multistart" / "1_job"
        self.assertFalse((archived_root / "station_data_s2").exists())
        self.assertTrue((archived_root / "station_data_s2.installed.yaml").is_file())
        self.assertTrue((archived_root / "interviews" / "station_data_s2.interview.yamll").is_file())
        installed_metadata = state.load_yaml_mapping(archived_root / "station_data_s2.installed.yaml")
        self.assertTrue(installed_metadata["installed_as_live_station_data"])
        self.assertEqual("interviews/station_data_s2.interview.yamll", installed_metadata["interview"])
        self.assertFalse((self.tmpdir / "station_data" / "rooms" / "research" / "storage" / "tmp").exists())
        self.assertFalse((self.tmpdir / "station_data" / "rooms" / "research" / "storage" / "shared" / "tmp").exists())
        self.assertFalse((self.tmpdir / "station_data" / "sync").exists())
        archived_admin_link = archived_root / "admin" / "station_data_s2"
        self.assertFalse(os.path.lexists(archived_admin_link))
        self.assertFalse(any(path.name.startswith("_unexpected_live_station_data_") for path in archived_root.iterdir()))

    def test_finalize_deletes_unexpected_live_station_data_without_config(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        selected = job_dir / "station_data_s2"
        selected.mkdir(parents=True)
        (selected / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "selected", "current_tick": 3}),
            encoding="utf-8",
        )
        live = self.tmpdir / "station_data"
        live.mkdir(parents=True)
        (live / "manual_note.txt").write_text("discard me", encoding="utf-8")
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 2,
            "branches": [{"seed": 2, "status": "completed"}],
        }
        state.save_job_state(job_dir, payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "finalizing", "job_dir": str(job_dir)})

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.object(ctrl, "_post_guidance_message"):
            with mock.patch.object(ctrl, "_create_manual_backup"):
                with mock.patch.object(ctrl, "_restart_normal_station"):
                    ctrl.finalize_job(job_dir, payload, 2)

        archived_root = self.tmpdir / "station_data" / "multistart" / "1_job"
        self.assertFalse(any(path.name.startswith("_unexpected_live_station_data_") for path in archived_root.iterdir()))
        self.assertFalse((archived_root / "manual_note.txt").exists())

    def test_finalize_deletes_unexpected_live_station_data_at_tick_one(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        selected = job_dir / "station_data_s2"
        selected.mkdir(parents=True)
        (selected / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "selected", "current_tick": 3}),
            encoding="utf-8",
        )
        live = self.tmpdir / "station_data"
        live.mkdir(parents=True)
        (live / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "stray", "current_tick": 1}),
            encoding="utf-8",
        )
        (live / "early_note.txt").write_text("discard me", encoding="utf-8")
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 2,
            "branches": [{"seed": 2, "status": "completed"}],
        }
        state.save_job_state(job_dir, payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "finalizing", "job_dir": str(job_dir)})

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.object(ctrl, "_post_guidance_message"):
            with mock.patch.object(ctrl, "_create_manual_backup"):
                with mock.patch.object(ctrl, "_restart_normal_station"):
                    ctrl.finalize_job(job_dir, payload, 2)

        archived_root = self.tmpdir / "station_data" / "multistart" / "1_job"
        self.assertFalse(any(path.name.startswith("_unexpected_live_station_data_") for path in archived_root.iterdir()))
        self.assertFalse((archived_root / "early_note.txt").exists())

    def test_finalize_blocks_unexpected_live_station_data_after_tick_one(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        selected = job_dir / "station_data_s2"
        selected.mkdir(parents=True)
        (selected / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "selected", "current_tick": 3}),
            encoding="utf-8",
        )
        live = self.tmpdir / "station_data"
        live.mkdir(parents=True)
        (live / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "real", "current_tick": 2}),
            encoding="utf-8",
        )
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 2,
            "branches": [{"seed": 2, "status": "completed"}],
        }
        state.save_job_state(job_dir, payload)

        ctrl = controller.Controller(self.tmpdir)
        with self.assertRaisesRegex(RuntimeError, "cannot finalize while live station_data exists"):
            ctrl.finalize_job(job_dir, payload, 2)

    def test_finalize_removes_archived_origin_before_manual_backup(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        selected = job_dir / "station_data_s1"
        selected.mkdir(parents=True)
        (selected / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "selected", "current_tick": 12}),
            encoding="utf-8",
        )
        origin = job_dir / "origin_station_data"
        origin.mkdir(parents=True)
        protected = origin / "protected.txt"
        protected.write_text("already backed up\n", encoding="utf-8")
        protected.chmod(0o400)
        admin_dir = job_dir / "admin"
        admin_dir.mkdir()
        (admin_dir / "origin_station_data").symlink_to(os.path.join("..", "origin_station_data"), target_is_directory=True)
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 1,
            "branches": [{"seed": 1, "status": "completed"}],
        }
        state.save_job_state(job_dir, payload)
        state.save_current_job(self.tmpdir, {"job_id": "job", "status": "finalizing", "job_dir": str(job_dir)})

        ctrl = controller.Controller(self.tmpdir)
        archive_root = self.tmpdir / "station_data" / "multistart" / "1_job"

        def backup_after_origin_removal(archive_name):
            self.assertEqual("1_job", archive_name)
            self.assertFalse((archive_root / "origin_station_data").exists())
            self.assertFalse(os.path.lexists(archive_root / "admin" / "origin_station_data"))
            return {"station_id": "selected", "tick": 12, "manifest_path": str(self.tmpdir / "backup.json")}

        with mock.patch.object(ctrl, "_post_guidance_message"):
            with mock.patch.object(ctrl, "_create_manual_backup", side_effect=backup_after_origin_removal):
                with mock.patch.object(ctrl, "_restart_normal_station"):
                    ctrl.finalize_job(job_dir, payload, 1)

        archived = state.load_yaml_mapping(archive_root / "state.yaml")
        self.assertTrue(archived["finalization_steps"]["origin_station_data_removed_before_backup"])
        self.assertTrue(archived["origin_station_data_removal"]["removed"])
        self.assertFalse((archive_root / "origin_station_data").exists())

    def test_prune_archived_branch_data_after_verified_backup_handles_read_only_files(self):
        archive_root = self.tmpdir / "station_data" / "multistart" / "1_job"
        branch1 = archive_root / "station_data_s1"
        branch2 = archive_root / "station_data_s2"
        for seed, branch in ((1, branch1), (2, branch2)):
            system_dir = branch / "rooms" / "research" / "storage" / "system"
            system_dir.mkdir(parents=True)
            (branch / "station_config.yaml").write_text(
                yaml.safe_dump({"station_id": f"station-{seed}", "current_tick": 42 + seed}),
                encoding="utf-8",
            )
            (branch / "interview.yamll").write_text(f"agent_name: Seed {seed}\n", encoding="utf-8")
            protected = system_dir / "readonly.py"
            protected.write_text("content\n", encoding="utf-8")
            protected.chmod(0o400)
            system_dir.chmod(0o500)
        admin_dir = archive_root / "admin"
        admin_dir.mkdir()
        (admin_dir / "station_data_s1").symlink_to(os.path.join("..", "station_data_s1"), target_is_directory=True)

        manifest_path = self.tmpdir / "backup" / "station-id" / "snapshots" / "tick_50.json"
        manifest_path.parent.mkdir(parents=True)
        manifest = {
            "files": [
                {"path": "multistart/1_job/station_data_s1/station_config.yaml"},
                {"path": "multistart/1_job/station_data_s2/station_config.yaml"},
            ],
            "symlinks": [],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        payload = {
            "selected_seed": 2,
            "branches": [{"seed": 1, "status": "completed"}, {"seed": 2, "status": "completed"}],
        }

        ctrl = controller.Controller(self.tmpdir)
        result = ctrl._prune_archived_branch_data_after_backup(
            archive_root,
            payload,
            {"station_id": "station-id", "tick": 50, "manifest_path": str(manifest_path)},
        )

        self.assertTrue(result["success"], result)
        self.assertFalse(branch1.exists())
        self.assertFalse(branch2.exists())
        self.assertFalse((admin_dir / "station_data_s1").exists())
        self.assertFalse((admin_dir / "station_data_s1").is_symlink())
        self.assertTrue((archive_root / "station_data_s1.pruned.yaml").is_file())
        self.assertTrue((archive_root / "station_data_s2.pruned.yaml").is_file())
        self.assertTrue((archive_root / "interviews" / "station_data_s1.interview.yamll").is_file())
        branch_manifest = state.load_yaml_mapping(archive_root / "branch_archive_manifest.yaml")
        self.assertEqual("multistart/1_job", branch_manifest["restore"]["source_prefix"])
        self.assertEqual("multistart_1_job", branch_manifest["restore"]["default_output"])

    def test_create_job_removes_stale_root_interview_from_branch_copies(self):
        live = self._write_station_config(current_tick=0)
        (live / "interview.yamll").write_text("agent_name: Old\nanswer: stale\n", encoding="utf-8")
        index_dir = live / "index"
        index_dir.mkdir()
        (index_dir / "station_index.sqlite3").write_text("stale root index", encoding="utf-8")
        research_tmp = live / "rooms" / "research" / "storage" / "tmp" / "lineage"
        research_tmp.mkdir(parents=True)
        (research_tmp / "scratch.txt").write_text("tmp", encoding="utf-8")
        shared_tmp = live / "rooms" / "research" / "storage" / "shared" / "tmp" / "workspace"
        shared_tmp.mkdir(parents=True)
        (shared_tmp / "scratch.txt").write_text("shared tmp", encoding="utf-8")
        sync_dir = live / "sync"
        sync_dir.mkdir()
        (sync_dir / "state.yaml").write_text("sync", encoding="utf-8")

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: ""}):
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)
        payload = state.load_job_state(job_dir)

        self.assertTrue((job_dir / "origin_station_data" / "interview.yamll").is_file())
        self.assertEqual([16, 16], [branch["target_tick"] for branch in payload["branches"]])
        self.assertFalse((job_dir / "origin_station_data" / "index").exists())
        self.assertFalse((job_dir / "origin_station_data" / "rooms" / "research" / "storage" / "tmp").exists())
        self.assertFalse((job_dir / "origin_station_data" / "rooms" / "research" / "storage" / "shared" / "tmp").exists())
        self.assertFalse((job_dir / "origin_station_data" / "sync").exists())
        self.assertFalse((job_dir / "station_data_s1" / "interview.yamll").exists())
        self.assertFalse((job_dir / "station_data_s2" / "interview.yamll").exists())
        self.assertFalse((job_dir / "station_data_s1" / "index").exists())
        self.assertFalse((job_dir / "station_data_s2" / "index").exists())
        self.assertFalse((job_dir / "station_data_s1" / "rooms" / "research" / "storage" / "tmp").exists())
        self.assertFalse((job_dir / "station_data_s2" / "rooms" / "research" / "storage" / "shared" / "tmp").exists())

    def test_create_job_persists_waiting_state_before_copying(self):
        self._write_station_config(current_tick=0)
        ctrl = controller.Controller(self.tmpdir)

        def inspect_durable_state(job_dir, payload):
            current = state.load_current_job(self.tmpdir)
            durable = state.load_job_state(job_dir)
            self.assertEqual("creating", current["status"])
            self.assertEqual("creating", durable["status"])
            self.assertEqual(["copy_pending"] * 8, [b["status"] for b in durable["branches"]])
            self.assertTrue((self.tmpdir / "station_data").is_dir())

        with mock.patch.object(ctrl, "_resume_job_creation", side_effect=inspect_durable_state):
            ctrl.create_job("init", 8, 4, 16, branch_tick=0)

    def test_branch_copy_fanout_runs_all_eight_seeds_concurrently(self):
        ctrl = controller.Controller(self.tmpdir)
        job_dir = paths.multistart_root(self.tmpdir) / "0_parallel"
        job_dir.mkdir(parents=True)
        payload = {"branches": [{"seed": seed} for seed in range(1, 9)]}
        barrier = threading.Barrier(8, timeout=5)
        worker_names = set()
        worker_lock = threading.Lock()

        def copy_one(*_args):
            with worker_lock:
                worker_names.add(threading.current_thread().name)
            barrier.wait()

        with mock.patch.object(ctrl, "_branch_copy_is_complete", return_value=False):
            with mock.patch.object(ctrl, "_copy_one_branch", side_effect=copy_one) as copy:
                ctrl._copy_all_branches_parallel(
                    job_dir,
                    payload,
                    self.tmpdir / "origin",
                    "station-id",
                )

        self.assertEqual(8, copy.call_count)
        self.assertEqual(8, len(worker_names))

    def test_stagnation_waiting_services_start_before_branch_copy_creation(self):
        self._write_station_config(current_tick=42)
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        state.save_yaml_mapping(
            paths.pending_stagnation_path(self.tmpdir),
            {"type": "stagnation", "branch_tick": 42},
        )
        ctrl = controller.Controller(self.tmpdir)
        order = []

        with mock.patch("station.constants.MULTISTART_STAGNATION_SEEDS", 8):
            with mock.patch.object(ctrl, "_ensure_disk_space_for_branch_copy"):
                with mock.patch.object(ctrl, "_wait_for_live_station_quiescent"):
                    with mock.patch.object(ctrl, "_stop_normal_services_for_job", side_effect=lambda: order.append("stop")):
                        with mock.patch.object(ctrl, "_start_waiting_services", side_effect=lambda: order.append("waiting")):
                            with mock.patch.object(ctrl, "create_job", side_effect=lambda *_args, **_kwargs: order.append("copy")):
                                ctrl._check_pending_stagnation_request()

        self.assertEqual(["stop", "waiting", "copy"], order)

    def test_parallel_storage_copy_manifest_retains_all_eight_seed_records(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        storage.mkdir(parents=True)
        (storage / "artifact.txt").write_text("origin", encoding="utf-8")
        storage_base = self.tmpdir / "research_storage"

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(storage_base)}):
            job_dir = controller.Controller(self.tmpdir).create_job(
                "init", 8, 4, 16, branch_tick=0
            )

        manifest = state.load_yaml_mapping(job_dir / research_storage.JOB_MANIFEST_FILENAME)
        self.assertEqual({str(seed) for seed in range(1, 9)}, set(manifest["seeds"]))
        self.assertEqual({"ready"}, {info["status"] for info in manifest["seeds"].values()})

    def test_create_job_materializes_external_research_storage_per_branch(self):
        live = self._write_station_config(current_tick=0)
        external_storage = self.tmpdir / "external_research_storage"
        seed_bank = external_storage / "shared" / "seed_bank"
        seed_bank.mkdir(parents=True)
        (seed_bank / "index.sqlite").write_text("origin-index", encoding="utf-8")
        research_root = live / "rooms" / "research"
        research_root.mkdir(parents=True)
        (research_root / "storage").symlink_to(external_storage, target_is_directory=True)
        (live / "constant_config.yaml").write_text(
            "RESEARCH_SEED_BANK_ENABLED: true\n",
            encoding="utf-8",
        )

        ctrl = controller.Controller(self.tmpdir)
        job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        origin_storage = job_dir / "origin_station_data" / "rooms" / "research" / "storage"
        branch_one = job_dir / "station_data_s1" / "rooms" / "research" / "storage"
        branch_two = job_dir / "station_data_s2" / "rooms" / "research" / "storage"
        self.assertTrue(origin_storage.is_symlink())
        self.assertFalse(branch_one.is_symlink())
        self.assertFalse(branch_two.is_symlink())
        self.assertEqual("origin-index", (branch_one / "shared" / "seed_bank" / "index.sqlite").read_text())
        self.assertEqual("origin-index", (branch_two / "shared" / "seed_bank" / "index.sqlite").read_text())

        (branch_one / "shared" / "seed_bank" / "index.sqlite").write_text("branch-one", encoding="utf-8")
        self.assertEqual("origin-index", (branch_two / "shared" / "seed_bank" / "index.sqlite").read_text())
        self.assertEqual("origin-index", (external_storage / "shared" / "seed_bank" / "index.sqlite").read_text())

    def test_create_job_isolates_existing_storage_link_when_managed_base_is_unset(self):
        live = self._write_station_config(current_tick=0)
        external_storage = self.tmpdir / "external_research_storage"
        external_storage.mkdir(parents=True)
        research_root = live / "rooms" / "research"
        research_root.mkdir(parents=True)
        (research_root / "storage").symlink_to(external_storage, target_is_directory=True)
        (live / "constant_config.yaml").write_text(
            "RESEARCH_SEED_BANK_ENABLED: false\n",
            encoding="utf-8",
        )

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: ""}):
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        branch_one = job_dir / "station_data_s1" / "rooms" / "research" / "storage"
        branch_two = job_dir / "station_data_s2" / "rooms" / "research" / "storage"
        self.assertFalse(branch_one.is_symlink())
        self.assertFalse(branch_two.is_symlink())
        self.assertNotEqual(branch_one.resolve(), branch_two.resolve())
        self.assertFalse((job_dir / research_storage.JOB_MANIFEST_FILENAME).exists())

    def test_create_job_places_each_branch_in_uuid_storage_allocations(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        artifact = storage / "lineages" / "alpha" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("origin", encoding="utf-8")
        seed_bank_index = storage / "shared" / "seed_bank" / "index.sqlite"
        seed_bank_index.parent.mkdir(parents=True)
        seed_bank_index.write_text("origin-seed-bank", encoding="utf-8")
        storage_base = self.tmpdir / "research_storage"

        with mock.patch.dict(
            os.environ,
            {research_storage.BASE_PATH_ENV: str(storage_base)},
        ):
            ctrl = controller.Controller(self.tmpdir)
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        branch_one = job_dir / "station_data_s1" / "rooms" / "research" / "storage"
        branch_two = job_dir / "station_data_s2" / "rooms" / "research" / "storage"
        origin_storage = job_dir / "origin_station_data" / "rooms" / "research" / "storage"
        self.assertFalse(origin_storage.is_symlink())
        self.assertTrue(branch_one.is_symlink())
        self.assertTrue(branch_two.is_symlink())
        self.assertNotEqual(branch_one.resolve(), branch_two.resolve())
        self.assertNotEqual(origin_storage.resolve(), branch_one.resolve())
        self.assertTrue(research_storage.path_is_within(branch_one.resolve(), storage_base))
        self.assertTrue(research_storage.path_is_within(branch_two.resolve(), storage_base))
        self.assertEqual("origin", (branch_one / "lineages" / "alpha" / "artifact.txt").read_text())
        self.assertEqual("origin", (branch_two / "lineages" / "alpha" / "artifact.txt").read_text())

        (branch_one / "lineages" / "alpha" / "artifact.txt").write_text("seed-one", encoding="utf-8")
        self.assertEqual("origin", (branch_two / "lineages" / "alpha" / "artifact.txt").read_text())
        (branch_one / "shared" / "seed_bank" / "index.sqlite").write_text(
            "seed-one-bank",
            encoding="utf-8",
        )
        self.assertEqual(
            "origin-seed-bank",
            (branch_two / "shared" / "seed_bank" / "index.sqlite").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            "origin-seed-bank",
            (
                job_dir
                / "origin_station_data"
                / "rooms"
                / "research"
                / "storage"
                / "shared"
                / "seed_bank"
                / "index.sqlite"
            ).read_text(encoding="utf-8"),
        )
        self.assertEqual(
            "origin",
            (
                job_dir
                / "origin_station_data"
                / "rooms"
                / "research"
                / "storage"
                / "lineages"
                / "alpha"
                / "artifact.txt"
            ).read_text(),
        )
        manifest = state.load_yaml_mapping(job_dir / research_storage.JOB_MANIFEST_FILENAME)
        self.assertEqual({"1", "2"}, set(manifest["seeds"]))
        self.assertNotIn("origin", manifest)

    def test_multistart_uses_yaml_research_storage_base_when_environment_is_absent(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        storage.mkdir(parents=True)
        (storage / "artifact.txt").write_text("origin", encoding="utf-8")
        storage_base = self.tmpdir / "yaml_storage"
        (live / "constant_config.yaml").write_text(
            yaml.safe_dump({"RESEARCH_STORAGE_BASE_PATH": str(storage_base)}),
            encoding="utf-8",
        )

        with mock.patch.dict(os.environ, {}, clear=True):
            ctrl = controller.Controller(self.tmpdir)
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        manifest = state.load_yaml_mapping(job_dir / research_storage.JOB_MANIFEST_FILENAME)
        self.assertEqual(str(storage_base), manifest["base_path"])
        self.assertTrue(
            research_storage.path_is_within(
                research_storage.research_storage_path(job_dir / "station_data_s1").resolve(),
                storage_base,
            )
        )

    def test_create_job_relocates_existing_managed_origin_when_base_changes(self):
        live = self._write_station_config(current_tick=0)
        old_base = self.tmpdir / "old_storage"
        old_target = old_base / "old-live"
        old_target.mkdir(parents=True)
        (old_target / "artifact.txt").write_text("origin", encoding="utf-8")
        research_storage.write_allocation_marker(
            old_target,
            {"kind": "live", "station_id": "station-id"},
        )
        research_root = live / "rooms" / "research"
        research_root.mkdir(parents=True)
        research_storage.research_storage_path(live).symlink_to(
            old_target,
            target_is_directory=True,
        )
        new_base = self.tmpdir / "new_storage"

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(new_base)}):
            job_dir = controller.Controller(self.tmpdir).create_job(
                "init", 2, 2, 16, branch_tick=0
            )

        origin_storage = research_storage.research_storage_path(job_dir / "origin_station_data")
        self.assertTrue(origin_storage.is_symlink())
        self.assertTrue(research_storage.path_is_within(origin_storage.resolve(), new_base))
        self.assertEqual("origin", (origin_storage / "artifact.txt").read_text(encoding="utf-8"))
        self.assertFalse(old_target.exists())
        self.assertFalse(research_storage.allocation_marker_path(old_target).exists())
        for seed in (1, 2):
            seed_storage = research_storage.research_storage_path(job_dir / f"station_data_s{seed}")
            self.assertTrue(research_storage.path_is_within(seed_storage.resolve(), new_base))

    def test_interrupted_creation_keeps_valid_manifest_base_after_env_change(self):
        live = self._write_station_config(current_tick=0)
        storage = research_storage.research_storage_path(live)
        storage.mkdir(parents=True)
        (storage / "artifact.txt").write_text("origin", encoding="utf-8")
        first_base = self.tmpdir / "first_storage"
        second_base = self.tmpdir / "second_storage"

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(first_base)}):
            ctrl = controller.Controller(self.tmpdir)
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        manifest_path = job_dir / research_storage.JOB_MANIFEST_FILENAME
        manifest = state.load_yaml_mapping(manifest_path)
        seed2_target = Path(manifest["seeds"].pop("2")["target"])
        shutil.rmtree(seed2_target)
        research_storage.remove_allocation_marker(seed2_target)
        state.save_yaml_mapping(manifest_path, manifest)
        shutil.rmtree(job_dir / "station_data_s2")

        payload = state.load_job_state(job_dir)
        payload["status"] = "creating"
        for branch in payload["branches"]:
            if branch["seed"] == 2:
                branch["status"] = "copy_pending"
                branch["copy_status"] = "pending"
        state.save_job_state(job_dir, payload)

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(second_base)}):
            ctrl._resume_job_creation(job_dir, payload)

        resumed_manifest = state.load_yaml_mapping(manifest_path)
        self.assertEqual(str(first_base), resumed_manifest["base_path"])
        for seed in (1, 2):
            seed_storage = research_storage.research_storage_path(job_dir / f"station_data_s{seed}")
            self.assertTrue(research_storage.path_is_within(seed_storage.resolve(), first_base))
            self.assertEqual("origin", (seed_storage / "artifact.txt").read_text(encoding="utf-8"))
        self.assertFalse(second_base.exists())

    def test_restored_active_job_reconciles_regular_storage_into_current_base(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        artifact = storage / "lineages" / "alpha" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("origin", encoding="utf-8")
        first_base = self.tmpdir / "first_storage"
        second_base = self.tmpdir / "restored_storage"

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(first_base)}):
            ctrl = controller.Controller(self.tmpdir)
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        data_roots = [job_dir / "station_data_s1", job_dir / "station_data_s2"]
        payload = state.load_job_state(job_dir)
        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(second_base)}):
            ctrl._reconcile_job_storage_allocations(job_dir, payload)
        self.assertTrue(
            all(
                research_storage.path_is_within(
                    research_storage.research_storage_path(data_root).resolve(),
                    first_base,
                )
                for data_root in data_roots
            )
        )

        for data_root in data_roots:
            storage_path = research_storage.research_storage_path(data_root)
            allocation = storage_path.resolve()
            restored = storage_path.parent / ".restored_storage"
            shutil.copytree(allocation, restored, symlinks=True)
            storage_path.unlink()
            os.replace(restored, storage_path)
            shutil.rmtree(allocation)
            research_storage.remove_allocation_marker(allocation)

        with mock.patch.dict(os.environ, {research_storage.BASE_PATH_ENV: str(second_base)}):
            ctrl._reconcile_job_storage_allocations(job_dir, payload)

        resolved_targets = []
        for data_root in data_roots:
            storage_path = research_storage.research_storage_path(data_root)
            self.assertTrue(storage_path.is_symlink())
            self.assertTrue(research_storage.path_is_within(storage_path.resolve(), second_base))
            self.assertEqual("origin", (storage_path / "lineages" / "alpha" / "artifact.txt").read_text())
            resolved_targets.append(storage_path.resolve())
        self.assertEqual(2, len(set(resolved_targets)))

        manifest = state.load_yaml_mapping(job_dir / research_storage.JOB_MANIFEST_FILENAME)
        self.assertEqual(str(second_base), manifest["base_path"])
        self.assertEqual(
            set(resolved_targets),
            {Path(info["target"]) for info in manifest["seeds"].values()},
        )

    def test_storage_base_preflight_keeps_live_data_when_root_is_not_writable(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        storage.mkdir(parents=True)
        (storage / "artifact.txt").write_text("origin", encoding="utf-8")
        storage_base = self.tmpdir / "research_storage"

        with mock.patch.dict(
            os.environ,
            {research_storage.BASE_PATH_ENV: str(storage_base)},
        ):
            with mock.patch(
                "station.multistart.controller.os.open",
                side_effect=OSError(30, "Read-only file system"),
            ):
                ctrl = controller.Controller(self.tmpdir)
                with self.assertRaisesRegex(controller.MultistartDiskSpaceError, "not writable"):
                    ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        self.assertTrue((live / "station_config.yaml").is_file())
        self.assertFalse(any(paths.multistart_root(self.tmpdir).glob("0_*")))

    def test_storage_allocation_copy_failure_preserves_recoverable_creating_job(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        storage.mkdir(parents=True)
        (storage / "artifact.txt").write_text("origin", encoding="utf-8")
        storage_base = self.tmpdir / "research_storage"

        with mock.patch.dict(
            os.environ,
            {research_storage.BASE_PATH_ENV: str(storage_base)},
        ):
            ctrl = controller.Controller(self.tmpdir)
            with mock.patch.object(
                ctrl,
                "_install_branch_storage_allocation",
                side_effect=RuntimeError("remote copy failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "remote copy failed"):
                    ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        self.assertFalse(live.exists())
        current = state.load_current_job(self.tmpdir)
        self.assertEqual("creating", current["status"])
        job_dir = Path(current["job_dir"])
        self.assertTrue((job_dir / "origin_station_data" / "station_config.yaml").is_file())
        self.assertEqual(
            "origin",
            (
                job_dir
                / "origin_station_data"
                / "rooms"
                / "research"
                / "storage"
                / "artifact.txt"
            ).read_text(),
        )
        payload = state.load_job_state(job_dir)
        self.assertEqual("creating", payload["status"])
        self.assertIn("remote copy failed", payload["creation_error"])

    def test_storage_allocation_cannot_be_nested_inside_source_storage(self):
        source_storage = self.tmpdir / "source" / "rooms" / "research" / "storage"
        source_storage.mkdir(parents=True)
        (source_storage / "artifact.txt").write_text("origin", encoding="utf-8")
        branch_root = self.tmpdir / "branch"
        storage_base = source_storage / "allocations"
        ctrl = controller.Controller(self.tmpdir)

        with self.assertRaisesRegex(RuntimeError, "inside its source"):
            ctrl._install_branch_storage_allocation(
                source_storage,
                branch_root,
                job_path=self.tmpdir / "job",
                job_id="job-id",
                station_id="station-id",
                seed=1,
                storage_base=storage_base,
            )

        self.assertFalse(branch_root.exists())

    def test_selected_storage_allocation_is_promoted_and_obsolete_allocations_are_removed(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        artifact = storage / "lineages" / "alpha" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("origin", encoding="utf-8")
        storage_base = self.tmpdir / "research_storage"

        with mock.patch.dict(
            os.environ,
            {research_storage.BASE_PATH_ENV: str(storage_base)},
        ):
            ctrl = controller.Controller(self.tmpdir)
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        selected = job_dir / "station_data_s2"
        selected_storage = research_storage.research_storage_path(selected)
        self.assertTrue(selected_storage.is_symlink())
        selected_target = selected_storage.resolve()
        ctrl._promote_selected_research_storage(job_dir, selected, 2)
        ctrl._promote_selected_research_storage(job_dir, selected, 2)
        self.assertTrue(selected_storage.is_symlink())
        self.assertEqual(selected_target, selected_storage.resolve())
        self.assertEqual("origin", (selected_storage / "lineages" / "alpha" / "artifact.txt").read_text())
        selected_marker = research_storage.read_allocation_marker(selected_target)
        self.assertEqual("live", selected_marker["kind"])
        self.assertEqual(job_dir.name.split("_", 1)[1], selected_marker["promoted_from_job_id"])

        archive_root = self.tmpdir / "archive"
        archive_root.mkdir()
        shutil.copy2(
            job_dir / research_storage.JOB_MANIFEST_FILENAME,
            archive_root / research_storage.JOB_MANIFEST_FILENAME,
        )
        result = ctrl._remove_job_seed_storage_allocations(
            archive_root,
            preserve_selected=True,
            include_origin=True,
        )
        self.assertTrue(result["success"], result)
        self.assertTrue(selected_target.exists())
        self.assertEqual(1, len(result["removed"]))
        repeated = ctrl._remove_job_seed_storage_allocations(
            archive_root,
            preserve_selected=True,
            include_origin=True,
        )
        self.assertTrue(repeated["success"], repeated)
        self.assertEqual(1, len(repeated["already_missing"]))

    def test_storage_cleanup_removes_read_only_system_tree_and_preserves_selected(self):
        storage_base = self.tmpdir / "research_storage"
        selected_target = storage_base / "selected"
        selected_target.mkdir(parents=True)
        obsolete_target = storage_base / "obsolete"
        build_dir = obsolete_target / "system" / "_algorithmic_build"
        build_dir.mkdir(parents=True)
        compiled = build_dir / "epoch_book_b_n36_c_search.so"
        compiled.write_bytes(b"compiled")
        compiled.chmod(0o444)
        build_dir.chmod(0o555)
        build_dir.parent.chmod(0o555)
        research_storage.write_allocation_marker(
            obsolete_target,
            {
                "kind": "multistart_seed",
                "station_id": "station-id",
                "job_id": "job-id",
                "seed": 2,
            },
        )
        archive_root = self.tmpdir / "archive"
        archive_root.mkdir()
        state.save_yaml_mapping(
            archive_root / research_storage.JOB_MANIFEST_FILENAME,
            {
                "base_path": str(storage_base),
                "station_id": "station-id",
                "job_id": "job-id",
                "selected_seed": 1,
                "seeds": {
                    "1": {"target": str(selected_target)},
                    "2": {"target": str(obsolete_target)},
                },
            },
        )

        real_chmod = os.chmod

        def nfs_chmod_denied(*_args, **_kwargs):
            raise PermissionError("NFS rejects chmod")

        def sudo_remove(command, **_kwargs):
            target = Path(command[-1])
            for root, dirs, _files in os.walk(target):
                real_chmod(root, 0o700)
                for dirname in dirs:
                    child = Path(root) / dirname
                    if not child.is_symlink():
                        real_chmod(child, 0o700)
            shutil.rmtree(target)
            return mock.Mock(returncode=0)

        with mock.patch("station.research_storage.os.chmod", side_effect=nfs_chmod_denied):
            with mock.patch("station.research_storage.subprocess.run", side_effect=sudo_remove) as sudo:
                result = controller.Controller(self.tmpdir)._remove_job_seed_storage_allocations(
                    archive_root,
                    preserve_selected=True,
                    include_origin=True,
                )
        sudo.assert_called_once()

        self.assertTrue(result["success"], result)
        self.assertFalse(obsolete_target.exists())
        self.assertFalse(research_storage.allocation_marker_path(obsolete_target).exists())
        self.assertTrue(selected_target.exists())

    def test_finalize_promotes_selected_storage_and_cleans_obsolete_allocations(self):
        live = self._write_station_config(current_tick=0)
        storage = live / "rooms" / "research" / "storage"
        artifact = storage / "lineages" / "alpha" / "artifact.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("origin", encoding="utf-8")
        storage_base = self.tmpdir / "research_storage"

        with mock.patch.dict(
            os.environ,
            {research_storage.BASE_PATH_ENV: str(storage_base)},
        ):
            ctrl = controller.Controller(self.tmpdir)
            job_dir = ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        payload = state.load_job_state(job_dir)
        payload["status"] = "finalizing"
        payload["selected_seed"] = 2
        for branch in payload["branches"]:
            branch["status"] = "completed"
        state.save_job_state(job_dir, payload)
        storage_manifest = state.load_yaml_mapping(job_dir / research_storage.JOB_MANIFEST_FILENAME)
        selected_target = Path(storage_manifest["seeds"]["2"]["target"])
        selected_system = selected_target / "system"
        selected_system.mkdir(parents=True, exist_ok=True)
        selected_client = selected_system / "seed_bank.py"
        selected_client.write_text("# branch-private frozen client\n", encoding="utf-8")
        selected_system.chmod(0o555)
        obsolete_targets = {
            Path(storage_manifest["seeds"]["1"]["target"]),
        }
        archive_name = job_dir.name
        backup_manifest = self.tmpdir / "finalization_manifest.json"
        backup_manifest.write_text(
            json.dumps({
                "files": [
                    {
                        "path": (
                            f"multistart/{archive_name}/station_data_s1/"
                            "rooms/research/storage/lineages/alpha/artifact.txt"
                        )
                    }
                ],
                "symlinks": [],
            }),
            encoding="utf-8",
        )

        with mock.patch.object(ctrl, "_post_guidance_message"):
            with mock.patch.object(
                ctrl,
                "_create_manual_backup",
                return_value={"station_id": "station-id", "tick": 16, "manifest_path": str(backup_manifest)},
            ):
                with mock.patch.object(ctrl, "_restart_normal_station"):
                    ctrl.finalize_job(job_dir, payload, 2)

        installed_storage = self.tmpdir / "station_data" / "rooms" / "research" / "storage"
        self.assertTrue(installed_storage.is_symlink())
        self.assertEqual(selected_target, installed_storage.resolve())
        self.assertEqual("origin", (installed_storage / "lineages" / "alpha" / "artifact.txt").read_text())
        self.assertTrue((installed_storage / "system" / "seed_bank.py").is_file())
        self.assertFalse((installed_storage / "system" / "seed_bank.py").is_symlink())
        self.assertEqual(
            "# branch-private frozen client\n",
            (installed_storage / "system" / "seed_bank.py").read_text(encoding="utf-8"),
        )
        self.assertTrue(selected_target.exists())
        self.assertTrue(all(not path.exists() for path in obsolete_targets))
        archived = state.load_yaml_mapping(
            self.tmpdir / "station_data" / "multistart" / archive_name / "state.yaml"
        )
        self.assertTrue(archived["finalization_steps"]["obsolete_research_storage_removed"])

    def test_create_job_disk_guard_halts_before_moving_live_station_data(self):
        live = self._write_station_config(current_tick=0)
        (live / "large.bin").write_bytes(b"x" * 128)
        ctrl = controller.Controller(self.tmpdir)

        disk_usage = mock.Mock(total=1000, used=949, free=51)
        with mock.patch("station.multistart.controller.shutil.disk_usage", return_value=disk_usage):
            with self.assertRaisesRegex(RuntimeError, "multistart waiting for disk space.*retry automatically"):
                ctrl.create_job("init", 2, 2, 16, branch_tick=0)

        self.assertTrue((self.tmpdir / "station_data" / "station_config.yaml").is_file())
        root = paths.multistart_root(self.tmpdir)
        self.assertTrue(root.is_dir())
        self.assertEqual([], list(root.iterdir()))

    def test_init_disk_guard_keeps_pending_request_and_retries(self):
        live = self._write_station_config(current_tick=0)
        (live / "large.bin").write_bytes(b"x" * 128)
        ctrl = controller.Controller(self.tmpdir)

        blocked_usage = mock.Mock(total=1000, used=949, free=51)
        with mock.patch("station.constants.MULTISTART_INIT_SEEDS", 2):
            with mock.patch("station.multistart.controller.shutil.disk_usage", return_value=blocked_usage):
                self.assertTrue(ctrl.start_init_job_if_needed())

            pending = state.load_yaml_mapping(paths.pending_init_path(self.tmpdir))
            self.assertEqual("blocked_disk_space", pending["status"])
            self.assertTrue((live / "station_config.yaml").is_file())
            self.assertEqual({}, state.load_current_job(self.tmpdir))

            available_usage = mock.Mock(total=10_000_000, used=0, free=10_000_000)
            with mock.patch("station.multistart.controller.shutil.disk_usage", return_value=available_usage):
                self.assertTrue(ctrl._check_pending_init_request())

        self.assertFalse(paths.pending_init_path(self.tmpdir).exists())
        current = state.load_current_job(self.tmpdir)
        self.assertEqual("init", current["mode"])
        self.assertFalse(live.exists())

    def test_finalize_does_not_require_duplicate_audit_copy_space(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        selected = job_dir / "station_data_s2"
        selected.mkdir(parents=True)
        (selected / "station_config.yaml").write_text(
            yaml.safe_dump({"station_id": "selected", "current_tick": 3}),
            encoding="utf-8",
        )
        (selected / "large.bin").write_bytes(b"x" * 128)
        payload = {
            "job_id": "job",
            "status": "finalizing",
            "selected_seed": 2,
            "branches": [{"seed": 2, "status": "completed"}],
        }

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch("station.multistart.controller.shutil.disk_usage", side_effect=AssertionError("no final copy check")):
            with mock.patch.object(ctrl, "_post_guidance_message"):
                with mock.patch.object(ctrl, "_create_manual_backup"):
                    with mock.patch.object(ctrl, "_restart_normal_station"):
                        ctrl.finalize_job(job_dir, payload, 2)

        live = self.tmpdir / "station_data"
        archived_root = live / "multistart" / "1_job"
        self.assertTrue((live / "station_config.yaml").is_file())
        self.assertTrue((archived_root / "station_data_s2.installed.yaml").is_file())
        self.assertFalse((archived_root / "station_data_s2").exists())
        self.assertFalse((archived_root / "_audit_selected").exists())

    def test_pid_running_rejects_unrelated_process_pid(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        pid_path = paths.controller_pid_path(self.tmpdir)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self.assertFalse(controller.pid_running(pid_path))

    def test_launch_branch_sets_branch_data_root_env_before_python_imports(self):
        job_dir = self.tmpdir / "station_multistart" / "1_job"
        job_dir.mkdir(parents=True)
        payload = {
            "mode": "init",
            "roll_ticks": 40,
            "branch_tick": 1,
            "branches": [{"seed": 2, "status": "pending"}],
        }
        state.save_job_state(job_dir, payload)
        branch_dir = state.branch_dir(job_dir, 2)

        class DummyProcess:
            pid = 12345

        ctrl = controller.Controller(self.tmpdir)
        with mock.patch("station.multistart.controller.subprocess.Popen", return_value=DummyProcess()) as popen:
            ctrl._launch_branch(job_dir, payload, 2)

        launch_env = popen.call_args.kwargs["env"]
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(str(branch_dir), launch_env["STATION_BASE_DATA_PATH"])
        self.assertEqual("1", launch_env["STATION_MULTISTART_BRANCH"])
        self.assertEqual("2", launch_env["STATION_MULTISTART_SEED"])
        self.assertEqual("1", launch_env["STATION_DISABLE_BACKUPS"])
        self.assertEqual("False", launch_env["AUTO_START"])

    def test_force_stop_always_runs_repo_scoped_cleanup(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        paths.controller_pid_path(self.tmpdir).write_text("12345", encoding="utf-8")
        with mock.patch("station.multistart.controller.ipc.request_stop", return_value={"success": True}):
            with mock.patch(
                "station.multistart.controller._force_stop_multistart_processes",
                return_value={12345},
            ) as force_stop:
                with mock.patch("station.multistart.controller.pid_running", return_value=False):
                    rc = controller.main(["stop", "--repo", str(self.tmpdir), "--force"])

        self.assertEqual(0, rc)
        force_stop.assert_called_once_with(self.tmpdir.resolve(), include_controller=True)

    def test_force_stop_runs_cleanup_when_ipc_raises(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        paths.controller_pid_path(self.tmpdir).write_text("12345", encoding="utf-8")
        with mock.patch("station.multistart.controller.ipc.request_stop", side_effect=PermissionError("denied")):
            with mock.patch(
                "station.multistart.controller._force_stop_multistart_processes",
                return_value={12345},
            ) as force_stop:
                with mock.patch("station.multistart.controller.pid_running", return_value=False):
                    rc = controller.main(["stop", "--repo", str(self.tmpdir), "--force"])

        self.assertEqual(0, rc)
        force_stop.assert_called_once_with(self.tmpdir.resolve(), include_controller=True)

    def test_force_stop_fails_when_repo_multistart_processes_survive(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        with mock.patch("station.multistart.controller.ipc.request_stop", return_value={"success": True}):
            with mock.patch("station.multistart.controller._force_stop_multistart_processes", return_value={12345}):
                with mock.patch("station.multistart.controller.pid_running", return_value=False):
                    with mock.patch("station.multistart.controller.find_running_controller_pid", return_value=None):
                        with mock.patch(
                            "station.multistart.controller._multistart_process_groups",
                            return_value={12345},
                        ):
                            stdout = io.StringIO()
                            with contextlib.redirect_stdout(stdout):
                                rc = controller.main(["stop", "--repo", str(self.tmpdir), "--force"])

        self.assertEqual(1, rc)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["success"])
        self.assertEqual([12345], payload["active_process_groups"])

    def test_graceful_stop_succeeds_when_no_controller_is_running(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        with mock.patch(
            "station.multistart.controller.ipc.request_stop",
            return_value={"success": False, "error": "controller socket not found"},
        ):
            with mock.patch("station.multistart.controller.pid_running", return_value=False):
                with mock.patch("station.multistart.controller.find_running_controller_pid", return_value=None):
                    stdout = io.StringIO()
                    with contextlib.redirect_stdout(stdout):
                        rc = controller.main(["stop", "--repo", str(self.tmpdir)])

        self.assertEqual(0, rc)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["success"])
        self.assertIn("no running multistart controller", payload["message"])

    def test_graceful_stop_times_out_with_active_branch_details(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "12_job"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "job", "status": "running", "job_dir": str(job_dir)},
        )
        state.save_job_state(
            job_dir,
            {
                "branches": [
                    {"seed": 3, "status": "running", "pid": 999999},
                ],
            },
        )
        with mock.patch("station.multistart.controller.ipc.request_stop", return_value={"success": True}):
            with mock.patch("station.multistart.controller.pid_running", return_value=True):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    rc = controller.main(["stop", "--repo", str(self.tmpdir), "--timeout-seconds", "0"])

        self.assertEqual(1, rc)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["success"])
        self.assertIn("timed out", payload["error"])
        self.assertEqual(["s3 status=running pid=999999 alive=false"], payload["active_branches"])

    def test_graceful_stop_does_not_exit_while_repo_multistart_processes_remain(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        with mock.patch("station.multistart.controller.ipc.request_stop", return_value={"success": True}):
            with mock.patch("station.multistart.controller.pid_running", return_value=False):
                with mock.patch("station.multistart.controller.find_running_controller_pid", return_value=None):
                    with mock.patch(
                        "station.multistart.controller._multistart_process_groups",
                        return_value={12345},
                    ):
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            rc = controller.main(["stop", "--repo", str(self.tmpdir), "--timeout-seconds", "0"])

        self.assertEqual(1, rc)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["success"])
        self.assertEqual([12345], payload["active_process_groups"])

    def test_graceful_stop_recovers_unreachable_controller_for_active_job(self):
        root = paths.multistart_root(self.tmpdir)
        root.mkdir()
        job_dir = root / "12_job"
        job_dir.mkdir()
        state.save_current_job(
            self.tmpdir,
            {"job_id": "job", "status": "running", "job_dir": str(job_dir)},
        )
        state.save_job_state(job_dir, {"branches": []})
        stop_responses = [
            {"success": False, "error": "controller IPC stop failed: [Errno 111] Connection refused"},
            {"success": True},
        ]
        with mock.patch("station.multistart.controller.ipc.request_stop", side_effect=stop_responses) as request_stop:
            with mock.patch("station.multistart.controller.ipc.request_status", return_value={"success": True}):
                with mock.patch(
                    "station.multistart.controller._terminate_unresponsive_controllers",
                    return_value=[999],
                ) as terminate:
                    with mock.patch("station.multistart.controller.start_detached", return_value=12345) as start:
                        with mock.patch("station.multistart.controller._multistart_shutdown_complete", return_value=True):
                            stdout = io.StringIO()
                            with contextlib.redirect_stdout(stdout):
                                rc = controller.main(["stop", "--repo", str(self.tmpdir)])

        self.assertEqual(0, rc)
        self.assertEqual(2, request_stop.call_count)
        terminate.assert_called_once_with(self.tmpdir.resolve())
        start.assert_called_once_with(self.tmpdir.resolve(), init=False)
        self.assertTrue(json.loads(stdout.getvalue())["success"])

    def test_restart_normal_station_uses_auto_start_flag(self):
        (self.tmpdir / "start.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        ctrl = controller.Controller(self.tmpdir)
        with mock.patch("station.multistart.controller.subprocess.Popen") as popen:
            ctrl._restart_normal_station()
        command = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertEqual([str(self.tmpdir / "start.sh"), "-s"], command)
        self.assertEqual("1", env["STATION_MULTISTART_SKIP_CONTROLLER_START"])
        self.assertEqual("1", env["STATION_MULTISTART_SKIP_HOOK"])

    def test_admin_guidance_announcement_is_agent_facing_previous_stations(self):
        message = controller.admin.guidance_announcement("Concrete guidance.")
        self.assertIn("previous stations", message)
        self.assertNotIn("previous branches", message)

    def test_live_station_quiescent_requires_all_pending_background_flags_clear(self):
        ctrl = controller.Controller(self.tmpdir)
        payload = {
            "success": True,
            "statistics": {
                "running_experiments_count": 0,
                "queued_experiments_count": 0,
                "drainable_running_jobs_count": 1,
                "drainable_queued_jobs_count": 0,
                "pending_research_evaluations": False,
                "pending_coder_sessions": True,
                "pending_external_reports": False,
                "pending_archive_surveys": False,
                "pending_archive_evaluations": False,
            },
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with mock.patch("station.multistart.controller.urllib.request.urlopen", return_value=Response()):
            self.assertFalse(ctrl._live_station_quiescent_once())

        payload["statistics"]["pending_coder_sessions"] = False
        payload["statistics"]["drainable_running_jobs_count"] = 0
        with mock.patch("station.multistart.controller.urllib.request.urlopen", return_value=Response()):
            self.assertTrue(ctrl._live_station_quiescent_once())


if __name__ == "__main__":
    unittest.main()
