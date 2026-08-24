from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from station import constants, file_io_utils, research_storage
from station.multistart import admin, interviews, ipc, paths, state, waiting


POLL_SECONDS = 5.0
FORCE_STOP_TERM_SECONDS = 5.0
FORCE_STOP_KILL_SECONDS = 5.0
DEFAULT_GRACEFUL_STOP_TIMEOUT_SECONDS = 7200.0
GRACEFUL_STOP_STATUS_SECONDS = 60.0
CONTROLLER_RECOVERY_TIMEOUT_SECONDS = 30.0
CONTROLLER_IPC_SELF_HEAL_GRACE_SECONDS = 2.0
DISK_SPACE_HALT_FRACTION = 0.95
BRANCH_COPY_EXCLUDED_TOP_LEVEL = frozenset({"multistart"})


class MultistartDiskSpaceError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)


def _read_proc_cwd(pid: int) -> str:
    try:
        return str(Path(f"/proc/{pid}/cwd").resolve())
    except Exception:
        return ""


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_is_repo_multistart_process(pid: int, repo: Path) -> bool:
    if pid <= 1 or pid == os.getpid():
        return False
    root_text = str(paths.multistart_root(repo).resolve())
    cmdline = _read_proc_cmdline(pid)
    cwd = _read_proc_cwd(pid)
    return (
        root_text in cmdline
        or cwd == root_text
        or cwd.startswith(root_text + os.sep)
    )


def _multistart_process_groups(repo: Path, *, include_controller: bool) -> set[int]:
    repo = repo.resolve()
    groups: set[int] = set()
    controller_pid = find_running_controller_pid(repo)
    if include_controller and controller_pid is not None:
        try:
            groups.add(os.getpgid(controller_pid))
        except OSError:
            pass
    for proc_path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_path.name)
        except ValueError:
            continue
        if not _pid_is_repo_multistart_process(pid, repo):
            continue
        if not include_controller and controller_pid is not None:
            try:
                if os.getpgid(pid) == os.getpgid(controller_pid):
                    continue
            except OSError:
                continue
        try:
            groups.add(os.getpgid(pid))
        except OSError:
            continue
    groups.discard(os.getpgrp())
    return groups


def _wait_process_groups_dead(groups: set[int], deadline: float) -> None:
    while time.monotonic() < deadline:
        alive = False
        for pgid in groups:
            try:
                os.killpg(pgid, 0)
                alive = True
                break
            except OSError:
                continue
        if not alive:
            return
        time.sleep(0.2)


def _force_stop_multistart_processes(
    repo: Path,
    *,
    include_controller: bool,
    log: Any | None = None,
) -> set[int]:
    groups = _multistart_process_groups(repo, include_controller=include_controller)
    if not groups:
        return set()
    if log is not None:
        log(f"force stopping multistart process group(s): {sorted(groups)}")
    for pgid in sorted(groups):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            if log is not None:
                log(f"warning: could not TERM process group {pgid}: {exc}")
    _wait_process_groups_dead(groups, time.monotonic() + FORCE_STOP_TERM_SECONDS)
    for pgid in sorted(groups):
        try:
            os.killpg(pgid, 0)
        except OSError:
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            if log is not None:
                log(f"warning: could not KILL process group {pgid}: {exc}")
    _wait_process_groups_dead(groups, time.monotonic() + FORCE_STOP_KILL_SECONDS)
    return groups


class Controller:
    def __init__(self, repo: Path):
        self.repo = repo.resolve()
        self.stop_requested = threading.Event()
        self.force_stop_requested = threading.Event()
        self.runtime_api_payload: dict[str, Any] | None = None
        self._server_thread: threading.Thread | None = None
        self._server_socket: socket.socket | None = None
        self._ipc_lock = threading.Lock()
        self._ipc_generation = 0
        self._job_state_lock = threading.Lock()
        self._storage_manifest_lock = threading.Lock()

    def log(self, message: str) -> None:
        paths.multistart_root(self.repo).mkdir(parents=True, exist_ok=True)
        line = f"{state.utc_now()} {message.rstrip()}\n"
        with paths.controller_log_path(self.repo).open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[multistart-controller] {message}", flush=True)

    def write_pid(self) -> None:
        paths.multistart_root(self.repo).mkdir(parents=True, exist_ok=True)
        paths.controller_pid_path(self.repo).write_text(str(os.getpid()), encoding="utf-8")

    def start_ipc(self) -> None:
        root = paths.multistart_root(self.repo)
        root.mkdir(parents=True, exist_ok=True)
        sock_path = paths.controller_sock_path(self.repo)

        def bind_server() -> socket.socket:
            if sock_path.exists():
                sock_path.unlink()
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(sock_path))
                server.listen(8)
                server.settimeout(1.0)
                return server
            except Exception:
                server.close()
                raise

        with self._ipc_lock:
            self._ipc_generation += 1
            generation = self._ipc_generation
            previous_server = self._server_socket
            if previous_server is not None:
                try:
                    previous_server.close()
                except OSError:
                    pass
            server = bind_server()
            self._server_socket = server

        def serve(initial_server: socket.socket) -> None:
            active_server: socket.socket | None = initial_server
            while not self.stop_requested.is_set():
                with self._ipc_lock:
                    if generation != self._ipc_generation:
                        break
                if active_server is None:
                    try:
                        active_server = bind_server()
                        with self._ipc_lock:
                            if generation != self._ipc_generation:
                                active_server.close()
                                break
                            self._server_socket = active_server
                        self.log("controller IPC socket restored")
                    except OSError as exc:
                        self.log(f"controller IPC socket restore failed; retrying: {exc}")
                        self.stop_requested.wait(1.0)
                        continue
                try:
                    conn, _addr = active_server.accept()
                except socket.timeout:
                    if not sock_path.exists():
                        self.log("controller IPC socket path disappeared; recreating it")
                        try:
                            active_server.close()
                        except OSError:
                            pass
                        active_server = None
                    continue
                except OSError as exc:
                    if not self.stop_requested.is_set():
                        self.log(f"controller IPC listener failed; recreating it: {exc}")
                    try:
                        active_server.close()
                    except OSError:
                        pass
                    active_server = None
                    continue
                with conn:
                    try:
                        response = self._handle_connection(conn)
                    except Exception as exc:
                        self.log(f"controller IPC request failed without stopping listener: {exc}")
                        response = {"success": False, "error": f"controller request failed: {exc}"}
                    self._send_ipc_response(conn, response)

            if active_server is not None:
                try:
                    active_server.close()
                except OSError:
                    pass
            with self._ipc_lock:
                if generation == self._ipc_generation:
                    self._server_socket = None

        self._server_thread = threading.Thread(target=serve, args=(server,), daemon=True)
        self._server_thread.start()

    def _send_ipc_response(self, conn: socket.socket, response: dict[str, Any]) -> bool:
        try:
            conn.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
            return True
        except (BrokenPipeError, ConnectionResetError) as exc:
            self.log(f"IPC client disconnected before receiving response: {exc}")
            return False
        except OSError as exc:
            self.log(f"IPC response send failed: {exc}")
            return False

    def _handle_connection(self, conn: socket.socket) -> dict[str, Any]:
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).split(b"\n", 1)[0]
        try:
            message = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return {"success": False, "error": f"invalid message: {exc}"}
        if not isinstance(message, dict):
            return {"success": False, "error": "message must be an object"}

        msg_type = message.get("type")
        if msg_type == "status":
            return {"success": True, "status": waiting.public_status(self.repo)}
        if msg_type == "runtime_api_update":
            payload = message.get("payload")
            self.runtime_api_payload = payload if isinstance(payload, dict) else None
            if self.runtime_api_payload is not None:
                try:
                    from station import runtime_api_config

                    runtime_api_config.apply_update(self.runtime_api_payload)
                except Exception as exc:
                    self.log(f"warning: could not apply runtime API update to controller environment: {exc}")
            return {"success": True, "message": "runtime API payload accepted for future branch workers"}
        if msg_type == "pause_branches":
            return self._set_branch_pause(paused=True)
        if msg_type == "resume_branches":
            return self._set_branch_pause(paused=False)
        if msg_type == "stop":
            self.stop_requested.set()
            if bool(message.get("force")):
                self.force_stop_requested.set()
                self._kill_recorded_branch_workers()
            else:
                self._request_graceful_shutdown()
            return {"success": True, "message": "controller stop requested"}
        return {"success": False, "error": f"unknown message type: {msg_type}"}

    def _request_graceful_shutdown(self) -> dict[str, Any]:
        response = self._set_branch_pause(paused=True)
        current = state.load_current_job(self.repo)
        job_path = Path(str(current.get("job_dir") or ""))
        if not job_path.is_dir():
            return response
        payload = state.load_job_state(job_path)
        if not payload:
            return response
        payload[state.CONTROL_KEY] = state.CONTROL_PAUSED
        payload[state.SHUTDOWN_REQUESTED_KEY] = True
        payload[state.SHUTDOWN_REQUESTED_AT_KEY] = state.utc_now()
        state.save_job_state(job_path, payload)
        state.append_job_log(job_path, "graceful shutdown requested; branch rolling paused")
        response["shutdown_requested"] = True
        return response

    def _set_branch_pause(self, *, paused: bool) -> dict[str, Any]:
        current = state.load_current_job(self.repo)
        job_path = Path(str(current.get("job_dir") or ""))
        if not job_path.is_dir():
            return {"success": False, "error": "active multistart job not found"}
        payload = state.load_job_state(job_path)
        if not payload:
            return {"success": False, "error": "active multistart job state not found"}
        control = state.CONTROL_PAUSED if paused else state.CONTROL_RUNNING
        payload[state.CONTROL_KEY] = control
        if not paused:
            payload.pop(state.SHUTDOWN_REQUESTED_KEY, None)
            payload.pop(state.SHUTDOWN_REQUESTED_AT_KEY, None)
            self._reset_incomplete_branches_for_resume(job_path, payload)
            payload = state.load_job_state(job_path)
            payload["status"] = "running"
            current["status"] = "running"
            state.save_current_job(self.repo, current)
        branches = payload.get("branches") if isinstance(payload.get("branches"), list) else []
        affected = 0
        skipped_done = 0
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            status_text = str(branch.get("status") or "")
            if status_text in {"completed", "failed", "interviewing", "waiting_quiescent"}:
                skipped_done += 1
                continue
            current_tick = _safe_int(branch.get("current_tick"))
            target_tick = _safe_int(branch.get("target_tick"))
            if not paused and current_tick is not None and target_tick is not None and current_tick >= target_tick:
                skipped_done += 1
                continue
            if status_text in {"pending", "running", "paused"}:
                affected += 1
        state.save_job_state(job_path, payload)
        state.append_job_log(job_path, f"{'paused' if paused else 'resumed'} branch rolling control; affected={affected}")
        return {
            "success": True,
            "message": "Pause requested for running/pending branches." if paused else "Resume requested for paused/pending branches.",
            "control": control,
            "affected": affected,
            "skipped": skipped_done,
            "status": waiting.public_status(self.repo),
        }

    def cleanup_ipc(self) -> None:
        sock_path = paths.controller_sock_path(self.repo)
        pid_path = paths.controller_pid_path(self.repo)
        with self._ipc_lock:
            self._ipc_generation += 1
            server = self._server_socket
            self._server_socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        server_thread = self._server_thread
        if server_thread is not None and server_thread is not threading.current_thread():
            server_thread.join(timeout=2.0)
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass

    def _kill_recorded_branch_workers(self) -> None:
        current = state.load_current_job(self.repo)
        job_path = Path(str(current.get("job_dir") or ""))
        if not job_path.is_dir():
            return
        payload = state.load_job_state(job_path)
        for branch in payload.get("branches") or []:
            if not isinstance(branch, dict):
                continue
            pid = branch.get("pid")
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            try:
                os.kill(pid_int, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                self.log(f"warning: could not stop branch worker {pid_int}: {exc}")
        _force_stop_multistart_processes(self.repo, include_controller=False, log=self.log)

    def start_init_job_if_needed(self) -> bool:
        if waiting.active_job(self.repo):
            self.log("active job already exists; controller will resume")
            return True

        from station import constants

        seeds = int(getattr(constants, "MULTISTART_INIT_SEEDS", 0) or 0)
        if seeds <= 1:
            return False
        if paths.pending_stagnation_path(self.repo).is_file():
            self.log("init multistart skipped because a stagnation request is pending")
            return False
        live = paths.live_station_data_path(self.repo)
        if not live.is_dir():
            self.log("init multistart enabled but station_data is missing; waiting page/resume only")
            return False

        config = state.read_station_config(live)
        current_tick = int(config.get("current_tick") or 0)
        if current_tick > 1:
            self.log(f"init multistart skipped because station_data current_tick={current_tick}")
            return False
        if (live / "multistart").is_dir():
            self.log("init multistart skipped because station_data/multistart already exists")
            return False

        max_parallel = int(getattr(constants, "MULTISTART_INIT_MAX_PARALLEL", 4) or 1)
        roll_ticks = int(getattr(constants, "MULTISTART_INIT_ROLL_TICKS", 40) or 0)
        request_path = paths.pending_init_path(self.repo)
        if not request_path.is_file():
            state.save_yaml_mapping(request_path, {
                "type": "init",
                "mode": "init",
                "status": "pending",
                "branch_tick": max(0, current_tick),
                "seed_count": seeds,
                "max_parallel": max_parallel,
                "roll_ticks": roll_ticks,
                "created_at": state.utc_now(),
            })
            self.log("queued init multistart request")
        self._check_pending_init_request()
        return True

    def _check_pending_init_request(self) -> bool:
        if waiting.active_job(self.repo) or paths.pending_stagnation_path(self.repo).is_file():
            return False
        request_path = paths.pending_init_path(self.repo)
        if not request_path.is_file():
            return False
        request = state.load_yaml_mapping(request_path)
        if not request:
            return False

        try:
            from station import constants

            seeds = int(getattr(constants, "MULTISTART_INIT_SEEDS", 0) or 0)
            if seeds <= 1:
                request_path.unlink(missing_ok=True)
                return False
            live = paths.live_station_data_path(self.repo)
            if not live.is_dir():
                self.log("init multistart request is pending but station_data is missing")
                return False
            config = state.read_station_config(live)
            current_tick = int(config.get("current_tick") or 0)
            if current_tick > 1 or (live / "multistart").is_dir():
                request_path.unlink(missing_ok=True)
                self.log("discarded stale init multistart request because the station is no longer fresh")
                return False

            max_parallel = int(getattr(constants, "MULTISTART_INIT_MAX_PARALLEL", 4) or 1)
            roll_ticks = int(getattr(constants, "MULTISTART_INIT_ROLL_TICKS", 40) or 0)
            self.create_job(
                "init",
                seeds,
                max_parallel,
                roll_ticks,
                branch_tick=max(0, current_tick),
            )
            request_path.unlink(missing_ok=True)
            return True
        except MultistartDiskSpaceError as exc:
            self._mark_pending_init_blocked(request_path, request, str(exc), exc.details)
            self.log(f"init multistart blocked before moving live station data; will retry: {exc}")
            return False
        except Exception as exc:
            self.log(f"failed to create init multistart job; will retry: {exc}\n{traceback.format_exc()}")
            return False

    def _mark_pending_init_blocked(
        self,
        request_path: Path,
        request: dict[str, Any],
        message: str,
        disk_details: dict[str, Any],
    ) -> None:
        updated = dict(request)
        updated["status"] = "blocked_disk_space"
        updated["message"] = message
        updated.setdefault("blocked_at", state.utc_now())
        updated["last_checked_at"] = state.utc_now()
        updated["disk_space"] = disk_details
        state.save_yaml_mapping(request_path, updated)

    def create_job(self, mode: str, seed_count: int, max_parallel: int, roll_ticks: int, *, branch_tick: int) -> Path:
        root = paths.multistart_root(self.repo)
        root.mkdir(parents=True, exist_ok=True)
        job_id = state.new_job_id()
        job_path = state.job_dir(self.repo, branch_tick, job_id)
        if job_path.exists():
            raise RuntimeError(f"job path already exists: {job_path}")
        seed_count = max(0, int(seed_count))
        max_parallel = max(1, min(int(max_parallel), seed_count))
        origin = state.origin_dir(job_path)
        live = paths.live_station_data_path(self.repo)
        if not live.is_dir():
            raise RuntimeError("station_data is required to create a multistart job")
        source_config = state.read_station_config(live)
        self._ensure_disk_space_for_branch_copy(
            live,
            copies=seed_count,
            context=f"creating {seed_count} multistart branch copy/copies",
        )
        job_path.mkdir(parents=True)
        created_at = state.utc_now()
        target_tick = int(branch_tick) + max(0, int(roll_ticks))
        branches = [
            {
                "seed": seed,
                "data_root": str(state.branch_dir(job_path, seed)),
                "status": "copy_pending",
                "copy_status": "pending",
                "log_path": str(job_path / f"branch_s{seed}.log"),
                "target_tick": target_tick,
            }
            for seed in range(1, seed_count + 1)
        ]
        payload = {
            "job_id": job_id,
            "mode": mode,
            "status": "creating",
            "station_name": source_config.get("station_name"),
            "origin_station_id": source_config.get("station_id"),
            "branch_tick": int(branch_tick),
            "seed_count": seed_count,
            "max_parallel": max_parallel,
            "roll_ticks": int(roll_ticks),
            "job_dir": str(job_path),
            "created_at": created_at,
            "branches": branches,
            "selected_seed": None,
        }
        state.save_job_state(job_path, payload)
        state.save_current_job(self.repo, {
            "job_id": job_id,
            "mode": mode,
            "status": "creating",
            "branch_tick": int(branch_tick),
            "seed_count": seed_count,
            "max_parallel": max_parallel,
            "roll_ticks": int(roll_ticks),
            "job_dir": str(job_path),
            "created_at": created_at,
        })
        state.append_job_log(job_path, f"creating {mode} multistart job with {seed_count} parallel branch copies")
        try:
            self._resume_job_creation(job_path, payload)
            return job_path
        except Exception as exc:
            self._record_job_creation_error(job_path, exc)
            raise

    def _resume_job_creation(
        self,
        job_path: Path,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload = payload or state.load_job_state(job_path)
        if not payload:
            raise RuntimeError(f"missing job state during creation: {job_path}")
        job_id = str(payload.get("job_id") or job_path.name)
        origin = state.origin_dir(job_path)
        live = paths.live_station_data_path(self.repo)
        if not origin.is_dir():
            if not live.is_dir():
                raise RuntimeError("both live station_data and origin_station_data are missing during job creation")
            shutil.move(str(live), str(origin))
            state.append_job_log(job_path, "moved live station_data into recoverable origin_station_data")
        elif live.exists():
            raise RuntimeError("both live station_data and origin_station_data exist during job creation")

        self._remove_rebuildable_runtime_state_recursive(origin)
        config = state.read_station_config(origin)
        station_id = str(config.get("station_id") or "station")
        configured_storage_base = self._configured_research_storage_base(origin)
        storage_base = self._preferred_job_storage_base(
            job_path,
            station_id=station_id,
            job_id=job_id,
            configured_base=configured_storage_base,
        )
        self._ensure_origin_storage_allocation(
            origin,
            job_path=job_path,
            job_id=job_id,
            station_id=station_id,
            storage_base=storage_base,
        )
        self._copy_all_branches_parallel(
            job_path,
            payload,
            origin,
            station_id,
            storage_base,
        )

        payload = state.load_job_state(job_path)
        payload["status"] = "created"
        payload["creation_completed_at"] = state.utc_now()
        payload.pop("creation_error", None)
        storage_manifest = state.load_yaml_mapping(
            job_path / research_storage.JOB_MANIFEST_FILENAME
        )
        if storage_manifest:
            payload["research_storage_allocations"] = {
                "enabled": True,
                "base_path": storage_manifest.get("base_path"),
                "seed_count": len(storage_manifest.get("seeds") or {}),
            }
        state.save_job_state(job_path, payload)
        current = state.load_current_job(self.repo)
        current["status"] = "running"
        current["creation_completed_at"] = payload["creation_completed_at"]
        state.save_current_job(self.repo, current)
        state.append_job_log(job_path, f"created {payload.get('mode')} multistart job with {payload.get('seed_count')} branches")
        self.log(f"created job {job_path}")

    def _record_job_creation_error(self, job_path: Path, exc: Exception) -> None:
        with self._job_state_lock:
            payload = state.load_job_state(job_path)
            if payload:
                payload["status"] = "creating"
                payload["creation_error"] = str(exc)
                payload["creation_error_at"] = state.utc_now()
                state.save_job_state(job_path, payload)
            current = state.load_current_job(self.repo)
            if current:
                current["status"] = "creating"
                current["message"] = str(exc)
                state.save_current_job(self.repo, current)
        state.append_job_log(job_path, f"branch-copy creation pass failed and will resume: {exc}")

    def _copy_all_branches_parallel(
        self,
        job_path: Path,
        payload: dict[str, Any],
        origin: Path,
        station_id: str,
        storage_base: Path | None = None,
    ) -> None:
        branches = [branch for branch in payload.get("branches") or [] if isinstance(branch, dict)]
        seeds = [int(branch.get("seed") or 0) for branch in branches if int(branch.get("seed") or 0) > 0]
        incomplete = [seed for seed in seeds if not self._branch_copy_is_complete(job_path, seed)]
        if not incomplete:
            return
        state.append_job_log(job_path, f"copying {len(incomplete)} branch tree(s) concurrently: {incomplete}")
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=len(incomplete), thread_name_prefix="multistart-copy") as pool:
            futures = {
                pool.submit(
                    self._copy_one_branch,
                    job_path,
                    payload,
                    origin,
                    station_id,
                    seed,
                    storage_base,
                ): seed
                for seed in incomplete
            }
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"s{seed}: {exc}")
        if failures:
            raise RuntimeError("parallel branch copy failed: " + "; ".join(failures))

    def _copy_one_branch(
        self,
        job_path: Path,
        payload: dict[str, Any],
        origin: Path,
        station_id: str,
        seed: int,
        storage_base: Path | None,
    ) -> None:
        branch_path = state.branch_dir(job_path, seed)
        self._update_branch_copy_record(
            job_path,
            seed,
            status="copying",
            copy_status="copying",
            copy_started_at=state.utc_now(),
            error=None,
        )
        try:
            if branch_path.exists() or branch_path.is_symlink():
                if branch_path.is_dir() and not branch_path.is_symlink():
                    research_storage.remove_tree_allow_read_only(branch_path)
                else:
                    branch_path.unlink()
            self._copy_branch_source_tree(
                origin,
                branch_path,
                job_path=job_path,
                job_id=str(payload.get("job_id") or job_path.name),
                station_id=station_id,
                seed=seed,
                storage_base=storage_base,
            )
            self._remove_rebuildable_runtime_state_recursive(branch_path)
            stale_interview = branch_path / interviews.INTERVIEW_FILENAME
            if stale_interview.exists():
                stale_interview.unlink()
            self._update_branch_copy_record(
                job_path,
                seed,
                status="pending",
                copy_status="complete",
                copy_completed_at=state.utc_now(),
                error=None,
            )
            state.append_job_log(job_path, f"completed branch copy s{seed}")
        except Exception as exc:
            self._update_branch_copy_record(
                job_path,
                seed,
                status="copy_failed",
                copy_status="failed",
                copy_failed_at=state.utc_now(),
                error=str(exc),
            )
            raise

    def _update_branch_copy_record(self, job_path: Path, seed: int, **updates: Any) -> None:
        with self._job_state_lock:
            payload = state.load_job_state(job_path)
            branches = payload.get("branches") if isinstance(payload.get("branches"), list) else []
            for branch in branches:
                if isinstance(branch, dict) and int(branch.get("seed") or 0) == int(seed):
                    for key, value in updates.items():
                        if value is None:
                            branch.pop(key, None)
                        else:
                            branch[key] = value
                    break
            payload["branches"] = branches
            payload["updated_at"] = state.utc_now()
            state.save_job_state(job_path, payload)

    def _branch_copy_is_complete(self, job_path: Path, seed: int) -> bool:
        payload = state.load_job_state(job_path)
        branches = payload.get("branches") if isinstance(payload.get("branches"), list) else []
        branch = next(
            (
                item for item in branches
                if isinstance(item, dict) and int(item.get("seed") or 0) == int(seed)
            ),
            None,
        )
        if not isinstance(branch, dict) or branch.get("copy_status") != "complete":
            return False
        branch_path = state.branch_dir(job_path, seed)
        if not branch_path.is_dir() or not (branch_path / constants.STATION_CONFIG_FILENAME).is_file():
            return False
        origin_storage = research_storage.research_storage_path(state.origin_dir(job_path))
        if not os.path.lexists(origin_storage):
            return True
        storage_path = research_storage.research_storage_path(branch_path)
        return storage_path.exists()

    def run_loop(self) -> int:
        existing_pid = find_running_controller_pid(self.repo, exclude_pid=os.getpid())
        if existing_pid is not None:
            paths.controller_pid_path(self.repo).write_text(str(existing_pid), encoding="utf-8")
            self.log(f"controller already running for repo at pid {existing_pid}; exiting duplicate")
            return 0
        self.write_pid()
        try:
            self.start_ipc()
            if not waiting.active_job(self.repo):
                self.start_init_job_if_needed()
            halted_job_key: tuple[str, str] | None = None
            while not self.stop_requested.is_set():
                current: dict[str, Any] = {}
                try:
                    current = state.load_current_job(self.repo)
                    if current:
                        status = str(current.get("status") or "").lower()
                        job_key = (
                            str(current.get("job_id") or ""),
                            str(current.get("job_dir") or ""),
                        )
                        if status != "failed" and halted_job_key == job_key:
                            halted_job_key = None
                        controller_halted_here = status == "failed" and halted_job_key == job_key
                        if status not in waiting.TERMINAL_STATUSES and not controller_halted_here:
                            self._run_or_resume_job(Path(str(current.get("job_dir"))))
                    self._check_pending_init_request()
                    self._check_pending_stagnation_request()
                except Exception as exc:
                    halted_job_key = self._record_controller_job_halt(current, exc)
                    self.log(
                        "multistart job halted; controller IPC remains available for Resume: "
                        f"{exc}\n{traceback.format_exc()}"
                    )
                time.sleep(POLL_SECONDS)
            return 0
        except Exception as exc:
            self.log(f"controller startup failed: {exc}\n{traceback.format_exc()}")
            return 1
        finally:
            self.cleanup_ipc()

    def _record_controller_job_halt(
        self,
        current: dict[str, Any],
        exc: Exception,
    ) -> tuple[str, str] | None:
        """Persist a resumable halt while leaving the controller and IPC alive."""
        try:
            latest = state.load_current_job(self.repo) or current
        except Exception as load_exc:
            self.log(f"warning: could not reload current job while halting: {load_exc}")
            latest = current
        if not latest:
            return None
        job_path_text = str(latest.get("job_dir") or "")
        job_path = Path(job_path_text) if job_path_text else None
        job_key = (str(latest.get("job_id") or ""), job_path_text)
        error_message = str(exc) or exc.__class__.__name__
        halted_at = state.utc_now()
        try:
            with self._job_state_lock:
                payload = state.load_job_state(job_path) if job_path is not None and job_path.is_dir() else {}
                if payload and str(payload.get("status") or "").lower() not in waiting.TERMINAL_STATUSES:
                    payload["status"] = "failed"
                    payload.setdefault("failure_reason", error_message)
                    payload["controller_halted_at"] = halted_at
                    state.save_job_state(job_path, payload)

                persisted_current = state.load_current_job(self.repo) or latest
                if str(persisted_current.get("status") or "").lower() not in waiting.TERMINAL_STATUSES:
                    persisted_current["status"] = "failed"
                    persisted_current["message"] = error_message
                    persisted_current["controller_halted_at"] = halted_at
                    state.save_current_job(self.repo, persisted_current)
            if job_path is not None and job_path.is_dir():
                state.append_job_log(
                    job_path,
                    f"controller kept alive after job halt; Resume remains available: {error_message}",
                )
        except Exception as persist_exc:
            self.log(f"warning: could not persist controller halt metadata: {persist_exc}")
        return job_key

    def _check_pending_stagnation_request(self) -> None:
        if waiting.active_job(self.repo):
            return
        request_path = paths.pending_stagnation_path(self.repo)
        if not request_path.is_file():
            return
        request = state.load_yaml_mapping(request_path)
        if not request:
            return
        try:
            from station import constants

            seeds = int(getattr(constants, "MULTISTART_STAGNATION_SEEDS", 0) or 0)
            if seeds <= 1:
                request_path.unlink(missing_ok=True)
                return
            max_parallel = int(getattr(constants, "MULTISTART_STAGNATION_MAX_PARALLEL", 4) or 1)
            roll_ticks = int(getattr(constants, "MULTISTART_STAGNATION_ROLL_TICKS", 40) or 0)
            branch_tick = int(request.get("branch_tick") or request.get("current_tick") or 0)
            live = paths.live_station_data_path(self.repo)
            self._ensure_disk_space_for_branch_copy(
                live,
                copies=seeds,
                context=f"creating {seeds} multistart branch copy/copies",
            )
            self._wait_for_live_station_quiescent()
            self._stop_normal_services_for_job()
            self._start_waiting_services()
            self.create_job("stagnation", seeds, max_parallel, roll_ticks, branch_tick=branch_tick)
            request_path.unlink(missing_ok=True)
        except MultistartDiskSpaceError as exc:
            self._mark_pending_stagnation_blocked(request_path, request, str(exc), exc.details)
            self.log(f"stagnation multistart blocked before stopping live station: {exc}")
        except Exception as exc:
            self.log(f"failed to create stagnation job: {exc}\n{traceback.format_exc()}")

    def _mark_pending_stagnation_blocked(
        self,
        request_path: Path,
        request: dict[str, Any],
        message: str,
        disk_details: dict[str, Any],
    ) -> None:
        updated = dict(request)
        updated["status"] = "blocked_disk_space"
        updated["message"] = message
        updated["blocked_at"] = state.utc_now()
        updated["disk_space"] = disk_details
        state.save_yaml_mapping(request_path, updated)

        config_path = paths.live_station_data_path(self.repo) / "station_config.yaml"
        config = state.load_yaml_mapping(config_path)
        if config:
            config["multistart_stagnation_pending"] = updated
            state.save_yaml_mapping(config_path, config)

    def _wait_for_live_station_quiescent(self) -> None:
        deadline = time.monotonic() + 24 * 60 * 60
        while not self.stop_requested.is_set():
            if self._live_station_quiescent_once():
                return
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for live station coder/evaluator work to finish")
            time.sleep(POLL_SECONDS)

    def _live_station_quiescent_once(self) -> bool:
        port = os.environ.get("FLASK_PORT", "5000")
        username = os.environ.get("FLASK_AUTH_USERNAME", "admin")
        password = os.environ.get("FLASK_AUTH_PASSWORD", "changeme")
        url = f"http://127.0.0.1:{port}/api/station/statistics"
        request = urllib.request.Request(url)
        import base64

        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            self.log(f"waiting for live station statistics before branch copy: {exc}")
            return False
        stats = payload.get("statistics") if isinstance(payload, dict) else {}
        try:
            running = int(stats.get(
                "drainable_running_jobs_count",
                stats.get("running_experiments_count"),
            ) or 0)
            queued = int(stats.get(
                "drainable_queued_jobs_count",
                stats.get("queued_experiments_count"),
            ) or 0)
        except (TypeError, ValueError):
            return False
        pending_fields = (
            "pending_research_evaluations",
            "pending_coder_sessions",
            "pending_external_reports",
            "pending_archive_surveys",
            "pending_archive_evaluations",
        )
        return running == 0 and queued == 0 and not any(bool(stats.get(field)) for field in pending_fields)

    def _stop_normal_services_for_job(self) -> None:
        script = self.repo / "stop.sh"
        if not script.is_file():
            return
        env = os.environ.copy()
        env["STATION_MULTISTART_SKIP_CONTROLLER_STOP"] = "1"
        subprocess.run([str(script), "--force"], cwd=self.repo, env=env, check=False)

    def _start_waiting_services(self) -> None:
        script = self.repo / "start.sh"
        if not script.is_file():
            return
        env = os.environ.copy()
        env["STATION_MULTISTART_WAIT_ONLY"] = "1"
        env["STATION_MULTISTART_SKIP_CONTROLLER_START"] = "1"
        subprocess.run([str(script)], cwd=self.repo, env=env, check=False)

    def _run_or_resume_job(self, job_path: Path) -> None:
        if not job_path.is_dir():
            raise RuntimeError(f"current job directory missing: {job_path}")
        payload = state.load_job_state(job_path)
        if not payload:
            raise RuntimeError(f"missing job state: {job_path}")
        if str(payload.get("status")) == "creating":
            try:
                self._resume_job_creation(job_path, payload)
            except Exception as exc:
                self._record_job_creation_error(job_path, exc)
                self.log(f"multistart branch copying will retry: {exc}")
                return
            payload = state.load_job_state(job_path)
        self._reconcile_job_storage_allocations(job_path, payload)
        payload = state.load_job_state(job_path)
        if str(payload.get("status")) == "complete":
            return
        if str(payload.get("status")) == "finalizing":
            selected_seed = payload.get("selected_seed")
            if selected_seed is None:
                raise RuntimeError("job is finalizing but selected_seed is missing")
            self.finalize_job(job_path, payload, int(selected_seed))
            return

        self._reset_incomplete_branches_for_resume(job_path, payload)
        payload = state.load_job_state(job_path)
        payload["status"] = "running"
        state.save_job_state(job_path, payload)
        current = state.load_current_job(self.repo)
        current["status"] = "running"
        state.save_current_job(self.repo, current)

        branches = [branch for branch in payload.get("branches") or [] if isinstance(branch, dict)]
        max_parallel = max(1, int(payload.get("max_parallel") or 1))
        processes: dict[int, subprocess.Popen] = {}
        while not self.stop_requested.is_set():
            payload = state.load_job_state(job_path)
            branches = [branch for branch in payload.get("branches") or [] if isinstance(branch, dict)]
            for seed, process in list(processes.items()):
                rc = process.poll()
                if rc is not None:
                    processes.pop(seed, None)

            self._reset_dead_active_branches(job_path, payload)
            payload = state.load_job_state(job_path)
            branches = [branch for branch in payload.get("branches") or [] if isinstance(branch, dict)]
            completed = [b for b in branches if b.get("status") == "completed"]
            failed = [b for b in branches if b.get("status") == "failed"]
            active_external = [
                branch for branch in branches
                if branch.get("status") in {"running", "paused", "waiting_quiescent", "interviewing"}
                and self._branch_pid_alive(branch)
                and int(branch.get("seed") or 0) not in processes
            ]
            if failed:
                if processes or active_external:
                    time.sleep(POLL_SECONDS)
                    continue
                break
            if len(completed) + len(failed) >= len(branches):
                break
            if state.job_paused(payload):
                time.sleep(POLL_SECONDS)
                continue

            running_seeds = set(processes)
            available_slots = max(0, max_parallel - len(processes) - len(active_external))
            for branch in branches:
                if available_slots <= 0:
                    break
                if branch.get("status") != "pending":
                    continue
                seed = int(branch.get("seed"))
                if seed in running_seeds:
                    continue
                processes[seed] = self._launch_branch(job_path, payload, seed)
                available_slots -= 1
            time.sleep(POLL_SECONDS)

        if self.stop_requested.is_set():
            if self.force_stop_requested.is_set():
                self._terminate_branch_processes(processes)
                return
            for process in processes.values():
                process.wait()
            self._wait_for_recorded_active_branches(job_path)
            return

        for process in processes.values():
            process.wait()

        payload = state.load_job_state(job_path)
        branches = [branch for branch in payload.get("branches") or [] if isinstance(branch, dict)]
        failed = [b for b in branches if b.get("status") == "failed"]
        if failed:
            payload["status"] = "failed"
            payload["failure_reason"] = f"{len(failed)} branch worker(s) failed"
            state.save_job_state(job_path, payload)
            current = state.load_current_job(self.repo)
            current["status"] = "failed"
            state.save_current_job(self.repo, current)
            raise RuntimeError(payload["failure_reason"])

        self._verify_interviews_before_selection(job_path, payload, branches)

        payload["status"] = "selecting"
        state.save_job_state(job_path, payload)
        current = state.load_current_job(self.repo)
        if current:
            current["status"] = "selecting"
            state.save_current_job(self.repo, current)
        try:
            selected_seed = admin.run_selection(payload, job_path)
        except admin.AdminSelectionAttemptsExhausted as exc:
            payload = state.load_job_state(job_path)
            payload["status"] = "failed"
            payload["failure_reason"] = str(exc)
            state.save_job_state(job_path, payload)
            current = state.load_current_job(self.repo)
            if current:
                current["status"] = "failed"
                current["message"] = str(exc)
                state.save_current_job(self.repo, current)
            state.append_job_log(job_path, f"admin selection halted after retry exhaustion: {exc}")
            raise
        payload["selected_seed"] = selected_seed
        payload["status"] = "finalizing"
        state.save_job_state(job_path, payload)
        current = state.load_current_job(self.repo)
        if current:
            current["status"] = "finalizing"
            state.save_current_job(self.repo, current)
        self.finalize_job(job_path, payload, selected_seed)

    def _terminate_branch_processes(self, processes: dict[int, subprocess.Popen]) -> set[int]:
        terminated: set[int] = set()
        for seed, process in processes.items():
            if process.poll() is not None:
                continue
            try:
                process.terminate()
                terminated.add(seed)
            except OSError:
                pass
        deadline = time.monotonic() + 30.0
        for seed, process in processes.items():
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.2)
            if process.poll() is None:
                try:
                    process.kill()
                    terminated.add(seed)
                except OSError:
                    pass
        return terminated

    def _mark_terminated_branches_pending(self, job_path: Path, seeds: set[int]) -> None:
        payload = state.load_job_state(job_path)
        branches = payload.get("branches")
        if not isinstance(branches, list):
            return
        changed = False
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            seed = int(branch.get("seed") or 0)
            if seed not in seeds:
                continue
            if branch.get("status") not in {"running", "paused", "waiting_quiescent", "interviewing"}:
                continue
            branch["previous_status"] = branch.get("status")
            branch["status"] = "pending"
            branch["pid"] = None
            branch["halt_reason"] = "halted because another branch failed"
            changed = True
        if changed:
            state.append_job_log(job_path, f"marked halted branch(es) pending for resume: {sorted(seeds)}")
            state.save_job_state(job_path, payload)

    def _wait_for_recorded_active_branches(self, job_path: Path) -> None:
        while True:
            payload = state.load_job_state(job_path)
            branches = payload.get("branches")
            if not isinstance(branches, list):
                return
            if bool(payload.get(state.SHUTDOWN_REQUESTED_KEY)):
                self._stop_quiescent_paused_branch_workers(job_path, payload, branches)
                payload = state.load_job_state(job_path)
                branches = payload.get("branches")
                if not isinstance(branches, list):
                    return
            active = [
                branch for branch in branches
                if isinstance(branch, dict)
                and branch.get("status") in {"running", "paused", "waiting_quiescent", "interviewing"}
                and self._branch_pid_alive(branch)
            ]
            if not active:
                return
            state.append_job_log(
                job_path,
                "graceful stop waiting for active branch worker(s): "
                + ", ".join(f"s{branch.get('seed')} pid={branch.get('pid')}" for branch in active),
            )
            time.sleep(POLL_SECONDS)

    def _stop_quiescent_paused_branch_workers(
        self,
        job_path: Path,
        payload: dict[str, Any],
        branches: list[Any],
    ) -> None:
        changed = False
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            if branch.get("status") not in {"paused", "waiting_quiescent"}:
                continue
            pid = _safe_int(branch.get("pid"))
            if pid is None or not self._branch_pid_alive(branch):
                if branch.get("pid") is not None:
                    branch["pid"] = None
                    changed = True
                continue
            if self._branch_has_background_work(job_path, branch):
                continue
            self.log(f"stopping quiescent paused branch worker s{branch.get('seed')} pid={pid}")
            self._terminate_pid(pid)
            branch["pid"] = None
            branch["status"] = "paused"
            branch["shutdown_requested"] = True
            branch["shutdown_stopped_at"] = time.time()
            branch["pause_reason"] = "graceful multistart shutdown"
            changed = True
        if changed:
            payload["branches"] = branches
            state.save_job_state(job_path, payload)

    def _branch_has_background_work(self, job_path: Path, branch: dict[str, Any]) -> bool:
        seed = _safe_int(branch.get("seed")) or 0
        data_root = Path(str(branch.get("data_root") or state.branch_dir(job_path, seed)))
        evals = waiting._evaluation_summary(data_root)
        if any(
            int(evals.get(key) or 0) > 0
            for key in ("active_coders", "running", "queued")
        ):
            return True
        return self._branch_has_background_work_from_yaml(data_root)

    def _branch_has_background_work_from_yaml(self, data_root: Path) -> bool:
        evaluations_dir = data_root / "rooms" / "research" / "evaluations"
        if not evaluations_dir.is_dir():
            return False
        active_statuses = {
            "queued",
            "running",
            "coder_running",
            "attempt_queued",
            "attempt_running",
            "pending_resume",
            "resuming",
        }
        try:
            paths_iter = list(evaluations_dir.glob("*.yaml"))
        except OSError:
            return True
        for eval_path in paths_iter:
            try:
                record = file_io_utils.load_yaml(str(eval_path))
            except Exception:
                return True
            if not isinstance(record, dict):
                continue
            if str(record.get("status") or record.get("top_level_status") or "").lower() in active_statuses:
                return True
            coder = record.get("coder")
            if isinstance(coder, dict):
                coder_status = str(coder.get("status") or "").lower()
                if bool(coder.get("active")):
                    return True
                if coder_status in active_statuses:
                    return True
                pid = _safe_int(coder.get("pid"))
                if pid is not None and _pid_exists(pid):
                    return True
            current_attempt = record.get("current_attempt")
            if isinstance(current_attempt, dict):
                if str(current_attempt.get("status") or "").lower() in active_statuses:
                    return True
            attempts = record.get("attempts")
            if isinstance(attempts, list):
                for attempt in attempts:
                    if isinstance(attempt, dict) and str(attempt.get("status") or "").lower() in active_statuses:
                        return True
        return False

    def _terminate_pid(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            self.log(f"warning: could not TERM pid {pid}: {exc}")
            return
        deadline = time.monotonic() + FORCE_STOP_TERM_SECONDS
        while _pid_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if not _pid_exists(pid):
            return
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as exc:
            self.log(f"warning: could not KILL pid {pid}: {exc}")

    def _verify_interviews_before_selection(
        self,
        job_path: Path,
        payload: dict[str, Any],
        branches: list[dict[str, Any]],
    ) -> None:
        missing: list[int] = []
        for branch in branches:
            if branch.get("status") != "completed":
                continue
            seed = int(branch.get("seed") or 0)
            interview_path = state.branch_dir(job_path, seed) / interviews.INTERVIEW_FILENAME
            if not interview_path.is_file() or interview_path.stat().st_size <= 0:
                branch["status"] = "failed"
                branch["error"] = f"{interviews.INTERVIEW_FILENAME} missing before admin selection"
                branch["pid"] = None
                missing.append(seed)
        if not missing:
            return

        payload["branches"] = branches
        payload["status"] = "failed"
        payload["failure_reason"] = "missing branch interview files before admin selection"
        state.save_job_state(job_path, payload)
        current = state.load_current_job(self.repo)
        if current:
            current["status"] = "failed"
            state.save_current_job(self.repo, current)
        raise RuntimeError(f"missing {interviews.INTERVIEW_FILENAME} for seed(s): {missing}")

    def _reset_incomplete_branches_for_resume(self, job_path: Path, payload: dict[str, Any]) -> None:
        changed = False
        if state.job_control(payload) != state.CONTROL_PAUSED:
            for key in (state.SHUTDOWN_REQUESTED_KEY, state.SHUTDOWN_REQUESTED_AT_KEY):
                if key in payload:
                    payload.pop(key, None)
                    changed = True
        branches = payload.get("branches")
        if not isinstance(branches, list):
            if changed:
                state.append_job_log(job_path, "cleared stale multistart shutdown flag for resume")
                state.save_job_state(job_path, payload)
            return
        resumable_statuses = {"running", "paused", "waiting_quiescent", "interviewing", "failed", "pending"}
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            status = branch.get("status")
            if status not in resumable_statuses:
                continue
            pid = branch.get("pid")
            pid_alive = False
            try:
                if pid is not None:
                    os.kill(int(pid), 0)
                    pid_alive = True
            except Exception:
                pid_alive = False
            if pid_alive and status != "failed":
                continue
            attempts = int(branch.get("attempts") or 0)
            seed = int(branch.get("seed") or 0)
            if seed > 0:
                if branch.get("reset_data_on_resume"):
                    self._restore_branch_data_from_origin(job_path, seed)
                    for key in (
                        "current_tick",
                        "start_tick",
                        "interview_base_tick",
                        "top_evaluation_id",
                        "top_score",
                        "top_sort_key",
                        "init_agents_spawned",
                    ):
                        branch.pop(key, None)
                else:
                    self._remove_rebuildable_indexes_recursive(state.branch_dir(job_path, seed))
            should_count_attempt = status != "pending" or pid is not None or branch.get("previous_status") is not None
            if should_count_attempt:
                branch["attempts"] = attempts + 1
            branch["previous_status"] = branch.get("status")
            branch["status"] = "pending"
            branch["pid"] = None
            for key in (
                "error",
                "traceback",
                "failed_at",
                "started_at",
                "launched_at",
                "completed_at",
                "reset_data_on_resume",
                "shutdown_requested",
                "shutdown_stopped_at",
                "pause_requested",
                "pause_reason",
                "paused_at",
                "resumed_at",
            ):
                branch.pop(key, None)
            changed = True
        if changed:
            state.append_job_log(job_path, "reset incomplete branch workers for resume")
            state.save_job_state(job_path, payload)

    def _restore_branch_data_from_origin(self, job_path: Path, seed: int) -> None:
        origin = state.origin_dir(job_path)
        branch_path = state.branch_dir(job_path, seed)
        if not origin.is_dir():
            self._remove_rebuildable_indexes_recursive(branch_path)
            state.append_job_log(
                job_path,
                f"warning: could not reset branch s{seed} from origin; origin_station_data missing",
            )
            return
        if branch_path.exists():
            research_storage.remove_tree_allow_read_only(branch_path)
        payload = state.load_job_state(job_path)
        config = state.read_station_config(origin)
        station_id = str(payload.get("origin_station_id") or config.get("station_id") or "station")
        job_id = str(payload.get("job_id") or job_path.name)
        storage_base = self._preferred_job_storage_base(
            job_path,
            station_id=station_id,
            job_id=job_id,
            configured_base=self._configured_research_storage_base(origin),
        )
        self._copy_branch_source_tree(
            origin,
            branch_path,
            job_path=job_path,
            job_id=job_id,
            station_id=station_id,
            seed=seed,
            storage_base=storage_base,
        )
        self._remove_rebuildable_runtime_state_recursive(branch_path)
        stale_interview = branch_path / interviews.INTERVIEW_FILENAME
        if stale_interview.exists():
            stale_interview.unlink()
        state.append_job_log(job_path, f"reset branch s{seed} data from origin_station_data")

    def _branch_pid_alive(self, branch: dict[str, Any]) -> bool:
        pid = branch.get("pid")
        try:
            if pid is None:
                return False
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    def _reset_dead_active_branches(self, job_path: Path, payload: dict[str, Any]) -> None:
        changed = False
        branches = payload.get("branches")
        if not isinstance(branches, list):
            return
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            if branch.get("status") not in {"running", "paused", "waiting_quiescent", "interviewing"}:
                continue
            if self._branch_pid_alive(branch):
                continue
            attempts = int(branch.get("attempts") or 0)
            branch["attempts"] = attempts + 1
            branch["previous_status"] = branch.get("status")
            branch["status"] = "pending"
            branch["pid"] = None
            changed = True
        if changed:
            state.append_job_log(job_path, "reset dead active branch workers")
            state.save_job_state(job_path, payload)

    def _update_branch_record(self, job_path: Path, seed: int, **updates: Any) -> None:
        payload = state.load_job_state(job_path)
        branches = payload.get("branches")
        if not isinstance(branches, list):
            branches = []
        for branch in branches:
            if isinstance(branch, dict) and int(branch.get("seed") or 0) == seed:
                branch.update(updates)
                break
        payload["branches"] = branches
        state.save_job_state(job_path, payload)

    def _launch_branch(self, job_path: Path, payload: dict[str, Any], seed: int) -> subprocess.Popen:
        python = sys.executable
        log_path = job_path / f"branch_s{seed}.log"
        command = [
            python,
            "-m",
            "station.multistart.branch_worker",
            "--mode",
            str(payload.get("mode")),
            "--seed",
            str(seed),
            "--data-root",
            str(state.branch_dir(job_path, seed)),
            "--job-dir",
            str(job_path),
            "--roll-ticks",
            str(payload.get("roll_ticks") or 0),
            "--branch-tick",
            str(payload.get("branch_tick") or 0),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["STATION_REPO_ROOT"] = str(self.repo)
        env["STATION_BASE_DATA_PATH"] = str(state.branch_dir(job_path, seed))
        env["STATION_MULTISTART_BRANCH"] = "1"
        env["STATION_MULTISTART_SEED"] = str(seed)
        env["STATION_DISABLE_BACKUPS"] = "1"
        env.setdefault("AUTO_START", "False")
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=self.repo,
                env=env,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
            )
        self._update_branch_record(job_path, seed, status="running", pid=process.pid, launched_at=time.time())
        state.append_job_log(job_path, f"launched branch s{seed} pid={process.pid}")
        return process

    def finalize_job(self, job_path: Path, payload: dict[str, Any], selected_seed: int) -> None:
        live = paths.live_station_data_path(self.repo)
        archive_name = job_path.name
        if live.exists():
            archive_root = live / "multistart" / archive_name
            if archive_root.is_dir():
                self._complete_installed_job(job_path, payload, archive_name, live)
                return
            if self._can_resume_after_selected_branch_install(job_path, selected_seed):
                self._install_remaining_job_archive(job_path, payload, selected_seed, archive_root)
                self._complete_installed_job(job_path, payload, archive_name, live)
                return
            if self._should_discard_unexpected_live_station_data(live):
                self._remove_unexpected_live_station_data(live)
                state.append_job_log(job_path, "deleted unexpected live station_data with no config or tick <= 1")
            else:
                raise RuntimeError(f"cannot finalize while live station_data exists: {live}")
        selected_dir = state.branch_dir(job_path, selected_seed)
        if not selected_dir.is_dir():
            raise RuntimeError(f"selected branch missing: {selected_dir}")

        audit_tmp = job_path / "_audit_selected"
        if audit_tmp.exists():
            research_storage.remove_tree_allow_read_only(audit_tmp)
        selected_interview = selected_dir / interviews.INTERVIEW_FILENAME
        self._promote_selected_research_storage(job_path, selected_dir, selected_seed)
        self._record_selected_branch_install(job_path, selected_dir, payload, selected_seed)
        if selected_interview.exists():
            selected_interview.unlink()
        self._rewrite_selected_branch_root(selected_dir, live)
        self._remove_rebuildable_runtime_state_recursive(selected_dir)
        shutil.move(str(selected_dir), str(live))
        self._remove_rebuildable_runtime_state_recursive(live)

        archive_target = live / "multistart" / archive_name
        self._install_remaining_job_archive(job_path, payload, selected_seed, archive_target)
        self._complete_installed_job(job_path, payload, archive_name, live)

    def _promote_selected_research_storage(
        self,
        job_path: Path,
        selected_dir: Path,
        selected_seed: int,
    ) -> None:
        manifest_path = job_path / research_storage.JOB_MANIFEST_FILENAME
        manifest = state.load_yaml_mapping(manifest_path)
        seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
        seed_info = seeds.get(str(selected_seed)) if isinstance(seeds, dict) else None
        if not isinstance(seed_info, dict):
            return

        storage_path = research_storage.research_storage_path(selected_dir)
        target_raw = str(seed_info.get("target") or "").strip()
        base_raw = str(manifest.get("base_path") or "").strip()
        target = Path(target_raw) if target_raw else Path("/__missing_research_storage_target__")
        base_path = Path(base_raw) if base_raw else None
        if storage_path.exists() and not storage_path.is_symlink():
            seed_info["installed_as_local"] = True
            seed_info["promoted_at"] = state.utc_now()
            manifest["selected_seed"] = int(selected_seed)
            manifest["seeds"] = seeds
            state.save_yaml_mapping(manifest_path, manifest)
            return
        if not target.is_dir():
            raise RuntimeError(f"selected branch Research storage allocation is missing: {target}")
        if base_path is None or not research_storage.path_is_within(target, base_path):
            raise RuntimeError(f"selected branch Research storage is outside its configured base: {target}")
        if storage_path.resolve() != target.resolve():
            raise RuntimeError(
                f"selected branch Research storage link does not match its allocation: {storage_path}"
            )
        station_id = str(manifest.get("station_id") or "station")
        job_id = str(manifest.get("job_id") or job_path.name)
        marker = research_storage.read_allocation_marker(target)
        already_promoted = (
            str(marker.get("kind") or "") == "live"
            and str(marker.get("station_id") or "") == station_id
            and str(marker.get("promoted_from_job_id") or "") == job_id
            and str(marker.get("promoted_from_seed") or "") == str(selected_seed)
        )
        if not already_promoted and not research_storage.marker_matches(
            target,
            station_id=station_id,
            job_id=job_id,
            seed=selected_seed,
            kinds={"multistart_seed"},
        ):
            raise RuntimeError(f"selected branch Research storage marker mismatch: {target}")

        if not already_promoted:
            marker.update({
                "kind": "live",
                "station_id": station_id,
                "promoted_from_job_id": job_id,
                "promoted_from_seed": int(selected_seed),
                "promoted_at": state.utc_now(),
            })
            marker.pop("job_id", None)
            marker.pop("seed", None)
            research_storage.write_allocation_marker(target, marker)

        seed_info["promoted_as_live"] = True
        seed_info["promoted_at"] = state.utc_now()
        manifest["selected_seed"] = int(selected_seed)
        manifest["seeds"] = seeds
        state.save_yaml_mapping(manifest_path, manifest)
        state.append_job_log(job_path, f"promoted Research storage allocation for selected branch s{selected_seed}")

    def _can_resume_after_selected_branch_install(self, job_path: Path, selected_seed: int) -> bool:
        if not job_path.is_dir():
            return False
        if state.branch_dir(job_path, selected_seed).exists():
            return False
        return (job_path / f"station_data_s{selected_seed}.installed.yaml").is_file()

    def _record_selected_branch_install(
        self,
        job_path: Path,
        selected_dir: Path,
        payload: dict[str, Any],
        selected_seed: int,
    ) -> None:
        branch_name = f"station_data_s{selected_seed}"
        interviews_dir = job_path / "interviews"
        interviews_dir.mkdir(parents=True, exist_ok=True)
        interview_source = selected_dir / interviews.INTERVIEW_FILENAME
        interview_copy = interviews_dir / f"{branch_name}.{interviews.INTERVIEW_FILENAME}"
        if interview_source.exists():
            shutil.copy2(interview_source, interview_copy)

        config = state.read_station_config(selected_dir)
        metadata = {
            "seed": selected_seed,
            "branch_dir": branch_name,
            "selected": True,
            "installed_as_live_station_data": True,
            "live_data_root": "station_data",
            "station_name": config.get("station_name"),
            "station_id": config.get("station_id"),
            "current_tick": config.get("current_tick"),
            "interview": str(interview_copy.relative_to(job_path)) if interview_copy.exists() else None,
            "recorded_at": state.utc_now(),
        }
        state.save_yaml_mapping(job_path / f"{branch_name}.installed.yaml", metadata)

        payload["selected_branch_install"] = metadata
        branches = payload.get("branches")
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict) and _safe_int(branch.get("seed")) == selected_seed:
                    branch["installed_as_live_station_data"] = True
                    branch["installed_live_data_root"] = "station_data"
                    break
        state.save_job_state(job_path, payload)

    def _install_remaining_job_archive(
        self,
        job_path: Path,
        payload: dict[str, Any],
        selected_seed: int,
        archive_target: Path,
    ) -> None:
        if archive_target.exists() or archive_target.is_symlink():
            raise RuntimeError(f"multistart archive target already exists: {archive_target}")
        selected_link = job_path / state.ADMIN_DIR_NAME / f"station_data_s{selected_seed}"
        if os.path.lexists(selected_link):
            if selected_link.is_dir() and not selected_link.is_symlink():
                research_storage.remove_tree_allow_read_only(selected_link)
            else:
                selected_link.unlink()
        self._remove_rebuildable_runtime_state_recursive(job_path)
        self._repair_archived_admin_branch_links(job_path)
        state.save_job_state(job_path, payload)
        archive_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(job_path), str(archive_target))

    def _rewrite_selected_branch_root(self, selected_dir: Path, live: Path) -> None:
        old_root = str(selected_dir.resolve())
        new_root = str(live.resolve())
        if old_root == new_root:
            return

        text_suffixes = {
            ".cfg",
            ".conf",
            ".csv",
            ".json",
            ".jsonl",
            ".log",
            ".md",
            ".py",
            ".sh",
            ".txt",
            ".yaml",
            ".yamll",
            ".yml",
        }
        skipped_roots = {
            selected_dir / "multistart",
        }
        rewritten = 0
        for path in selected_dir.rglob("*"):
            if path.is_dir() or path.is_symlink() or path.suffix.lower() not in text_suffixes:
                continue
            if any(root in path.parents for root in skipped_roots):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if old_root not in content:
                continue
            file_io_utils.save_text(content.replace(old_root, new_root), str(path))
            rewritten += 1
        if rewritten:
            self.log(f"rewrote selected branch root path in {rewritten} file(s)")

    def _remove_rebuildable_runtime_state_recursive(self, root: Path) -> None:
        self._remove_rebuildable_indexes_recursive(root)
        self._remove_transient_runtime_dirs_recursive(root)

    def _remove_rebuildable_indexes_recursive(self, root: Path) -> None:
        if not root.exists():
            return
        index_dirs: set[Path] = set()
        for path in root.rglob(f"{constants.STATION_INDEX_DB_FILENAME}*"):
            if path.parent.name == constants.STATION_INDEX_DIR_NAME:
                index_dirs.add(path.parent)
        direct_index_dir = root / constants.STATION_INDEX_DIR_NAME
        if direct_index_dir.is_dir():
            index_dirs.add(direct_index_dir)
        for index_dir in sorted(index_dirs, key=lambda item: len(item.parts), reverse=True):
            try:
                research_storage.remove_tree_allow_read_only(index_dir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(f"failed to remove rebuildable station index directory {index_dir}") from exc

    def _remove_transient_runtime_dirs_recursive(self, root: Path) -> None:
        if not root.exists():
            return
        transient_dirs: set[Path] = set()
        for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
            current = Path(dirpath)
            kept_dirnames = []
            for dirname in dirnames:
                child = current / dirname
                if child.is_symlink():
                    kept_dirnames.append(dirname)
                    continue
                relative = child.relative_to(root).as_posix()
                is_station_sync = (
                    child.name == constants.PARALLEL_TICK_STATE_DIR_NAME
                    and (child.parent / constants.STATION_CONFIG_FILENAME).is_file()
                )
                is_research_tmp = (
                    relative.endswith("rooms/research/storage/tmp")
                    or relative.endswith("rooms/research/storage/shared/tmp")
                )
                if is_station_sync or is_research_tmp:
                    transient_dirs.add(child)
                    continue
                kept_dirnames.append(dirname)
            dirnames[:] = kept_dirnames

        for transient_dir in sorted(transient_dirs, key=lambda item: len(item.parts), reverse=True):
            try:
                research_storage.remove_tree_allow_read_only(transient_dir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(f"failed to remove transient runtime directory {transient_dir}") from exc

    def _repair_archived_admin_branch_links(self, archive_root: Path) -> None:
        admin_dir = archive_root / state.ADMIN_DIR_NAME
        if not admin_dir.is_dir():
            return
        repaired = 0
        for branch_dir in sorted(archive_root.glob("station_data_s*")):
            if not branch_dir.is_dir() or branch_dir.is_symlink():
                continue
            link_path = admin_dir / branch_dir.name
            desired_target = os.path.relpath(branch_dir, admin_dir)
            if os.path.lexists(link_path):
                if link_path.is_symlink():
                    try:
                        if os.readlink(link_path) == desired_target:
                            continue
                    except OSError:
                        pass
                    link_path.unlink()
                elif link_path.is_dir():
                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
            link_path.symlink_to(desired_target, target_is_directory=True)
            repaired += 1
        if repaired:
            self.log(f"repaired archived admin branch link(s): {repaired}")

    def _should_discard_unexpected_live_station_data(self, live: Path) -> bool:
        if not live.is_dir():
            return False
        config_path = live / "station_config.yaml"
        if not config_path.is_file():
            return True
        config = state.read_station_config(live)
        try:
            current_tick = int(config.get("current_tick") or 0)
        except (TypeError, ValueError):
            current_tick = 0
        return current_tick <= 1

    def _remove_unexpected_live_station_data(self, live: Path) -> None:
        if live.is_dir() and not live.is_symlink():
            research_storage.remove_tree_allow_read_only(live)
        else:
            live.unlink()

    def _complete_installed_job(self, job_path: Path, payload: dict[str, Any], archive_name: str, live: Path) -> None:
        archive_root = live / "multistart" / archive_name
        self._repair_archived_admin_branch_links(archive_root)
        archive_state_path = archive_root / "state.yaml"
        archived_payload = state.load_yaml_mapping(archive_state_path)
        if archived_payload:
            payload.update(archived_payload)
        steps = payload.setdefault("finalization_steps", {})
        if not isinstance(steps, dict):
            steps = {}
            payload["finalization_steps"] = steps

        if not steps.get("guidance_posted"):
            self._post_guidance_message(live, archive_name)
            steps["guidance_posted"] = True
            state.save_yaml_mapping(archive_state_path, payload)

        if not steps.get("origin_station_data_removed_before_backup"):
            origin_result = self._remove_archived_origin_station_data_before_backup(archive_root)
            payload["origin_station_data_removal"] = origin_result
            steps["origin_station_data_removed_before_backup"] = True
            state.save_yaml_mapping(archive_state_path, payload)

        if not steps.get("manual_backup_created"):
            backup_ref = self._create_manual_backup(archive_name)
            steps["manual_backup_created"] = True
            if isinstance(backup_ref, dict):
                payload["finalization_backup"] = backup_ref
            state.save_yaml_mapping(archive_state_path, payload)
        else:
            backup_ref = payload.get("finalization_backup") or self._infer_finalization_backup_reference(live, archive_name)
            if isinstance(backup_ref, dict) and not payload.get("finalization_backup"):
                payload["finalization_backup"] = backup_ref

        if not steps.get("archived_branch_data_pruned"):
            prune_result = self._prune_archived_branch_data_after_backup(archive_root, payload, backup_ref)
            payload["archived_branch_data_prune"] = prune_result
            if prune_result.get("success"):
                steps["archived_branch_data_pruned"] = True
            else:
                steps["archived_branch_data_prune_attempted"] = True
            state.save_yaml_mapping(archive_state_path, payload)

        if steps.get("archived_branch_data_pruned") and not steps.get("obsolete_research_storage_removed"):
            cleanup_result = self._remove_job_seed_storage_allocations(
                archive_root,
                preserve_selected=True,
                include_origin=True,
            )
            payload["research_storage_cleanup"] = cleanup_result
            if cleanup_result.get("success"):
                steps["obsolete_research_storage_removed"] = True
            else:
                steps["research_storage_cleanup_attempted"] = True
            state.save_yaml_mapping(archive_state_path, payload)

        payload["status"] = "complete"
        payload["completed_at"] = payload.get("completed_at") or state.utc_now()
        state.save_yaml_mapping(archive_state_path, payload)
        state.clear_current_job(self.repo)
        if job_path.exists() and job_path.resolve() != archive_root.resolve():
            try:
                research_storage.remove_tree_allow_read_only(job_path)
            except OSError as exc:
                self.log(f"failed to remove temporary multistart job folder {job_path}: {exc}")
        self._restart_normal_station()
        self.log(f"finalized job {archive_name} with selected branch s{payload.get('selected_seed')}")

    def _remove_archived_origin_station_data_before_backup(self, archive_root: Path) -> dict[str, Any]:
        origin = archive_root / state.ORIGIN_DIR_NAME
        admin_origin_link = archive_root / state.ADMIN_DIR_NAME / state.ORIGIN_DIR_NAME
        removed = False
        fallback_used = False
        errors: list[str] = []

        if os.path.lexists(admin_origin_link):
            try:
                if admin_origin_link.is_dir() and not admin_origin_link.is_symlink():
                    research_storage.remove_tree_allow_read_only(admin_origin_link)
                else:
                    admin_origin_link.unlink()
            except OSError as exc:
                errors.append(f"admin origin link: {exc}")

        if origin.exists():
            try:
                research_storage.remove_tree_allow_read_only(origin)
                removed = True
            except OSError as exc:
                errors.append(str(exc))
                fallback_used = True
                result = subprocess.run(["sudo", "rm", "-rf", str(origin)], cwd=self.repo, check=False)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"failed to remove archived origin_station_data before final backup: sudo rm exited {result.returncode}"
                    ) from exc
                removed = True
        elif os.path.lexists(origin):
            try:
                origin.unlink()
                removed = True
            except OSError as exc:
                errors.append(str(exc))
                fallback_used = True
                result = subprocess.run(["sudo", "rm", "-rf", str(origin)], cwd=self.repo, check=False)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"failed to remove archived origin_station_data symlink before final backup: sudo rm exited {result.returncode}"
                    ) from exc
                removed = True

        if os.path.lexists(origin):
            raise RuntimeError(f"origin_station_data still exists after removal attempt: {origin}")

        if removed:
            self.log(f"removed archived origin_station_data before final backup from {archive_root.name}")
        return {
            "removed": removed,
            "fallback_sudo_rm_used": fallback_used,
            "removed_at": state.utc_now() if removed else None,
            "path": f"multistart/{archive_root.name}/{state.ORIGIN_DIR_NAME}",
            "admin_link_removed": not os.path.lexists(admin_origin_link),
            "errors": errors,
        }

    def _create_manual_backup(self, archive_name: str | None = None) -> dict[str, Any]:
        env = os.environ.copy()
        env.pop("STATION_BASE_DATA_PATH", None)
        result_path = paths.multistart_root(self.repo) / f".manual_backup_{os.getpid()}_{int(time.time())}.json"
        snapshot_suffix = f"multistart_{archive_name}" if archive_name else None
        code = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from station.station import Station\n"
            "from station.backup_utils import create_backup\n"
            "s = Station()\n"
            "tick = s._get_current_tick()\n"
            "snapshot_suffix = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None\n"
            "manifest_path = create_backup(tick, 'multistart' if snapshot_suffix else 'manual', s, snapshot_suffix=snapshot_suffix)\n"
            "Path(sys.argv[1]).write_text(json.dumps({"
            "'station_id': s.station_id, "
            "'tick': tick, "
            "'manifest_path': manifest_path"
            "}), encoding='utf-8')\n"
        )
        command = [
            sys.executable,
            "-c",
            code,
            str(result_path),
            snapshot_suffix or "",
        ]
        try:
            result = subprocess.run(command, cwd=self.repo, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError("manual backup failed after multistart finalization")
            data = json.loads(result_path.read_text(encoding="utf-8"))
            manifest_path = data.get("manifest_path")
            if manifest_path:
                manifest = Path(str(manifest_path))
                if not manifest.is_absolute():
                    manifest = self.repo / manifest
                data["manifest_path"] = str(manifest)
            return data
        finally:
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass

    def _infer_finalization_backup_reference(self, live: Path, archive_name: str | None = None) -> dict[str, Any] | None:
        config = state.read_station_config(live)
        station_id = config.get("station_id")
        tick = _safe_int(config.get("current_tick"))
        if not station_id or tick is None:
            return None
        snapshots_dir = self.repo / constants.BACKUP_BASE_DIR / str(station_id) / "snapshots"
        manifest_path = None
        if archive_name:
            candidate = snapshots_dir / f"tick_{tick}_multistart_{archive_name}.json"
            if candidate.is_file():
                manifest_path = candidate
        if manifest_path is None:
            manifest_path = snapshots_dir / f"tick_{tick}.json"
        if not manifest_path.is_file():
            return None
        return {
            "station_id": station_id,
            "tick": tick,
            "manifest_path": str(manifest_path),
            "inferred": True,
        }

    def _prune_archived_branch_data_after_backup(
        self,
        archive_root: Path,
        payload: dict[str, Any],
        backup_ref: Any,
    ) -> dict[str, Any]:
        branch_dirs = [
            path for path in sorted(archive_root.glob("station_data_s*"))
            if path.is_dir() and not path.is_symlink()
        ]
        if not branch_dirs:
            manifest_path = archive_root / "branch_archive_manifest.yaml"
            return {
                "success": manifest_path.exists(),
                "already_pruned": manifest_path.exists(),
                "reason": None if manifest_path.exists() else "no archived branch directories found",
            }

        backup = backup_ref if isinstance(backup_ref, dict) else payload.get("finalization_backup")
        if not isinstance(backup, dict):
            return {"success": False, "reason": "missing finalization backup reference"}

        backup_check = self._check_backup_contains_archived_branches(archive_root, branch_dirs, backup)
        if not backup_check.get("success"):
            return backup_check

        selected_seed = _safe_int(payload.get("selected_seed"))
        branch_records = {}
        for branch in payload.get("branches") or []:
            if isinstance(branch, dict):
                seed = _safe_int(branch.get("seed"))
                if seed is not None:
                    branch_records[seed] = branch

        interviews_dir = archive_root / "interviews"
        interviews_dir.mkdir(parents=True, exist_ok=True)
        branches: dict[str, Any] = {}
        errors: list[str] = []
        removed_count = 0
        removed_bytes = 0

        for branch_dir in branch_dirs:
            seed = _safe_int(branch_dir.name.removeprefix("station_data_s"))
            seed_key = f"s{seed}" if seed is not None else branch_dir.name
            branch_record = branch_records.get(seed or -1, {})
            config = state.read_station_config(branch_dir)
            branch_bytes = self._tree_size_bytes(branch_dir)
            interview_source = branch_dir / interviews.INTERVIEW_FILENAME
            interview_copy = None
            if interview_source.exists():
                interview_copy = interviews_dir / f"{branch_dir.name}.{interviews.INTERVIEW_FILENAME}"
                try:
                    shutil.copy2(interview_source, interview_copy)
                except OSError as exc:
                    errors.append(f"{branch_dir.name}: failed to preserve interview: {exc}")

            metadata = {
                "seed": seed,
                "branch_dir": branch_dir.name,
                "selected": seed == selected_seed,
                "status": branch_record.get("status"),
                "note": branch_record.get("note"),
                "station_name": config.get("station_name"),
                "station_id": config.get("station_id"),
                "current_tick": config.get("current_tick"),
                "backup_path": f"multistart/{archive_root.name}/{branch_dir.name}",
                "interview": str(interview_copy.relative_to(archive_root)) if interview_copy else None,
                "pruned_after_backup": True,
                "pruned_bytes": branch_bytes,
            }
            state.save_yaml_mapping(archive_root / f"{branch_dir.name}.pruned.yaml", metadata)
            branches[seed_key] = metadata

            try:
                research_storage.remove_tree_allow_read_only(branch_dir)
                removed_count += 1
                removed_bytes += branch_bytes
            except OSError as exc:
                errors.append(f"{branch_dir.name}: {exc}")

        self._remove_pruned_admin_branch_links(archive_root)
        manifest = {
            "version": 1,
            "archive_name": archive_root.name,
            "pruned_at": state.utc_now(),
            "backup": backup,
            "restore": {
                "source_prefix": f"multistart/{archive_root.name}",
                "default_output": f"multistart_{archive_root.name}",
                "command": (
                    f"bash scripts/restore.sh --multistart-job {archive_root.name} "
                    f"{backup.get('station_id')} {backup.get('tick')}"
                ),
            },
            "branches": branches,
            "removed_branch_directories": removed_count,
            "removed_bytes": removed_bytes,
            "errors": errors,
        }
        state.save_yaml_mapping(archive_root / "branch_archive_manifest.yaml", manifest)
        self.log(
            f"pruned {removed_count} archived multistart branch folder(s) from {archive_root.name}; "
            f"removed {removed_bytes / (1024 ** 3):.2f} GiB"
        )
        return {
            "success": not errors,
            "backup_verified": True,
            "removed_branch_directories": removed_count,
            "removed_bytes": removed_bytes,
            "errors": errors,
        }

    def _check_backup_contains_archived_branches(
        self,
        archive_root: Path,
        branch_dirs: list[Path],
        backup: dict[str, Any],
    ) -> dict[str, Any]:
        manifest_path = backup.get("manifest_path")
        if not manifest_path:
            return {"success": False, "reason": "backup reference has no manifest_path"}
        manifest = Path(str(manifest_path))
        if not manifest.is_absolute():
            manifest = self.repo / manifest
        if not manifest.is_file():
            return {"success": False, "reason": f"backup manifest not found: {manifest}"}
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"success": False, "reason": f"backup manifest unreadable: {exc}"}

        paths_in_backup = {
            str(entry.get("path") or "").replace("\\", "/")
            for entry in data.get("files", [])
            if isinstance(entry, dict)
        }
        paths_in_backup.update(
            str(entry.get("path") or "").replace("\\", "/")
            for entry in data.get("symlinks", [])
            if isinstance(entry, dict)
        )
        missing = []
        for branch_dir in branch_dirs:
            prefix = f"multistart/{archive_root.name}/{branch_dir.name}/"
            if not any(path.startswith(prefix) for path in paths_in_backup):
                missing.append(branch_dir.name)
        if missing:
            return {
                "success": False,
                "reason": "backup manifest does not contain all archived branch folders",
                "missing_branches": missing,
            }
        return {"success": True}

    def _tree_size_bytes(self, root: Path) -> int:
        total = 0
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _copy_branch_source_tree(
        self,
        source: Path,
        target: Path,
        *,
        job_path: Path | None = None,
        job_id: str | None = None,
        station_id: str | None = None,
        seed: int | None = None,
        storage_base: Path | None,
    ) -> None:
        source_storage = research_storage.resolved_research_storage_path(source)
        use_managed_storage = storage_base is not None and source_storage is not None
        source_storage_linked = research_storage.research_storage_path(source).is_symlink()
        source_root = source.resolve()
        research_root = (
            source / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_RESEARCH
        ).resolve(strict=False)

        def ignore(src: str, names: list[str]) -> set[str]:
            src_path = Path(src).resolve(strict=False)
            ignored = {
                name
                for name in names
                if src_path == source_root and name in BRANCH_COPY_EXCLUDED_TOP_LEVEL
            }
            if use_managed_storage and src_path == research_root:
                ignored.add(constants.RESEARCH_STORAGE_DIR)
            return ignored

        shutil.copytree(source, target, symlinks=True, ignore=ignore)
        if use_managed_storage:
            if job_path is None or seed is None:
                raise RuntimeError("managed multistart storage requires job and seed context")
            self._install_branch_storage_allocation(
                source_storage,
                target,
                job_path=job_path,
                job_id=str(job_id or job_path.name),
                station_id=str(station_id or "station"),
                seed=int(seed),
                storage_base=storage_base,
            )
        elif source_storage_linked:
            self._materialize_linked_research_storage(source, target)

    @staticmethod
    def _configured_research_storage_base(source: Path | None = None) -> Path | None:
        config_value = getattr(constants, "RESEARCH_STORAGE_BASE_PATH", None)
        if source is not None:
            source_config = state.load_yaml_mapping(Path(source) / "constant_config.yaml")
            if "RESEARCH_STORAGE_BASE_PATH" in source_config:
                config_value = source_config.get("RESEARCH_STORAGE_BASE_PATH")
        try:
            return research_storage.configured_base_path(config_value)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    @staticmethod
    def _preferred_job_storage_base(
        job_path: Path,
        *,
        station_id: str,
        job_id: str,
        configured_base: Path | None,
    ) -> Path | None:
        """Keep a live job on its valid manifest base across env changes."""
        manifest = state.load_yaml_mapping(job_path / research_storage.JOB_MANIFEST_FILENAME)
        base_raw = str(manifest.get("base_path") or "").strip()
        if not base_raw:
            return configured_base
        manifest_base = Path(base_raw)

        origin = manifest.get("origin") if isinstance(manifest.get("origin"), dict) else {}
        origin_raw = str(origin.get("target") or "").strip()
        if origin_raw:
            origin_target = Path(origin_raw)
            if (
                origin_target.is_dir()
                and research_storage.path_is_within(origin_target, manifest_base)
                and research_storage.marker_matches(
                    origin_target,
                    station_id=station_id,
                    kinds={"live"},
                )
            ):
                return manifest_base

        seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
        for seed_raw, seed_info in seeds.items():
            if not isinstance(seed_info, dict):
                continue
            seed = _safe_int(seed_raw)
            target_raw = str(seed_info.get("target") or "").strip()
            if seed is None or not target_raw:
                continue
            target = Path(target_raw)
            marker = research_storage.read_allocation_marker(target)
            promoted = (
                str(marker.get("kind") or "") == "live"
                and str(marker.get("station_id") or "") == station_id
                and str(marker.get("promoted_from_job_id") or "") == job_id
                and str(marker.get("promoted_from_seed") or "") == str(seed)
            )
            if (
                target.is_dir()
                and research_storage.path_is_within(target, manifest_base)
                and (
                    promoted
                    or research_storage.marker_matches(
                        target,
                        station_id=station_id,
                        job_id=job_id,
                        seed=seed,
                        kinds={"multistart_seed"},
                    )
                )
            ):
                return manifest_base
        return configured_base

    def _ensure_origin_storage_allocation(
        self,
        origin: Path,
        *,
        job_path: Path,
        job_id: str,
        station_id: str,
        storage_base: Path | None = None,
    ) -> None:
        storage_base = storage_base or self._configured_research_storage_base(origin)
        storage_path = research_storage.research_storage_path(origin)
        if storage_base is None or not storage_path.exists():
            return
        if not storage_path.is_symlink():
            return
        target = research_storage.relocate_storage_symlink(
            storage_path,
            storage_base,
            marker_payload={
                "kind": "live",
                "station_id": station_id,
                "created_by": "multistart_origin_relocation",
            },
            remove_tree=research_storage.remove_tree_allow_read_only,
        )
        self._record_origin_storage_allocation(
            job_path,
            storage_base=storage_base,
            station_id=station_id,
            job_id=job_id,
            target=target,
            owned=research_storage.marker_matches(
                target,
                station_id=station_id,
                kinds={"live"},
            ),
        )

    def _record_origin_storage_allocation(
        self,
        job_path: Path,
        *,
        storage_base: Path,
        station_id: str,
        job_id: str,
        target: Path,
        owned: bool,
    ) -> None:
        manifest_path = job_path / research_storage.JOB_MANIFEST_FILENAME
        manifest = state.load_yaml_mapping(manifest_path)
        manifest.update({
            "format_version": 1,
            "environment_variable": research_storage.BASE_PATH_ENV,
            "base_path": str(storage_base),
            "station_id": station_id,
            "job_id": job_id,
            "origin": {
                "target": str(target),
                "owned": bool(owned),
                "recorded_at": state.utc_now(),
            },
            "seeds": manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {},
        })
        state.save_yaml_mapping(manifest_path, manifest)

    def _install_branch_storage_allocation(
        self,
        source_storage: Path,
        branch_root: Path,
        *,
        job_path: Path,
        job_id: str,
        station_id: str,
        seed: int,
        storage_base: Path,
    ) -> None:
        storage_base.mkdir(parents=True, exist_ok=True)
        manifest_path = job_path / research_storage.JOB_MANIFEST_FILENAME
        with self._storage_manifest_lock:
            manifest = state.load_yaml_mapping(manifest_path)
            seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
            seed_info = seeds.get(str(seed)) if isinstance(seeds.get(str(seed)), dict) else {}
            target_raw = str(seed_info.get("target") or "").strip()
            target_storage = (
                Path(target_raw)
                if target_raw
                else research_storage.new_allocation_path(storage_base)
            )
            if (
                not research_storage.path_is_within(target_storage, storage_base)
                or target_storage.resolve(strict=False) == source_storage.resolve(strict=False)
            ):
                target_storage = research_storage.new_allocation_path(storage_base)
            if research_storage.path_is_within(source_storage, target_storage):
                raise RuntimeError(f"Research storage allocation contains its source: {target_storage}")
            if research_storage.path_is_within(target_storage, source_storage):
                raise RuntimeError(f"Research storage allocation is inside its source: {target_storage}")

            seeds[str(seed)] = {
                "target": str(target_storage),
                "source": str(source_storage),
                "status": "allocating",
                "updated_at": state.utc_now(),
            }
            manifest.update({
                "format_version": 1,
                "environment_variable": research_storage.BASE_PATH_ENV,
                "base_path": str(storage_base),
                "station_id": station_id,
                "job_id": job_id,
                "seeds": seeds,
            })
            state.save_yaml_mapping(manifest_path, manifest)
        research_storage.write_allocation_marker(
            target_storage,
            {
                "kind": "multistart_seed",
                "station_id": station_id,
                "job_id": job_id,
                "seed": int(seed),
                "repo": str(self.repo),
                "created_at": state.utc_now(),
            },
        )

        for stale in storage_base.glob(f".{target_storage.name}.*.tmp"):
            research_storage.remove_tree_allow_read_only(stale)
        temporary = storage_base / f".{target_storage.name}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            shutil.copytree(source_storage, temporary, symlinks=True)
            if target_storage.exists() or target_storage.is_symlink():
                if target_storage.is_dir() and not target_storage.is_symlink():
                    research_storage.remove_tree_allow_read_only(target_storage)
                else:
                    target_storage.unlink()
            os.replace(temporary, target_storage)
        finally:
            if temporary.exists():
                research_storage.remove_tree_allow_read_only(temporary)

        branch_storage = research_storage.research_storage_path(branch_root)
        branch_storage.parent.mkdir(parents=True, exist_ok=True)
        if branch_storage.exists() or branch_storage.is_symlink():
            if branch_storage.is_dir() and not branch_storage.is_symlink():
                research_storage.remove_tree_allow_read_only(branch_storage)
            else:
                branch_storage.unlink()
        branch_storage.symlink_to(target_storage, target_is_directory=True)
        allocation_bytes = self._tree_size_bytes(target_storage)
        with self._storage_manifest_lock:
            manifest = state.load_yaml_mapping(manifest_path)
            seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
            seed_info = seeds.get(str(seed)) if isinstance(seeds.get(str(seed)), dict) else {}
            seed_info.update({
                "target": str(target_storage),
                "source": str(source_storage),
                "bytes": allocation_bytes,
                "status": "ready",
                "updated_at": state.utc_now(),
            })
            seeds[str(seed)] = seed_info
            manifest.update({
                "format_version": 1,
                "environment_variable": research_storage.BASE_PATH_ENV,
                "base_path": str(storage_base),
                "station_id": station_id,
                "job_id": job_id,
                "seeds": seeds,
            })
            state.save_yaml_mapping(manifest_path, manifest)

    def _reconcile_job_storage_allocations(self, job_path: Path, payload: dict[str, Any]) -> None:
        origin = state.origin_dir(job_path)
        origin_config = state.read_station_config(origin) if origin.is_dir() else {}
        station_id = str(payload.get("origin_station_id") or origin_config.get("station_id") or "station")
        job_id = str(payload.get("job_id") or job_path.name)
        source_for_config = origin if origin.exists() else None
        storage_base = self._preferred_job_storage_base(
            job_path,
            station_id=station_id,
            job_id=job_id,
            configured_base=self._configured_research_storage_base(source_for_config),
        )
        if storage_base is None:
            return
        if origin.is_dir():
            self._ensure_origin_storage_allocation(
                origin,
                job_path=job_path,
                job_id=job_id,
                station_id=station_id,
                storage_base=storage_base,
            )
        for branch in payload.get("branches") or []:
            if not isinstance(branch, dict):
                continue
            seed = _safe_int(branch.get("seed"))
            if seed is None:
                continue
            branch_root = state.branch_dir(job_path, seed)
            storage_path = research_storage.research_storage_path(branch_root)
            if not branch_root.is_dir() or not os.path.lexists(storage_path):
                continue
            manifest = state.load_yaml_mapping(job_path / research_storage.JOB_MANIFEST_FILENAME)
            seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
            seed_info = seeds.get(str(seed)) if isinstance(seeds.get(str(seed)), dict) else {}
            expected_raw = str(seed_info.get("target") or "").strip()
            if storage_path.is_symlink() and storage_path.exists():
                resolved = storage_path.resolve()
                marker = research_storage.read_allocation_marker(resolved)
                promoted = (
                    str(marker.get("kind") or "") == "live"
                    and str(marker.get("station_id") or "") == station_id
                    and str(marker.get("promoted_from_job_id") or "") == job_id
                    and str(marker.get("promoted_from_seed") or "") == str(seed)
                )
                if (
                    expected_raw
                    and resolved == Path(expected_raw).resolve(strict=False)
                    and research_storage.path_is_within(resolved, storage_base)
                    and (
                        promoted
                        or research_storage.marker_matches(
                            resolved,
                            station_id=station_id,
                            job_id=job_id,
                            seed=seed,
                            kinds={"multistart_seed"},
                        )
                    )
                ):
                    continue
                source_storage = resolved
            elif storage_path.is_symlink():
                raise RuntimeError(f"branch s{seed} Research storage link is broken: {storage_path}")
            else:
                source_storage = storage_path
            self._install_branch_storage_allocation(
                source_storage,
                branch_root,
                job_path=job_path,
                job_id=job_id,
                station_id=station_id,
                seed=seed,
                storage_base=storage_base,
            )

    def _remove_job_seed_storage_allocations(
        self,
        job_path: Path,
        *,
        station_id: str | None = None,
        job_id: str | None = None,
        preserve_selected: bool,
        include_origin: bool,
    ) -> dict[str, Any]:
        return research_storage.remove_job_allocations(
            job_path,
            station_id=station_id,
            job_id=job_id,
            preserve_selected=preserve_selected,
            include_origin=include_origin,
            sudo_fallback=True,
            cwd=self.repo,
        )

    def _materialize_linked_research_storage(self, source: Path, target: Path) -> None:
        """Give each multistart branch a private Research storage tree.

        Research storage may be a symlink into RESEARCH_STORAGE_BASE_PATH (or
        another location). A normal symlink-preserving branch copy would make
        branches share SQLite/artifact writes. Materialize only this storage
        link while preserving unrelated symlinks.
        """
        relative_storage = Path(
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH,
            constants.RESEARCH_STORAGE_DIR,
        )
        source_storage = source / relative_storage
        target_storage = target / relative_storage
        if not source_storage.is_symlink():
            return
        resolved_storage = source_storage.resolve()
        if not resolved_storage.is_dir():
            raise RuntimeError(
                f"linked Research storage target is missing: {source_storage} -> {resolved_storage}"
            )
        if target_storage.is_symlink():
            target_storage.unlink()
        elif target_storage.exists():
            research_storage.remove_tree_allow_read_only(target_storage)
        shutil.copytree(resolved_storage, target_storage, symlinks=True)

    def _tree_size_bytes_for_branch_copy(self, root: Path, *, exclude_research_storage: bool = False) -> int:
        total = 0
        excluded_roots = {root / name for name in BRANCH_COPY_EXCLUDED_TOP_LEVEL}
        research_storage_path = research_storage.research_storage_path(root)
        for path in root.rglob("*"):
            if any(path == excluded or excluded in path.parents for excluded in excluded_roots):
                continue
            if exclude_research_storage and (
                path == research_storage_path or research_storage_path in path.parents
            ):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        if research_storage_path.is_symlink() and not exclude_research_storage:
            resolved_storage = research_storage_path.resolve()
            if resolved_storage.is_dir():
                total += self._tree_size_bytes(resolved_storage)
        return total

    def _ensure_disk_space_for_branch_copy(self, source: Path, *, copies: int, context: str) -> None:
        try:
            storage_base = self._configured_research_storage_base(source)
        except RuntimeError as exc:
            raise MultistartDiskSpaceError(
                f"invalid {research_storage.BASE_PATH_ENV}: {exc}; "
                "the controller will retry automatically.",
                {
                    "context": context,
                    "environment_variable": research_storage.BASE_PATH_ENV,
                    "research_storage_error": str(exc),
                },
            ) from exc
        source_storage = research_storage.resolved_research_storage_path(source)
        use_managed_storage = storage_base is not None and source_storage is not None
        storage_is_separate_filesystem = False
        if use_managed_storage:
            self._ensure_storage_base_for_branch_copy(
                storage_base,
                source_storage,
                copies=copies,
                context=context,
            )
            try:
                storage_is_separate_filesystem = os.stat(storage_base).st_dev != os.stat(source).st_dev
            except OSError:
                storage_is_separate_filesystem = False
        self._ensure_disk_space_for_copy(
            source,
            copies=copies,
            context=context,
            source_bytes=self._tree_size_bytes_for_branch_copy(
                source,
                exclude_research_storage=use_managed_storage and storage_is_separate_filesystem,
            ),
            excluded_top_level=sorted(BRANCH_COPY_EXCLUDED_TOP_LEVEL),
        )

    def _ensure_storage_base_for_branch_copy(
        self,
        storage_base: Path,
        source_storage: Path,
        *,
        copies: int,
        context: str,
    ) -> None:
        probe: Path | None = None
        try:
            storage_base.mkdir(parents=True, exist_ok=True)
            probe = storage_base / f".station_multistart_write_probe.{os.getpid()}.{time.time_ns()}"
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            probe.unlink()
        except OSError as exc:
            details = {
                "context": context,
                "research_storage_base_path": str(storage_base),
                "research_storage_error": str(exc),
                "environment_variable": research_storage.BASE_PATH_ENV,
            }
            raise MultistartDiskSpaceError(
                f"multistart Research storage base is not writable: {storage_base}: {exc}. "
                f"Set {research_storage.BASE_PATH_ENV} to a writable absolute path; "
                "the controller will retry automatically.",
                details,
            ) from exc
        finally:
            if probe is not None:
                try:
                    probe.unlink(missing_ok=True)
                except OSError:
                    pass

        copies = max(0, int(copies))
        source_bytes = self._tree_size_bytes(source_storage)
        required = source_bytes * copies
        usage = shutil.disk_usage(storage_base)
        projected_used = usage.used + required
        threshold = int(usage.total * DISK_SPACE_HALT_FRACTION)
        if projected_used <= threshold:
            return
        must_free = max(0, projected_used - threshold)
        details = {
            "context": context,
            "research_storage_base_path": str(storage_base),
            "storage_source_bytes": source_bytes,
            "storage_estimated_copy_bytes": required,
            "storage_current_used_bytes": usage.used,
            "storage_available_bytes": usage.free,
            "storage_total_bytes": usage.total,
            "storage_projected_used_bytes": projected_used,
            "storage_threshold_bytes": threshold,
            "storage_must_free_bytes": must_free,
            "environment_variable": research_storage.BASE_PATH_ENV,
        }
        raise MultistartDiskSpaceError(
            "multistart waiting for Research storage space before "
            f"{context}: {self._format_bytes(required)} is required at {storage_base}, "
            f"which would exceed the {DISK_SPACE_HALT_FRACTION * 100:.0f}% safety limit. "
            f"Free at least {self._format_bytes(must_free)} there; the controller will retry automatically.",
            details,
        )

    def _ensure_disk_space_for_copy(
        self,
        source: Path,
        *,
        copies: int,
        context: str,
        source_bytes: int | None = None,
        excluded_top_level: list[str] | None = None,
    ) -> None:
        copies = max(0, int(copies))
        if copies <= 0:
            return
        if source_bytes is None:
            source_bytes = self._tree_size_bytes(source)
        estimated_bytes = source_bytes * copies
        if estimated_bytes <= 0:
            return
        usage = shutil.disk_usage(source if source.exists() else self.repo)
        projected_used = usage.used + estimated_bytes
        threshold = int(usage.total * DISK_SPACE_HALT_FRACTION)
        if projected_used <= threshold:
            return
        projected_percent = 100.0 * projected_used / usage.total if usage.total else 100.0
        current_percent = 100.0 * usage.used / usage.total if usage.total else 100.0
        must_free = max(0, projected_used - threshold)
        details = {
            "context": context,
            "copies": copies,
            "source_bytes": source_bytes,
            "excluded_top_level": excluded_top_level or [],
            "estimated_copy_bytes": estimated_bytes,
            "current_used_bytes": usage.used,
            "available_bytes": usage.free,
            "total_bytes": usage.total,
            "projected_used_bytes": projected_used,
            "threshold_bytes": threshold,
            "must_free_bytes": must_free,
            "current_used_percent": round(current_percent, 3),
            "projected_used_percent": round(projected_percent, 3),
            "safety_limit_percent": round(DISK_SPACE_HALT_FRACTION * 100.0, 3),
            "source_size": self._format_bytes(source_bytes),
            "estimated_copy_size": self._format_bytes(estimated_bytes),
            "current_used": self._format_bytes(usage.used),
            "available": self._format_bytes(usage.free),
            "total": self._format_bytes(usage.total),
            "projected_used": self._format_bytes(projected_used),
            "threshold": self._format_bytes(threshold),
            "must_free": self._format_bytes(must_free),
        }
        raise MultistartDiskSpaceError(
            "multistart waiting for disk space before "
            f"{context}: estimated copy size {self._format_bytes(estimated_bytes)} would raise disk usage "
            f"from {current_percent:.1f}% to {projected_percent:.1f}% "
            f"({self._format_bytes(projected_used)} / {self._format_bytes(usage.total)}), "
            f"above the {DISK_SPACE_HALT_FRACTION * 100:.0f}% safety limit. "
            f"Free at least {self._format_bytes(must_free)} or reduce multistart seed count; "
            "the controller will retry automatically.",
            details,
        )

    def _format_bytes(self, value: int) -> str:
        amount = float(max(0, value))
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024.0 or unit == "TiB":
                return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
            amount /= 1024.0
        return f"{amount:.1f} TiB"

    def _remove_pruned_admin_branch_links(self, archive_root: Path) -> None:
        admin_dir = archive_root / state.ADMIN_DIR_NAME
        if not admin_dir.is_dir():
            return
        for link_path in sorted(admin_dir.glob("station_data_s*")):
            try:
                if link_path.is_symlink():
                    link_path.unlink()
            except OSError as exc:
                self.log(f"failed to remove pruned admin branch link {link_path}: {exc}")

    def _post_guidance_message(self, live: Path, archive_name: str) -> None:
        guidance_path = live / "multistart" / archive_name / "admin" / admin.ADMIN_REPORTS_DIR_NAME / admin.ADMIN_GUIDANCE_REPORT_FILENAME
        try:
            guidance = guidance_path.read_text(encoding="utf-8")
        except OSError:
            return
        message = admin.guidance_announcement(guidance)
        env = os.environ.copy()
        env["STATION_BASE_DATA_PATH"] = str(live)
        tmp_message = guidance_path.with_name("_guidance_message.tmp")
        script = """
from pathlib import Path
from station import constants
from station.station import Station
from station.multistart.interviews import active_recursive_agent_names

s = Station()
msg = Path(MSG_PATH).read_text(encoding="utf-8")
for name in active_recursive_agent_names(Path(DATA_ROOT)):
    agent_data = s.agent_module.load_agent_data(name)
    if not agent_data:
        continue
    s.agent_module.add_pending_notification(
        agent_data,
        msg,
        protected_context_kind=constants.PROTECTED_CONTEXT_KIND_ARCHITECT_MESSAGE,
        protected_context_source="multistart_guidance",
        protected_context_title="Guidance from previous stations",
        protected_context_tick=s._get_current_tick(),
    )
    s.agent_module.save_agent_data(name, agent_data)
""".replace("MSG_PATH", repr(str(tmp_message))).replace("DATA_ROOT", repr(str(live)))
        command = [
            sys.executable,
            "-c",
            script,
        ]
        try:
            tmp_message.write_text(message, encoding="utf-8")
            subprocess.run(command, cwd=self.repo, env=env, check=False)
        finally:
            try:
                tmp_message.unlink()
            except OSError:
                pass

    def _restart_normal_station(self) -> None:
        script = self.repo / "start.sh"
        if not script.is_file():
            return
        env = os.environ.copy()
        env["STATION_MULTISTART_SKIP_CONTROLLER_START"] = "1"
        env["STATION_MULTISTART_SKIP_HOOK"] = "1"
        subprocess.Popen([str(script), "-s"], cwd=self.repo, env=env)


def start_detached(repo: Path, *, init: bool = False) -> int:
    root = paths.multistart_root(repo)
    root.mkdir(parents=True, exist_ok=True)
    existing_pid = find_running_controller_pid(repo)
    if existing_pid is not None:
        paths.controller_pid_path(repo).write_text(str(existing_pid), encoding="utf-8")
        return existing_pid
    log_path = paths.controller_log_path(repo)
    command = [sys.executable, "-m", "station.multistart.controller", "run", "--repo", str(repo)]
    if init:
        command.append("--init")
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
        )
    paths.controller_pid_path(repo).write_text(str(process.pid), encoding="utf-8")
    return process.pid


def _controller_cmdline_matches(args: list[str], repo: Path) -> bool:
    if "station.multistart.controller" not in args:
        return False
    if "run" not in args:
        return False
    repo_text = str(repo.resolve())
    for idx, arg in enumerate(args):
        if arg == "--repo" and idx + 1 < len(args):
            try:
                return str(Path(args[idx + 1]).resolve()) == repo_text
            except Exception:
                return args[idx + 1] == repo_text
        if arg.startswith("--repo="):
            try:
                return str(Path(arg.split("=", 1)[1]).resolve()) == repo_text
            except Exception:
                return arg.split("=", 1)[1] == repo_text
    return False


def _pid_matches_controller(pid: int, repo: Path) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    args = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return _controller_cmdline_matches(args, repo)


def find_running_controller_pid(repo: Path, *, exclude_pid: int | None = None) -> int | None:
    repo = repo.resolve()
    pids: list[int] = []
    for proc_path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_path.name)
        except ValueError:
            continue
        if exclude_pid is not None and pid == exclude_pid:
            continue
        if _pid_matches_controller(pid, repo):
            pids.append(pid)
    return min(pids) if pids else None


def pid_running(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    repo = pid_path.parent.parent.resolve()
    return _pid_matches_controller(pid, repo)


def _recorded_active_branch_descriptions(repo: Path) -> list[str]:
    current = state.load_current_job(repo)
    job_path = Path(str(current.get("job_dir") or ""))
    if not job_path.is_dir():
        return []
    payload = state.load_job_state(job_path)
    branches = payload.get("branches")
    if not isinstance(branches, list):
        return []
    descriptions: list[str] = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        status_text = str(branch.get("status") or "")
        if status_text not in {"running", "paused", "waiting_quiescent", "interviewing"}:
            continue
        pid = branch.get("pid")
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            pid_int = None
        alive = pid_int is not None and _pid_exists(pid_int)
        descriptions.append(
            f"s{branch.get('seed')} status={status_text} pid={pid or 'none'} alive={str(alive).lower()}"
        )
    return descriptions


def _parse_timeout_seconds(value: Any) -> float:
    try:
        timeout_seconds = float(value)
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_GRACEFUL_STOP_TIMEOUT_SECONDS
    return max(0.0, timeout_seconds)


def _multistart_shutdown_complete(repo: Path, pid_path: Path) -> bool:
    if pid_running(pid_path):
        return False
    if find_running_controller_pid(repo) is not None:
        return False
    return not _multistart_process_groups(repo, include_controller=True)


def _active_or_recoverable_multistart_state(repo: Path) -> bool:
    current = state.load_current_job(repo)
    if current and str(current.get("status") or "") not in waiting.TERMINAL_STATUSES:
        return True
    return bool(_multistart_process_groups(repo, include_controller=True))


def _running_controller_pids(repo: Path) -> list[int]:
    pids: set[int] = set()
    pid_path = paths.controller_pid_path(repo)
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        pid = None
    if pid is not None and _pid_matches_controller(pid, repo):
        pids.add(pid)
    found = find_running_controller_pid(repo)
    if found is not None:
        pids.add(found)
    return sorted(pids)


def _terminate_unresponsive_controllers(repo: Path) -> list[int]:
    pids = _running_controller_pids(repo)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            continue
    deadline = time.monotonic() + FORCE_STOP_TERM_SECONDS
    while time.monotonic() < deadline:
        if not any(_pid_matches_controller(pid, repo) for pid in pids):
            break
        time.sleep(0.2)
    for pid in pids:
        if not _pid_matches_controller(pid, repo):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            continue
    for cleanup_path in (paths.controller_pid_path(repo), paths.controller_sock_path(repo)):
        try:
            cleanup_path.unlink()
        except OSError:
            pass
    return pids


def _recover_controller(repo: Path, *, accept_responsive_controller: bool) -> dict[str, Any]:
    """Serialize replacement of a missing or unresponsive controller."""
    repo = repo.resolve()
    root = paths.multistart_root(repo)
    root.mkdir(parents=True, exist_ok=True)
    recovery_lock_path = root / "controller.recovery.lock"
    with recovery_lock_path.open("a+", encoding="utf-8") as recovery_lock:
        fcntl.flock(recovery_lock.fileno(), fcntl.LOCK_EX)
        if accept_responsive_controller:
            grace_deadline = time.monotonic() + CONTROLLER_IPC_SELF_HEAL_GRACE_SECONDS
            while True:
                try:
                    current_response = ipc.request_status(repo=repo)
                except Exception as exc:
                    current_response = {"success": False, "error": str(exc)}
                if current_response.get("success") is True:
                    return {
                        "success": True,
                        "pid": find_running_controller_pid(repo),
                        "already_running": True,
                        "terminated_controller_pids": [],
                    }
                if find_running_controller_pid(repo) is None or time.monotonic() >= grace_deadline:
                    break
                time.sleep(0.2)

        terminated_pids = _terminate_unresponsive_controllers(repo)
        try:
            pid = start_detached(repo, init=False)
        except Exception as exc:
            return {
                "success": False,
                "error": f"could not start recovery multistart controller: {exc}",
                "terminated_controller_pids": terminated_pids,
            }

        deadline = time.monotonic() + CONTROLLER_RECOVERY_TIMEOUT_SECONDS
        last_response: dict[str, Any] = {"success": False, "error": "controller recovery did not respond"}
        while time.monotonic() < deadline:
            try:
                last_response = ipc.request_status(repo=repo)
            except Exception as exc:
                last_response = {"success": False, "error": str(exc)}
            if last_response.get("success") is True:
                return {
                    "success": True,
                    "pid": pid,
                    "terminated_controller_pids": terminated_pids,
                }
            time.sleep(0.5)
        return {
            "success": False,
            "error": "recovery multistart controller did not accept IPC before timeout",
            "pid": pid,
            "terminated_controller_pids": terminated_pids,
            "last_response": last_response,
        }


def recover_controller(repo: Path) -> dict[str, Any]:
    """Recover Resume IPC, retaining a controller that responds during the recovery race."""
    return _recover_controller(repo, accept_responsive_controller=True)


def _recover_controller_for_stop(repo: Path) -> dict[str, Any]:
    return _recover_controller(repo, accept_responsive_controller=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Station multistart controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo", default=os.getcwd())
    run_parser.add_argument("--init", action="store_true")
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--repo", default=os.getcwd())
    start_parser.add_argument("--init", action="store_true")
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--repo", default=os.getcwd())
    stop_parser.add_argument("--force", action="store_true")
    stop_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=os.environ.get("MULTISTART_STOP_TIMEOUT_SECONDS", str(DEFAULT_GRACEFUL_STOP_TIMEOUT_SECONDS)),
        help="Maximum seconds to wait for graceful multistart shutdown before failing.",
    )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--repo", default=os.getcwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(getattr(args, "repo", os.getcwd())).resolve()
    if args.command == "start":
        pid_path = paths.controller_pid_path(repo)
        if pid_running(pid_path):
            print(pid_path.read_text(encoding="utf-8").strip())
            return 0
        existing_pid = find_running_controller_pid(repo)
        if existing_pid is not None:
            pid_path.write_text(str(existing_pid), encoding="utf-8")
            print(existing_pid)
            return 0
        print(start_detached(repo, init=bool(getattr(args, "init", False))))
        return 0
    if args.command == "stop":
        pid_path = paths.controller_pid_path(repo)
        try:
            response = ipc.request_stop(repo=repo, force=bool(args.force))
        except Exception as exc:
            response = {"success": False, "error": f"controller IPC stop failed: {exc}"}
        if response.get("success") is not True and not bool(args.force) and _active_or_recoverable_multistart_state(repo):
            print(
                f"multistart controller IPC unavailable ({response.get('error')}); "
                "starting recovery controller",
                file=sys.stderr,
                flush=True,
            )
            recovery = _recover_controller_for_stop(repo)
            if recovery.get("success") is True:
                print(
                    f"recovery multistart controller started pid={recovery.get('pid')}; requesting stop",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    response = ipc.request_stop(repo=repo, force=False)
                except Exception as exc:
                    response = {"success": False, "error": f"controller IPC stop failed after recovery: {exc}"}
            else:
                response = dict(response)
                response["recovery"] = recovery
        if bool(args.force):
            groups = _force_stop_multistart_processes(repo, include_controller=True)
            if not groups and response.get("success") is not True:
                pid = None
                if pid_running(pid_path):
                    try:
                        pid = int(pid_path.read_text(encoding="utf-8").strip())
                    except Exception:
                        pid = None
                if pid is None:
                    pid = find_running_controller_pid(repo)
                if pid is not None:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        deadline = time.monotonic() + FORCE_STOP_TERM_SECONDS
                        while _pid_exists(pid) and time.monotonic() < deadline:
                            time.sleep(0.2)
                        if _pid_exists(pid):
                            os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
            if not pid_running(pid_path):
                for cleanup_path in (pid_path, paths.controller_sock_path(repo)):
                    try:
                        cleanup_path.unlink()
                    except OSError:
                        pass
            remaining_groups = sorted(_multistart_process_groups(repo, include_controller=True))
            remaining_controller = find_running_controller_pid(repo)
            if remaining_controller is not None or remaining_groups:
                print(json.dumps({
                    "success": False,
                    "error": "multistart processes survived force stop",
                    "controller_pid": remaining_controller,
                    "active_process_groups": remaining_groups,
                }))
                return 1
            print(json.dumps(response if response.get("success") else {"success": True, "force_stopped": True}))
            return 0
        if response.get("success") is not True:
            active_groups = sorted(_multistart_process_groups(repo, include_controller=True))
            if not pid_running(pid_path) and find_running_controller_pid(repo) is None and not active_groups:
                print(json.dumps({"success": True, "message": "no running multistart controller found"}))
                return 0
            response = dict(response)
            if active_groups:
                response["active_process_groups"] = active_groups
            print(json.dumps(response))
            return 1

        timeout_seconds = _parse_timeout_seconds(getattr(args, "timeout_seconds", None))
        deadline = time.monotonic() + timeout_seconds
        next_report = 0.0
        print(
            f"waiting for multistart controller to stop gracefully "
            f"(timeout={timeout_seconds:.0f}s)",
            file=sys.stderr,
            flush=True,
        )
        while not _multistart_shutdown_complete(repo, pid_path):
            now = time.monotonic()
            if now >= deadline:
                active = _recorded_active_branch_descriptions(repo)
                active_groups = sorted(_multistart_process_groups(repo, include_controller=True))
                timeout_response = {
                    "success": False,
                    "error": "timed out waiting for multistart controller to stop gracefully",
                    "timeout_seconds": timeout_seconds,
                    "active_branches": active,
                    "active_process_groups": active_groups,
                    "hint": "Use ./stop.sh --force or ./start.sh -s --force only if you want to bypass the graceful wait.",
                }
                print(json.dumps(timeout_response))
                return 1
            if now >= next_report:
                active = _recorded_active_branch_descriptions(repo)
                active_groups = sorted(_multistart_process_groups(repo, include_controller=True))
                if active:
                    print(
                        "still waiting for multistart branch worker(s): " + "; ".join(active),
                        file=sys.stderr,
                        flush=True,
                    )
                elif active_groups:
                    print(
                        "still waiting for multistart process group(s): "
                        + ", ".join(str(group) for group in active_groups),
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        "still waiting for multistart controller process to exit",
                        file=sys.stderr,
                        flush=True,
                    )
                next_report = now + GRACEFUL_STOP_STATUS_SECONDS
            time.sleep(min(POLL_SECONDS, max(0.1, deadline - now)))
        print(json.dumps(response))
        return 0
    if args.command == "status":
        print(json.dumps(waiting.public_status(repo), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        return Controller(repo).run_loop()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
