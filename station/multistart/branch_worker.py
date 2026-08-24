from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _configure_env(data_root: Path) -> None:
    os.environ["STATION_BASE_DATA_PATH"] = str(data_root)
    os.environ["STATION_MULTISTART_BRANCH"] = "1"
    os.environ["STATION_DISABLE_BACKUPS"] = "1"
    os.environ.setdefault("AUTO_START", "False")


def _update_branch(job_dir: Path, seed: int, **updates: Any) -> None:
    from station.multistart import state

    payload = state.load_job_state(job_dir)
    branches = payload.get("branches")
    if not isinstance(branches, list):
        branches = []
    for branch in branches:
        if isinstance(branch, dict) and int(branch.get("seed") or 0) == seed:
            branch.update(updates)
            break
    else:
        branch = {"seed": seed}
        branch.update(updates)
        branches.append(branch)
    payload["branches"] = branches
    state.save_job_state(job_dir, payload)


def _load_branch(job_dir: Path, seed: int) -> dict[str, Any]:
    from station.multistart import state

    payload = state.load_job_state(job_dir)
    for branch in payload.get("branches") or []:
        if isinstance(branch, dict) and int(branch.get("seed") or 0) == seed:
            return dict(branch)
    return {}


def _stop_background_services(station: Any) -> None:
    for name in (
        "stop_auto_research_evaluator",
        "stop_auto_external_reporter",
        "stop_auto_archive_evaluator",
        "stop_auto_archive_surveyor",
    ):
        func = getattr(station, name, None)
        if callable(func):
            try:
                func()
            except Exception as exc:
                print(f"[multistart] warning: {name} failed: {exc}", flush=True)


def _quiescent(station: Any) -> bool:
    checks = (
        "has_pending_research_evaluations",
        "has_pending_coder_sessions",
        "has_drainable_external_reports",
        "has_pending_archive_surveys",
    )
    for name in checks:
        func = getattr(station, name, None)
        if callable(func):
            try:
                if func():
                    return False
            except Exception as exc:
                print(f"[multistart] warning: quiesce check {name} failed: {exc}", flush=True)
                return False
    return True


def _wait_until_quiescent(station: Any, poll_seconds: float) -> None:
    while not _quiescent(station):
        time.sleep(max(1.0, poll_seconds))


def _shutdown_requested(job_dir: Path) -> bool:
    from station.multistart import state

    payload = state.load_job_state(job_dir)
    return bool(payload.get(state.SHUTDOWN_REQUESTED_KEY))


def _drain_and_exit_for_shutdown(job_dir: Path, seed: int, station: Any, poll_seconds: float) -> None:
    _update_branch(
        job_dir,
        seed,
        status="waiting_quiescent",
        shutdown_requested=True,
        current_tick=int(station._get_current_tick()),
    )
    _wait_until_quiescent(station, poll_seconds)
    _update_branch(
        job_dir,
        seed,
        status="paused",
        shutdown_requested=True,
        shutdown_stopped_at=time.time(),
        pause_reason="graceful multistart shutdown",
        current_tick=int(station._get_current_tick()),
        pid=None,
    )


def _wait_if_paused(job_dir: Path, seed: int, station: Any, target_tick: int, poll_seconds: float) -> bool:
    from station.multistart import state

    marked_paused = False
    while True:
        payload = state.load_job_state(job_dir)
        if bool(payload.get(state.SHUTDOWN_REQUESTED_KEY)):
            _drain_and_exit_for_shutdown(job_dir, seed, station, poll_seconds)
            return True

        if not state.job_paused(payload):
            if marked_paused and int(station._get_current_tick()) < int(target_tick):
                _update_branch(
                    job_dir,
                    seed,
                    status="running",
                    pause_requested=False,
                    pause_reason=None,
                    resumed_at=time.time(),
                )
            return False

        current_tick = int(station._get_current_tick())
        if current_tick >= int(target_tick):
            return False

        if not marked_paused:
            _update_branch(
                job_dir,
                seed,
                status="paused",
                pause_requested=True,
                pause_reason="manual multistart pause",
                paused_at=time.time(),
                current_tick=current_tick,
            )
            marked_paused = True
        time.sleep(max(1.0, poll_seconds))


def _ensure_runnable_branch(orchestrator: Any, job_dir: Path, seed: int) -> None:
    turn_order = list(getattr(orchestrator, "agent_turn_order", []) or [])
    if turn_order:
        return
    _update_branch(job_dir, seed, reset_data_on_resume=True)
    raise RuntimeError("branch has no active agents in turn order after initialization")


def run_branch(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    _configure_env(data_root)

    from station.multistart import interviews
    from station.station import Station
    from station.station_runner import Orchestrator

    _update_branch(job_dir, args.seed, status="running", pid=os.getpid(), started_at=time.time())
    station = None
    orchestrator = None
    try:
        station = Station()
        orchestrator = Orchestrator(station, auto_prepare_on_init=True)
        if args.mode == "init":
            spawned = orchestrator.try_init_agents_for_multistart_branch()
            if spawned:
                _update_branch(job_dir, args.seed, init_agents_spawned=spawned)
        _ensure_runnable_branch(orchestrator, job_dir, args.seed)

        start_tick = int(station._get_current_tick())
        branch_record = _load_branch(job_dir, args.seed)
        target_tick = branch_record.get("target_tick")
        try:
            target_tick = int(target_tick)
        except (TypeError, ValueError):
            target_tick = int(args.branch_tick) + max(0, int(args.roll_ticks))
        _update_branch(job_dir, args.seed, start_tick=start_tick, target_tick=target_tick, current_tick=start_tick)

        if args.mode == "stagnation":
            station.check_stagnation()

        orchestrator.is_running = True
        while int(station._get_current_tick()) < target_tick:
            if _wait_if_paused(job_dir, args.seed, station, target_tick, args.poll_seconds):
                return 0
            if int(station._get_current_tick()) >= target_tick:
                break
            if _shutdown_requested(job_dir):
                _drain_and_exit_for_shutdown(job_dir, args.seed, station, args.poll_seconds)
                return 0
            if not orchestrator.run_single_tick():
                raise RuntimeError("branch tick runner stopped before target tick")
            current_tick = int(station._get_current_tick())
            _update_branch(job_dir, args.seed, current_tick=current_tick)
            if orchestrator.is_paused:
                raise RuntimeError(f"branch paused unexpectedly: {orchestrator.get_pause_reason()}")

        if _shutdown_requested(job_dir):
            _drain_and_exit_for_shutdown(job_dir, args.seed, station, args.poll_seconds)
            return 0

        interview_base_tick = max(1, int(station._get_current_tick()) - 1)
        _update_branch(job_dir, args.seed, status="interviewing", interview_base_tick=interview_base_tick)
        if not interviews.run_interviews(orchestrator, data_root, base_tick=interview_base_tick, branch_tick=int(args.branch_tick)):
            raise RuntimeError("branch interviews failed")

        _update_branch(job_dir, args.seed, status="waiting_quiescent", current_tick=int(station._get_current_tick()))
        _wait_until_quiescent(station, args.poll_seconds)

        config = station.config if isinstance(getattr(station, "config", None), dict) else {}
        _update_branch(
            job_dir,
            args.seed,
            status="completed",
            current_tick=int(station._get_current_tick()),
            top_evaluation_id=config.get("top_evaluation_id"),
            top_score=config.get("top_score"),
            top_sort_key=config.get("top_sort_key"),
            completed_at=time.time(),
            pid=None,
        )
        return 0
    except Exception as exc:
        _update_branch(
            job_dir,
            args.seed,
            status="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
            failed_at=time.time(),
            pid=None,
        )
        print(f"[multistart] branch worker failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    finally:
        if orchestrator is not None:
            try:
                orchestrator.is_running = False
                orchestrator.stop_orchestration()
            except Exception:
                pass
        if station is not None:
            _stop_background_services(station)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one multistart branch headlessly.")
    parser.add_argument("--mode", choices=["init", "stagnation"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--roll-ticks", type=int, required=True)
    parser.add_argument("--branch-tick", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_branch(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
