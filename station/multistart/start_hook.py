from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from station.multistart import controller, paths, waiting


BOOTSTRAP_STAGNATION_STATUS = 21


def _fresh_init_candidate(repo: Path) -> bool:
    live = paths.live_station_data_path(repo)
    if not live.is_dir():
        return False
    config = __import__("station.multistart.state", fromlist=["read_station_config"]).read_station_config(live)
    try:
        current_tick = int(config.get("current_tick") or 0)
    except (TypeError, ValueError):
        current_tick = 0
    return current_tick <= 1 and not (live / "multistart").exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="start.sh multistart hook")
    parser.add_argument("--repo", default=os.getcwd())
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()

    if os.environ.get("STATION_MULTISTART_WAIT_ONLY") == "1":
        print("waiting")
        return 20

    active_job = waiting.active_job(repo)
    if os.environ.get("STATION_MULTISTART_SKIP_CONTROLLER_START") == "1":
        waiting_active = waiting.waiting_mode_active(repo)
        print("waiting" if waiting_active else "normal")
        return 20 if waiting_active else 0

    if active_job:
        pid_path = paths.controller_pid_path(repo)
        if not controller.pid_running(pid_path):
            controller.start_detached(repo, init=False)
        print("waiting")
        return 20

    if waiting.waiting_mode_active(repo):
        pid_path = paths.controller_pid_path(repo)
        if not controller.pid_running(pid_path):
            controller.start_detached(repo, init=paths.pending_init_path(repo).is_file())
        print("waiting")
        return 20

    from station import constants

    init_enabled = int(getattr(constants, "MULTISTART_INIT_SEEDS", 0) or 0) > 1
    stagnation_enabled = int(getattr(constants, "MULTISTART_STAGNATION_SEEDS", 0) or 0) > 1
    if not init_enabled and not stagnation_enabled:
        print("normal")
        return 0

    pending_stagnation = paths.pending_stagnation_path(repo).is_file()
    init_candidate = init_enabled and not pending_stagnation and _fresh_init_candidate(repo)
    pid_path = paths.controller_pid_path(repo)
    if not controller.pid_running(pid_path):
        controller.start_detached(repo, init=init_candidate)

    # A pending stagnation request is deliberately processed by the detached
    # controller after the normal station API is back online.  Starting the
    # controller before Gunicorn is expected: it waits for live station
    # statistics before copying branches.  Do not make the startup hook wait
    # for the request to clear, because that creates a restart deadlock (the
    # controller needs the API, while this hook would refuse to start it).
    if init_candidate:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if waiting.waiting_mode_active(repo):
                print("waiting")
                return 20
            time.sleep(0.5)

    if pending_stagnation and paths.pending_stagnation_path(repo).is_file():
        print("stagnation multistart request is pending; bootstrapping the live API for the controller")
        return BOOTSTRAP_STAGNATION_STATUS

    if waiting.waiting_mode_active(repo):
        print("waiting")
        return 20

    print("normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
