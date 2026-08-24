from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from station.multistart import paths


DEFAULT_TIMEOUT = 3.0


def send_message(message: dict[str, Any], *, repo: Path | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    sock_path = paths.controller_sock_path(repo)
    if not sock_path.exists():
        return {
            "success": False,
            "error": "controller socket not found",
            "controller_unavailable": True,
        }

    payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
    connected = False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(sock_path))
            connected = True
            client.sendall(payload)
            chunks: list[bytes] = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except socket.timeout as exc:
        return {
            "success": False,
            "error": f"controller IPC timed out: {exc}",
            "controller_unavailable": not connected,
        }
    except OSError as exc:
        return {
            "success": False,
            "error": f"controller IPC unavailable: {exc}",
            "controller_unavailable": True,
        }

    raw = b"".join(chunks).split(b"\n", 1)[0]
    if not raw:
        return {
            "success": False,
            "error": "empty controller response",
            "controller_unavailable": True,
        }
    try:
        response = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {"success": False, "error": f"invalid controller response: {exc}"}
    return response if isinstance(response, dict) else {"success": False, "error": "controller response is not an object"}


def notify_runtime_api_update(payload: dict[str, Any], *, repo: Path | None = None) -> dict[str, Any]:
    return send_message({"type": "runtime_api_update", "payload": payload}, repo=repo)


def request_stop(*, repo: Path | None = None, force: bool = False) -> dict[str, Any]:
    return send_message({"type": "stop", "force": bool(force)}, repo=repo)


def request_status(*, repo: Path | None = None) -> dict[str, Any]:
    return send_message({"type": "status"}, repo=repo)


def request_pause_branches(*, repo: Path | None = None) -> dict[str, Any]:
    return send_message({"type": "pause_branches"}, repo=repo)


def request_resume_branches(
    *,
    repo: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    response = send_message({"type": "resume_branches"}, repo=repo, timeout=timeout)
    if response.get("success") is True or not response.get("controller_unavailable"):
        return response

    repo_path = (repo or paths.repo_root()).resolve()
    original_error = str(response.get("error") or "controller IPC unavailable")
    try:
        # Imported lazily because the controller module itself imports this IPC module.
        from station.multistart import controller

        recovery = controller.recover_controller(repo_path)
    except Exception as exc:
        recovery = {"success": False, "error": str(exc)}
    if recovery.get("success") is not True:
        return {
            "success": False,
            "error": (
                f"{original_error}; automatic controller recovery failed: "
                f"{recovery.get('error') or 'unknown recovery error'}"
            ),
            "controller_unavailable": True,
            "recovery": recovery,
        }

    resumed = send_message({"type": "resume_branches"}, repo=repo_path, timeout=timeout)
    if isinstance(resumed, dict):
        resumed["controller_recovered"] = True
    return resumed
