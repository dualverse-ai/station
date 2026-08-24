"""Shared local Station dashboard API access."""

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .repo import parse_env_file, read_station_https_port


DEFAULT_FLASK_PORT = 5000
DEFAULT_HTTPS_PORT = 8443


@dataclass(frozen=True)
class ApiEndpoint:
    base_url: str
    auth_header: str | None


def _positive_int(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def auth_header(env_values: dict[str, str]) -> str | None:
    username = os.environ.get("FLASK_AUTH_USERNAME") or env_values.get("FLASK_AUTH_USERNAME") or "admin"
    password = os.environ.get("FLASK_AUTH_PASSWORD") or env_values.get("FLASK_AUTH_PASSWORD") or "changeme"
    if not username and not password:
        return None
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def candidate_base_urls(repo: Path) -> list[str]:
    env_values = parse_env_file(repo)
    urls: list[str] = []

    https_port = (
        _positive_int(os.environ.get("NGINX_HTTPS_PORT"))
        or _positive_int(env_values.get("NGINX_HTTPS_PORT"))
        or read_station_https_port(repo)
        or DEFAULT_HTTPS_PORT
    )
    urls.append(f"https://127.0.0.1:{https_port}")

    http_port = _positive_int(os.environ.get("NGINX_HTTP_PORT")) or _positive_int(env_values.get("NGINX_HTTP_PORT"))
    if http_port:
        urls.append(f"http://127.0.0.1:{http_port}")

    flask_port = _positive_int(os.environ.get("FLASK_PORT")) or _positive_int(env_values.get("FLASK_PORT")) or DEFAULT_FLASK_PORT
    urls.append(f"http://127.0.0.1:{flask_port}")
    return urls


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context()),
    )


def request_json(
    endpoint: ApiEndpoint,
    path: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if endpoint.auth_header:
        headers["Authorization"] = endpoint.auth_header
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(endpoint.base_url + path, data=data, headers=headers, method=method)
    with _opener().open(request, timeout=timeout) as response:
        raw = response.read(2_000_000).decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("API response is not a JSON object")
    return parsed


def find_endpoint(repo: Path, probe_path: str, timeout: float = 10.0) -> tuple[ApiEndpoint, dict[str, Any]] | None:
    env_values = parse_env_file(repo)
    header = auth_header(env_values)
    for base_url in candidate_base_urls(repo):
        endpoint = ApiEndpoint(base_url=base_url, auth_header=header)
        try:
            response = request_json(endpoint, probe_path, timeout=timeout)
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, ValueError):
            continue
        return endpoint, response
    return None
