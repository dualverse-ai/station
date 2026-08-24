from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from typing import Any

from station_tools.config import ToolsConfig
from station_tools.frontend_api import find_endpoint, request_json
from station_tools.repo import read_station_metadata
from station_tools.selectors import select_repos, targets_or_current


PROVIDERS = (
    (
        "openai",
        ("OPENAI_API_KEY",),
        "OPENAI_BASE_URL",
        "OPENAI_HTTP_PROXY",
        "OPENAI_HTTPS_PROXY",
        ("BACKUP_OPENAI_API_KEY",),
        "BACKUP_OPENAI_BASE_URL",
        "BACKUP_OPENAI_HTTP_PROXY",
        "BACKUP_OPENAI_HTTPS_PROXY",
    ),
    (
        "claude",
        ("ANTHROPIC_API_KEY",),
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_HTTP_PROXY",
        "ANTHROPIC_HTTPS_PROXY",
        ("BACKUP_ANTHROPIC_API_KEY",),
        "BACKUP_ANTHROPIC_BASE_URL",
        "BACKUP_ANTHROPIC_HTTP_PROXY",
        "BACKUP_ANTHROPIC_HTTPS_PROXY",
    ),
    (
        "gemini",
        ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_GEMINI_HTTP_PROXY",
        "GOOGLE_GEMINI_HTTPS_PROXY",
        ("BACKUP_GOOGLE_API_KEY", "BACKUP_GEMINI_API_KEY"),
        "BACKUP_GOOGLE_GEMINI_BASE_URL",
        "BACKUP_GOOGLE_GEMINI_HTTP_PROXY",
        "BACKUP_GOOGLE_GEMINI_HTTPS_PROXY",
    ),
    (
        "grok",
        ("XAI_API_KEY",),
        "XAI_BASE_URL",
        "XAI_HTTP_PROXY",
        "XAI_HTTPS_PROXY",
        ("BACKUP_XAI_API_KEY",),
        "BACKUP_XAI_BASE_URL",
        "BACKUP_XAI_HTTP_PROXY",
        "BACKUP_XAI_HTTPS_PROXY",
    ),
)


def add_parser(subparsers) -> None:
    parser: ArgumentParser = subparsers.add_parser("refresh-api", help="Refresh runtime API credentials from the current shell env")
    parser.add_argument("targets", nargs="*", help="Station ids, suffixes, names, or paths")
    parser.add_argument("--dry-run", action="store_true", help="Show updates without sending API requests")
    parser.add_argument("--timeout", type=float, default=10.0, help="API timeout in seconds")
    parser.add_argument(
        "--include-station-proxy",
        action="store_true",
        default=os.environ.get("REFRESH_STATION_API_INCLUDE_STATION_PROXY", "0") == "1",
        help="Refresh station proxy fields from LLM/HTTP proxy env vars",
    )
    parser.set_defaults(func=run)


def _first_present(names: tuple[str, ...]) -> tuple[str, str] | None:
    for name in names:
        if name in os.environ:
            return name, os.environ.get(name, "")
    return None


def _split_semicolons(value: str) -> list[str]:
    if value == "":
        return []
    return [item.strip() for item in value.split(";")]


def _backup_entries(
    key_names: tuple[str, ...],
    base_url_name: str,
    http_proxy_name: str,
    https_proxy_name: str,
) -> tuple[str, list[dict[str, str]]] | None:
    present = _first_present(key_names)
    if not present:
        return None
    key_env, raw_keys = present
    keys = _split_semicolons(raw_keys)
    base_urls = _split_semicolons(os.environ.get(base_url_name, "")) if base_url_name in os.environ else []
    http_proxies = _split_semicolons(os.environ.get(http_proxy_name, "")) if http_proxy_name in os.environ else []
    https_proxies = _split_semicolons(os.environ.get(https_proxy_name, "")) if https_proxy_name in os.environ else []
    entries = []
    for index, key in enumerate(keys):
        if not key:
            continue
        entries.append(
            {
                "api_key": key,
                "base_url": base_urls[index] if index < len(base_urls) else "",
                "http_proxy": http_proxies[index] if index < len(http_proxies) else "",
                "https_proxy": https_proxies[index] if index < len(https_proxies) else "",
            }
        )
    return key_env, entries


def _build_payloads(include_station_proxy: bool) -> list[tuple[str, dict[str, Any], tuple[str, ...]]]:
    payloads: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []
    for (
        provider,
        api_key_names,
        base_url,
        http_proxy,
        https_proxy,
        backup_key_names,
        backup_base_url,
        backup_http_proxy,
        backup_https_proxy,
    ) in PROVIDERS:
        payload: dict[str, Any] = {"target": "provider", "provider": provider}
        fields: list[str] = []
        present = _first_present(api_key_names)
        if present:
            field, value = present
            payload["api_key"] = value
            fields.append(field)
        for payload_key, env_name in (("base_url", base_url), ("http_proxy", http_proxy), ("https_proxy", https_proxy)):
            if env_name in os.environ:
                payload[payload_key] = os.environ.get(env_name, "")
                fields.append(env_name)
        backup = _backup_entries(backup_key_names, backup_base_url, backup_http_proxy, backup_https_proxy)
        if backup:
            field, entries = backup
            payload["backup_endpoints"] = entries
            fields.append(field)
            for env_name in (backup_base_url, backup_http_proxy, backup_https_proxy):
                if env_name in os.environ:
                    fields.append(env_name)
        if len(payload) > 2:
            payloads.append((provider, payload, tuple(fields)))

    for target, env_names in (
        ("codex", ("CODEX_API_KEY", "CODEX_BASE_URL", "CODEX_HTTP_PROXY", "CODEX_HTTPS_PROXY")),
        ("external_counter", ("EXTERNAL_OPENAI_API_KEY", "EXTERNAL_OPENAI_BASE_URL", "EXTERNAL_HTTP_PROXY", "EXTERNAL_HTTPS_PROXY")),
    ):
        payload = {"target": target}
        fields = []
        for payload_key, env_name in zip(("api_key", "base_url", "http_proxy", "https_proxy"), env_names):
            if env_name in os.environ:
                payload[payload_key] = os.environ.get(env_name, "")
                fields.append(env_name)
        if len(payload) > 1:
            payloads.append((target, payload, tuple(fields)))

    if include_station_proxy:
        payload = {"target": "station_proxy"}
        fields = []
        for payload_key, candidates in (
            ("http_proxy", ("LLM_HTTP_PROXY", "HTTP_PROXY", "http_proxy")),
            ("https_proxy", ("LLM_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy")),
        ):
            present = _first_present(candidates)
            if present:
                field, value = present
                payload[payload_key] = value
                fields.append(field)
        if len(payload) > 1:
            payloads.append(("station_proxy", payload, tuple(fields)))
    return payloads


def _message(response: dict) -> str:
    generation = response.get("config", {}).get("generation") if isinstance(response.get("config"), dict) else None
    message = str(response.get("message") or response.get("error") or response)
    return f"{message} generation={generation}" if generation is not None else message


def run(args: Namespace, config: ToolsConfig) -> int:
    payloads = _build_payloads(args.include_station_proxy)
    if not payloads:
        print("error: no supported API env vars are present in this shell")
        return 1

    selection = select_repos(targets_or_current(args.targets), config.station_patterns)
    if not selection.repos:
        print("no valid station repos selected")
        return 1

    refreshed: list[str] = []
    unreachable: list[str] = []
    failed: list[str] = []
    for repo in selection.repos:
        meta = read_station_metadata(repo)
        label = f"{meta.station_name} ({repo.name})"
        found = find_endpoint(repo, "/api/station/api_runtime_config", timeout=args.timeout)
        if not found:
            unreachable.append(f"{label} API not reachable on local ports")
            continue
        endpoint, _ = found
        for name, payload, fields in payloads:
            if args.dry_run:
                refreshed.append(f"{label} would update {name} from env fields: {','.join(fields)}")
                continue
            try:
                response = request_json(endpoint, "/api/station/api_runtime_config", method="PUT", payload=payload, timeout=args.timeout)
            except Exception as exc:
                failed.append(f"{label} failed updating {name}: {exc}")
                continue
            refreshed.append(f"{label} updated {name}: {_message(response)}")

    print("station refresh-api summary")
    print("Dry run:" if args.dry_run else "Updated:")
    for item in refreshed:
        print(f"  {item}")
    for title, items in (("API unreachable", unreachable), ("Skipped invalid paths", list(selection.skipped)), ("Failed", failed)):
        if items:
            print(f"{title}:")
            for item in items:
                print(f"  {item}")
    return 1 if unreachable or failed else 0
