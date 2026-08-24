#!/usr/bin/env python3
"""
Live command-line monitor for Station status and Research Center CPU/GPU allocations.

The coordination files only contain current allocations. Recent usage is sampled
while this monitor is running.

Usage:
    scripts/monitor_station.py
    scripts/monitor_station.py --once
    scripts/monitor_station.py --station-glob "~/station*"
    scripts/monitor_station.py --root /tmp --interval 1
    scripts/monitor_station.py --gpu-ids 0-7 --cpu-ids 0-95
"""

from __future__ import annotations

import argparse
import ast
import base64
import glob
import ipaddress
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - Station runs on Linux.
    fcntl = None  # type: ignore[assignment]


DEFAULT_ROOT = Path("/tmp")
DEFAULT_GPU_FILE_NAME = "station_gpu_used.json"
DEFAULT_CPU_FILE_NAME = "station_cpu_used.json"
DEFAULT_EXTERNAL_SCHEME = "https"
DEFAULT_EXTERNAL_HOST = "auto"
ENV_NGINX_HTTPS_PORT_KEY = "NGINX_HTTPS_PORT"
ENV_NGINX_HTTP_PORT_KEY = "NGINX_HTTP_PORT"
ENV_FLASK_PORT_KEY = "FLASK_PORT"
ENV_FLASK_AUTH_USERNAME_KEY = "FLASK_AUTH_USERNAME"
ENV_FLASK_AUTH_PASSWORD_KEY = "FLASK_AUTH_PASSWORD"
DEFAULT_FLASK_PORT = 5000
DEFAULT_LOCAL_API_TIMEOUT_SECONDS = 0.35
EXTERNAL_HOST_ENV_VARS = ("STATION_EXTERNAL_HOST", "STATION_PUBLIC_HOST", "PUBLIC_HOST")
PUBLIC_IP_PROBE_URLS = (
    "https://api.ipify.org",
    "https://checkip.amazonaws.com",
    "https://ifconfig.me/ip",
)
PENDING_EXTERNAL_HOST = "__pending_external_host__"
CONFIG_KEYS = {
    "RESEARCH_EVAL_GPU_COORD_FILE",
    "RESEARCH_EVAL_CPU_COORD_FILE",
    "RESEARCH_EVAL_AVAILABLE_GPUS",
    "RESEARCH_EVAL_AVAILABLE_CPUS",
    "STAGNATION_THRESHOLD_TICKS",
}
MAX_STATUS_COLUMN_WIDTH = 80
MULTISTART_ROOT_NAME = "station_multistart"
MULTISTART_CURRENT_JOB_FILENAME = "current_job.yaml"
MULTISTART_LEGACY_CURRENT_JOB_FILENAME = "current_job"
MULTISTART_PENDING_INIT_FILENAME = "pending_init.yaml"
MULTISTART_PENDING_STAGNATION_FILENAME = "pending_stagnation.yaml"
MULTISTART_TERMINAL_STATUSES = {"complete", "completed", "cancelled", "canceled"}
AGENT_AWAITING_HUMAN_INTERVENTION_FLAG = "awaiting_human_intervention"
AGENT_HUMAN_INTERACTION_ID_KEY = "human_interaction_id"
AGENT_HUMAN_INTERACTION_IDS_KEY = "human_interaction_ids"
HUMAN_INTERVENTION_KEYS = {
    AGENT_AWAITING_HUMAN_INTERVENTION_FLAG,
    AGENT_HUMAN_INTERACTION_ID_KEY,
    AGENT_HUMAN_INTERACTION_IDS_KEY,
}
HUMAN_INTERVENTION_LIST_KEYS = {AGENT_HUMAN_INTERACTION_IDS_KEY}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class Allocation:
    resource: str
    key: str
    units: Tuple[int, ...]
    station_id: str
    eval_id: str
    start_time: Optional[float]
    start_time_str: str


@dataclass(frozen=True)
class ResourceSnapshot:
    resource: str
    path: Path
    allocations: Tuple[Allocation, ...]
    used_units: frozenset[int]
    last_updated: Optional[float]
    last_updated_str: str
    error: Optional[str] = None


@dataclass(frozen=True)
class UsageSample:
    timestamp: float
    cpu_used: int
    gpu_used: int
    cpu_load_percent: Optional[float]
    gpu_load_percent: Optional[float]
    ram_used_percent: Optional[float]
    gpu_memory_percent: Optional[float]


@dataclass(frozen=True)
class StationSnapshot:
    station_id: str
    name: str
    path: Path
    display_path: str
    external_url: Optional[str]
    station_status: str
    status_summary: str
    human_intervention_count: int
    current_tick: Optional[int]
    seconds_since_last_tick: Optional[float]
    ticks_since_last_breakthrough: Optional[int]
    top_score: Any
    top_tick: Optional[int]
    next_stagnation_tick: Optional[int]
    active_coders: int
    error: Optional[str] = None


@dataclass(frozen=True)
class MultistartTopRecord:
    sort_key: Tuple[Any, ...]
    score: Any
    top_tick: Optional[int]
    evaluation_id: Any
    seed: int


@dataclass(frozen=True)
class MonitorConfig:
    cpu_path: Path
    gpu_path: Path
    station_patterns: Tuple[str, ...]
    cpu_ids: Optional[Tuple[int, ...]]
    gpu_ids: Optional[Tuple[int, ...]]
    cpu_total: Optional[int]
    gpu_total: Optional[int]
    interval: float
    history_size: int
    chart_window_seconds: float
    chart_buckets: int
    chart_height: int
    show_chart: bool
    lock_timeout: float
    external_scheme: str
    external_host: str
    external_port_base: Optional[int]
    external_port_step: int
    show_external_links: bool
    use_local_api_status: bool
    local_api_timeout: float
    once: bool
    clear: bool
    color: bool


class Palette:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""


_EXTERNAL_HOST_LOCK = threading.Lock()
_EXTERNAL_HOST_PROBE_STARTED = False
_EXTERNAL_HOST_PROBE_DONE = False
_EXTERNAL_HOST_PROBE_RESULT = ""


def parse_id_spec(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    ids: List[int] = []
    seen: Set[int] = set()
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for item in values:
            if item not in seen:
                ids.append(item)
                seen.add(item)
    return tuple(ids)


def parse_int_list(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return parse_id_spec(stripped)
    if isinstance(value, Iterable):
        ids: List[int] = []
        seen: Set[int] = set()
        for item in value:
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                continue
            if parsed not in seen:
                ids.append(parsed)
                seen.add(parsed)
        return tuple(ids) if ids else None
    return None


def parse_yaml_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    lower = stripped.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null", "~"}:
        return None
    if stripped[0:1] in {"[", "{", "'", '"'}:
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped.strip("'\"")
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped.strip("'\"")


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def load_station_defaults(repo_root: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    constants_path = repo_root / "station" / "constants.py"
    if constants_path.exists():
        assignment_pattern = re.compile(r"^([A-Z0-9_]+)\s*=\s*(.+?)\s*(?:#.*)?$")
        for raw_line in constants_path.read_text(encoding="utf-8").splitlines():
            match = assignment_pattern.match(raw_line.strip())
            if not match:
                continue
            key, raw_value = match.groups()
            if key not in CONFIG_KEYS:
                continue
            try:
                defaults[key] = ast.literal_eval(raw_value.strip())
            except (ValueError, SyntaxError):
                defaults[key] = raw_value.strip().strip("'\"")

    config_path = repo_root / "station_data" / "constant_config.yaml"
    if config_path.exists():
        key_pattern = re.compile(r"^([A-Z0-9_]+)\s*:\s*(.*)$")
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = strip_inline_comment(raw_line).strip()
            if not line:
                continue
            match = key_pattern.match(line)
            if not match:
                continue
            key, raw_value = match.groups()
            if key in CONFIG_KEYS:
                defaults[key] = parse_yaml_scalar(raw_value)
    return defaults


def detect_nvidia_gpu_ids() -> Optional[Tuple[int, ...]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    ids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line.split()[0].strip(",")))
        except ValueError:
            return None
    return tuple(ids) if ids else None


def build_path(root: Path, explicit_path: Optional[str], config_path: Any, fallback_name: str) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    if config_path and root == DEFAULT_ROOT:
        return Path(str(config_path)).expanduser()
    return root / fallback_name


def int_tuple_from_total(total: Optional[int]) -> Optional[Tuple[int, ...]]:
    if total is None:
        return None
    if total < 0:
        raise ValueError("resource totals must be non-negative")
    return tuple(range(total))


def clean_external_host(value: Any) -> str:
    raw_value = str(value or "").strip().rstrip("/")
    if not raw_value:
        return ""
    if raw_value.lower() in {"auto", "probe"}:
        return "auto"
    if raw_value.lower() in {"none", "off", "false"}:
        return ""
    if "://" in raw_value:
        parsed = urllib.parse.urlsplit(raw_value)
        return parsed.hostname or ""
    if raw_value.startswith("[") and raw_value.endswith("]"):
        return raw_value[1:-1]
    return raw_value


def format_url_host(host: str) -> str:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    if ip.version == 6:
        return f"[{host}]"
    return host


def global_ip_or_none(candidate: Any) -> Optional[str]:
    host = clean_external_host(candidate)
    if not host or host == "auto":
        return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_global:
        return str(ip)
    return None


def run_probe_command(command: Sequence[str], timeout: float) -> str:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=max(0.2, timeout),
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def detect_global_ip_from_ip_addr(timeout: float) -> Optional[str]:
    output = run_probe_command(["ip", "-j", "-4", "address", "show", "scope", "global"], timeout)
    if output:
        try:
            interfaces = json.loads(output)
        except json.JSONDecodeError:
            interfaces = []
        if isinstance(interfaces, list):
            for interface in interfaces:
                if not isinstance(interface, dict):
                    continue
                for addr_info in interface.get("addr_info", []):
                    if not isinstance(addr_info, dict):
                        continue
                    detected = global_ip_or_none(addr_info.get("local"))
                    if detected:
                        return detected

    output = run_probe_command(["ip", "-4", "address", "show", "scope", "global"], timeout)
    for candidate in re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)", output):
        detected = global_ip_or_none(candidate)
        if detected:
            return detected
    return None


def detect_global_ip_from_hostname(timeout: float) -> Optional[str]:
    output = run_probe_command(["hostname", "-I"], timeout)
    for candidate in output.split():
        detected = global_ip_or_none(candidate)
        if detected:
            return detected
    return None


def detect_global_ip_from_web(timeout: float) -> Optional[str]:
    if timeout <= 0:
        return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in PUBLIC_IP_PROBE_URLS:
        request = urllib.request.Request(url, headers={"User-Agent": "station-monitor/1"})
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read(128).decode("utf-8", errors="replace")
        except (OSError, urllib.error.URLError, TimeoutError):
            continue
        for candidate in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", body):
            detected = global_ip_or_none(candidate)
            if detected:
                return detected
    return None


def detect_external_host(timeout: float) -> str:
    for env_name in EXTERNAL_HOST_ENV_VARS:
        env_host = clean_external_host(os.environ.get(env_name))
        if env_host and env_host != "auto":
            return env_host
    for probe in (
        detect_global_ip_from_ip_addr,
        detect_global_ip_from_hostname,
        detect_global_ip_from_web,
    ):
        detected = probe(timeout)
        if detected:
            return detected
    return ""


def start_external_host_probe(timeout: float) -> None:
    global _EXTERNAL_HOST_PROBE_STARTED, _EXTERNAL_HOST_PROBE_DONE, _EXTERNAL_HOST_PROBE_RESULT
    with _EXTERNAL_HOST_LOCK:
        if _EXTERNAL_HOST_PROBE_STARTED:
            return
        _EXTERNAL_HOST_PROBE_STARTED = True
        _EXTERNAL_HOST_PROBE_DONE = False
        _EXTERNAL_HOST_PROBE_RESULT = ""

    def worker() -> None:
        global _EXTERNAL_HOST_PROBE_DONE, _EXTERNAL_HOST_PROBE_RESULT
        try:
            result = detect_external_host(timeout)
        except Exception:
            result = ""
        with _EXTERNAL_HOST_LOCK:
            _EXTERNAL_HOST_PROBE_RESULT = result
            _EXTERNAL_HOST_PROBE_DONE = True

    thread = threading.Thread(target=worker, name="station-monitor-external-host-probe", daemon=True)
    thread.start()


def get_external_host_probe_state() -> Tuple[bool, str]:
    with _EXTERNAL_HOST_LOCK:
        return _EXTERNAL_HOST_PROBE_DONE, _EXTERNAL_HOST_PROBE_RESULT


def resolve_external_host(raw_host: Any, timeout: float) -> str:
    host = clean_external_host(raw_host)
    if not host:
        return ""
    if host == "auto":
        for env_name in EXTERNAL_HOST_ENV_VARS:
            env_host = clean_external_host(os.environ.get(env_name))
            if env_host and env_host != "auto":
                return env_host
        start_external_host_probe(timeout)
        return PENDING_EXTERNAL_HOST
    return host


def resolve_config(args: argparse.Namespace) -> MonitorConfig:
    repo_root = Path(__file__).resolve().parents[1]
    defaults = load_station_defaults(repo_root)
    root = Path(args.root).expanduser() if args.root else DEFAULT_ROOT
    station_patterns = (
        tuple(pattern for group in args.station_glob for pattern in group)
        if args.station_glob
        else (str(Path.home() / "station"), str(Path.home() / "station_*"))
    )
    chart_window_seconds = max(1.0, float(args.chart_window))
    chart_buckets = max(4, int(args.chart_buckets))
    chart_height = max(2, int(args.chart_height))
    chart_samples = int(chart_window_seconds / args.interval) + 2
    external_scheme = str(args.external_scheme or DEFAULT_EXTERNAL_SCHEME).strip().rstrip(":/") or DEFAULT_EXTERNAL_SCHEME
    external_host = "" if args.no_external_links else resolve_external_host(args.external_host, max(0.0, args.external_probe_timeout))
    external_port_base = args.external_port_base
    external_port_step = int(args.external_port_step)
    local_api_timeout = max(0.05, float(args.local_api_timeout))
    if external_port_base is not None and external_port_base < 0:
        raise ValueError("--external-port-base must be non-negative")
    if external_port_step < 0:
        raise ValueError("--external-port-step must be non-negative")

    gpu_ids = parse_id_spec(args.gpu_ids)
    if gpu_ids is None and args.gpu_total is not None:
        gpu_ids = int_tuple_from_total(args.gpu_total)
    if gpu_ids is None:
        gpu_ids = parse_int_list(defaults.get("RESEARCH_EVAL_AVAILABLE_GPUS"))
    if gpu_ids is None:
        gpu_ids = detect_nvidia_gpu_ids()

    cpu_ids = parse_id_spec(args.cpu_ids)
    if cpu_ids is None and args.cpu_total is not None:
        cpu_ids = int_tuple_from_total(args.cpu_total)
    if cpu_ids is None:
        cpu_ids = parse_int_list(defaults.get("RESEARCH_EVAL_AVAILABLE_CPUS"))

    cpu_total = len(cpu_ids) if cpu_ids is not None else args.cpu_total
    if cpu_total is None:
        cpu_total = os.cpu_count()
    gpu_total = len(gpu_ids) if gpu_ids is not None else args.gpu_total

    return MonitorConfig(
        cpu_path=build_path(root, args.cpu_file, defaults.get("RESEARCH_EVAL_CPU_COORD_FILE"), DEFAULT_CPU_FILE_NAME),
        gpu_path=build_path(root, args.gpu_file, defaults.get("RESEARCH_EVAL_GPU_COORD_FILE"), DEFAULT_GPU_FILE_NAME),
        station_patterns=station_patterns,
        cpu_ids=cpu_ids,
        gpu_ids=gpu_ids,
        cpu_total=cpu_total,
        gpu_total=gpu_total,
        interval=args.interval,
        history_size=max(1, args.history, chart_samples),
        chart_window_seconds=chart_window_seconds,
        chart_buckets=chart_buckets,
        chart_height=chart_height,
        show_chart=not args.no_chart,
        lock_timeout=max(0.0, args.lock_timeout),
        external_scheme=external_scheme,
        external_host=external_host,
        external_port_base=external_port_base,
        external_port_step=external_port_step,
        show_external_links=not args.no_external_links,
        use_local_api_status=not args.no_local_api_status,
        local_api_timeout=local_api_timeout,
        once=args.once,
        clear=not args.no_clear,
        color=(not args.no_color and sys.stdout.isatty()),
    )


def load_yaml_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "not a mapping"
    return data, None


def read_top_level_fields(path: Path, keys: Set[str], list_keys: Set[str]) -> Dict[str, Any]:
    if not path.exists():
        return {}
    values: Dict[str, Any] = {}
    active_list_key: Optional[str] = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped_line = raw_line.lstrip()
                if active_list_key and stripped_line.startswith("- "):
                    values.setdefault(active_list_key, []).append(parse_yaml_scalar(stripped_line[2:].strip()))
                    continue
                if active_list_key:
                    active_list_key = None
                    if len(values) == len(keys):
                        break
                if not raw_line or raw_line[0].isspace() or ":" not in raw_line:
                    if raw_line and raw_line.strip():
                        active_list_key = None
                    continue
                key, raw_value = raw_line.split(":", 1)
                key = key.strip()
                if key not in keys:
                    continue
                stripped_value = raw_value.strip()
                if key in list_keys and stripped_value == "":
                    values[key] = []
                    active_list_key = key
                else:
                    values[key] = parse_yaml_scalar(stripped_value)
                if len(values) == len(keys) and active_list_key is None:
                    break
    except OSError:
        return {}
    return values


def has_human_intervention_request(fields: Dict[str, Any]) -> bool:
    if fields.get(AGENT_AWAITING_HUMAN_INTERVENTION_FLAG, False):
        return True
    request_ids = fields.get(AGENT_HUMAN_INTERACTION_IDS_KEY)
    if isinstance(request_ids, list) and request_ids:
        return True
    return bool(fields.get(AGENT_HUMAN_INTERACTION_ID_KEY))


def human_intervention_count(data_root: Path, turn_order: Optional[Sequence[Any]] = None) -> int:
    agents_dir = data_root / "agents"
    if not agents_dir.is_dir():
        return 0
    if turn_order:
        agent_paths = [agents_dir / f"{agent_name}.yaml" for agent_name in turn_order if str(agent_name)]
    else:
        agent_paths = sorted(agents_dir.glob("*.yaml"))
    count = 0
    for agent_path in agent_paths:
        fields = read_top_level_fields(agent_path, HUMAN_INTERVENTION_KEYS, HUMAN_INTERVENTION_LIST_KEYS)
        if has_human_intervention_request(fields):
            count += 1
    return count


def coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def roman_to_int(value: str) -> Optional[int]:
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(str(value or "").upper()):
        number = numerals.get(char)
        if number is None:
            return None
        if number < previous:
            total -= number
        else:
            total += number
            previous = number
    return total if total > 0 else None


def stagnation_level(status: Any) -> Optional[int]:
    text = str(status or "").strip()
    if text == "Healthy":
        return 0
    match = re.match(r"^Stagnation\s+([IVXLCDM]+)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return roman_to_int(match.group(1))


def station_stagnation_threshold(station_path: Path) -> int:
    defaults = load_station_defaults(station_path)
    value = coerce_int(defaults.get("STAGNATION_THRESHOLD_TICKS"))
    return max(1, value or 250)


def current_stagnation_start_tick(config: Dict[str, Any], current_level: int) -> Optional[int]:
    current_status = str(config.get("station_status") or "").strip()
    history = config.get("status_history")
    if not isinstance(history, list):
        return None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "").strip() != current_status:
            continue
        if stagnation_level(entry.get("status")) != current_level:
            continue
        tick = coerce_int(entry.get("start_tick"))
        if tick is not None:
            return tick
    return None


def next_stagnation_tick(config: Dict[str, Any], current_tick: Optional[int], top_tick: Optional[int], threshold: int) -> Optional[int]:
    level = stagnation_level(config.get("station_status", "Healthy"))
    if level is None:
        return None
    counter = coerce_int(config.get("stagnation_counter"))
    if counter is not None and current_tick is not None:
        target_counter = (level + 1) * threshold
        return current_tick + max(0, target_counter - max(0, counter))
    due_tick: Optional[int]
    if level <= 0:
        due_tick = top_tick + threshold if top_tick is not None else None
    else:
        start_tick = current_stagnation_start_tick(config, level)
        if start_tick is not None:
            due_tick = start_tick + threshold
        elif top_tick is not None:
            due_tick = top_tick + ((level + 1) * threshold)
        elif current_tick is not None:
            due_tick = current_tick + threshold
        else:
            due_tick = None
    if due_tick is not None and current_tick is not None and due_tick < current_tick:
        return current_tick
    return due_tick


def coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_top_sort_key(sort_key: Any, score: Any) -> Optional[Tuple[Any, ...]]:
    if sort_key is not None:
        raw_items = tuple(sort_key) if isinstance(sort_key, (list, tuple)) else (sort_key,)
        normalized_items: List[Any] = []
        for item in raw_items:
            if isinstance(item, bool):
                normalized_items.append(int(item))
                continue
            if isinstance(item, (int, float)):
                if isinstance(item, float) and math.isnan(item):
                    normalized_items = []
                    break
                normalized_items.append(item)
                continue
            text = str(item or "").strip()
            if not text or text.startswith("*"):
                normalized_items = []
                break
            try:
                numeric_item: Any = int(text)
            except ValueError:
                try:
                    numeric_item = float(text)
                except ValueError:
                    normalized_items = []
                    break
            if isinstance(numeric_item, float) and math.isnan(numeric_item):
                normalized_items = []
                break
            normalized_items.append(numeric_item)
        if normalized_items:
            return tuple(normalized_items)
    numeric_score = coerce_float(score)
    if numeric_score is None or math.isnan(numeric_score):
        return None
    return (numeric_score,)


def evaluation_id_rank(value: Any) -> Tuple[int, Any]:
    try:
        return (0, int(str(value)))
    except (TypeError, ValueError):
        return (1, str(value))


def multistart_top_record_is_better(candidate: MultistartTopRecord, current: MultistartTopRecord) -> bool:
    candidate_key = candidate.sort_key
    current_key = current.sort_key
    if candidate_key > current_key:
        return True
    if candidate_key < current_key:
        return False
    if candidate.top_tick is not None and current.top_tick is not None and candidate.top_tick != current.top_tick:
        return candidate.top_tick < current.top_tick
    if candidate.evaluation_id is not None and current.evaluation_id is not None:
        candidate_eval_rank = evaluation_id_rank(candidate.evaluation_id)
        current_eval_rank = evaluation_id_rank(current.evaluation_id)
        if candidate_eval_rank != current_eval_rank:
            return candidate_eval_rank < current_eval_rank
    return candidate.seed < current.seed


def read_cpu_times() -> Optional[Tuple[int, int]]:
    try:
        first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = first_line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(values) < 5:
        return None
    idle = values[3] + values[4]
    total = sum(values)
    return total, idle


def sample_cpu_load_percent(previous: Optional[Tuple[int, int]]) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    current = read_cpu_times()
    if previous is None or current is None:
        return None, current
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None, current
    busy_delta = max(0, total_delta - max(0, idle_delta))
    return max(0.0, min(100.0, 100.0 * busy_delta / total_delta)), current


def sample_ram_used_percent() -> Optional[float]:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    values: Dict[str, float] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        value = coerce_float(parts[0] if parts else None)
        if value is not None:
            values[key] = value
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None or total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total - available) / total))


def sample_gpu_metrics(timeout: float = 0.8) -> Tuple[Optional[float], Optional[float]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if result.returncode != 0:
        return None, None
    utilization_values: List[float] = []
    memory_values: List[float] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        utilization = coerce_float(parts[0])
        memory_used = coerce_float(parts[1])
        memory_total = coerce_float(parts[2])
        if utilization is not None:
            utilization_values.append(utilization)
        if memory_used is not None and memory_total is not None and memory_total > 0:
            memory_values.append(100.0 * memory_used / memory_total)
    utilization_percent = max(0.0, min(100.0, mean(utilization_values))) if utilization_values else None
    memory_percent = max(0.0, min(100.0, mean(memory_values))) if memory_values else None
    return utilization_percent, memory_percent


def format_agent_list_for_status(agent_names: Sequence[Any], max_visible: int = 5) -> str:
    names = [str(name) for name in agent_names if str(name)]
    if not names:
        return "none"
    visible_names = names[:max_visible]
    hidden_count = len(names) - max_visible
    suffix = f" +{hidden_count} more" if hidden_count > 0 else ""
    return ", ".join(visible_names) + suffix


def format_parallel_tick_status(parallel_tick_status: Optional[Dict[str, Any]]) -> str:
    if not parallel_tick_status:
        return ""
    if parallel_tick_status.get("error"):
        return f"Parallel status unavailable: {parallel_tick_status['error']}"
    if not parallel_tick_status.get("active"):
        return ""

    counts = parallel_tick_status.get("counts") if isinstance(parallel_tick_status.get("counts"), dict) else {}
    preparing = parallel_tick_status.get("preparing_station_response") or []
    waiting = parallel_tick_status.get("waiting_for_response") or []
    pending_commit = parallel_tick_status.get("response_received_pending_commit") or []
    internal_running = parallel_tick_status.get("internal_action_running") or []
    total = coerce_int(counts.get("total")) or 0
    prepared = coerce_int(counts.get("observation_prepared")) or 0
    responses_received = coerce_int(counts.get("response_received")) or 0
    committed = coerce_int(counts.get("committed")) or 0

    if preparing:
        return f"Preparing station responses ({prepared}/{total} ready)"
    if waiting:
        return f"Waiting for {format_agent_list_for_status(waiting, 4)} ({responses_received}/{total} done)"
    if pending_commit:
        return f"Committing {len(pending_commit)} response(s) ({committed}/{total} done)"
    if internal_running:
        return f"Waiting for internal action: {format_agent_list_for_status(internal_running, 4)}"
    if total > 0:
        return f"Committed {committed}/{total} responses"
    return "No active agents"


def format_orchestrator_status_summary(status: Dict[str, Any]) -> str:
    wait_reasons = status.get("waiting_reasons") if isinstance(status.get("waiting_reasons"), dict) else {}
    parallel_status = format_parallel_tick_status(status.get("parallel_tick_status"))
    lines: List[str] = []
    if status.get("is_waiting"):
        lines.append("Waiting")
    elif status.get("is_paused"):
        lines.append("Paused")
    elif status.get("is_running"):
        lines.append("Running")
    elif status.get("is_prepared"):
        lines.append("Prepared (Idle)")
    else:
        lines.append("Idle / Stopped")
    lines.extend(str(reason) for reason in wait_reasons.values() if str(reason))
    pause_reason = str(status.get("pause_reason") or "").strip()
    if status.get("is_paused") and pause_reason:
        lines.append(pause_reason)
    if parallel_status:
        lines.append(parallel_status)
    return "; ".join(lines)


def pending_multistart_disk_space_summary(
    station_path: Path,
    config: Dict[str, Any],
) -> str:
    root = station_path / MULTISTART_ROOT_NAME
    candidates = (
        (root / MULTISTART_PENDING_STAGNATION_FILENAME, "Stagnation multistart"),
        (root / MULTISTART_PENDING_INIT_FILENAME, "Multistart initialization"),
    )
    current_tick = coerce_int(config.get("current_tick"))

    for pending_path, label in candidates:
        pending, error = load_yaml_file(pending_path)
        if error or not pending or str(pending.get("status") or "").strip() != "blocked_disk_space":
            continue
        branch_tick = coerce_int(pending.get("branch_tick"))
        if current_tick is not None and branch_tick is not None and current_tick != branch_tick:
            continue

        disk_space = pending.get("disk_space") if isinstance(pending.get("disk_space"), dict) else {}
        must_free_bytes = coerce_float(disk_space.get("must_free_bytes"))
        if must_free_bytes is not None and must_free_bytes > 0:
            must_free_gib = must_free_bytes / (1024.0 ** 3)
            return f"{label} blocked: insufficient disk space; est. extra {must_free_gib:.1f} GiB needed"

        must_free = str(disk_space.get("must_free") or "").strip()
        if must_free:
            return f"{label} blocked: insufficient disk space; est. extra {must_free} needed"
        return f"{label} blocked: insufficient disk space"
    return ""


def build_file_status_summary(config: Dict[str, Any]) -> str:
    return "Status API unavailable"


def build_runtime_status_summary(
    station_path: Path,
    station_data_path: Path,
    config: Dict[str, Any],
    monitor_config: MonitorConfig,
) -> Tuple[str, int]:
    if monitor_config.use_local_api_status:
        statistics = fetch_local_station_statistics(station_path, monitor_config.local_api_timeout)
        human_count_from_statistics: Optional[int] = None
        if statistics is not None:
            pending = statistics.get("pending_human_requests")
            if isinstance(pending, dict):
                agents = pending.get("agents")
                if isinstance(agents, list):
                    human_count_from_statistics = len(agents)
        status = fetch_local_orchestrator_status(station_path, monitor_config.local_api_timeout)
        if status is not None:
            agents = status.get("agents_awaiting_human")
            human_count = (
                human_count_from_statistics
                if human_count_from_statistics is not None
                else len(agents) if isinstance(agents, list) else 0
            )
            summary = format_orchestrator_status_summary(status)
            if status.get("is_paused") or status.get("is_waiting"):
                disk_summary = pending_multistart_disk_space_summary(station_path, config)
                if disk_summary:
                    summary = disk_summary
            return summary, human_count
    return build_file_status_summary(config), human_intervention_count(
        station_data_path,
        config.get("agent_turn_order") if isinstance(config.get("agent_turn_order"), list) else None,
    )


def parse_env_file(station_path: Path) -> Dict[str, str]:
    env_path = station_path / ".env"
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = parse_env_value(raw_value)
        if key:
            values[key] = value
    return values


def local_api_port_for_station(env_values: Dict[str, str]) -> int:
    raw_port = env_values.get(ENV_FLASK_PORT_KEY) or os.environ.get(ENV_FLASK_PORT_KEY)
    port = coerce_int(raw_port)
    return port if port and port > 0 else DEFAULT_FLASK_PORT


def positive_env_port(env_values: Dict[str, str], key: str) -> Optional[int]:
    port = coerce_int(env_values.get(key) or os.environ.get(key))
    return port if port and port > 0 else None


def local_api_urls_for_station(station_path: Path, env_values: Dict[str, str], api_path: str) -> List[str]:
    api_path = "/" + api_path.lstrip("/")
    urls = [f"http://127.0.0.1:{local_api_port_for_station(env_values)}{api_path}"]
    http_port = positive_env_port(env_values, ENV_NGINX_HTTP_PORT_KEY)
    if http_port:
        urls.append(f"http://127.0.0.1:{http_port}{api_path}")
    https_port = positive_env_port(env_values, ENV_NGINX_HTTPS_PORT_KEY)
    if https_port is None:
        https_port = read_station_https_port(station_path)
    if https_port:
        urls.append(f"https://127.0.0.1:{https_port}{api_path}")
    return urls


def local_api_auth_header(env_values: Dict[str, str]) -> Optional[str]:
    username = env_values.get(ENV_FLASK_AUTH_USERNAME_KEY) or os.environ.get(ENV_FLASK_AUTH_USERNAME_KEY) or "admin"
    password = env_values.get(ENV_FLASK_AUTH_PASSWORD_KEY) or os.environ.get(ENV_FLASK_AUTH_PASSWORD_KEY) or "changeme"
    if not username and not password:
        return None
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def fetch_local_api_payload(station_path: Path, api_path: str, timeout: float) -> Optional[Dict[str, Any]]:
    env_values = parse_env_file(station_path)
    auth_header = local_api_auth_header(env_values)
    ssl_context = ssl._create_unverified_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl_context),
    )
    for url in local_api_urls_for_station(station_path, env_values, api_path):
        request = urllib.request.Request(url)
        if auth_header:
            request.add_header("Authorization", auth_header)
        request.add_header("Accept", "application/json")
        try:
            with opener.open(request, timeout=timeout) as response:
                payload = json.loads(response.read(1_000_000).decode("utf-8", errors="replace"))
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("success"):
            continue
        return payload
    return None


def fetch_local_orchestrator_status(station_path: Path, timeout: float) -> Optional[Dict[str, Any]]:
    payload = fetch_local_api_payload(station_path, "/api/orchestrator/status", timeout)
    if not payload:
        return None
    status = payload.get("status")
    if isinstance(status, dict):
        return status
    return None


def fetch_local_station_statistics(station_path: Path, timeout: float) -> Optional[Dict[str, Any]]:
    payload = fetch_local_api_payload(station_path, "/api/station/statistics", timeout)
    if not payload:
        return None
    statistics = payload.get("statistics")
    if isinstance(statistics, dict):
        return statistics
    return payload


def build_station_status_summary(
    station_path: Path,
    config: Dict[str, Any],
    monitor_config: MonitorConfig,
) -> str:
    if monitor_config.use_local_api_status:
        status = fetch_local_orchestrator_status(station_path, monitor_config.local_api_timeout)
        if status is not None:
            return format_orchestrator_status_summary(status)
    return build_file_status_summary(config)


def display_path(path: Path) -> str:
    resolved = path.expanduser()
    try:
        relative = resolved.relative_to(Path.home())
        return "~" if str(relative) == "." else "~/" + str(relative)
    except ValueError:
        return str(resolved)


def discover_station_paths(patterns: Sequence[str]) -> Tuple[Path, ...]:
    paths: List[Path] = []
    seen: Set[Path] = set()
    for pattern in patterns:
        expanded_pattern = os.path.expandvars(os.path.expanduser(pattern))
        matched = glob.glob(expanded_pattern)
        candidates = matched if matched else [expanded_pattern]
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_dir():
                continue
            config_path = path / "station_data" / "station_config.yaml"
            if not config_path.exists() and not active_multistart_job(path):
                continue
            try:
                key = path.resolve()
            except OSError:
                key = path
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return tuple(sorted(paths, key=lambda item: display_path(item)))


def multistart_root(station_path: Path) -> Path:
    return station_path / MULTISTART_ROOT_NAME


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    data, error = load_yaml_file(path)
    if error or not isinstance(data, dict):
        return {}
    return data


def load_current_multistart_job(station_path: Path) -> Dict[str, Any]:
    root = multistart_root(station_path)
    for filename in (MULTISTART_CURRENT_JOB_FILENAME, MULTISTART_LEGACY_CURRENT_JOB_FILENAME):
        data = load_yaml_mapping(root / filename)
        if data:
            return data
    return {}


def active_multistart_job(station_path: Path) -> Optional[Dict[str, Any]]:
    job = load_current_multistart_job(station_path)
    if not job:
        return None
    status = str(job.get("status") or "").lower()
    if status in MULTISTART_TERMINAL_STATUSES:
        return None
    job_dir = job.get("job_dir")
    if job_dir and not Path(str(job_dir)).exists():
        return None
    return job


def load_multistart_detail(station_path: Path, job: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    job_dir = job.get("job_dir")
    if job_dir:
        job_path = Path(str(job_dir))
    else:
        branch_tick = coerce_int(job.get("branch_tick"))
        job_id = str(job.get("job_id") or "").strip()
        job_path = multistart_root(station_path) / f"{branch_tick}_{job_id}" if branch_tick and job_id else Path()
    detail = load_yaml_mapping(job_path / "state.yaml") if job_path else {}
    return job_path, detail


def multistart_stage(status_text: str, counts: Dict[str, int], seed_count: Any, selected_seed: Any) -> str:
    normalized = str(status_text or "").lower()
    total = coerce_int(seed_count) or 0
    completed = int(counts.get("completed") or 0)
    failed = int(counts.get("failed") or 0)
    if normalized == "selecting":
        return "admin running"
    if normalized == "finalizing":
        return f"finalizing seed {selected_seed}" if selected_seed else "finalizing"
    if normalized == "failed" or failed:
        return f"halted; {failed} failed" if failed else "halted"
    if total > 0 and completed >= total:
        return "pending admin"
    if counts.get("interviewing"):
        return f"interviewing {counts['interviewing']}/{total}"
    if counts.get("waiting_quiescent"):
        return f"waiting jobs {counts['waiting_quiescent']}/{total}"
    if counts.get("running"):
        return f"rolling branches {counts['running']}/{total}"
    if counts.get("pending"):
        return f"pending branches {counts['pending']}/{total}"
    return normalized or "pending"


def research_evaluation_summary(data_root: Path) -> Dict[str, int]:
    evaluations_dir = data_root / "rooms" / "research" / "evaluations"
    db_path = data_root / "index" / "station_index.sqlite3"
    if not evaluations_dir.is_dir() or not db_path.is_file():
        return {"active_coders": 0}
    try:
        scope = str(evaluations_dir.resolve())
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.2)
        try:
            conn.execute("PRAGMA query_only = ON")
            row = conn.execute(
                """
                SELECT SUM(CASE WHEN active_coder = 1 OR coder_active = 1 THEN 1 ELSE 0 END) AS active_coders
                FROM research_evaluations
                WHERE evaluations_dir = ?
                """,
                (scope,),
            ).fetchone()
            return {"active_coders": int((row[0] if row else 0) or 0)}
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return {"active_coders": 0}


def read_multistart_snapshot(
    station_path: Path,
    now: float,
    external_url: Optional[str],
    monitor_config: MonitorConfig,
    job: Dict[str, Any],
) -> StationSnapshot:
    job_path, detail = load_multistart_detail(station_path, job)
    branches = detail.get("branches") if isinstance(detail.get("branches"), list) else []
    branch_configs: List[Dict[str, Any]] = []
    branch_ticks: List[int] = []
    breakthrough_counters: List[int] = []
    tick_timestamps: List[float] = []
    top_ticks: List[int] = []
    top_records: List[MultistartTopRecord] = []
    next_stagnation_ticks: List[int] = []
    active_coders = 0
    human_count = 0
    stagnation_threshold = station_stagnation_threshold(station_path)
    counts: Dict[str, int] = {
        "pending": 0,
        "running": 0,
        "waiting_quiescent": 0,
        "interviewing": 0,
        "completed": 0,
        "failed": 0,
    }

    for raw_branch in branches:
        if not isinstance(raw_branch, dict):
            continue
        status_text = str(raw_branch.get("status") or "pending")
        counts[status_text] = counts.get(status_text, 0) + 1
        seed = coerce_int(raw_branch.get("seed")) or 0
        data_root = Path(str(raw_branch.get("data_root") or (job_path / f"station_data_s{seed}")))
        config_path = data_root / "station_config.yaml"
        config = load_yaml_mapping(config_path)
        if config:
            branch_configs.append(config)
        current_tick = coerce_int(config.get("current_tick")) if config else None
        if current_tick is None:
            current_tick = coerce_int(raw_branch.get("current_tick"))
        if current_tick is not None:
            branch_ticks.append(current_tick)
        breakthrough_counter = coerce_int(config.get("stagnation_counter")) if config else None
        if breakthrough_counter is not None:
            breakthrough_counters.append(max(0, breakthrough_counter))
        timestamp = latest_tick_timestamp(load_yaml_mapping(data_root / "tick_timing_state.yaml"), current_tick)
        if timestamp is None:
            timestamp = coerce_float(raw_branch.get("completed_at") or raw_branch.get("started_at"))
        if timestamp is None and config_path.exists():
            try:
                timestamp = config_path.stat().st_mtime
            except OSError:
                timestamp = None
        if timestamp is not None:
            tick_timestamps.append(timestamp)
        top_tick = coerce_int(config.get("top_tick")) if config else None
        if top_tick is not None:
            top_ticks.append(top_tick)
        top_score = config.get("top_score") if config else raw_branch.get("top_score")
        top_sort_key = config.get("top_sort_key") if config else raw_branch.get("top_sort_key")
        normalized_top_sort_key = normalize_top_sort_key(top_sort_key, top_score)
        if normalized_top_sort_key is not None:
            top_records.append(
                MultistartTopRecord(
                    sort_key=normalized_top_sort_key,
                    score=top_score,
                    top_tick=top_tick,
                    evaluation_id=(
                        config.get("top_evaluation_id") if config else raw_branch.get("top_evaluation_id")
                    ),
                    seed=seed,
                )
            )
        branch_next_stagnation_tick = next_stagnation_tick(config, current_tick, top_tick, stagnation_threshold) if config else None
        if branch_next_stagnation_tick is not None:
            next_stagnation_ticks.append(branch_next_stagnation_tick)
        eval_summary = research_evaluation_summary(data_root)
        active_coders += int(eval_summary.get("active_coders") or 0)
        turn_order = config.get("agent_turn_order") if isinstance(config.get("agent_turn_order"), list) else None
        human_count += human_intervention_count(data_root, turn_order)

    station_name = str(detail.get("station_name") or job.get("station_name") or "").strip() or station_path.name
    station_id = str(detail.get("origin_station_id") or job.get("origin_station_id") or "").strip()
    if not station_id:
        for config in branch_configs:
            station_id = str(config.get("station_id") or "").strip()
            if station_id:
                break
    status_text = str(detail.get("status") or job.get("status") or "")
    seed_count = detail.get("seed_count") or job.get("seed_count")
    selected_seed = detail.get("selected_seed") or job.get("selected_seed")
    stage = multistart_stage(status_text, counts, seed_count, selected_seed)
    branch_tick = coerce_int(detail.get("branch_tick") or job.get("branch_tick"))
    roll_ticks = coerce_int(detail.get("roll_ticks") or job.get("roll_ticks"))
    target_tick = branch_tick + roll_ticks if branch_tick is not None and roll_ticks is not None else None
    current_tick = max(branch_ticks) if branch_ticks else branch_tick
    if top_records:
        best_top_record = top_records[0]
        for candidate_top_record in top_records[1:]:
            if multistart_top_record_is_better(candidate_top_record, best_top_record):
                best_top_record = candidate_top_record
        top_score = best_top_record.score
        top_tick = best_top_record.top_tick
        best_seed = best_top_record.seed
    else:
        top_score = None
        top_tick = max(top_ticks) if top_ticks else None
        best_seed = None
    next_stag_tick = min(next_stagnation_ticks) if next_stagnation_ticks else None
    seconds_since_last_tick = max(0.0, now - max(tick_timestamps)) if tick_timestamps else None
    ticks_since_breakthrough = (
        max(breakthrough_counters)
        if breakthrough_counters
        else max(0, current_tick - top_tick) if current_tick is not None and top_tick is not None else None
    )
    counts_text = ",".join(
        f"{key[:1]}={value}"
        for key, value in (
            ("running", counts.get("running", 0)),
            ("completed", counts.get("completed", 0)),
            ("failed", counts.get("failed", 0)),
            ("pending", counts.get("pending", 0)),
        )
        if value
    )
    progress = f"{current_tick}/{target_tick}" if current_tick is not None and target_tick is not None else "-"
    best_text = f"; best=s{best_seed}" if best_seed else ""
    status_summary = f"multistart {stage}; tick {progress}; {counts_text or 'no branches'}{best_text}"

    return StationSnapshot(
        station_id=station_id,
        name=station_name,
        path=station_path,
        display_path=display_path(station_path),
        external_url=external_url,
        station_status="multistart",
        status_summary=status_summary,
        human_intervention_count=human_count,
        current_tick=current_tick,
        seconds_since_last_tick=seconds_since_last_tick,
        ticks_since_last_breakthrough=ticks_since_breakthrough,
        top_score=top_score,
        top_tick=top_tick,
        next_stagnation_tick=next_stag_tick,
        active_coders=active_coders,
    )


def station_port_offset(path: Path, fallback_index: int) -> int:
    match = re.fullmatch(r"station(?:_(\d+))?", path.name)
    if not match:
        return fallback_index
    suffix = match.group(1)
    if suffix is None:
        return 0
    try:
        return max(0, int(suffix) - 1)
    except ValueError:
        return fallback_index


def parse_env_value(raw_value: str) -> str:
    value = strip_inline_comment(raw_value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_station_env_https_port(station_path: Path) -> Optional[int]:
    env_path = station_path / ".env"
    if not env_path.exists():
        return None
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() != ENV_NGINX_HTTPS_PORT_KEY:
            continue
        return coerce_int(parse_env_value(raw_value))
    return None


def read_station_https_port(station_path: Path) -> Optional[int]:
    nginx_config = station_path / "deployment" / "nginx.conf"
    if not nginx_config.exists():
        return None
    try:
        lines = nginx_config.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    listen_pattern = re.compile(
        r"\blisten\s+(?:\[[^\]]+\]:|[^:;\s]+:)?(?P<port>\d+)\b[^;]*\bssl\b"
    )
    for line in lines:
        match = listen_pattern.search(line)
        if not match:
            continue
        try:
            return int(match.group("port"))
        except ValueError:
            return None
    return None


def station_external_port(config: MonitorConfig, station_path: Path, station_index: int) -> Optional[int]:
    configured_port = read_station_env_https_port(station_path)
    if configured_port is None:
        configured_port = read_station_https_port(station_path)
    if configured_port is not None:
        return configured_port
    if config.external_port_base is None:
        return None
    return config.external_port_base + config.external_port_step * station_port_offset(station_path, station_index)


def build_external_url(config: MonitorConfig, station_path: Path, station_index: int) -> Optional[str]:
    if not config.show_external_links or not config.external_host:
        return None
    if config.external_host == PENDING_EXTERNAL_HOST:
        _probe_done, detected_host = get_external_host_probe_state()
        if not detected_host:
            return "pending"
        host = format_url_host(detected_host)
    else:
        host = format_url_host(config.external_host)
    port = station_external_port(config, station_path, station_index)
    if port is None:
        return f"{config.external_scheme}://{host}/"
    return f"{config.external_scheme}://{host}:{port}/"


def latest_tick_timestamp(timing_state: Dict[str, Any], current_tick: Optional[int]) -> Optional[float]:
    latest_tick = timing_state.get("latest_tick")
    latest_start = coerce_float(timing_state.get("latest_tick_started_timestamp"))
    if latest_start is not None and (
        current_tick is None or latest_tick in {current_tick, str(current_tick)}
    ):
        return latest_start

    recent_ticks = timing_state.get("recent_ticks")
    if isinstance(recent_ticks, list):
        for item in reversed(recent_ticks):
            if not isinstance(item, dict):
                continue
            if current_tick is not None and item.get("tick") not in {current_tick, str(current_tick)}:
                continue
            ended = coerce_float(item.get("ended_timestamp"))
            if ended is not None:
                return ended
            started = coerce_float(item.get("started_timestamp"))
            if started is not None:
                return started

    latest_end = coerce_float(timing_state.get("latest_tick_ended_timestamp"))
    return latest_end


def read_station_snapshot(
    station_path: Path,
    now: float,
    external_url: Optional[str],
    monitor_config: MonitorConfig,
) -> StationSnapshot:
    station_data_path = station_path / "station_data"
    config, config_error = load_yaml_file(station_data_path / "station_config.yaml")
    if config_error or config is None:
        return StationSnapshot(
            station_id="",
            name=station_path.name,
            path=station_path,
            display_path=display_path(station_path),
            external_url=external_url,
            station_status="-",
            status_summary=f"Unavailable: {config_error}",
            human_intervention_count=0,
            current_tick=None,
            seconds_since_last_tick=None,
            ticks_since_last_breakthrough=None,
            top_score=None,
            top_tick=None,
            next_stagnation_tick=None,
            active_coders=0,
            error=config_error,
        )

    timing_state, _timing_error = load_yaml_file(station_data_path / "tick_timing_state.yaml")
    timing_state = timing_state or {}

    current_tick = coerce_int(config.get("current_tick"))
    top_tick = coerce_int(config.get("top_tick"))
    tick_timestamp = latest_tick_timestamp(timing_state, current_tick)
    seconds_since_last_tick = max(0.0, now - tick_timestamp) if tick_timestamp is not None else None
    ticks_since_last_breakthrough = coerce_int(config.get("stagnation_counter"))
    if ticks_since_last_breakthrough is not None:
        ticks_since_last_breakthrough = max(0, ticks_since_last_breakthrough)
    elif current_tick is not None and top_tick is not None:
        ticks_since_last_breakthrough = max(0, current_tick - top_tick)
    station_id = str(config.get("station_id") or "").strip()
    name = str(config.get("station_name") or "").strip() or station_path.name
    next_stag_tick = next_stagnation_tick(
        config,
        current_tick,
        top_tick,
        station_stagnation_threshold(station_path),
    )
    eval_summary = research_evaluation_summary(station_data_path)
    running_status, human_count = build_runtime_status_summary(station_path, station_data_path, config, monitor_config)

    return StationSnapshot(
        station_id=station_id,
        name=name,
        path=station_path,
        display_path=display_path(station_path),
        external_url=external_url,
        station_status=str(config.get("station_status") or "").strip() or "-",
        status_summary=running_status,
        human_intervention_count=human_count,
        current_tick=current_tick,
        seconds_since_last_tick=seconds_since_last_tick,
        ticks_since_last_breakthrough=ticks_since_last_breakthrough,
        top_score=config.get("top_score"),
        top_tick=top_tick,
        next_stagnation_tick=next_stag_tick,
        active_coders=int(eval_summary.get("active_coders") or 0),
    )


def collect_station_snapshots(config: MonitorConfig, now: float) -> Tuple[StationSnapshot, ...]:
    snapshots: List[StationSnapshot] = []
    for index, path in enumerate(discover_station_paths(config.station_patterns)):
        external_url = build_external_url(config, path, index)
        job = active_multistart_job(path)
        if job:
            snapshots.append(read_multistart_snapshot(path, now, external_url, config, job))
        else:
            snapshots.append(read_station_snapshot(path, now, external_url, config))
    return tuple(snapshots)


def station_name_map(stations: Sequence[StationSnapshot]) -> Dict[str, str]:
    return {
        station.station_id: station.name
        for station in stations
        if station.station_id
    }


def read_json_with_lock(path: Path, lock_timeout: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            if fcntl is not None:
                start = time.monotonic()
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() - start >= lock_timeout:
                            return None, "lock timeout"
                        time.sleep(0.02)
            try:
                content = handle.read().strip()
                if not content:
                    return {"allocations": {}}, None
                return json.loads(content), None
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    except OSError as exc:
        return None, str(exc)


def parse_units(raw_units: Any) -> Tuple[int, ...]:
    if not isinstance(raw_units, list):
        return tuple()
    units: List[int] = []
    for item in raw_units:
        try:
            units.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(units)))


def parse_snapshot(resource: str, path: Path, unit_field: str, lock_timeout: float) -> ResourceSnapshot:
    data, error = read_json_with_lock(path, lock_timeout)
    if error:
        return ResourceSnapshot(
            resource=resource,
            path=path,
            allocations=tuple(),
            used_units=frozenset(),
            last_updated=None,
            last_updated_str="",
            error=error,
        )

    assert data is not None
    raw_allocations = data.get("allocations", {})
    allocations: List[Allocation] = []
    used_units: Set[int] = set()
    if isinstance(raw_allocations, dict):
        for key, raw_info in raw_allocations.items():
            if not isinstance(raw_info, dict):
                continue
            station_id = str(raw_info.get("station_id") or "").strip()
            eval_id = str(raw_info.get("eval_id") or "").strip()
            if (not station_id or not eval_id) and ":" in str(key):
                key_station, key_eval = str(key).split(":", 1)
                station_id = station_id or key_station
                eval_id = eval_id or key_eval
            units = parse_units(raw_info.get(unit_field))
            used_units.update(units)
            start_time_raw = raw_info.get("start_time")
            try:
                start_time = float(start_time_raw) if start_time_raw is not None else None
            except (TypeError, ValueError):
                start_time = None
            allocations.append(
                Allocation(
                    resource=resource,
                    key=str(key),
                    units=units,
                    station_id=station_id or "?",
                    eval_id=eval_id or "?",
                    start_time=start_time,
                    start_time_str=str(raw_info.get("start_time_str") or ""),
                )
            )

    last_updated_raw = data.get("last_updated")
    try:
        last_updated = float(last_updated_raw) if last_updated_raw is not None else None
    except (TypeError, ValueError):
        last_updated = None

    return ResourceSnapshot(
        resource=resource,
        path=path,
        allocations=tuple(sorted(allocations, key=lambda item: (item.station_id, item.eval_id, item.key))),
        used_units=frozenset(used_units),
        last_updated=last_updated,
        last_updated_str=str(data.get("last_updated_str") or ""),
    )


def collect_snapshots(config: MonitorConfig) -> Dict[str, ResourceSnapshot]:
    return {
        "GPU": parse_snapshot("GPU", config.gpu_path, "gpus", config.lock_timeout),
        "CPU": parse_snapshot("CPU", config.cpu_path, "cpus", config.lock_timeout),
    }


def compact_ranges(units: Sequence[int]) -> str:
    if not units:
        return "-"
    sorted_units = sorted(set(units))
    ranges: List[str] = []
    start = sorted_units[0]
    end = sorted_units[0]
    for value in sorted_units[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(str(start) if start == end else f"{start}-{end}")
        start = end = value
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_time(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def truncate(value: str, width: int) -> str:
    value = str(value)
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "."


def format_optional_int(value: Optional[int]) -> str:
    return str(value) if value is not None else "-"


def format_top_score(score: Any) -> str:
    if score is None or score == "":
        return "-"
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            return str(score)
        if numeric_score.is_integer():
            return str(int(numeric_score))
        decimal_places = 8
        factor = 10 ** decimal_places
        truncated = math.trunc(numeric_score * factor) / factor
        return f"{truncated:.{decimal_places}f}"
    return str(score)


def station_display_name(station_id: str, name_by_id: Dict[str, str]) -> str:
    return name_by_id.get(station_id, station_id or "?")


def color_ticks_since_breakthrough(value: Optional[int], text: str, palette: Palette) -> str:
    if value is None or not palette.enabled:
        return text
    if value > 500:
        color = palette.red
    elif value > 100:
        color = palette.yellow
    else:
        color = palette.green
    return f"{color}{text}{palette.reset}"


def color_tick_age(seconds: Optional[float], text: str, palette: Palette) -> str:
    if seconds is None or not palette.enabled:
        return text
    if seconds > 3 * 3600:
        color = palette.red
    elif seconds >= 3600:
        color = palette.yellow
    else:
        color = palette.green
    return f"{color}{text}{palette.reset}"


def color_disk_usage(percent: float, text: str, palette: Palette) -> str:
    if not palette.enabled:
        return text
    if percent >= 90.0:
        color = palette.red
    elif percent >= 70.0:
        color = palette.yellow
    else:
        color = palette.green
    return f"{color}{text}{palette.reset}"


def render_bar(used: int, total: Optional[int], width: int, palette: Palette) -> str:
    width = max(8, width)
    if total is None or total <= 0:
        filled = min(width, used)
        color = palette.cyan
    else:
        ratio = used / total if total else 0
        filled = min(width, max(0, round(width * ratio)))
        if ratio >= 0.9:
            color = palette.red
        elif ratio >= 0.7:
            color = palette.yellow
        else:
            color = palette.green
    return "[" + color + ("#" * filled) + palette.reset + ("-" * (width - filled)) + "]"


def render_values(values: Sequence[int], width: int) -> str:
    if not values:
        return "-"
    output: List[str] = []
    used = 0
    for value in reversed(values):
        item = str(value)
        needed = len(item) + (1 if output else 0)
        if output and used + needed > width:
            break
        output.append(item)
        used += needed
    return " ".join(reversed(output))


def format_capacity(used: int, total: Optional[int]) -> str:
    if total is None:
        return f"{used}/?"
    return f"{used}/{total}"


def render_resource_line(
    label: str,
    used: int,
    total: Optional[int],
    values: Sequence[int],
    term_width: int,
    palette: Palette,
) -> str:
    bar_width = min(36, max(12, term_width - 90))
    samples_width = min(36, max(12, term_width - 88))
    avg = mean(values) if values else 0.0
    peak = max(values) if values else 0
    return (
        f"{palette.bold}{label:<3}{palette.reset} "
        f"{format_capacity(used, total):>8} "
        f"{render_bar(used, total, bar_width, palette)} "
        f"recent avg {avg:5.1f} peak {peak:<4} samples {render_values(values, samples_width)}"
    )


def format_window_label(seconds: float) -> str:
    seconds_int = int(seconds)
    if seconds_int >= 3600 and seconds_int % 3600 == 0:
        return f"{seconds_int // 3600}h"
    if seconds_int >= 60 and seconds_int % 60 == 0:
        return f"{seconds_int // 60}m"
    return f"{seconds_int}s"


def sample_resource_value(sample: UsageSample, resource: str) -> int:
    if resource == "CPU":
        return sample.cpu_used
    return sample.gpu_used


def sample_metric_value(sample: UsageSample, metric: str) -> Optional[float]:
    if metric == "cpu_load":
        return sample.cpu_load_percent
    if metric == "gpu_load":
        return sample.gpu_load_percent
    if metric == "ram":
        return sample.ram_used_percent
    if metric == "gpu_memory":
        return sample.gpu_memory_percent
    return None


def infer_active_usage_at(snapshot: ResourceSnapshot, timestamp: float) -> Optional[float]:
    used_units: Set[int] = set()
    inferred = False
    for allocation in snapshot.allocations:
        if allocation.start_time is None:
            continue
        if allocation.start_time <= timestamp:
            used_units.update(allocation.units)
            inferred = True
    if not inferred:
        return None
    return float(len(used_units))


def bucket_usage_values(
    samples: Sequence[UsageSample],
    snapshot: ResourceSnapshot,
    resource: str,
    now: float,
    window_seconds: float,
    bucket_count: int,
) -> List[Optional[float]]:
    bucket_count = max(1, bucket_count)
    bucket_width = window_seconds / bucket_count
    window_start = now - window_seconds
    sums = [0.0 for _ in range(bucket_count)]
    counts = [0 for _ in range(bucket_count)]

    for sample in samples:
        if sample.timestamp < window_start:
            continue
        offset = sample.timestamp - window_start
        index = int(offset / bucket_width) if bucket_width > 0 else bucket_count - 1
        index = min(bucket_count - 1, max(0, index))
        sums[index] += sample_resource_value(sample, resource)
        counts[index] += 1

    values: List[Optional[float]] = []
    for index, (total, count) in enumerate(zip(sums, counts)):
        if count:
            values.append(total / count)
            continue
        bucket_center = window_start + (index + 0.5) * bucket_width
        values.append(infer_active_usage_at(snapshot, bucket_center))
    return values


def bucket_metric_values(
    samples: Sequence[UsageSample],
    metric: str,
    now: float,
    window_seconds: float,
    bucket_count: int,
) -> List[Optional[float]]:
    bucket_count = max(1, bucket_count)
    bucket_width = window_seconds / bucket_count
    window_start = now - window_seconds
    sums = [0.0 for _ in range(bucket_count)]
    counts = [0 for _ in range(bucket_count)]

    for sample in samples:
        value = sample_metric_value(sample, metric)
        if value is None or sample.timestamp < window_start:
            continue
        offset = sample.timestamp - window_start
        index = int(offset / bucket_width) if bucket_width > 0 else bucket_count - 1
        index = min(bucket_count - 1, max(0, index))
        sums[index] += value
        counts[index] += 1

    return [total / count if count else None for total, count in zip(sums, counts)]


def format_axis_number(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.1f}"


def render_compact_chart(
    title: str,
    values: Sequence[Optional[float]],
    now_value: Optional[float],
    total: Optional[int],
    height: int,
    width: int,
    window_label: str,
    color: str,
    palette: Palette,
) -> List[str]:
    width = max(22, width)
    label_width = 5
    plot_width = max(8, width - label_width - 3)
    observed = [value for value in values if value is not None]
    peak = max(observed) if observed else 0.0
    scale = float(total) if total and total > 0 else peak
    scale = max(scale, peak, 1.0)
    total_label = str(total) if total is not None else "?"
    now_label = "-" if now_value is None else format_axis_number(float(now_value))
    visible_values = list(values[-plot_width:])
    if len(visible_values) < plot_width:
        visible_values = [None for _ in range(plot_width - len(visible_values))] + visible_values

    lines = [truncate(f"{title} now {now_label}/{total_label} pk {format_axis_number(peak)}", width)]
    for row in range(height, 0, -1):
        threshold = ((row - 0.5) / height) * scale
        label_value = (row / height) * scale
        chars: List[str] = []
        for value in visible_values:
            if value is None:
                chars.append(" ")
            elif value > 0 and value >= threshold:
                chars.append("#")
            else:
                chars.append(" ")
        bar = "".join(chars)
        lines.append(f"{format_axis_number(label_value):>{label_width}} |{color}{bar}{palette.reset}|")

    baseline_chars = ["." if value is not None else " " for value in visible_values]
    baseline = "".join(baseline_chars)
    lines.append(f"{'0':>{label_width}} |{palette.dim}{baseline}{palette.reset}|")
    axis_inner = f"{'-' + window_label:<{max(3, plot_width - 3)}}now"
    lines.append(f"{'':>{label_width}}  {truncate(axis_inner, plot_width + 1)}")
    return lines


def compose_chart_row(charts: Sequence[List[str]], width: int) -> List[str]:
    if not charts:
        return []
    height = max(len(chart) for chart in charts)
    padded: List[List[str]] = []
    for chart in charts:
        lines = chart + ["" for _ in range(height - len(chart))]
        padded.append([f"{line:<{width}}" for line in lines])
    return ["  ".join(row_parts) for row_parts in zip(*padded)]


def render_usage_chart(
    config: MonitorConfig,
    snapshots: Dict[str, ResourceSnapshot],
    samples: Sequence[UsageSample],
    term_width: int,
    palette: Palette,
) -> List[str]:
    if not config.show_chart:
        return []
    now = time.time()
    chart_width = max(22, (term_width - 4) // 3)
    bucket_count = max(8, chart_width - 8)
    window_label = format_window_label(config.chart_window_seconds)
    cpu_values = bucket_usage_values(samples, snapshots["CPU"], "CPU", now, config.chart_window_seconds, bucket_count)
    gpu_values = bucket_usage_values(samples, snapshots["GPU"], "GPU", now, config.chart_window_seconds, bucket_count)
    cpu_load_values = bucket_metric_values(samples, "cpu_load", now, config.chart_window_seconds, bucket_count)
    gpu_load_values = bucket_metric_values(samples, "gpu_load", now, config.chart_window_seconds, bucket_count)
    ram_values = bucket_metric_values(samples, "ram", now, config.chart_window_seconds, bucket_count)
    gpu_memory_values = bucket_metric_values(samples, "gpu_memory", now, config.chart_window_seconds, bucket_count)
    cpu_now = samples[-1].cpu_used if samples else 0
    gpu_now = samples[-1].gpu_used if samples else 0
    cpu_load_now = samples[-1].cpu_load_percent if samples else None
    gpu_load_now = samples[-1].gpu_load_percent if samples else None
    ram_now = samples[-1].ram_used_percent if samples else None
    gpu_memory_now = samples[-1].gpu_memory_percent if samples else None

    lines = [
        f"Last {window_label} usage charts (bucket avg, rolls right)",
        "blank = no sample; allocation charts infer active starts where possible; dots = observed zero",
    ]
    first_row = [
        render_compact_chart("CPU alloc", cpu_values, float(cpu_now), config.cpu_total, config.chart_height, chart_width, window_label, palette.green, palette),
        render_compact_chart("CPU load %", cpu_load_values, cpu_load_now, 100, config.chart_height, chart_width, window_label, palette.yellow, palette),
        render_compact_chart("RAM used %", ram_values, ram_now, 100, config.chart_height, chart_width, window_label, palette.cyan, palette),
    ]
    second_row = [
        render_compact_chart("GPU alloc", gpu_values, float(gpu_now), config.gpu_total, config.chart_height, chart_width, window_label, palette.cyan, palette),
        render_compact_chart("GPU load %", gpu_load_values, gpu_load_now, 100, config.chart_height, chart_width, window_label, palette.yellow, palette),
        render_compact_chart("GPU mem %", gpu_memory_values, gpu_memory_now, 100, config.chart_height, chart_width, window_label, palette.cyan, palette),
    ]
    lines.extend(compose_chart_row(first_row, chart_width))
    lines.append("")
    lines.extend(compose_chart_row(second_row, chart_width))
    return lines


def render_file_line(snapshot: ResourceSnapshot, now: float) -> str:
    if snapshot.error:
        return f"{snapshot.resource:<3} file {snapshot.path} ({snapshot.error})"
    updated = format_duration(now - snapshot.last_updated) if snapshot.last_updated else "-"
    updated_time = snapshot.last_updated_str or format_time(snapshot.last_updated)
    return f"{snapshot.resource:<3} file {snapshot.path} updated {updated} ago ({updated_time})"


def aggregate_by_station(snapshots: Dict[str, ResourceSnapshot]) -> List[Tuple[str, int, int, Set[str]]]:
    by_station: Dict[str, Dict[str, Any]] = {}
    for snapshot in snapshots.values():
        for allocation in snapshot.allocations:
            bucket = by_station.setdefault(allocation.station_id, {"CPU": 0, "GPU": 0, "evals": set()})
            bucket[allocation.resource] += len(allocation.units)
            if allocation.eval_id:
                bucket["evals"].add(allocation.eval_id)
    rows: List[Tuple[str, int, int, Set[str]]] = []
    for station_id, info in by_station.items():
        rows.append((station_id, int(info["CPU"]), int(info["GPU"]), set(info["evals"])))
    return sorted(rows, key=lambda row: (-row[1] - row[2], row[0]))


def visible_len(value: str) -> int:
    return len(ANSI_ESCAPE_RE.sub("", str(value)))


def pad_cell(value: str, width: int, *, right: bool = False) -> str:
    text = str(value)
    padding = max(0, width - visible_len(text))
    return (" " * padding + text) if right else (text + " " * padding)


def format_running_status(station: StationSnapshot, palette: Palette) -> str:
    prefix = (
        f"{station.human_intervention_count} human intervention needed; "
        if station.human_intervention_count
        else ""
    )
    text = prefix + (station.status_summary or "-")
    if not station.human_intervention_count or not palette.enabled:
        return text
    if text.startswith(prefix):
        return f"{palette.yellow}{prefix}{palette.reset}{text[len(prefix):]}"
    return f"{palette.yellow}{text}{palette.reset}"


def render_status_table(stations: Sequence[StationSnapshot], term_width: int, palette: Palette) -> List[str]:
    if not stations:
        return ["Station status monitor", "(no station folders found; use --station-glob to override)"]

    has_external_links = any(station.external_url for station in stations)
    station_width = max(len("STATION"), max(len(station.name) for station in stations))
    tick_width = max(len("TICK"), max(len(format_optional_int(station.current_tick)) for station in stations))
    age_width = max(
        len("AGE"),
        max(len(format_duration(station.seconds_since_last_tick)) for station in stations),
    )
    since_width = max(
        len("SINCE"),
        len("BT"),
        max(len(format_optional_int(station.ticks_since_last_breakthrough)) for station in stations),
    )
    score_width = max(len("SCORE"), max(len(format_top_score(station.top_score)) for station in stations))
    top_tick_width = max(len("TICK"), max(len(format_optional_int(station.top_tick)) for station in stations))
    next_width = max(len("STAG"), max(len(format_optional_int(station.next_stagnation_tick)) for station in stations))
    coders_width = max(len("CODERS"), max(len(str(station.active_coders)) for station in stations))
    status_width = max(len("STATUS"), max(len(station.station_status or "-") for station in stations))
    running_width = max(
        len("RUNNING"),
        max(
            len(
                (
                    f"{station.human_intervention_count} human intervention needed; "
                    if station.human_intervention_count
                    else ""
                )
                + (f"Unavailable: {station.error}" if station.error else station.status_summary or "-")
            )
            for station in stations
        ),
    )
    folder_width = max(len("FOLDER"), max(len(station.display_path) for station in stations))
    link_width = (
        max(len("LINK"), max(len(station.external_url or "") for station in stations))
        if has_external_links
        else 0
    )

    columns: List[Tuple[int, bool]] = [
        (station_width, False),
        (tick_width, True),
        (age_width, True),
        (since_width, True),
        (score_width, True),
        (top_tick_width, True),
        (next_width, True),
        (coders_width, True),
        (status_width, False),
        (running_width, False),
        (folder_width, False),
    ]
    if has_external_links:
        columns.append((link_width, False))

    def render_row(values: Sequence[str]) -> str:
        cells = [
            pad_cell(value, width, right=right)
            for value, (width, right) in zip(values, columns)
        ]
        return " | ".join(cells)

    header1_values = ["", "", "", "SINCE", "TOP", "TOP", "NEXT", "", "", "", ""]
    header2_values = [
        "STATION",
        "TICK",
        "AGE",
        "BT",
        "SCORE",
        "TICK",
        "STAG",
        "CODERS",
        "STATUS",
        "RUNNING",
        "FOLDER",
    ]
    if has_external_links:
        header1_values.append("")
        header2_values.append("LINK")
    header1 = render_row(header1_values)
    header2 = render_row(header2_values)
    lines = ["Station status monitor", header1, header2]

    for station in stations:
        if station.error:
            lines.append(
                render_row(
                    [
                        station.name,
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        f"Unavailable: {station.error}",
                        station.display_path,
                    ]
                    + ([station.external_url or ""] if has_external_links else [])
                )
            )
            continue

        tick_age = f"{format_duration(station.seconds_since_last_tick):>{age_width}}"
        since_breakthrough = f"{format_optional_int(station.ticks_since_last_breakthrough):>{since_width}}"
        running_text = format_running_status(station, palette)
        lines.append(
            render_row(
                [
                    station.name,
                    format_optional_int(station.current_tick),
                    color_tick_age(station.seconds_since_last_tick, tick_age, palette),
                    color_ticks_since_breakthrough(
                        station.ticks_since_last_breakthrough,
                        since_breakthrough,
                        palette,
                    ),
                    format_top_score(station.top_score),
                    format_optional_int(station.top_tick),
                    format_optional_int(station.next_stagnation_tick),
                    str(station.active_coders),
                    station.station_status or "-",
                    running_text,
                    station.display_path,
                ]
                + ([station.external_url or ""] if has_external_links else [])
            )
        )
    separator = "-+-".join("-" * width for width, _right in columns)
    lines.insert(3, separator)
    return lines


def render_station_table(snapshots: Dict[str, ResourceSnapshot], name_by_id: Dict[str, str]) -> List[str]:
    rows = aggregate_by_station(snapshots)
    lines = ["Allocations by station", "STATION              CPU  GPU  EVALS"]
    if not rows:
        lines.append("(no active allocations)")
        return lines
    for station_id, cpu_count, gpu_count, evals in rows:
        eval_text = ",".join(sorted(evals, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item)))
        station_label = station_display_name(station_id, name_by_id)
        lines.append(f"{truncate(station_label, 20):<20} {cpu_count:>3} {gpu_count:>4}  {truncate(eval_text, 40)}")
    return lines


def render_allocation_table(snapshots: Dict[str, ResourceSnapshot], now: float, name_by_id: Dict[str, str]) -> List[str]:
    allocations = sorted(
        [allocation for snapshot in snapshots.values() for allocation in snapshot.allocations],
        key=lambda item: (item.resource, item.station_id, item.eval_id, item.key),
    )
    lines = ["Current allocations", "RES  STATION              EVAL    COUNT  UNITS               AGE       START"]
    if not allocations:
        lines.append("(no active allocations)")
        return lines
    for allocation in allocations:
        age = format_duration(now - allocation.start_time) if allocation.start_time else "-"
        start = allocation.start_time_str or format_time(allocation.start_time)
        station_label = station_display_name(allocation.station_id, name_by_id)
        lines.append(
            f"{allocation.resource:<4} "
            f"{truncate(station_label, 20):<20} "
            f"{truncate(allocation.eval_id, 7):<7} "
            f"{len(allocation.units):>5}  "
            f"{truncate(compact_ranges(allocation.units), 18):<18} "
            f"{age:>8}  "
            f"{truncate(start, 19)}"
        )
    return lines


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.1f}"


def disk_usage_summary(stations: Sequence[StationSnapshot], palette: Palette) -> str:
    paths: List[Path] = []
    seen_devices: Set[int] = set()
    for station in stations:
        try:
            stat = station.path.stat()
        except OSError:
            continue
        if stat.st_dev in seen_devices:
            continue
        seen_devices.add(stat.st_dev)
        paths.append(station.path)
    if not paths:
        paths = [Path.home()]

    parts: List[str] = []
    for path in paths[:3]:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        percent = 100.0 * usage.used / usage.total if usage.total else 0.0
        percent_text = color_disk_usage(percent, f"{percent:.1f}%", palette)
        parts.append(
            f"{display_path(path)} {percent_text} "
            f"({format_gib(usage.used)}/{format_gib(usage.total)} GiB)"
        )
    if len(paths) > 3:
        parts.append(f"+{len(paths) - 3} disk(s)")
    return "Disk: " + ("; ".join(parts) if parts else "unavailable")


def render_dashboard(
    config: MonitorConfig,
    stations: Sequence[StationSnapshot],
    snapshots: Dict[str, ResourceSnapshot],
    samples: Deque[UsageSample],
) -> str:
    now = time.time()
    term_width = shutil.get_terminal_size((120, 30)).columns
    palette = Palette(config.color)
    name_by_id = station_name_map(stations)

    lines: List[str] = []
    title = "Station monitor"
    mode = "snapshot" if config.once else f"live every {config.interval:g}s"
    lines.append(
        f"{palette.bold}{title}{palette.reset}  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
        f"{mode}  {disk_usage_summary(stations, palette)}"
    )
    lines.append("")
    lines.extend(render_status_table(stations, term_width, palette))
    lines.append("")
    chart_lines = render_usage_chart(config, snapshots, list(samples), term_width, palette)
    if chart_lines:
        lines.extend(chart_lines)
        lines.append("")
    lines.extend(render_station_table(snapshots, name_by_id))
    lines.append("")
    lines.extend(render_allocation_table(snapshots, now, name_by_id))
    lines.append("")
    lines.append("Recent history is sampled while this process runs; coordination files store active allocations only.")
    if not config.once:
        lines.append("Press Ctrl-C to exit.")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Station status and CPU/GPU coordination files")
    parser.add_argument("--root", default=None, help="Directory containing station_cpu_used.json and station_gpu_used.json (default: /tmp)")
    parser.add_argument(
        "--station-glob",
        action="append",
        nargs="+",
        default=None,
        help='Station folder glob/path to monitor; accepts shell-expanded paths and is repeatable (default: "~/station" and "~/station_*")',
    )
    parser.add_argument("--cpu-file", default=None, help="Exact CPU coordination JSON path")
    parser.add_argument("--gpu-file", default=None, help="Exact GPU coordination JSON path")
    parser.add_argument("--cpu-ids", default=None, help='Coordinated CPU IDs, e.g. "0-95,128-191"')
    parser.add_argument("--gpu-ids", default=None, help='Coordinated GPU IDs, e.g. "0-7"')
    parser.add_argument("--cpu-total", type=int, default=None, help="CPU capacity if --cpu-ids is not provided")
    parser.add_argument("--gpu-total", type=int, default=None, help="GPU capacity if --gpu-ids is not provided")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds for live mode")
    parser.add_argument("--history", type=int, default=60, help="Number of samples to keep for recent usage")
    parser.add_argument("--chart-window", type=float, default=3600.0, help="Seconds of usage to show in the rolling chart")
    parser.add_argument("--chart-buckets", type=int, default=60, help="Maximum time buckets to draw in the rolling chart")
    parser.add_argument("--chart-height", type=int, default=8, help="Rows of height for each resource chart")
    parser.add_argument("--no-chart", action="store_true", help="Hide the rolling usage chart")
    parser.add_argument("--lock-timeout", type=float, default=1.0, help="Seconds to wait for a shared file lock")
    parser.add_argument("--external-host", default=DEFAULT_EXTERNAL_HOST, help="Host/IP to use for external station links")
    parser.add_argument("--external-scheme", default=DEFAULT_EXTERNAL_SCHEME, help="Scheme to use for external station links")
    parser.add_argument(
        "--external-port-base",
        type=int,
        default=None,
        help="Fallback HTTPS port for the base station folder when .env/nginx config has no port",
    )
    parser.add_argument("--external-port-step", type=int, default=1, help="Fallback port increment when --external-port-base is used")
    parser.add_argument("--external-probe-timeout", type=float, default=1.0, help="Seconds to wait for each external host probe")
    parser.add_argument("--no-external-links", action="store_true", help="Hide external station links")
    parser.add_argument(
        "--local-api-timeout",
        type=float,
        default=DEFAULT_LOCAL_API_TIMEOUT_SECONDS,
        help="Seconds to wait for each station's local /api/orchestrator/status request",
    )
    parser.add_argument(
        "--no-local-api-status",
        action="store_true",
        help="Do not fetch local dashboard status; show only station_config fallback status",
    )
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between live refreshes")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    samples: Deque[UsageSample] = deque(maxlen=config.history_size)
    previous_cpu_times: Optional[Tuple[int, int]] = None

    try:
        while True:
            now = time.time()
            snapshots = collect_snapshots(config)
            stations = collect_station_snapshots(config, now)
            cpu_load_percent, previous_cpu_times = sample_cpu_load_percent(previous_cpu_times)
            ram_used_percent = sample_ram_used_percent()
            gpu_load_percent, gpu_memory_percent = sample_gpu_metrics()
            samples.append(
                UsageSample(
                    timestamp=now,
                    cpu_used=len(snapshots["CPU"].used_units),
                    gpu_used=len(snapshots["GPU"].used_units),
                    cpu_load_percent=cpu_load_percent,
                    gpu_load_percent=gpu_load_percent,
                    ram_used_percent=ram_used_percent,
                    gpu_memory_percent=gpu_memory_percent,
                )
            )

            frame = render_dashboard(config, stations, snapshots, samples)
            if config.clear and not config.once and sys.stdout.isatty():
                print("\033[H\033[J", end="")
            print(frame, flush=True)

            if config.once:
                return 0
            time.sleep(config.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
