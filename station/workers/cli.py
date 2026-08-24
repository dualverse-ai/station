"""
Generic CLI worker backend definitions.

This module owns the shared Codex/Claude CLI launch layer. Specialized systems
such as the Research Center coder and the Archive surveyor provide their own
prompts, state machines, storage layouts, and report contracts on top of this
backend layer.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
import tomllib
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

from station import constants
from station import runtime_api_config
from station.eval_research.evaluation_helpers import resolve_conda_env


DEFAULT_CODEX_PROVIDER_DOMAIN = "api.openai.com"
CODEX_ALT_PROVIDER_NAME = "alt"


def _provider_domain_from_base_url(raw_base_url: object) -> Optional[str]:
    value = str(raw_base_url or "").strip()
    if not value:
        return None
    try:
        parsed = urlparse(value if "://" in value else f"//{value}")
        hostname = parsed.hostname
    except (TypeError, ValueError):
        return None
    if not hostname:
        return None
    try:
        return hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None


def get_codex_provider_domains(runtime_env: Optional[Dict[str, str]] = None) -> List[str]:
    """Return every trusted Codex API hostname available to restricted workers."""

    env = runtime_env if runtime_env is not None else os.environ
    domains = {DEFAULT_CODEX_PROVIDER_DOMAIN}

    for env_name in ("CODEX_BASE_URL", "OPENAI_BASE_URL"):
        explicit_domain = _provider_domain_from_base_url(env.get(env_name))
        if explicit_domain:
            domains.add(explicit_domain)

    codex_home = str(env.get("CODEX_HOME") or os.path.expanduser("~/.codex"))
    config_path = os.path.join(os.path.expanduser(codex_home), "config.toml")
    try:
        with open(config_path, "rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return sorted(domains)

    top_level_domain = _provider_domain_from_base_url(config.get("openai_base_url"))
    if top_level_domain:
        domains.add(top_level_domain)

    providers = config.get("model_providers")
    if isinstance(providers, dict):
        for provider in providers.values():
            if not isinstance(provider, dict):
                continue
            domain = _provider_domain_from_base_url(provider.get("base_url"))
            if domain:
                domains.add(domain)

    return sorted(domains)


def build_codex_restricted_web_config_args(
    runtime_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build the shared provider-only network policy for non-web Codex workers."""

    domain_rules = ", ".join(
        f"{json.dumps(domain)} = \"allow\""
        for domain in get_codex_provider_domains(runtime_env)
    )
    return [
        "--config",
        'web_search="disabled"',
        "--config",
        "sandbox_workspace_write.network_access=true",
        "--config",
        "features.network_proxy.enabled=true",
        "--config",
        f"features.network_proxy.domains={{ {domain_rules} }}",
    ]


def build_codex_alt_provider_config_args(
    runtime_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Build an explicit Codex provider from the dedicated CODEX_* environment."""

    env = runtime_env if runtime_env is not None else os.environ
    api_key = str(env.get("CODEX_API_KEY") or "").strip()
    base_url = str(env.get("CODEX_BASE_URL") or "").strip()
    if not api_key and not base_url:
        return []
    if not api_key or not base_url:
        missing = "CODEX_API_KEY" if not api_key else "CODEX_BASE_URL"
        raise ValueError(
            "Incomplete Codex API override: CODEX_API_KEY and CODEX_BASE_URL "
            f"must be set together (missing {missing})."
        )

    provider_config = (
        f"model_providers.{CODEX_ALT_PROVIDER_NAME}={{ "
        f'name="Alt", base_url={json.dumps(base_url)}, '
        'env_key="CODEX_API_KEY", wire_api="responses" }'
    )
    return [
        "--config",
        f'model_provider="{CODEX_ALT_PROVIDER_NAME}"',
        "--config",
        provider_config,
    ]


def _extend_add_dir_args(command: List[str], workspace_root: str, dir_paths: List[str]) -> None:
    workspace_real = os.path.realpath(workspace_root)
    for dir_path in dir_paths:
        if not dir_path:
            continue
        real_dir = os.path.realpath(dir_path)
        try:
            common_root = os.path.commonpath([workspace_real, real_dir])
        except ValueError:
            common_root = ""
        if common_root != workspace_real:
            command.extend(["--add-dir", real_dir])


def _env_override_or_unset(override_key: str) -> Optional[str]:
    return runtime_api_config.get_env_value(override_key)


def _resolve_workspace_root(workspace_root: Optional[str], research_root: Optional[str]) -> str:
    resolved = workspace_root or research_root
    if not resolved:
        raise ValueError("workspace_root is required")
    return resolved


def candidate_js_bin_dirs() -> List[str]:
    candidates: List[str] = []
    raw_candidates = glob.glob("/home/*/.nvm/versions/node/*/bin") + glob.glob(
        os.path.expanduser("~/.nvm/versions/node/*/bin")
    )
    for candidate in raw_candidates:
        if os.path.isdir(candidate) and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def build_cli_worker_runtime_env(conda_env_name: str) -> Dict[str, str]:
    env = os.environ.copy()
    if constants.LLM_HTTP_PROXY:
        env["http_proxy"] = constants.LLM_HTTP_PROXY
        env["HTTP_PROXY"] = constants.LLM_HTTP_PROXY
    if constants.LLM_HTTPS_PROXY:
        env["https_proxy"] = constants.LLM_HTTPS_PROXY
        env["HTTPS_PROXY"] = constants.LLM_HTTPS_PROXY
    path_parts = [part for part in str(env.get("PATH", "")).split(os.pathsep) if part]
    for bin_dir in reversed(candidate_js_bin_dirs()):
        if bin_dir not in path_parts:
            path_parts.insert(0, bin_dir)
    env["PATH"] = os.pathsep.join(path_parts)
    resolve_conda_env(conda_env_name, env)
    return env


def apply_codex_proxy_overrides(env: Dict[str, str]) -> None:
    codex_proxies = runtime_api_config.get_codex_proxy_overrides()
    http_proxy = codex_proxies.get("http_proxy")
    https_proxy = codex_proxies.get("https_proxy")
    if http_proxy:
        env["http_proxy"] = http_proxy
        env["HTTP_PROXY"] = http_proxy
    if https_proxy:
        env["https_proxy"] = https_proxy
        env["HTTPS_PROXY"] = https_proxy


def detect_cli_worker_executable(backend: str, env: Dict[str, str]) -> str:
    backend = backend.lower()
    env_var = "CODEX_BIN_PATH" if backend == "codex" else "CLAUDE_BIN_PATH"
    candidate = env.get(env_var) or os.environ.get(env_var)
    if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
        return candidate

    executable_name = "codex" if backend == "codex" else "claude"
    candidate = shutil.which(executable_name, path=env.get("PATH"))
    if candidate:
        return candidate
    for bin_dir in candidate_js_bin_dirs():
        candidate = os.path.join(bin_dir, executable_name)
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(f"{backend} executable not found. Set {env_var} or add it to PATH.")


@dataclass(frozen=True)
class PreparedCliWorkerLaunch:
    backend: str
    command: List[str]
    stdin_text: Optional[str]
    transcript_path: str
    stderr_path: str
    last_message_path: Optional[str]
    transcript_format: str
    env_overrides: Optional[Dict[str, Optional[str]]] = None


@dataclass(frozen=True)
class TranscriptGrowthCheck:
    applies: bool
    current_size: int
    last_growth_timestamp: float
    idle_seconds: float
    timeout_seconds: float
    timed_out: bool


def _safe_transcript_size(transcript_path: str) -> int:
    if not transcript_path:
        return 0
    try:
        return int(os.path.getsize(transcript_path))
    except OSError:
        return 0


def check_cli_worker_transcript_growth_timeout(
    *,
    backend: str,
    transcript_path: str,
    last_size: int,
    last_growth_timestamp: float,
    now: Optional[float] = None,
) -> TranscriptGrowthCheck:
    timeout_seconds = float(getattr(constants, "CLI_WORKER_TRANSCRIPT_IDLE_TIMEOUT_SECONDS", 0) or 0)
    current_time = time.time() if now is None else float(now)
    current_size = _safe_transcript_size(transcript_path)
    try:
        previous_size = int(last_size or 0)
    except (TypeError, ValueError):
        previous_size = 0
    try:
        previous_growth_timestamp = float(last_growth_timestamp or current_time)
    except (TypeError, ValueError):
        previous_growth_timestamp = current_time
    if previous_growth_timestamp <= 0:
        previous_growth_timestamp = current_time

    # Transcript-growth liveness is Codex-specific. Claude also writes a
    # transcript, but its streaming behavior is different enough that the
    # generic CLI timeout should not infer liveness from size growth there.
    applies = str(backend or "").strip().lower() == "codex" and timeout_seconds > 0
    if not applies:
        return TranscriptGrowthCheck(
            applies=False,
            current_size=current_size,
            last_growth_timestamp=previous_growth_timestamp,
            idle_seconds=0.0,
            timeout_seconds=timeout_seconds,
            timed_out=False,
        )

    if current_size != previous_size:
        return TranscriptGrowthCheck(
            applies=True,
            current_size=current_size,
            last_growth_timestamp=current_time,
            idle_seconds=0.0,
            timeout_seconds=timeout_seconds,
            timed_out=False,
        )

    idle_seconds = max(0.0, current_time - previous_growth_timestamp)
    return TranscriptGrowthCheck(
        applies=True,
        current_size=current_size,
        last_growth_timestamp=previous_growth_timestamp,
        idle_seconds=idle_seconds,
        timeout_seconds=timeout_seconds,
        timed_out=idle_seconds >= timeout_seconds,
    )


class BaseCliWorkerBackend:
    backend_name = "base"
    transcript_filename = "stdout.txt"
    transcript_format = "text"
    supports_resume = False

    def prepare_launch(
        self,
        *,
        executable: str,
        run_dir: str,
        model_name: Optional[str],
        storage_root: str,
        prompt: str,
        workspace_root: Optional[str] = None,
        research_root: Optional[str] = None,
        extra_allowed_roots: Optional[List[str]] = None,
        network_access: bool = False,
    ) -> PreparedCliWorkerLaunch:
        raise NotImplementedError

    def prepare_resume_launch(
        self,
        *,
        executable: str,
        run_dir: str,
        model_name: Optional[str],
        storage_root: str,
        resume_token: str,
        prompt: str,
        workspace_root: Optional[str] = None,
        research_root: Optional[str] = None,
        extra_allowed_roots: Optional[List[str]] = None,
        network_access: bool = False,
    ) -> PreparedCliWorkerLaunch:
        raise NotImplementedError(f"{self.backend_name} backend does not support resume")

    def extract_resume_token(self, transcript_path: str) -> Optional[str]:
        return None


class CodexCliWorkerBackend(BaseCliWorkerBackend):
    backend_name = "codex"
    transcript_filename = "transcript.jsonl"
    transcript_format = "jsonl"
    supports_resume = True

    @staticmethod
    def _build_env_overrides(
        runtime_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Optional[str]]:
        if build_codex_alt_provider_config_args(runtime_env):
            return {
                "OPENAI_BASE_URL": None,
                "OPENAI_API_KEY": None,
                "CODEX_ACCESS_TOKEN": None,
                "CODEX_AUTH": None,
            }
        # Preserve the pre-existing no-override behavior exactly: when neither
        # dedicated CODEX_* variable is set, remove inherited OPENAI_* values.
        return {
            "OPENAI_BASE_URL": None,
            "OPENAI_API_KEY": None,
        }

    def prepare_launch(
        self,
        *,
        executable: str,
        run_dir: str,
        model_name: Optional[str],
        storage_root: str,
        prompt: str,
        workspace_root: Optional[str] = None,
        research_root: Optional[str] = None,
        extra_allowed_roots: Optional[List[str]] = None,
        network_access: bool = False,
        runtime_env: Optional[Dict[str, str]] = None,
    ) -> PreparedCliWorkerLaunch:
        resolved_workspace_root = _resolve_workspace_root(workspace_root, research_root)
        last_message_path = os.path.join(run_dir, "last_message.txt")
        command = [executable]
        command.extend(build_codex_alt_provider_config_args(runtime_env))
        if network_access:
            command.extend([
                "--search",
                "--config",
                "sandbox_workspace_write.network_access=true",
            ])
        else:
            command.extend(build_codex_restricted_web_config_args(runtime_env))
        command.extend([
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--cd",
            resolved_workspace_root,
            "--json",
            "--output-last-message",
            last_message_path,
        ])
        if model_name:
            command.extend(["--model", model_name])

        _extend_add_dir_args(command, resolved_workspace_root, [storage_root] + list(extra_allowed_roots or []))
        command.append("-")

        return PreparedCliWorkerLaunch(
            backend=self.backend_name,
            command=command,
            stdin_text=prompt,
            transcript_path=os.path.join(run_dir, self.transcript_filename),
            stderr_path=os.path.join(run_dir, "stderr.txt"),
            last_message_path=last_message_path,
            transcript_format=self.transcript_format,
            env_overrides=self._build_env_overrides(runtime_env),
        )

    def prepare_resume_launch(
        self,
        *,
        executable: str,
        run_dir: str,
        model_name: Optional[str],
        storage_root: str,
        resume_token: str,
        prompt: str,
        workspace_root: Optional[str] = None,
        research_root: Optional[str] = None,
        extra_allowed_roots: Optional[List[str]] = None,
        network_access: bool = False,
        runtime_env: Optional[Dict[str, str]] = None,
    ) -> PreparedCliWorkerLaunch:
        resolved_workspace_root = _resolve_workspace_root(workspace_root, research_root)
        last_message_path = os.path.join(run_dir, "last_message.txt")
        command = [executable]
        command.extend(build_codex_alt_provider_config_args(runtime_env))
        if network_access:
            command.extend([
                "--search",
                "--config",
                "sandbox_workspace_write.network_access=true",
            ])
        else:
            command.extend(build_codex_restricted_web_config_args(runtime_env))
        command.extend([
            "--cd",
            resolved_workspace_root,
            "--sandbox",
            "workspace-write",
        ])
        # `codex exec resume --add-dir ...` is rejected by the resume
        # subcommand parser, but the same workspace flags are accepted as
        # top-level Codex options before `exec resume`.
        _extend_add_dir_args(command, resolved_workspace_root, [storage_root] + list(extra_allowed_roots or []))
        command.extend([
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--full-auto",
            "--json",
            "--output-last-message",
            last_message_path,
        ])
        if model_name:
            command.extend(["--model", model_name])
        command.extend([resume_token, "-"])

        return PreparedCliWorkerLaunch(
            backend=self.backend_name,
            command=command,
            stdin_text=prompt,
            transcript_path=os.path.join(run_dir, self.transcript_filename),
            stderr_path=os.path.join(run_dir, "stderr.txt"),
            last_message_path=last_message_path,
            transcript_format=self.transcript_format,
            env_overrides=self._build_env_overrides(runtime_env),
        )

    def extract_resume_token(self, transcript_path: str) -> Optional[str]:
        if not transcript_path or not os.path.exists(transcript_path):
            return None
        try:
            with open(transcript_path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if str(payload.get("type", "")).strip() != "thread.started":
                        continue
                    token = str(payload.get("thread_id", "")).strip()
                    if token:
                        return token
        except Exception:
            return None
        return None


class ClaudeCliWorkerBackend(BaseCliWorkerBackend):
    backend_name = "claude"
    transcript_filename = "transcript.jsonl"
    transcript_format = "jsonl"

    @staticmethod
    def _build_env_overrides() -> Dict[str, Optional[str]]:
        return {
            "ANTHROPIC_BASE_URL": _env_override_or_unset("CLAUDE_CODE_BASE_URL"),
            "ANTHROPIC_API_KEY": _env_override_or_unset("CLAUDE_CODE_API_KEY"),
        }

    def prepare_launch(
        self,
        *,
        executable: str,
        run_dir: str,
        model_name: Optional[str],
        storage_root: str,
        prompt: str,
        workspace_root: Optional[str] = None,
        research_root: Optional[str] = None,
        extra_allowed_roots: Optional[List[str]] = None,
        network_access: bool = False,
    ) -> PreparedCliWorkerLaunch:
        resolved_workspace_root = _resolve_workspace_root(workspace_root, research_root)
        command = [
            executable,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "bypassPermissions",
        ]
        if model_name:
            command.extend(["--model", model_name])
        _extend_add_dir_args(command, resolved_workspace_root, [storage_root] + list(extra_allowed_roots or []))
        command.append(prompt)
        return PreparedCliWorkerLaunch(
            backend=self.backend_name,
            command=command,
            stdin_text=None,
            transcript_path=os.path.join(run_dir, self.transcript_filename),
            stderr_path=os.path.join(run_dir, "stderr.txt"),
            last_message_path=None,
            transcript_format=self.transcript_format,
            env_overrides=self._build_env_overrides(),
        )


def get_cli_worker_backend(backend_name: str) -> BaseCliWorkerBackend:
    backend = str(backend_name or "").strip().lower()
    if backend == "codex":
        return CodexCliWorkerBackend()
    if backend == "claude":
        return ClaudeCliWorkerBackend()
    raise ValueError(f"Unsupported CLI worker backend: {backend_name}")
