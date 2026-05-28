"""Thread-safe runtime API/proxy configuration for the active Station process.

This module intentionally does not persist API keys. It reflects launch-time
environment values until the dashboard applies an in-memory runtime override.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from station import constants


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    label: str
    base_url_env: str
    api_key_env: str
    http_proxy_env: str
    https_proxy_env: str
    default_base_url: str


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        provider_id="openai",
        label="OpenAI",
        base_url_env="OPENAI_BASE_URL",
        api_key_env="OPENAI_API_KEY",
        http_proxy_env="OPENAI_HTTP_PROXY",
        https_proxy_env="OPENAI_HTTPS_PROXY",
        default_base_url="https://api.openai.com/v1",
    ),
    "claude": ProviderSpec(
        provider_id="claude",
        label="Claude",
        base_url_env="ANTHROPIC_BASE_URL",
        api_key_env="ANTHROPIC_API_KEY",
        http_proxy_env="ANTHROPIC_HTTP_PROXY",
        https_proxy_env="ANTHROPIC_HTTPS_PROXY",
        default_base_url="https://api.anthropic.com",
    ),
    "gemini": ProviderSpec(
        provider_id="gemini",
        label="Gemini",
        base_url_env="GOOGLE_GEMINI_BASE_URL",
        api_key_env="GOOGLE_API_KEY",
        http_proxy_env="GOOGLE_GEMINI_HTTP_PROXY",
        https_proxy_env="GOOGLE_GEMINI_HTTPS_PROXY",
        default_base_url="https://generativelanguage.googleapis.com",
    ),
    "grok": ProviderSpec(
        provider_id="grok",
        label="Grok",
        base_url_env="XAI_BASE_URL",
        api_key_env="XAI_API_KEY",
        http_proxy_env="XAI_HTTP_PROXY",
        https_proxy_env="XAI_HTTPS_PROXY",
        default_base_url="https://api.x.ai",
    ),
}

CODEX_CONFIG = {
    "base_url_env": "CODEX_BASE_URL",
    "api_key_env": "CODEX_API_KEY",
    "http_proxy_env": "CODEX_HTTP_PROXY",
    "https_proxy_env": "CODEX_HTTPS_PROXY",
    "default_base_url": "",
}

EXTERNAL_COUNTER_CONFIG = {
    "base_url_env": "EXTERNAL_OPENAI_BASE_URL",
    "api_key_env": "EXTERNAL_OPENAI_API_KEY",
    "http_proxy_env": "EXTERNAL_HTTP_PROXY",
    "https_proxy_env": "EXTERNAL_HTTPS_PROXY",
    "default_base_url": "https://api.openai.com/v1",
}

_LOCK = threading.RLock()
_GENERATION = 0
PROVIDER_FALLBACK_WINDOW = 10
PROVIDER_FALLBACK_FAILURE_THRESHOLD = 0.7
PROVIDER_FALLBACK_RESULT_MAX_AGE_SECONDS = 3600
PROVIDER_BASE_RECOVERY_CHECK_INTERVAL_SECONDS = 1800
_PROVIDER_FALLBACK_STATE: Dict[str, Dict[str, Any]] = {}


def get_generation() -> int:
    with _LOCK:
        return _GENERATION


def _bump_generation() -> int:
    global _GENERATION
    _GENERATION += 1
    return _GENERATION


def get_env_value(name: str) -> Optional[str]:
    with _LOCK:
        return os.environ.get(name)


def get_env_values(names: Iterable[str]) -> Dict[str, Optional[str]]:
    with _LOCK:
        return {name: os.environ.get(name) for name in names}


def _backup_env_name(env_name: str) -> str:
    return f"BACKUP_{env_name}"


def _split_backup_env(env_name: str) -> List[str]:
    raw_value = os.environ.get(_backup_env_name(env_name))
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(";")]


def _split_backup_value(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(";")]


def _validate_provider_backup_values(
    spec: ProviderSpec,
    values: Dict[str, Optional[str]],
) -> List[str]:
    api_keys = _split_backup_value(values.get(_backup_env_name(spec.api_key_env)))
    base_urls = _split_backup_value(values.get(_backup_env_name(spec.base_url_env)))
    http_proxies = _split_backup_value(values.get(_backup_env_name(spec.http_proxy_env)))
    https_proxies = _split_backup_value(values.get(_backup_env_name(spec.https_proxy_env)))

    optional_fields = [
        (_backup_env_name(spec.base_url_env), base_urls),
        (_backup_env_name(spec.http_proxy_env), http_proxies),
        (_backup_env_name(spec.https_proxy_env), https_proxies),
    ]
    errors: List[str] = []
    optional_present = any(values.get(name) not in (None, "") for name, _items in optional_fields)

    if not api_keys:
        if optional_present:
            errors.append(
                f"{spec.provider_id}: backup Base URL/proxy values require "
                f"{_backup_env_name(spec.api_key_env)}."
            )
        return errors

    if any(not item for item in api_keys):
        errors.append(f"{spec.provider_id}: {_backup_env_name(spec.api_key_env)} cannot contain blank entries.")

    for env_name, items in optional_fields:
        if items and len(items) != len(api_keys):
            errors.append(
                f"{spec.provider_id}: {env_name} has {len(items)} entr"
                f"{'y' if len(items) == 1 else 'ies'}, but {_backup_env_name(spec.api_key_env)} "
                f"has {len(api_keys)}. Use ';' delimiters so every backup endpoint has one value; "
                "blank values are allowed for Base URL/proxy."
            )

    return errors


def validate_provider_backup_env_config() -> None:
    """Validate all BACKUP_* provider env vars and raise on malformed config."""
    with _LOCK:
        all_errors: List[str] = []
        for spec in PROVIDER_SPECS.values():
            values = {
                _backup_env_name(spec.api_key_env): os.environ.get(_backup_env_name(spec.api_key_env)),
                _backup_env_name(spec.base_url_env): os.environ.get(_backup_env_name(spec.base_url_env)),
                _backup_env_name(spec.http_proxy_env): os.environ.get(_backup_env_name(spec.http_proxy_env)),
                _backup_env_name(spec.https_proxy_env): os.environ.get(_backup_env_name(spec.https_proxy_env)),
            }
            all_errors.extend(_validate_provider_backup_values(spec, values))
        if all_errors:
            raise ValueError("Invalid provider backup API configuration: " + " ".join(all_errors))


def _build_provider_endpoints_locked(provider_id: str) -> List[Dict[str, Any]]:
    spec = PROVIDER_SPECS.get(str(provider_id or "").strip().lower())
    if not spec:
        return []
    backup_env_values = {
        _backup_env_name(spec.api_key_env): os.environ.get(_backup_env_name(spec.api_key_env)),
        _backup_env_name(spec.base_url_env): os.environ.get(_backup_env_name(spec.base_url_env)),
        _backup_env_name(spec.http_proxy_env): os.environ.get(_backup_env_name(spec.http_proxy_env)),
        _backup_env_name(spec.https_proxy_env): os.environ.get(_backup_env_name(spec.https_proxy_env)),
    }
    backup_errors = _validate_provider_backup_values(spec, backup_env_values)
    if backup_errors:
        raise ValueError("Invalid provider backup API configuration: " + " ".join(backup_errors))

    base_endpoint = {
        "provider_id": spec.provider_id,
        "index": 0,
        "name": "base",
        "is_base": True,
        "base_url": os.environ.get(spec.base_url_env),
        "api_key": os.environ.get(spec.api_key_env),
        "api_key_env": spec.api_key_env,
        "base_url_env": spec.base_url_env,
        "http_proxy": os.environ.get(spec.http_proxy_env) or getattr(constants, "LLM_HTTP_PROXY", None),
        "https_proxy": os.environ.get(spec.https_proxy_env) or getattr(constants, "LLM_HTTPS_PROXY", None),
        "http_proxy_env": spec.http_proxy_env,
        "https_proxy_env": spec.https_proxy_env,
        "valid": bool(os.environ.get(spec.api_key_env)),
    }
    endpoints: List[Dict[str, Any]] = [base_endpoint]

    backup_base_urls = _split_backup_env(spec.base_url_env)
    backup_api_keys = _split_backup_env(spec.api_key_env)
    backup_http_proxies = _split_backup_env(spec.http_proxy_env)
    backup_https_proxies = _split_backup_env(spec.https_proxy_env)
    backup_count = max(
        len(backup_base_urls),
        len(backup_api_keys),
        len(backup_http_proxies),
        len(backup_https_proxies),
    )

    for offset in range(backup_count):
        api_key = backup_api_keys[offset] if offset < len(backup_api_keys) else ""
        if not api_key:
            continue
        endpoints.append({
            "provider_id": spec.provider_id,
            "index": offset + 1,
            "name": f"backup_{offset + 1}",
            "is_base": False,
            "base_url": backup_base_urls[offset] if offset < len(backup_base_urls) else "",
            "api_key": api_key,
            "api_key_env": _backup_env_name(spec.api_key_env),
            "base_url_env": _backup_env_name(spec.base_url_env),
            "http_proxy": (
                backup_http_proxies[offset]
                if offset < len(backup_http_proxies) and backup_http_proxies[offset]
                else getattr(constants, "LLM_HTTP_PROXY", None)
            ),
            "https_proxy": (
                backup_https_proxies[offset]
                if offset < len(backup_https_proxies) and backup_https_proxies[offset]
                else getattr(constants, "LLM_HTTPS_PROXY", None)
            ),
            "http_proxy_env": _backup_env_name(spec.http_proxy_env),
            "https_proxy_env": _backup_env_name(spec.https_proxy_env),
            "valid": True,
        })
    return endpoints


def _get_provider_state_locked(provider_id: str) -> Dict[str, Any]:
    state = _PROVIDER_FALLBACK_STATE.setdefault(str(provider_id).lower(), {
        "default_index": 0,
        "endpoint_results": {},
        "last_base_probe_at": 0.0,
        "base_probe_in_progress": False,
        "default_reason": "initial",
        "default_changed_at": 0.0,
    })
    endpoints = _build_provider_endpoints_locked(provider_id)
    valid_indices = {endpoint["index"] for endpoint in endpoints if endpoint.get("valid", True)}
    if valid_indices and state.get("default_index", 0) not in valid_indices:
        state["default_index"] = min(valid_indices) if valid_indices else 0
        state["default_reason"] = "fallback_config_changed"
        state["default_changed_at"] = time.time()
    elif not valid_indices and state.get("default_index", 0) != 0:
        state["default_index"] = 0
        state["default_reason"] = "fallback_config_changed"
        state["default_changed_at"] = time.time()
    return state


def _select_provider_endpoint_locked(provider_id: str, endpoint_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
    endpoints = _build_provider_endpoints_locked(provider_id)
    if not endpoints:
        return None
    state = _get_provider_state_locked(provider_id)
    selected_index = state.get("default_index", 0) if endpoint_index is None else endpoint_index
    for endpoint in endpoints:
        if int(endpoint.get("index", -1)) == int(selected_index) and endpoint.get("valid", True):
            return endpoint
    for endpoint in endpoints:
        if endpoint.get("valid", True):
            return endpoint
    return endpoints[0]


def _endpoint_public_info(endpoint: Dict[str, Any], endpoint_count: int, configured: bool) -> Dict[str, Any]:
    return {
        "provider_id": endpoint.get("provider_id"),
        "index": endpoint.get("index", 0),
        "name": endpoint.get("name", "base"),
        "is_base": bool(endpoint.get("is_base")),
        "configured": configured,
        "endpoint_count": endpoint_count,
    }


def _build_config_snapshot_locked(
    env_names: Iterable[str],
    provider_id: Optional[str],
    endpoint_index: Optional[int] = None,
) -> Dict[str, Any]:
    spec = PROVIDER_SPECS.get(str(provider_id or "").strip().lower()) if provider_id else None
    if not spec:
        proxy_values = {
            "http_proxy": getattr(constants, "LLM_HTTP_PROXY", None),
            "https_proxy": getattr(constants, "LLM_HTTPS_PROXY", None),
        }
        return {
            "generation": _GENERATION,
            "env": {name: os.environ.get(name) for name in env_names},
            "http_proxy": proxy_values.get("http_proxy"),
            "https_proxy": proxy_values.get("https_proxy"),
        }

    endpoints = _build_provider_endpoints_locked(spec.provider_id)
    endpoint = _select_provider_endpoint_locked(spec.provider_id, endpoint_index)
    if endpoint is None:
        endpoint = {
            "index": 0,
            "name": "base",
            "is_base": True,
            "provider_id": spec.provider_id,
            "base_url": os.environ.get(spec.base_url_env),
            "api_key": os.environ.get(spec.api_key_env),
            "http_proxy": os.environ.get(spec.http_proxy_env) or getattr(constants, "LLM_HTTP_PROXY", None),
            "https_proxy": os.environ.get(spec.https_proxy_env) or getattr(constants, "LLM_HTTPS_PROXY", None),
        }
        endpoints = [endpoint]

    env: Dict[str, Optional[str]] = {}
    for name in env_names:
        if name == spec.api_key_env:
            env[name] = endpoint.get("api_key")
        elif name == spec.base_url_env:
            env[name] = endpoint.get("base_url")
        elif name == spec.http_proxy_env:
            env[name] = endpoint.get("http_proxy")
        elif name == spec.https_proxy_env:
            env[name] = endpoint.get("https_proxy")
        else:
            env[name] = os.environ.get(name)

    return {
        "generation": _GENERATION,
        "env": env,
        "http_proxy": endpoint.get("http_proxy"),
        "https_proxy": endpoint.get("https_proxy"),
        "provider_endpoint": _endpoint_public_info(
            endpoint,
            endpoint_count=len(endpoints),
            configured=len(endpoints) > 1,
        ),
    }


def get_provider_default_endpoint(provider_id: str) -> Dict[str, Any]:
    with _LOCK:
        return _build_config_snapshot_locked((), provider_id).get("provider_endpoint", {})


def _append_provider_result_locked(provider_id: str, endpoint_index: int, success: bool) -> None:
    state = _get_provider_state_locked(provider_id)
    results_by_endpoint = state.setdefault("endpoint_results", {})
    key = str(int(endpoint_index))
    history = results_by_endpoint.setdefault(key, [])
    cutoff = time.time() - PROVIDER_FALLBACK_RESULT_MAX_AGE_SECONDS
    history[:] = [
        item for item in history
        if isinstance(item, dict) and float(item.get("at", 0.0)) >= cutoff
    ]
    history.append({"at": time.time(), "success": bool(success)})
    if len(history) > PROVIDER_FALLBACK_WINDOW:
        del history[:-PROVIDER_FALLBACK_WINDOW]


def _next_provider_endpoint_index_locked(provider_id: str, current_index: int) -> int:
    endpoints = [
        endpoint for endpoint in _build_provider_endpoints_locked(provider_id)
        if endpoint.get("valid", True)
    ]
    if not endpoints:
        return 0
    indices = [int(endpoint["index"]) for endpoint in endpoints]
    if current_index not in indices:
        current_index = _get_provider_state_locked(provider_id).get("default_index", 0)
    try:
        position = indices.index(int(current_index))
    except ValueError:
        position = 0
    return indices[(position + 1) % len(indices)]


def _maybe_promote_default_after_failure_locked(provider_id: str, failed_endpoint_index: int) -> bool:
    state = _get_provider_state_locked(provider_id)
    if int(state.get("default_index", 0)) != int(failed_endpoint_index):
        return False
    endpoints = [
        endpoint for endpoint in _build_provider_endpoints_locked(provider_id)
        if endpoint.get("valid", True)
    ]
    if len(endpoints) <= 1:
        return False
    history = state.get("endpoint_results", {}).get(str(int(failed_endpoint_index)), [])
    cutoff = time.time() - PROVIDER_FALLBACK_RESULT_MAX_AGE_SECONDS
    history = [
        item for item in history
        if isinstance(item, dict) and float(item.get("at", 0.0)) >= cutoff
    ]
    if len(history) < PROVIDER_FALLBACK_WINDOW:
        return False
    failure_count = sum(1 for item in history[-PROVIDER_FALLBACK_WINDOW:] if not item.get("success"))
    if (failure_count / PROVIDER_FALLBACK_WINDOW) <= PROVIDER_FALLBACK_FAILURE_THRESHOLD:
        return False
    next_index = _next_provider_endpoint_index_locked(provider_id, failed_endpoint_index)
    if next_index == failed_endpoint_index:
        return False
    state["default_index"] = next_index
    state["default_reason"] = "rolling_failure_rate"
    state["default_changed_at"] = time.time()
    state["last_base_probe_at"] = time.time()
    _bump_generation()
    return True


def record_provider_success(provider_id: str, endpoint_index: Optional[int]) -> None:
    with _LOCK:
        endpoints = _build_provider_endpoints_locked(provider_id)
        if len(endpoints) <= 1:
            return
        selected = _select_provider_endpoint_locked(provider_id, endpoint_index)
        if selected is None:
            return
        _append_provider_result_locked(provider_id, int(selected["index"]), True)


def record_provider_failure(
    provider_id: str,
    endpoint_index: Optional[int],
    promote_default: bool = False,
) -> None:
    with _LOCK:
        endpoints = _build_provider_endpoints_locked(provider_id)
        if len(endpoints) <= 1:
            return
        selected = _select_provider_endpoint_locked(provider_id, endpoint_index)
        failed_index = int(selected["index"]) if selected else int(_get_provider_state_locked(provider_id).get("default_index", 0))
        _append_provider_result_locked(provider_id, failed_index, False)
        if promote_default:
            _maybe_promote_default_after_failure_locked(provider_id, failed_index)


def record_provider_failure_and_get_retry_snapshot(
    provider_id: str,
    endpoint_index: Optional[int],
    env_names: Iterable[str],
) -> Optional[Dict[str, Any]]:
    with _LOCK:
        endpoints = _build_provider_endpoints_locked(provider_id)
        if len(endpoints) <= 1:
            return None
        selected = _select_provider_endpoint_locked(provider_id, endpoint_index)
        failed_index = int(selected["index"]) if selected else int(_get_provider_state_locked(provider_id).get("default_index", 0))
        _append_provider_result_locked(provider_id, failed_index, False)
        _maybe_promote_default_after_failure_locked(provider_id, failed_index)
        valid_indices = [int(endpoint["index"]) for endpoint in endpoints if endpoint.get("valid", True)]
        try:
            failed_position = valid_indices.index(failed_index)
        except ValueError:
            failed_position = 0
        retry_index = valid_indices[(failed_position + 1) % len(valid_indices)] if valid_indices else failed_index
        if retry_index == failed_index:
            return None
        retry_snapshot = _build_config_snapshot_locked(env_names, provider_id, endpoint_index=retry_index)
        retry_snapshot["provider_endpoint_cycle_wrapped"] = failed_position + 1 >= len(valid_indices)
        return retry_snapshot


def claim_provider_base_recovery_probe(provider_id: str, env_names: Iterable[str]) -> Optional[Dict[str, Any]]:
    with _LOCK:
        endpoints = _build_provider_endpoints_locked(provider_id)
        if len(endpoints) <= 1:
            return None
        base_endpoint = next((endpoint for endpoint in endpoints if int(endpoint.get("index", -1)) == 0), None)
        if not base_endpoint or not base_endpoint.get("valid", True):
            return None
        state = _get_provider_state_locked(provider_id)
        if int(state.get("default_index", 0)) == 0:
            return None
        if state.get("base_probe_in_progress"):
            return None
        now = time.time()
        last_probe = float(state.get("last_base_probe_at") or 0.0)
        if last_probe and now - last_probe < PROVIDER_BASE_RECOVERY_CHECK_INTERVAL_SECONDS:
            return None
        state["base_probe_in_progress"] = True
        state["last_base_probe_at"] = now
        return _build_config_snapshot_locked(env_names, provider_id, endpoint_index=0)


def complete_provider_base_recovery_probe(provider_id: str, success: bool) -> None:
    with _LOCK:
        state = _get_provider_state_locked(provider_id)
        state["base_probe_in_progress"] = False
        _append_provider_result_locked(provider_id, 0, bool(success))
        if success and int(state.get("default_index", 0)) != 0:
            state["default_index"] = 0
            state["default_reason"] = "base_recovery_probe"
            state["default_changed_at"] = time.time()
            _bump_generation()


def get_station_proxy_snapshot() -> Dict[str, Any]:
    with _LOCK:
        return {
            "generation": _GENERATION,
            "http_proxy": getattr(constants, "LLM_HTTP_PROXY", None),
            "https_proxy": getattr(constants, "LLM_HTTPS_PROXY", None),
        }


def get_provider_proxy_values(provider_id: str) -> Dict[str, Optional[str]]:
    spec = PROVIDER_SPECS.get(str(provider_id or "").strip().lower())
    if not spec:
        return {
            "http_proxy": getattr(constants, "LLM_HTTP_PROXY", None),
            "https_proxy": getattr(constants, "LLM_HTTPS_PROXY", None),
        }
    with _LOCK:
        snapshot = _build_config_snapshot_locked((), spec.provider_id)
        return {
            "http_proxy": snapshot.get("http_proxy"),
            "https_proxy": snapshot.get("https_proxy"),
        }


def get_config_snapshot(
    env_names: Iterable[str] = (),
    provider_id: Optional[str] = None,
    endpoint_index: Optional[int] = None,
) -> Dict[str, Any]:
    with _LOCK:
        return _build_config_snapshot_locked(env_names, provider_id, endpoint_index=endpoint_index)


def _set_or_unset_env(name: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(name, None)
        return
    os.environ[name] = value


def _clean_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string value.")
    stripped = value.strip()
    return stripped or None


def _clean_direct_string(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Expected a string value.")
    return value.strip()


def mask_secret(value: Optional[str], visible_chars: int = 8) -> Dict[str, Any]:
    if not value:
        return {"present": False, "masked": "", "visible_prefix": ""}
    secret = str(value)
    prefix_len = min(max(1, visible_chars), len(secret))
    if len(secret) <= visible_chars:
        prefix_len = max(1, min(2, len(secret)))
    prefix = secret[:prefix_len]
    return {
        "present": True,
        "masked": f"{prefix}...",
        "visible_prefix": prefix,
    }


def _masked_env_field(env_name: str) -> Dict[str, Any]:
    value = os.environ.get(env_name)
    masked = mask_secret(value)
    return {
        "env": env_name,
        "present": masked["present"],
        "masked": masked["masked"],
    }


def _backup_entries_public_config(spec: ProviderSpec) -> List[Dict[str, Any]]:
    api_keys = _split_backup_env(spec.api_key_env)
    base_urls = _split_backup_env(spec.base_url_env)
    http_proxies = _split_backup_env(spec.http_proxy_env)
    https_proxies = _split_backup_env(spec.https_proxy_env)
    entries: List[Dict[str, Any]] = []
    for index, api_key in enumerate(api_keys):
        masked = mask_secret(api_key)
        entries.append({
            "index": index + 1,
            "api_key": {
                "present": masked["present"],
                "masked": masked["masked"],
            },
            "base_url": base_urls[index] if index < len(base_urls) else "",
            "http_proxy": http_proxies[index] if index < len(http_proxies) else "",
            "https_proxy": https_proxies[index] if index < len(https_proxies) else "",
        })
    return entries


def _build_default_provider_public_config(spec: ProviderSpec) -> Dict[str, Any]:
    endpoints = _build_provider_endpoints_locked(spec.provider_id)
    state = _get_provider_state_locked(spec.provider_id)
    current_endpoint = _select_provider_endpoint_locked(spec.provider_id)
    return {
        "id": spec.provider_id,
        "label": spec.label,
        "mode": "default",
        "fallback_configured": len(endpoints) > 1,
        "backup_endpoint_count": max(0, len(endpoints) - 1),
        "current_default_endpoint": _endpoint_public_info(
            current_endpoint or endpoints[0],
            endpoint_count=len(endpoints),
            configured=len(endpoints) > 1,
        ) if endpoints else {},
        "current_default_reason": state.get("default_reason", ""),
        "base_url_env": spec.base_url_env,
        "api_key_env": spec.api_key_env,
        "http_proxy_env": spec.http_proxy_env,
        "https_proxy_env": spec.https_proxy_env,
        "backup_base_url_env": _backup_env_name(spec.base_url_env),
        "backup_api_key_env": _backup_env_name(spec.api_key_env),
        "backup_http_proxy_env": _backup_env_name(spec.http_proxy_env),
        "backup_https_proxy_env": _backup_env_name(spec.https_proxy_env),
        "base_url": os.environ.get(spec.base_url_env, ""),
        "default_base_url": spec.default_base_url,
        "http_proxy": os.environ.get(spec.http_proxy_env, ""),
        "https_proxy": os.environ.get(spec.https_proxy_env, ""),
        "api_key": _masked_env_field(spec.api_key_env),
        "backup_api_key": _masked_env_field(_backup_env_name(spec.api_key_env)),
        "backup_endpoints": _backup_entries_public_config(spec),
        "endpoints": [],
    }


def build_public_config() -> Dict[str, Any]:
    with _LOCK:
        providers = [_build_default_provider_public_config(spec) for spec in PROVIDER_SPECS.values()]

        codex_available = (
            bool(getattr(constants, "RESEARCH_CENTER_ENABLED", False))
            and str(getattr(constants, "RESEARCH_CODER_BACKEND", "codex")).strip().lower() == "codex"
        )
        external_available = bool(
            getattr(constants, "EXTERNAL_COUNTER_ENABLED", False)
            and getattr(constants, "AUTO_EVAL_EXTERNAL_REPORT", False)
        )

        return {
            "generation": _GENERATION,
            "station_proxy": {
                "http_proxy_env": "HTTP_PROXY / http_proxy",
                "https_proxy_env": "HTTPS_PROXY / https_proxy",
                "http_proxy": getattr(constants, "LLM_HTTP_PROXY", None) or "",
                "https_proxy": getattr(constants, "LLM_HTTPS_PROXY", None) or "",
            },
            "providers": providers,
            "codex": {
                "available": codex_available,
                "base_url_env": CODEX_CONFIG["base_url_env"],
                "api_key_env": CODEX_CONFIG["api_key_env"],
                "base_url": os.environ.get(CODEX_CONFIG["base_url_env"], ""),
                "default_base_url": CODEX_CONFIG["default_base_url"],
                "api_key": _masked_env_field(CODEX_CONFIG["api_key_env"]),
                "http_proxy_env": CODEX_CONFIG["http_proxy_env"],
                "https_proxy_env": CODEX_CONFIG["https_proxy_env"],
                "http_proxy": os.environ.get(CODEX_CONFIG["http_proxy_env"], ""),
                "https_proxy": os.environ.get(CODEX_CONFIG["https_proxy_env"], ""),
            },
            "external_counter": {
                "available": external_available,
                "base_url_env": EXTERNAL_COUNTER_CONFIG["base_url_env"],
                "api_key_env": EXTERNAL_COUNTER_CONFIG["api_key_env"],
                "http_proxy_env": EXTERNAL_COUNTER_CONFIG["http_proxy_env"],
                "https_proxy_env": EXTERNAL_COUNTER_CONFIG["https_proxy_env"],
                "base_url": os.environ.get(EXTERNAL_COUNTER_CONFIG["base_url_env"], ""),
                "default_base_url": EXTERNAL_COUNTER_CONFIG["default_base_url"],
                "http_proxy": os.environ.get(EXTERNAL_COUNTER_CONFIG["http_proxy_env"], ""),
                "https_proxy": os.environ.get(EXTERNAL_COUNTER_CONFIG["https_proxy_env"], ""),
                "api_key": _masked_env_field(EXTERNAL_COUNTER_CONFIG["api_key_env"]),
            },
        }


def _set_station_proxy(http_proxy: str, https_proxy: str) -> None:
    http_value = http_proxy or None
    https_value = https_proxy or None
    constants.LLM_HTTP_PROXY = http_value
    constants.LLM_HTTPS_PROXY = https_value
    for env_name in ("http_proxy", "HTTP_PROXY"):
        _set_or_unset_env(env_name, http_value)
    for env_name in ("https_proxy", "HTTPS_PROXY"):
        _set_or_unset_env(env_name, https_value)
    grpc_value = https_value or http_value
    _set_or_unset_env("grpc_proxy", grpc_value)


def _set_api_key_from_payload(env_name: str, payload: Dict[str, Any]) -> None:
    if "api_key" not in payload:
        return
    api_key = payload.get("api_key")
    if not isinstance(api_key, str):
        raise ValueError("api_key must be a string when provided.")
    if api_key.strip():
        _set_or_unset_env(env_name, api_key.strip())


def _clean_backup_entry(value: Any, field_name: str, allow_blank: bool) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if ";" in cleaned:
        raise ValueError(f"{field_name} cannot contain ';'. Add another backup row instead.")
    if not allow_blank and not cleaned:
        raise ValueError(f"{field_name} is required for every backup endpoint.")
    return cleaned


def _normalize_backup_entries(raw_entries: Any, spec: ProviderSpec) -> List[Dict[str, str]]:
    if raw_entries in (None, ""):
        return []
    if not isinstance(raw_entries, list):
        raise ValueError("backup_endpoints must be a list.")
    current_api_keys = _split_backup_env(spec.api_key_env)
    normalized: List[Dict[str, str]] = []
    for index, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Backup endpoint #{index} must be an object.")
        api_key = _clean_backup_entry(raw_entry.get("api_key"), f"Backup endpoint #{index} API key", allow_blank=True)
        if not api_key:
            existing_index_raw = raw_entry.get("existing_index")
            try:
                existing_index = int(existing_index_raw)
            except Exception:
                existing_index = 0
            if existing_index < 1 or existing_index > len(current_api_keys):
                raise ValueError(f"Backup endpoint #{index} API key is required.")
            api_key = current_api_keys[existing_index - 1]
            if not api_key:
                raise ValueError(f"Backup endpoint #{index} API key is required.")
        normalized.append({
            "api_key": api_key,
            "base_url": _clean_backup_entry(raw_entry.get("base_url"), f"Backup endpoint #{index} Base URL", allow_blank=True),
            "http_proxy": _clean_backup_entry(raw_entry.get("http_proxy"), f"Backup endpoint #{index} HTTP Proxy", allow_blank=True),
            "https_proxy": _clean_backup_entry(raw_entry.get("https_proxy"), f"Backup endpoint #{index} HTTPS Proxy", allow_blank=True),
        })
    return normalized


def _apply_provider_backup_update(payload: Dict[str, Any], spec: ProviderSpec) -> None:
    backup_values = _build_provider_backup_values(payload, spec)
    if backup_values is None:
        return
    _set_provider_backup_values(backup_values)


def _build_provider_backup_values(payload: Dict[str, Any], spec: ProviderSpec) -> Optional[Dict[str, str]]:
    if "backup_endpoints" not in payload:
        return None
    entries = _normalize_backup_entries(payload.get("backup_endpoints"), spec)
    backup_values = {
        _backup_env_name(spec.api_key_env): ";".join(entry["api_key"] for entry in entries),
        _backup_env_name(spec.base_url_env): ";".join(entry["base_url"] for entry in entries),
        _backup_env_name(spec.http_proxy_env): ";".join(entry["http_proxy"] for entry in entries),
        _backup_env_name(spec.https_proxy_env): ";".join(entry["https_proxy"] for entry in entries),
    }
    errors = _validate_provider_backup_values(spec, backup_values)
    if errors:
        raise ValueError(" ".join(errors))
    return backup_values


def _set_provider_backup_values(backup_values: Dict[str, str]) -> None:
    for env_name, value in backup_values.items():
        _set_or_unset_env(env_name, value or None)


def _apply_proxy_config_update(payload: Dict[str, Any], http_proxy_env: str, https_proxy_env: str) -> None:
    if "http_proxy" in payload:
        _set_or_unset_env(http_proxy_env, _clean_optional_string(payload.get("http_proxy")))
    if "https_proxy" in payload:
        _set_or_unset_env(https_proxy_env, _clean_optional_string(payload.get("https_proxy")))


def apply_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    target = str(payload.get("target") or "").strip().lower()
    if not target:
        raise ValueError("target is required.")

    with _LOCK:
        if target == "station_proxy":
            _set_station_proxy(
                _clean_direct_string(payload.get("http_proxy")),
                _clean_direct_string(payload.get("https_proxy")),
            )
        elif target == "provider":
            _apply_provider_update(payload)
        elif target == "codex":
            _apply_env_config_update(
                payload=payload,
                base_url_env=CODEX_CONFIG["base_url_env"],
                api_key_env=CODEX_CONFIG["api_key_env"],
            )
            if "http_proxy" in payload:
                _set_or_unset_env(CODEX_CONFIG["http_proxy_env"], _clean_optional_string(payload.get("http_proxy")))
            if "https_proxy" in payload:
                _set_or_unset_env(CODEX_CONFIG["https_proxy_env"], _clean_optional_string(payload.get("https_proxy")))
        elif target == "external_counter":
            _apply_env_config_update(
                payload=payload,
                base_url_env=EXTERNAL_COUNTER_CONFIG["base_url_env"],
                api_key_env=EXTERNAL_COUNTER_CONFIG["api_key_env"],
            )
            _apply_proxy_config_update(
                payload=payload,
                http_proxy_env=EXTERNAL_COUNTER_CONFIG["http_proxy_env"],
                https_proxy_env=EXTERNAL_COUNTER_CONFIG["https_proxy_env"],
            )
        else:
            raise ValueError(f"Unsupported target: {target}")

        generation = _bump_generation()
        public_config = build_public_config()
        public_config["generation"] = generation
        return public_config


def _apply_env_config_update(payload: Dict[str, Any], base_url_env: str, api_key_env: str) -> None:
    if "base_url" in payload:
        _set_or_unset_env(base_url_env, _clean_optional_string(payload.get("base_url")))
    _set_api_key_from_payload(api_key_env, payload)


def _apply_provider_update(payload: Dict[str, Any]) -> None:
    provider_id = str(payload.get("provider") or "").strip().lower()
    spec = PROVIDER_SPECS.get(provider_id)
    if not spec:
        raise ValueError(f"Unsupported provider: {provider_id}")

    backup_values = _build_provider_backup_values(payload, spec)
    _apply_proxy_config_update(
        payload=payload,
        http_proxy_env=spec.http_proxy_env,
        https_proxy_env=spec.https_proxy_env,
    )
    _apply_env_config_update(
        payload=payload,
        base_url_env=spec.base_url_env,
        api_key_env=spec.api_key_env,
    )
    if backup_values is not None:
        _set_provider_backup_values(backup_values)


def get_codex_proxy_values() -> Dict[str, Optional[str]]:
    """Return Codex proxy values, falling back to Station proxy when unset."""
    with _LOCK:
        codex_http = os.environ.get(CODEX_CONFIG["http_proxy_env"])
        codex_https = os.environ.get(CODEX_CONFIG["https_proxy_env"])
        return {
            "http_proxy": codex_http or getattr(constants, "LLM_HTTP_PROXY", None),
            "https_proxy": codex_https or getattr(constants, "LLM_HTTPS_PROXY", None),
        }


def get_codex_proxy_overrides() -> Dict[str, Optional[str]]:
    """Return only explicit Codex proxy overrides, without Station fallback."""
    with _LOCK:
        return {
            "http_proxy": os.environ.get(CODEX_CONFIG["http_proxy_env"]),
            "https_proxy": os.environ.get(CODEX_CONFIG["https_proxy_env"]),
        }
