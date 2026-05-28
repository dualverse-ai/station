#!/usr/bin/env python3
"""Simple API connectivity test using Station connector + direct provider HTTP call."""

import argparse
import json
import os
import sys
import enum
import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
from station import file_io_utils


SYSTEM_PROMPT = "You are a concise assistant."
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_GPT_MODEL = "gpt-5.5"
DEFAULT_OPENAI_THINKING_LEVEL = "xhigh"
OPENAI_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
ANSI_RESET = "\033[0m"
ANSI_ANSWER = "\033[96m"
ANSI_THINKING = "\033[93m"
ANSI_STATION_META = "\033[92m"


def supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def colorize(text: str, color: str) -> str:
    if not text:
        return text
    if not supports_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def build_gemini_thinking_config(model_name: str) -> dict | None:
    model_prefix = (model_name or "").lower()
    if model_prefix.startswith("models/"):
        model_prefix = model_prefix[len("models/"):]
    if model_prefix.startswith("gemini-2.0"):
        return None
    if model_prefix.startswith("gemini-2.5"):
        return {"thinkingBudget": 24576, "includeThoughts": True}
    return {"includeThoughts": True, "thinkingLevel": "high"}


def extract_gemini_rest_parts(data: dict) -> tuple[str, str | None]:
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    for candidate in data.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text", "")
            if not text:
                continue
            if part.get("thought"):
                thinking_parts.append(text)
            else:
                answer_parts.append(text)

    answer_text = "".join(answer_parts)
    thinking_text = "\n".join(thinking_parts) if thinking_parts else None
    return answer_text, thinking_text


def extract_gemini_rest_token_info(data: dict) -> dict[str, int | None]:
    usage = data.get("usageMetadata") or {}
    return {
        "total_tokens_in_session": usage.get("totalTokenCount"),
        "last_exchange_prompt_tokens": usage.get("promptTokenCount"),
        "last_exchange_completion_tokens": usage.get("candidatesTokenCount"),
        "last_exchange_cached_tokens": usage.get("cachedContentTokenCount"),
        "last_exchange_thoughts_tokens": usage.get("thoughtsTokenCount"),
    }


def extract_gemini_sdk_parts(response: object) -> tuple[str, str | None]:
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", "")
            if not text:
                continue
            if getattr(part, "thought", False):
                thinking_parts.append(text)
            else:
                answer_parts.append(text)

    answer_text = "".join(answer_parts)
    thinking_text = "\n".join(thinking_parts) if thinking_parts else None
    return answer_text, thinking_text


def extract_gemini_sdk_token_info(response: object) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "total_tokens_in_session": getattr(usage, "total_token_count", None),
        "last_exchange_prompt_tokens": getattr(usage, "prompt_token_count", None),
        "last_exchange_completion_tokens": getattr(usage, "candidates_token_count", None),
        "last_exchange_cached_tokens": getattr(usage, "cached_content_token_count", None),
        "last_exchange_thoughts_tokens": getattr(usage, "thoughts_token_count", None),
    }


def to_jsonable(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_jsonable(vars(value))
        except Exception:
            pass
    return repr(value)


def extract_gemini_sdk_meta(response: object) -> dict:
    return {
        "response_id": getattr(response, "response_id", None),
        "model_version": getattr(response, "model_version", None),
        "usage_metadata": to_jsonable(getattr(response, "usage_metadata", None)),
        "raw": to_jsonable(response),
    }


def extract_claude_rest_parts(data: dict) -> tuple[str, str | None]:
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in data.get("content", []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            thinking = block.get("thinking")
            if thinking:
                thinking_parts.append(thinking)
        elif block_type == "text":
            text = block.get("text")
            if text:
                answer_parts.append(text)

    answer_text = "".join(answer_parts)
    thinking_text = "\n".join(thinking_parts) if thinking_parts else None
    return answer_text, thinking_text


def extract_claude_rest_token_info(data: dict) -> dict[str, int | None]:
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_read_input_tokens = usage.get("cache_read_input_tokens") or 0
    total_tokens_in_session = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens_in_session = int(input_tokens) + int(cache_read_input_tokens) + int(output_tokens)
    return {
        "total_tokens_in_session": total_tokens_in_session,
        "last_exchange_prompt_tokens": input_tokens,
        "last_exchange_completion_tokens": output_tokens,
        "last_exchange_cached_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
    }


def extract_claude_meta(data: dict) -> dict:
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "role": data.get("role"),
        "stop_reason": data.get("stop_reason"),
        "stop_sequence": data.get("stop_sequence"),
        "type": data.get("type"),
        "usage": data.get("usage"),
    }


def is_openai_reasoning_model(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    return not lowered.startswith(("gpt-3", "gpt3", "gpt-4", "gpt4"))


def validate_openai_thinking_level(model_name: str, openai_thinking_level: str | None) -> None:
    if openai_thinking_level is None:
        return

    normalized = openai_thinking_level.strip().lower()
    if normalized not in OPENAI_REASONING_EFFORTS:
        supported = ", ".join(OPENAI_REASONING_EFFORTS)
        raise ValueError(f"Unsupported OpenAI reasoning effort '{openai_thinking_level}'. Supported values: {supported}.")


def resolve_openai_thinking_level(model_name: str, openai_thinking_level: str | None) -> str | None:
    if not is_openai_reasoning_model(model_name):
        return None
    if openai_thinking_level is None:
        return DEFAULT_OPENAI_THINKING_LEVEL
    return openai_thinking_level.strip().lower()


def build_openai_reasoning_config(model_name: str, openai_thinking_level: str | None) -> dict | None:
    reasoning_effort = resolve_openai_thinking_level(model_name, openai_thinking_level)
    if reasoning_effort is None:
        return None
    return {
        "effort": reasoning_effort,
        "summary": "detailed",
    }


def extract_openai_responses_parts(data: dict) -> tuple[str, str | None]:
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content_item in item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if text:
                        answer_parts.append(text)
        elif item_type == "reasoning":
            summary = item.get("summary") or []
            if isinstance(summary, list):
                for summary_item in summary:
                    if isinstance(summary_item, dict):
                        text = summary_item.get("text")
                        if text:
                            thinking_parts.append(text)

    answer_text = "".join(answer_parts)
    thinking_text = "\n\n".join(thinking_parts) if thinking_parts else None
    return answer_text, thinking_text


def extract_openai_responses_token_info(data: dict) -> dict[str, int | None]:
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "total_tokens_in_session": total_tokens,
        "last_exchange_prompt_tokens": input_tokens,
        "last_exchange_completion_tokens": output_tokens,
        "last_exchange_cached_tokens": input_details.get("cached_tokens"),
        "last_exchange_thoughts_tokens": output_details.get("reasoning_tokens"),
    }


def extract_openai_responses_meta(data: dict) -> dict:
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "status": data.get("status"),
        "output": data.get("output"),
        "usage": data.get("usage"),
    }


def build_claude_thinking_payload(max_output_tokens: int, adaptive: bool = True) -> dict:
    if adaptive:
        return {"type": "adaptive"}
    thinking_budget = min(10000, int(max_output_tokens * 0.5))
    return {
        "type": "enabled",
        "budget_tokens": max(1024, thinking_budget),
    }


def build_claude_output_config(model_name: str) -> dict | None:
    lowered = (model_name or "").lower()
    if lowered.startswith("claude-opus-4-6"):
        return {"effort": "max"}
    if lowered.startswith("claude-opus-4-5") or lowered.startswith("claude-sonnet-4-6"):
        return {"effort": "high"}
    return None


def is_claude_adaptive_rejection(body_text: str) -> bool:
    lowered = (body_text or "").lower()
    indicators = [
        "adaptive",
        "invalid thinking type",
        "unknown parameter",
        "unsupported",
        "not enabled for this channel",
        "effort",
    ]
    return any(indicator in lowered for indicator in indicators)


def print_colored_block(prefix: str, label: str, text: str | None, color: str) -> None:
    if not text:
        return
    print(f"{prefix} {label}:")
    print(colorize(text.strip(), color))


def print_colored_json_line(prefix: str, label: str, payload: dict, color: str) -> None:
    json_text = json.dumps(payload, ensure_ascii=True)
    print(colorize(f"{prefix} {label}: {json_text}", color))


def print_reasoning_effort_line(prefix: str, reasoning_effort: str | None) -> None:
    if not reasoning_effort:
        return
    print(colorize(f"{prefix} reasoning_effort: {reasoning_effort}", ANSI_STATION_META))


def print_thinking_token_line(prefix: str, token_info: dict | None) -> None:
    if token_info is None:
        return
    thinking_tokens = token_info.get("last_exchange_thoughts_tokens")
    rendered = "not reported" if thinking_tokens is None else str(thinking_tokens)
    print(colorize(f"{prefix} thinking_tokens: {rendered}", ANSI_STATION_META))


def resolve_gemini_base_url() -> str:
    base_url = os.getenv("GOOGLE_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
    if "/v1" not in base_url:
        base_url = f"{base_url}/v1beta"
    return base_url


def resolve_claude_base_url() -> str:
    return os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")


def required_api_key(provider: str) -> str | None:
    if provider == "claude":
        return os.getenv("ANTHROPIC_API_KEY")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    return os.getenv("GOOGLE_API_KEY")


def print_station_persisted_metadata(history_file: Path) -> None:
    if not history_file.exists():
        print("[station] persisted_history: not found")
        return

    entries = file_io_utils.load_yaml_lines(str(history_file))
    model_entries = [entry for entry in entries if isinstance(entry, dict) and entry.get("role") == "model"]
    if not model_entries:
        print("[station] persisted_history: no model entries")
        return

    last_entry = model_entries[-1]
    api_metadata = last_entry.get("api_metadata") or {}
    ids = {
        "response_id": api_metadata.get("response_id"),
        "request_id": api_metadata.get("request_id"),
        "message_id": api_metadata.get("message_id"),
    }

    print_colored_json_line("[station]", "persisted_token_info", last_entry.get("token_info") or {}, ANSI_STATION_META)
    print_colored_json_line("[station]", "persisted_api_ids", ids, ANSI_STATION_META)
    print_colored_json_line("[station]", "persisted_api_metadata", api_metadata, ANSI_STATION_META)


def test_station_connector(
    provider: str,
    model_name: str,
    prompt: str,
    max_output_tokens: int,
    openai_thinking_level: str | None = None,
) -> bool:
    from station.llm_connectors.factory import create_llm_connector

    test_dir = REPO_ROOT / "tmp" / "api_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    history_file = test_dir / "llm_chat_history.yamll"
    if history_file.exists():
        history_file.unlink()

    reasoning_effort = None
    custom_api_params = None
    if provider == "openai":
        reasoning_effort = resolve_openai_thinking_level(model_name, openai_thinking_level)
        if reasoning_effort is not None:
            custom_api_params = {"openai_thinking_level": reasoning_effort}

    connector = create_llm_connector(
        model_class_name=provider,
        model_name=model_name,
        agent_name="api_test_agent",
        agent_data_path=str(test_dir),
        max_output_tokens=max_output_tokens,
        system_prompt=SYSTEM_PROMPT,
        temperature=1.0,
        custom_api_params=custom_api_params,
    )
    if connector is None:
        print(f"[station] FAIL: could not create {provider} connector")
        return False

    response_text, thinking_text, token_info = connector.send_message(prompt, current_tick=0)
    print("[station] OK")
    print(f"[station] provider: {provider}")
    print(f"[station] model: {model_name}")
    connector_source = inspect.getsourcefile(connector.__class__)
    if connector_source:
        print(f"[station] connector_source: {connector_source}")
    print_reasoning_effort_line("[station]", reasoning_effort)
    print(f"[station] prompt: {prompt}")
    print_colored_block("[station]", "answer", response_text, ANSI_ANSWER)
    print_colored_block("[station]", "thinking", thinking_text, ANSI_THINKING)
    print_colored_json_line("[station]", "token_info", token_info, ANSI_STATION_META)
    print_thinking_token_line("[station]", token_info)
    print_station_persisted_metadata(history_file)
    return True


def test_direct_gemini_post(model_name: str, prompt: str, timeout: int, max_output_tokens: int) -> bool:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[post] FAIL: GOOGLE_API_KEY is not set")
        return False

    base_url = resolve_gemini_base_url()
    url = f"{base_url}/models/{model_name}:generateContent?key={api_key}"
    generation_config = {"temperature": 1.0, "maxOutputTokens": max_output_tokens}
    thinking_config = build_gemini_thinking_config(model_name)
    if thinking_config is not None:
        generation_config["thinkingConfig"] = thinking_config
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    print(f"[post] provider: gemini")
    print(f"[post] base_url: {base_url}")
    resp = requests.post(url, json=payload, timeout=timeout)
    print(f"[post] status: {resp.status_code}")
    if not resp.ok:
        print(f"[post] FAIL body: {resp.text[:600]}")
        return False

    data = resp.json()
    response_text, thinking_text = extract_gemini_rest_parts(data)
    token_info = extract_gemini_rest_token_info(data)

    print("[post] OK")
    print(f"[post] model: {model_name}")
    print(f"[post] prompt: {prompt}")
    print_colored_block("[post]", "answer", response_text, ANSI_ANSWER)
    print_colored_block("[post]", "thinking", thinking_text, ANSI_THINKING)
    print_colored_json_line("[post]", "token_info", token_info, ANSI_STATION_META)
    return True


def test_direct_claude_post(model_name: str, prompt: str, timeout: int, max_output_tokens: int) -> bool:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[post] FAIL: ANTHROPIC_API_KEY is not set")
        return False

    base_url = resolve_claude_base_url()
    url = f"{base_url}/v1/messages"
    payload = {
        "model": model_name,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": 1.0,
        "thinking": build_claude_thinking_payload(max_output_tokens, adaptive=True),
    }
    output_config = build_claude_output_config(model_name)
    if output_config is not None:
        payload["output_config"] = output_config
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "extended-cache-ttl-2025-04-11",
    }

    print(f"[post] provider: claude")
    print(f"[post] base_url: {base_url}")
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if not resp.ok and is_claude_adaptive_rejection(resp.text):
        print("[post] adaptive thinking rejected; retrying with manual thinking mode")
        payload["thinking"] = build_claude_thinking_payload(max_output_tokens, adaptive=False)
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    print(f"[post] status: {resp.status_code}")
    if not resp.ok:
        print(f"[post] FAIL body: {resp.text[:600]}")
        return False

    data = resp.json()
    response_text, thinking_text = extract_claude_rest_parts(data)
    token_info = extract_claude_rest_token_info(data)
    meta_info = extract_claude_meta(data)

    print("[post] OK")
    print(f"[post] model: {model_name}")
    print(f"[post] prompt: {prompt}")
    print_colored_block("[post]", "answer", response_text, ANSI_ANSWER)
    print_colored_block("[post]", "thinking", thinking_text, ANSI_THINKING)
    print_colored_json_line("[post]", "token_info", token_info, ANSI_STATION_META)
    print_colored_json_line("[post]", "meta", meta_info, ANSI_STATION_META)
    return True


def test_direct_openai_post(
    model_name: str,
    prompt: str,
    timeout: int,
    max_output_tokens: int,
    openai_thinking_level: str | None,
) -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[post] FAIL: OPENAI_API_KEY is not set")
        return False

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    reasoning_effort = resolve_openai_thinking_level(model_name, openai_thinking_level)

    if is_openai_reasoning_model(model_name):
        url = f"{base_url}/responses"
        payload = {
            "model": model_name,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "reasoning": build_openai_reasoning_config(model_name, openai_thinking_level),
        }
        if max_output_tokens:
            payload["max_output_tokens"] = max_output_tokens
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }

        print("[post] provider: openai")
        print(f"[post] base_url: {base_url}")
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        print(f"[post] status: {resp.status_code}")
        if not resp.ok:
            print(f"[post] FAIL body: {resp.text[:600]}")
            return False

        data = resp.json()
        response_text, thinking_text = extract_openai_responses_parts(data)
        token_info = extract_openai_responses_token_info(data)
        meta_info = extract_openai_responses_meta(data)

        print("[post] OK")
        print(f"[post] model: {model_name}")
        print_reasoning_effort_line("[post]", reasoning_effort)
        print(f"[post] prompt: {prompt}")
        print_colored_block("[post]", "answer", response_text, ANSI_ANSWER)
        print_colored_block("[post]", "thinking", thinking_text, ANSI_THINKING)
        print_colored_json_line("[post]", "token_info", token_info, ANSI_STATION_META)
        print_thinking_token_line("[post]", token_info)
        print_colored_json_line("[post]", "meta", meta_info, ANSI_STATION_META)
        return True

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 1.0,
    }
    if max_output_tokens:
        payload["max_tokens"] = max_output_tokens
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }

    print("[post] provider: openai")
    print(f"[post] base_url: {base_url}")
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    print(f"[post] status: {resp.status_code}")
    if not resp.ok:
        print(f"[post] FAIL body: {resp.text[:600]}")
        return False

    data = resp.json()
    choice = ((data.get("choices") or [{}])[0]).get("message") or {}
    response_text = choice.get("content", "")
    token_usage = data.get("usage") or {}
    token_info = {
        "total_tokens_in_session": token_usage.get("total_tokens"),
        "last_exchange_prompt_tokens": token_usage.get("prompt_tokens"),
        "last_exchange_completion_tokens": token_usage.get("completion_tokens"),
        "last_exchange_cached_tokens": None,
        "last_exchange_thoughts_tokens": None,
    }

    print("[post] OK")
    print(f"[post] model: {model_name}")
    print(f"[post] prompt: {prompt}")
    print_colored_block("[post]", "answer", response_text, ANSI_ANSWER)
    print_colored_json_line("[post]", "token_info", token_info, ANSI_STATION_META)
    print_thinking_token_line("[post]", token_info)
    print_colored_json_line("[post]", "meta", {"id": data.get("id"), "model": data.get("model"), "usage": data.get("usage")}, ANSI_STATION_META)
    return True


def test_direct_post(
    provider: str,
    model_name: str,
    prompt: str,
    timeout: int,
    max_output_tokens: int,
    openai_thinking_level: str | None = None,
) -> bool:
    if provider == "claude":
        return test_direct_claude_post(model_name, prompt, timeout, max_output_tokens)
    if provider == "openai":
        return test_direct_openai_post(model_name, prompt, timeout, max_output_tokens, openai_thinking_level)
    return test_direct_gemini_post(model_name, prompt, timeout, max_output_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Gemini, Claude, or OpenAI API connection.")
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument("--claude", action="store_true", help="Use Anthropic Claude instead of Gemini.")
    provider_group.add_argument("--gemini", action="store_true", help="Explicitly use Gemini.")
    provider_group.add_argument("--gpt", action="store_true", help="Use OpenAI GPT instead of Gemini.")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--prompt",
        default="Where does the name Gemini originate from?",
    )
    parser.add_argument("--post-timeout", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument(
        "--openai-thinking-level",
        default=None,
        choices=OPENAI_REASONING_EFFORTS,
        help=(
            "OpenAI Responses API reasoning.effort override. "
            f"If omitted, reasoning models default here to {DEFAULT_OPENAI_THINKING_LEVEL}."
        ),
    )
    args = parser.parse_args()

    provider = "openai" if args.gpt else ("claude" if args.claude else "gemini")
    if args.model:
        model_name = args.model
    elif provider == "claude":
        model_name = DEFAULT_CLAUDE_MODEL
    elif provider == "openai":
        model_name = DEFAULT_GPT_MODEL
    else:
        model_name = DEFAULT_GEMINI_MODEL

    resolved_openai_thinking_level = None
    if provider == "openai":
        if is_openai_reasoning_model(model_name):
            validate_openai_thinking_level(model_name, args.openai_thinking_level)
            resolved_openai_thinking_level = resolve_openai_thinking_level(model_name, args.openai_thinking_level)
        elif args.openai_thinking_level is not None:
            print(f"warning: ignoring --openai-thinking-level for non-reasoning OpenAI model '{model_name}'")

    if provider == "claude":
        api_key_name = "ANTHROPIC_API_KEY"
        base_url = resolve_claude_base_url()
    elif provider == "openai":
        api_key_name = "OPENAI_API_KEY"
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    else:
        api_key_name = "GOOGLE_API_KEY"
        base_url = resolve_gemini_base_url()

    if not required_api_key(provider):
        print(f"{api_key_name} is not set.")
        return 2

    print(f"provider: {provider}")
    print(f"model: {model_name}")
    print(f"base_url: {base_url}")
    if resolved_openai_thinking_level is not None:
        print(f"openai_thinking_level: {resolved_openai_thinking_level}")

    ok_station = False
    ok_post = False
    try:
        ok_station = test_station_connector(
            provider,
            model_name,
            args.prompt,
            args.max_output_tokens,
            openai_thinking_level=resolved_openai_thinking_level,
        )
    except Exception as exc:
        print(f"[station] FAIL: {exc}")

    try:
        ok_post = test_direct_post(
            provider,
            model_name,
            args.prompt,
            args.post_timeout,
            args.max_output_tokens,
            openai_thinking_level=resolved_openai_thinking_level,
        )
    except Exception as exc:
        print(f"[post] FAIL: {exc}")

    if ok_station and ok_post:
        print("RESULT: PASS")
        return 0
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
