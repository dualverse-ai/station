"""Helpers for lightweight dashboard stream payloads."""

from __future__ import annotations


_STREAM_CONTENT_FIELDS = (
    "text_content",
    "content",
    "thinking_text",
    "human_message",
    "llm_response",
)


def sanitize_stream_event_payload(log_entry, selected_agent_name=None):
    """
    Keep live stream/poll payloads scoped to the dashboard's selected agent.

    Full prompt/response text is sent only when the browser has explicitly
    loaded that agent's dialogue view. Other agents keep event metadata, token
    stats, and status fields, but their message bodies are omitted.
    Applies only to transport payload, not persistent station dialogue logs.
    """
    if not isinstance(log_entry, dict):
        return log_entry

    data = log_entry.get("data")
    if not isinstance(data, dict):
        return log_entry

    sanitized = dict(log_entry)
    sanitized_data = dict(data)

    include_text = (
        bool(selected_agent_name)
        and selected_agent_name != "all"
        and sanitized_data.get("agent_name") == selected_agent_name
    )
    content_lengths = []
    for field in _STREAM_CONTENT_FIELDS:
        value = sanitized_data.get(field)
        if isinstance(value, str):
            content_lengths.append(len(value))
            if not include_text:
                sanitized_data.pop(field, None)

    sanitized_data.pop("text_preview", None)
    if content_lengths and not include_text:
        sanitized_data["content_omitted"] = True
        sanitized_data["full_length"] = max(content_lengths)
    elif include_text:
        sanitized_data.pop("content_omitted", None)

    sanitized["data"] = sanitized_data
    return sanitized
