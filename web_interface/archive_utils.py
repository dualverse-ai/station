"""Side-effect-free helpers for dashboard archive paper payloads."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from station import constants


def extract_archive_numeric_id(capsule_id_value: Any) -> Optional[int]:
    if capsule_id_value is None:
        return None
    match = re.search(r"(\d+)$", str(capsule_id_value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_reviewer_score_from_capsule(capsule_data: Dict[str, Any]) -> Optional[float]:
    messages = capsule_data.get(constants.CAPSULE_MESSAGES_KEY, [])
    for message in reversed(messages):
        content = str(message.get(constants.MESSAGE_CONTENT_KEY, "") or "")
        if "Reviewer Evaluation" not in content:
            continue
        match = re.search(r"\*\*Score:\*\*\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", content)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def build_all_abstracts_preview(capsules: List[Dict[str, Any]]) -> str:
    preview_parts: List[str] = []
    for capsule in capsules:
        numeric_id = capsule.get("numeric_id")
        title = capsule.get("title") or "Untitled"
        author = capsule.get("author") or "Unknown"
        accepted_tick = capsule.get("accepted_tick", "N/A")
        abstract = capsule.get("abstract") or "(No abstract available for this capsule.)"
        preview_parts.append(
            f"**Preview for Capsule #{numeric_id}: {title}**\n"
            f"Author: {author}, Created at Tick: {accepted_tick}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n---\n\n".join(preview_parts)


def build_archive_list_payload(capsule_module: Any) -> Dict[str, Any]:
    capsules: List[Dict[str, Any]] = []
    for capsule_data in capsule_module.list_capsules(constants.CAPSULE_TYPE_ARCHIVE, None):
        numeric_id = extract_archive_numeric_id(capsule_data.get(constants.CAPSULE_ID_KEY))
        if numeric_id is None:
            continue

        capsules.append({
            "numeric_id": numeric_id,
            "capsule_id": capsule_data.get(constants.CAPSULE_ID_KEY),
            "title": capsule_data.get(constants.CAPSULE_TITLE_KEY) or "Untitled",
            "author": capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY) or "Unknown",
            "accepted_tick": capsule_data.get(constants.CAPSULE_CREATED_AT_TICK_KEY),
            "word_count": capsule_data.get(constants.CAPSULE_WORD_COUNT_TOTAL_KEY, 0),
            "message_count": capsule_data.get("total_message_count", 0),
            "abstract": capsule_data.get(constants.CAPSULE_ABSTRACT_KEY) or "",
            "reviewer_score": capsule_data.get("reviewer_score"),
        })

    capsules.sort(key=lambda item: item["numeric_id"])
    return {
        "capsules": capsules,
        "all_abstracts_markdown": build_all_abstracts_preview(capsules),
    }


def build_archive_detail_payload(capsule_module: Any, numeric_id: int) -> Optional[Dict[str, Any]]:
    capsule_data = capsule_module.get_capsule(
        numeric_id,
        constants.CAPSULE_TYPE_ARCHIVE,
        include_deleted_capsule=False,
        include_deleted_messages=False,
    )
    if not capsule_data:
        return None

    return build_archive_detail_payload_from_capsule(capsule_data, numeric_id)


def build_archive_detail_payload_from_capsule(
    capsule_data: Dict[str, Any],
    numeric_id: Optional[int] = None,
) -> Dict[str, Any]:
    if numeric_id is None:
        numeric_id = extract_archive_numeric_id(capsule_data.get(constants.CAPSULE_ID_KEY)) or 0

    title = capsule_data.get(constants.CAPSULE_TITLE_KEY) or "Untitled"
    author = capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY) or "Unknown"
    accepted_tick = capsule_data.get(constants.CAPSULE_CREATED_AT_TICK_KEY)
    abstract = capsule_data.get(constants.CAPSULE_ABSTRACT_KEY) or ""
    reviewer_score = extract_reviewer_score_from_capsule(capsule_data)

    author_messages: List[Dict[str, Any]] = []
    reply_messages: List[Dict[str, Any]] = []
    for message in capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []):
        if message.get(constants.MESSAGE_IS_DELETED_KEY, False):
            continue
        if message.get(constants.MESSAGE_AUTHOR_NAME_KEY) == author:
            author_messages.append(message)
        else:
            reply_messages.append(message)

    paper_sections: List[str] = [f"# {title}", f"**Author:** {author}"]
    if accepted_tick is not None:
        paper_sections.append(f"**Accepted Tick:** {accepted_tick}")
    if reviewer_score is not None:
        paper_sections.append(f"**Reviewer Score:** {reviewer_score:g}/10")
    if abstract:
        paper_sections.append(f"## Abstract\n\n{abstract}")

    if author_messages:
        first_message = author_messages[0]
        first_content = str(first_message.get(constants.MESSAGE_CONTENT_KEY, "") or "").strip()
        if first_content:
            paper_sections.append(first_content)

        for idx, message in enumerate(author_messages[1:], start=2):
            title_suffix = message.get(constants.MESSAGE_TITLE_KEY)
            section_title = str(title_suffix).strip() if title_suffix else f"Continuation {idx - 1}"
            content = str(message.get(constants.MESSAGE_CONTENT_KEY, "") or "").strip()
            if not content:
                continue
            paper_sections.append(f"## {section_title}\n\n{content}")
    else:
        paper_sections.append("_No author manuscript content available._")

    reply_sections = _format_archive_reply_sections(reply_messages)
    if reply_sections:
        paper_sections.append("## Capsule Replies\n\n" + "\n\n---\n\n".join(reply_sections))

    paper_markdown = "\n\n".join(part for part in paper_sections if part)
    return {
        "numeric_id": numeric_id,
        "capsule_id": capsule_data.get(constants.CAPSULE_ID_KEY),
        "title": title,
        "author": author,
        "accepted_tick": accepted_tick,
        "abstract": abstract,
        "reviewer_score": reviewer_score,
        "reply_count": len(reply_sections),
        "paper_markdown": paper_markdown,
    }


def _format_archive_reply_sections(messages: List[Dict[str, Any]]) -> List[str]:
    sections: List[str] = []
    for idx, message in enumerate(messages, start=1):
        content = str(message.get(constants.MESSAGE_CONTENT_KEY, "") or "").strip()
        if not content:
            continue

        reply_author = message.get(constants.MESSAGE_AUTHOR_NAME_KEY) or "Unknown"
        posted_tick = message.get(constants.MESSAGE_POSTED_AT_TICK_KEY)
        title_suffix = str(message.get(constants.MESSAGE_TITLE_KEY) or "").strip()
        heading_parts = [title_suffix or f"Reply {idx}", f"by {reply_author}"]
        if posted_tick is not None:
            heading_parts.append(f"at Tick {posted_tick}")

        sections.append(f"### {' '.join(heading_parts)}\n\n{content}")
    return sections
