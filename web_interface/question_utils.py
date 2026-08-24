"""Side-effect-free helpers for dashboard Question Room payloads."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from station import constants


QUESTION_LIST_SORT_FIELDS = {
    "numeric_id": "numeric_id",
    "title": "title",
    "author": "author",
    "authored_tick": "created_at_tick",
    "last_activity_tick": "last_updated_at_tick",
    "status": "question_status",
    "net_upvote": "question_net_upvote",
    "message_count": "message_count",
}


def extract_question_numeric_id(capsule_id_value: Any) -> Optional[int]:
    if capsule_id_value is None:
        return None
    match = re.search(r"(\d+)$", str(capsule_id_value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def build_question_list_payload(
    capsule_module: Any,
    *,
    page: int = 1,
    page_size: int = 100,
    sort_by: str = "authored_tick",
    sort_direction: str = "desc",
) -> Dict[str, Any]:
    safe_page = max(1, int(page))
    safe_page_size = max(1, min(100, int(page_size)))
    safe_sort_by = sort_by if sort_by in QUESTION_LIST_SORT_FIELDS else "authored_tick"
    safe_sort_direction = "asc" if str(sort_direction).lower() == "asc" else "desc"

    metadata_rows, total = capsule_module.list_capsules_page(
        constants.CAPSULE_TYPE_QUESTION,
        None,
        page=safe_page,
        page_size=safe_page_size,
        sort_by=QUESTION_LIST_SORT_FIELDS[safe_sort_by],
        sort_direction=safe_sort_direction,
    )
    total_pages = max(1, math.ceil(total / safe_page_size))

    questions: List[Dict[str, Any]] = []
    for metadata in metadata_rows:
        numeric_id = extract_question_numeric_id(metadata.get(constants.CAPSULE_ID_KEY))
        if numeric_id is None:
            continue
        message_count = int(metadata.get("total_message_count", 0) or 0)
        questions.append({
            "numeric_id": numeric_id,
            "capsule_id": metadata.get(constants.CAPSULE_ID_KEY),
            "title": metadata.get(constants.CAPSULE_TITLE_KEY) or "Untitled",
            "author": metadata.get(constants.CAPSULE_AUTHOR_NAME_KEY) or "Unknown",
            "authored_tick": metadata.get(constants.CAPSULE_CREATED_AT_TICK_KEY),
            "last_activity_tick": metadata.get(constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY),
            "status": metadata.get(constants.QUESTION_STATUS_KEY) or constants.QUESTION_STATUS_PENDING,
            "net_upvote": int(metadata.get(constants.QUESTION_NET_UPVOTE_KEY, 0) or 0),
            "message_count": message_count,
            "reply_count": max(0, message_count - 1),
            "abstract": metadata.get(constants.CAPSULE_ABSTRACT_KEY) or "",
            "solved_by_message_id": metadata.get(constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY),
        })

    return {
        "questions": questions,
        "page": safe_page,
        "page_size": safe_page_size,
        "total": total,
        "total_pages": total_pages,
        "sort_by": safe_sort_by,
        "sort_direction": safe_sort_direction,
    }


def build_question_survey_preview(capsule_module: Any, *, limit: int = 500) -> str:
    """Build a bounded Surveyor preview from the SQLite capsule index."""
    safe_limit = max(1, min(1000, int(limit or 500)))
    metadata_rows, total = capsule_module.list_capsules_page(
        constants.CAPSULE_TYPE_QUESTION,
        None,
        page=1,
        page_size=safe_limit,
        sort_by="numeric_id",
        sort_direction="desc",
    )
    previews: List[str] = []
    for metadata in metadata_rows:
        numeric_id = extract_question_numeric_id(metadata.get(constants.CAPSULE_ID_KEY))
        if numeric_id is None:
            continue
        title = metadata.get(constants.CAPSULE_TITLE_KEY) or "Untitled"
        author = metadata.get(constants.CAPSULE_AUTHOR_NAME_KEY) or "Unknown"
        status = metadata.get(constants.QUESTION_STATUS_KEY) or constants.QUESTION_STATUS_PENDING
        net_upvote = int(metadata.get(constants.QUESTION_NET_UPVOTE_KEY, 0) or 0)
        abstract = metadata.get(constants.CAPSULE_ABSTRACT_KEY) or "(No abstract available.)"
        solved_by = metadata.get(constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY)
        solved_line = f"\nAccepted solution message: {solved_by}" if solved_by else ""
        previews.append(
            f"**Question #{numeric_id}: {title}**\n"
            f"Author: {author}, Status: {status}, Net Upvote: {net_upvote}{solved_line}\n"
            f"Abstract: {abstract}"
        )
    if not previews:
        return "No Question Room problems currently available."
    scope = f"Showing {len(previews)} of {int(total)} Question Room problems, newest first."
    if len(previews) >= int(total):
        scope = f"Showing all {len(previews)} Question Room problems, newest first."
    return scope + "\n\n" + "\n\n---\n\n".join(previews)


def build_question_detail_payload(capsule_module: Any, numeric_id: int) -> Optional[Dict[str, Any]]:
    capsule_data = capsule_module.get_capsule(
        numeric_id,
        constants.CAPSULE_TYPE_QUESTION,
        include_deleted_capsule=False,
        include_deleted_messages=False,
    )
    if not capsule_data:
        return None
    return build_question_detail_payload_from_capsule(capsule_data, numeric_id)


def build_question_detail_payload_from_capsule(
    capsule_data: Dict[str, Any],
    numeric_id: Optional[int] = None,
) -> Dict[str, Any]:
    if numeric_id is None:
        numeric_id = extract_question_numeric_id(capsule_data.get(constants.CAPSULE_ID_KEY)) or 0

    solved_by_message_id = capsule_data.get(constants.QUESTION_SOLVED_BY_MESSAGE_ID_KEY)
    messages: List[Dict[str, Any]] = []
    for message in capsule_data.get(constants.CAPSULE_MESSAGES_KEY, []):
        if not isinstance(message, dict) or message.get(constants.MESSAGE_IS_DELETED_KEY, False):
            continue
        message_id = message.get(constants.MESSAGE_ID_KEY)
        messages.append({
            "message_id": message_id,
            "author": message.get(constants.MESSAGE_AUTHOR_NAME_KEY) or "Unknown",
            "posted_tick": message.get(constants.MESSAGE_POSTED_AT_TICK_KEY),
            "title": message.get(constants.MESSAGE_TITLE_KEY) or "",
            "content": str(message.get(constants.MESSAGE_CONTENT_KEY, "") or ""),
            "solution_net_upvote": int(message.get(constants.QUESTION_SOLUTION_NET_UPVOTE_KEY, 0) or 0),
            "is_question": len(messages) == 0,
            "is_accepted": bool(message_id and message_id == solved_by_message_id),
        })

    return {
        "numeric_id": numeric_id,
        "capsule_id": capsule_data.get(constants.CAPSULE_ID_KEY),
        "title": capsule_data.get(constants.CAPSULE_TITLE_KEY) or "Untitled",
        "author": capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY) or "Unknown",
        "authored_tick": capsule_data.get(constants.CAPSULE_CREATED_AT_TICK_KEY),
        "last_activity_tick": capsule_data.get(constants.CAPSULE_LAST_UPDATED_AT_TICK_KEY),
        "status": capsule_data.get(constants.QUESTION_STATUS_KEY) or constants.QUESTION_STATUS_PENDING,
        "net_upvote": int(capsule_data.get(constants.QUESTION_NET_UPVOTE_KEY, 0) or 0),
        "abstract": capsule_data.get(constants.CAPSULE_ABSTRACT_KEY) or "",
        "solved_by_message_id": solved_by_message_id,
        "messages": messages,
    }
