from typing import Any, Optional


def normalize_optional_role_definition(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped_value = value.strip()
    return stripped_value or None
