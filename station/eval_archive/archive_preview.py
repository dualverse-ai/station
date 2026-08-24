from __future__ import annotations

import os
from pathlib import Path

from station import constants, file_io_utils


def build_archive_preview(archive_capsules_dir: str | Path) -> str:
    try:
        archive_root = Path(archive_capsules_dir)
        if not archive_root.is_dir():
            return "No archive papers currently available."
        capsule_files: list[tuple[int, Path]] = []
        for path in archive_root.iterdir():
            filename = path.name
            if not (filename.startswith("archive_") and filename.endswith(constants.YAML_EXTENSION)):
                continue
            try:
                capsule_id = int(filename.split("_", 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                continue
            capsule_files.append((capsule_id, path))
        capsule_files.sort(key=lambda item: item[0])

        previews: list[str] = []
        for capsule_id, path in capsule_files:
            capsule_data = file_io_utils.load_yaml(os.fspath(path))
            if not isinstance(capsule_data, dict):
                continue
            if capsule_data.get(constants.CAPSULE_IS_DELETED_KEY, False):
                continue
            title = capsule_data.get(constants.CAPSULE_TITLE_KEY, "Untitled")
            author = capsule_data.get(constants.CAPSULE_AUTHOR_NAME_KEY, "Unknown")
            created_tick = capsule_data.get(constants.CAPSULE_CREATED_AT_TICK_KEY, "N/A")
            abstract = capsule_data.get(constants.CAPSULE_ABSTRACT_KEY, "")
            preview = (
                f"**Archive #{capsule_id}: {title}**\n"
                f"Author: {author}, Created at Tick: {created_tick}\n"
                f"Abstract: {abstract if abstract else '(No abstract available.)'}"
            )
            previews.append(preview)
        return "\n\n---\n\n".join(previews) if previews else "No archive papers currently available."
    except Exception as exc:
        print(f"ArchivePreview: failed to load archive preview: {exc}")
        return "Error loading archive preview."
