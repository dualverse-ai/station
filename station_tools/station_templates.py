from __future__ import annotations

from argparse import ArgumentParser
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from station_tools.repo import load_yaml_mapping


DEFAULT_STATION_TEMPLATE_SOURCE = "example/station/default"
STATION_TEMPLATE_SOURCE_KEY = "station_template_source"
_TEMPLATE_ROOTS = ("example_private", "example")


def _active_multistart_job_dir(repo: Path) -> Path | None:
    multistart_root = repo / "station_multistart"
    for current_job_path in (multistart_root / "current_job.yaml", multistart_root / "current_job"):
        current_job = load_yaml_mapping(current_job_path)
        job_dir_value = current_job.get("job_dir")
        if not job_dir_value:
            continue
        job_dir = Path(str(job_dir_value))
        if not job_dir.is_absolute():
            job_dir = repo / job_dir
        try:
            job_dir.resolve().relative_to(multistart_root.resolve())
        except ValueError:
            continue
        if job_dir.is_dir():
            return job_dir
    return None


def _canonical_source(value: object, *, default: bool) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if default:
            return DEFAULT_STATION_TEMPLATE_SOURCE
        raise ValueError("station template cannot be empty")
    if not isinstance(value, str):
        raise ValueError("station template source must be a string")

    source = value.strip().replace("\\", "/")
    path = PurePosixPath(source)
    if path.is_absolute() or not path.parts or path.parts[0] not in _TEMPLATE_ROOTS:
        raise ValueError(
            "station template source must be example/station/<name> "
            "or example_private/station/<name>"
        )

    # Older Station releases stored templates in flat directories such as
    # example/station_default and example/station_gpt-5-5. Normalize those
    # persisted values before an update pulls the checkout with the new nested
    # template layout.
    if len(path.parts) == 2 and path.parts[1].startswith("station_"):
        name = path.parts[1][len("station_"):]
        if name in {"", ".", ".."}:
            raise ValueError("station template source contains an invalid directory name")
        return f"{path.parts[0]}/station/{name}"

    if len(path.parts) != 3 or path.parts[1] != "station":
        raise ValueError(
            "station template source must be example/station/<name> "
            "or example_private/station/<name>"
        )
    if path.parts[2] in {"", ".", ".."}:
        raise ValueError("station template source contains an invalid directory name")
    return path.as_posix()


def resolve_station_template(repo: Path, value: object = None) -> tuple[Path, str]:
    """Resolve an init CLI value to a template directory and canonical source."""
    if value is None or (isinstance(value, str) and not value.strip()):
        sources = (DEFAULT_STATION_TEMPLATE_SOURCE,)
    elif isinstance(value, str) and len(PurePosixPath(value.strip().replace("\\", "/")).parts) == 1:
        name = value.strip()
        if name in {"", ".", ".."}:
            raise ValueError("station template contains an invalid directory name")
        sources = tuple(f"{root}/station/{name}" for root in _TEMPLATE_ROOTS)
    else:
        sources = (_canonical_source(value, default=False),)

    searched: list[Path] = []
    for source in sources:
        canonical = _canonical_source(source, default=False)
        candidate = repo / canonical
        searched.append(candidate)
        if not candidate.is_dir():
            continue
        root = (repo / PurePosixPath(canonical).parts[0]).resolve()
        try:
            candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"station template resolves outside its allowed root: {candidate}") from exc
        return candidate, canonical

    searched_text = "\n".join(f"  {path}" for path in searched)
    raise FileNotFoundError(f"station template not found; searched:\n{searched_text}")


def configured_station_template_source(repo: Path) -> str:
    """Read the persisted canonical source, defaulting when it is absent."""
    config_paths = [repo / "station_data" / "station_config.yaml"]
    job_dir = _active_multistart_job_dir(repo)
    if job_dir is not None:
        config_paths.append(job_dir / "origin_station_data" / "station_config.yaml")

    for config_path in config_paths:
        config = load_yaml_mapping(config_path)
        if STATION_TEMPLATE_SOURCE_KEY in config:
            return _canonical_source(config[STATION_TEMPLATE_SOURCE_KEY], default=False)
    return DEFAULT_STATION_TEMPLATE_SOURCE


def station_data_roots_for_update(repo: Path) -> tuple[Path, ...]:
    """Return every persistent data root that an update must refresh."""
    job_dir = _active_multistart_job_dir(repo)
    if job_dir is None:
        return (repo / "station_data",)

    candidates = [job_dir / "origin_station_data"]
    job_state = load_yaml_mapping(job_dir / "state.yaml")
    branches = job_state.get("branches")
    if isinstance(branches, list):
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            seed = branch.get("seed")
            value = branch.get("data_root")
            if value:
                candidate = Path(str(value))
                if not candidate.is_absolute():
                    candidate = job_dir / candidate
            elif seed is not None:
                candidate = job_dir / f"station_data_s{seed}"
            else:
                continue
            candidates.append(candidate)

    candidates.extend(sorted(job_dir.glob("station_data_s*")))
    roots: list[Path] = []
    seen: set[Path] = set()
    resolved_job_dir = job_dir.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(resolved_job_dir)
        except ValueError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        roots.append(candidate)
    return tuple(roots)


def refresh_station_template_files(repo: Path, source: str) -> tuple[Path, ...]:
    """Copy prompt files and codex.md into live or active multistart data roots."""
    template_dir, _canonical = resolve_station_template(repo, source)
    source_files = sorted(template_dir.glob("*_prompts.yaml"))
    codex_path = template_dir / "codex.md"
    if codex_path.is_file():
        source_files.append(codex_path)

    roots = station_data_roots_for_update(repo)
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        for source_file in source_files:
            shutil.copy2(source_file, root / source_file.name)
    return roots


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Refresh Station template files in active data roots")
    parser.add_argument("repo")
    parser.add_argument("source")
    args = parser.parse_args(argv)
    roots = refresh_station_template_files(Path(args.repo), args.source)
    for root in roots:
        print(f"refreshed station template files in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
