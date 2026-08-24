from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


TEST_INIT_AGENTS = ["GPT-5.5", "Gemini 3.1 Pro"]
TEST_PAUSE_AFTER_TICK_END = 20

NO_MULTISTART_OVERRIDES: Dict[str, int] = {
    "MULTISTART_INIT_MAX_PARALLEL": 1,
    "MULTISTART_INIT_SEEDS": 0,
    "MULTISTART_INIT_ROLL_TICKS": 0,
    "MULTISTART_STAGNATION_MAX_PARALLEL": 1,
    "MULTISTART_STAGNATION_SEEDS": 0,
    "MULTISTART_STAGNATION_ROLL_TICKS": 0,
}

TEST_CONSTANT_OVERRIDES: Dict[str, int] = {
    **NO_MULTISTART_OVERRIDES,
    "PAUSE_AFTER_TICK_END": TEST_PAUSE_AFTER_TICK_END,
}


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return data


def _atomic_write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _update_constant_config(station_data: Path, overrides: Dict[str, Any]) -> None:
    constant_config_path = station_data / "constant_config.yaml"
    constant_config = _load_yaml_mapping(constant_config_path)
    constant_config.update(overrides)
    _atomic_write_yaml(constant_config_path, constant_config)


def apply_no_multistart(station_data: Path) -> None:
    _update_constant_config(station_data, NO_MULTISTART_OVERRIDES)


def apply_test_config(station_data: Path) -> None:
    _update_constant_config(station_data, TEST_CONSTANT_OVERRIDES)
    _atomic_write_yaml(station_data / "init_agents.yaml", TEST_INIT_AGENTS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Station startup config overrides")
    parser.add_argument("--station-data", default="station_data")
    parser.add_argument("--no-multistart", action="store_true", help="Disable init and stagnation multistart")
    parser.add_argument("--test", action="store_true", help="Apply quick-test startup overrides")
    args = parser.parse_args(argv)

    station_data = Path(args.station_data)
    if args.test:
        apply_test_config(station_data)
    elif args.no_multistart:
        apply_no_multistart(station_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
