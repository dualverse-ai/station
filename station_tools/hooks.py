"""Optional local shell hooks for private machine-specific behavior."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Mapping

from .config import ToolsConfig, build_hook_env


@dataclass(frozen=True)
class HookRunner:
    config: ToolsConfig
    disabled: bool = False

    def command_for(self, scope: str, name: str) -> str | None:
        if self.disabled:
            return None
        scoped = self.config.hooks.get(scope, {})
        return scoped.get(name) or self.config.hooks.get("global", {}).get(name)

    def run(self, scope: str, name: str, cwd: str | None = None, optional: bool = False) -> int:
        command = self.command_for(scope, name)
        if not command:
            return 0
        print(f"running {scope}.{name} hook")
        result = subprocess.run(
            ["bash", "-ic", command],
            cwd=cwd,
            env=build_hook_env(self.config),
            check=False,
        )
        if result.returncode != 0 and not optional:
            raise RuntimeError(f"{scope}.{name} hook failed with rc={result.returncode}")
        return result.returncode
