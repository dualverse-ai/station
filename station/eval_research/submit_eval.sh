#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "__STATION_PYTHON__" "$SCRIPT_DIR/_internal/submit_eval_cli_snapshot.py" "$@"
