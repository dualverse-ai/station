#!/usr/bin/env bash

# Poll a kissing-margin station and perform the one-time Phase 2 transition
# after an official exact certificate is recorded.

set -uo pipefail

POLL_SECONDS="${POLL_SECONDS:-60}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/station/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

REPO_ROOT="$(pwd -P)"
PACKAGE_ROOT="$REPO_ROOT/example/research_alpha_evolve/kissing_margin"
LIVE_RESEARCH_ROOT="$REPO_ROOT/station_data/rooms/research"
LIVE_EVALUATOR="$LIVE_RESEARCH_ROOT/evaluators/evaluator.py"

if [ ! -f "$REPO_ROOT/setup.py" ] || [ ! -f "$PACKAGE_ROOT/research/evaluators/evaluator_phase2.py" ]; then
  echo "Run this script from the target Station checkout, for example: cd ~/station"
  exit 1
fi

case "$POLL_SECONDS" in
  ''|*[!0-9]*) echo "POLL_SECONDS must be a positive integer."; exit 1 ;;
  0) echo "POLL_SECONDS must be a positive integer."; exit 1 ;;
esac

echo "Watching for an exact-certified Phase 1 result."

while true; do
  if [ -f "$LIVE_EVALUATOR" ] && grep -q 'PHASE_LABEL = "phase2"' "$LIVE_EVALUATOR"; then
    echo "Phase 2 is already active; nothing to do."
    exit 0
  fi

  probe_output="$($PYTHON_BIN - "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import numpy as np
from station import constants, file_io_utils

current_job = file_io_utils.load_yaml(str(repo / "station_multistart" / "current_job.yaml"))
if current_job:
    raise SystemExit(20)

research_root = repo / "station_data" / "rooms" / "research"
evaluations_dir = research_root / "evaluations"
if not evaluations_dir.is_dir():
    raise SystemExit(21)

package_root = repo / "example" / "research_alpha_evolve" / "kissing_margin"
sys.path.insert(0, str(package_root))
import replace_d

task_path = research_root / "research_task.md"
evaluator_path = research_root / "evaluators" / "evaluator.py"
baseline_path = research_root / "baseline.yamll"
try:
    dimension, target_count = replace_d.detect_current_values(
        [task_path.read_text(encoding="utf-8")],
        [evaluator_path.read_text(encoding="utf-8")],
        baseline_path.read_text(encoding="utf-8") if baseline_path.is_file() else "",
    )
except Exception:
    raise SystemExit(23)

def numeric_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return -1

accepted_certificates = {"exact rational certified", "symbolic exact certified"}
for evaluation_path in sorted(evaluations_dir.glob("*.yaml"), key=numeric_id, reverse=True):
    record = file_io_utils.load_yaml(str(evaluation_path))
    if not isinstance(record, dict):
        continue
    final = record.get("final")
    if not isinstance(final, dict) or str(final.get("status", "")).lower() != "completed":
        continue
    try:
        score = float(final.get("primary_score"))
    except (TypeError, ValueError):
        continue
    details = final.get(constants.EVALUATION_DETAILS_KEY)
    if not isinstance(details, dict):
        continue
    if score > 0.0 or details.get("N") != target_count or details.get("Dimension") != dimension:
        continue
    if str(details.get("Certified", "")).strip().lower() not in accepted_certificates:
        continue

    eval_id = evaluation_path.stem
    artifact = research_root / "storage" / "shared" / "submissions" / f"eval_{eval_id}.npz"
    if not artifact.is_file():
        continue
    try:
        with np.load(artifact, allow_pickle=False) as saved:
            certified = str(np.asarray(saved["certified"]).reshape(-1)[0]).strip().lower()
            status = str(np.asarray(saved["status"]).reshape(-1)[0]).strip().lower()
            count = int(np.asarray(saved["target_count"]).reshape(-1)[0])
            artifact_dimension = int(np.asarray(saved["dimension"]).reshape(-1)[0])
            has_exact_payload = "exact_vectors_repr" in saved.files
    except Exception:
        continue
    if certified not in accepted_certificates or status != "accepted":
        continue
    if count != target_count or artifact_dimension != dimension or not has_exact_payload:
        continue
    print(f"{eval_id}\t{artifact}\t{dimension}\t{target_count}")
    raise SystemExit(0)

raise SystemExit(22)
PY
)"
  probe_status=$?

  if [ "$probe_status" -eq 20 ]; then
    echo "Multistart is active; Stage 2 transition is invalid for now. Continuing to poll."
    sleep "$POLL_SECONDS"
    continue
  fi
  if [ "$probe_status" -eq 21 ] || [ "$probe_status" -eq 22 ]; then
    sleep "$POLL_SECONDS"
    continue
  fi
  if [ "$probe_status" -eq 23 ]; then
    echo "Could not detect the live kissing-task dimension and target count."
    exit 23
  fi
  if [ "$probe_status" -ne 0 ]; then
    echo "Could not inspect the live evaluation records (exit $probe_status)."
    exit "$probe_status"
  fi

  IFS=$'\t' read -r eval_id artifact dimension target_count <<< "$probe_output"
  echo "Found exact-certified Phase 1 evaluation $eval_id for N=$target_count, d=$dimension. Requesting transition."

  "$PYTHON_BIN" - "$REPO_ROOT" "$eval_id" "$artifact" "$POLL_SECONDS" "$dimension" "$target_count" <<'PY'
import os
import shutil
import sys
import tempfile
import time
import urllib.error
from pathlib import Path

repo = Path(sys.argv[1])
eval_id = sys.argv[2]
artifact = Path(sys.argv[3])
poll_seconds = max(1.0, float(sys.argv[4]))
dimension = int(sys.argv[5])
target_count = int(sys.argv[6])
sys.path.insert(0, str(repo))

from station import file_io_utils
from station_tools.frontend_api import find_endpoint, request_json

current_job_path = repo / "station_multistart" / "current_job.yaml"
if file_io_utils.load_yaml(str(current_job_path)):
    raise SystemExit(75)

package_root = repo / "example" / "research_alpha_evolve" / "kissing_margin"
research_root = repo / "station_data" / "rooms" / "research"
live_evaluator = research_root / "evaluators" / "evaluator.py"
phase2_evaluator = package_root / "research" / "evaluators" / "evaluator_phase2.py"
phase2_task = package_root / "research" / "research_task_phase2.md"
seed_path = research_root / "storage" / "system" / "v1_rational_certified_config.npz"

sys.path.insert(0, str(package_root))
import replace_d

phase2_evaluator_text = phase2_evaluator.read_text(encoding="utf-8")
phase2_task_text = phase2_task.read_text(encoding="utf-8")
if dimension != 11 or target_count != 594:
    phase2_evaluator_text = replace_d.update_evaluator(
        phase2_evaluator_text,
        dimension,
        target_count,
    )
    phase2_task_text = replace_d.update_task_text(
        phase2_task_text,
        dimension,
        target_count,
        target_count - 1,
    )

old_evaluator = live_evaluator.read_text(encoding="utf-8")
seed_path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=seed_path.parent, delete=False) as temporary:
    temporary_path = Path(temporary.name)
try:
    shutil.copyfile(artifact, temporary_path)
    os.replace(temporary_path, seed_path)
finally:
    temporary_path.unlink(missing_ok=True)

file_io_utils.save_text(phase2_evaluator_text, str(live_evaluator))

found = find_endpoint(repo, "/api/orchestrator/status", timeout=10.0)
if not found:
    file_io_utils.save_text(old_evaluator, str(live_evaluator))
    raise RuntimeError("Station API is not reachable; restored the Phase 1 evaluator.")
endpoint, _ = found

try:
    refresh = request_json(
        endpoint,
        "/api/station/research_evaluator/refresh",
        method="POST",
        timeout=10.0,
    )
except urllib.error.HTTPError as exc:
    file_io_utils.save_text(old_evaluator, str(live_evaluator))
    if exc.code in {409, 423} and file_io_utils.load_yaml(str(current_job_path)):
        raise SystemExit(75)
    raise RuntimeError(f"Evaluator refresh request failed with HTTP {exc.code}; restored the Phase 1 evaluator.") from exc
except Exception:
    file_io_utils.save_text(old_evaluator, str(live_evaluator))
    raise

if refresh.get("success") is not True:
    file_io_utils.save_text(old_evaluator, str(live_evaluator))
    raise RuntimeError(str(refresh.get("error") or refresh))

while True:
    if file_io_utils.load_yaml(str(current_job_path)):
        file_io_utils.save_text(old_evaluator, str(live_evaluator))
        raise SystemExit(75)
    refresh_status = request_json(
        endpoint,
        "/api/station/research_evaluator/refresh",
        timeout=10.0,
    )
    state = refresh_status.get("refresh") if isinstance(refresh_status.get("refresh"), dict) else {}
    state_name = str(state.get("status") or "")
    if state_name == "completed" and refresh_status.get("is_paused") is True:
        break
    if state_name == "failed":
        file_io_utils.save_text(old_evaluator, str(live_evaluator))
        raise RuntimeError(str(state.get("error") or "Evaluator refresh failed."))
    time.sleep(poll_seconds)

task_snapshot = request_json(endpoint, "/api/station/research_task_spec", timeout=10.0)
task_update = request_json(
    endpoint,
    "/api/station/research_task_spec",
    method="PUT",
    payload={
        "raw_markdown": phase2_task_text,
        "expected_revision": str(task_snapshot.get("revision") or ""),
    },
    timeout=10.0,
)
if task_update.get("success") is not True:
    raise RuntimeError(str(task_update.get("error") or task_update))

status_response = request_json(endpoint, "/api/orchestrator/status", timeout=10.0)
status = status_response.get("status") if isinstance(status_response.get("status"), dict) else {}
target_agents = status.get("turn_order") if isinstance(status.get("turn_order"), list) else []
if target_agents:
    message = """**Architect Message**

Congratulations on reaching an exact certificate. Phase 1 is officially solved, and the station has transitioned to Phase 2.

The goal is now to find exact certificates for configurations with N as large as possible. You may submit any configuration with more spheres than the Phase 1 target. Please run `read_task` again before your next Research Center submission.

Analysis of the certified Phase 1 configuration remains useful when it advances structural understanding, but the primary objective is now the strongest kissing-number lower bound you can prove."""
    sent = request_json(
        endpoint,
        "/api/station/send_system_message",
        method="POST",
        payload={"target_agents": target_agents, "message_content": message},
        timeout=30.0,
    )
    if sent.get("success") is not True:
        raise RuntimeError(str(sent.get("message") or sent))

resumed = request_json(endpoint, "/api/orchestrator/resume", method="POST", timeout=10.0)
if resumed.get("success") is not True:
    raise RuntimeError(str(resumed.get("message") or resumed))

print(f"Phase 2 transition completed from evaluation {eval_id} and the station resumed.")
PY
  transition_status=$?

  if [ "$transition_status" -eq 75 ]; then
    echo "Multistart began before the transition; continuing to poll."
    sleep "$POLL_SECONDS"
    continue
  fi
  if [ "$transition_status" -ne 0 ]; then
    echo "Stage 2 transition failed. If the refresh had already succeeded, the station remains paused for inspection."
    exit "$transition_status"
  fi
  exit 0
done
