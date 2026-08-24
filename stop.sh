#!/bin/bash
# Production stop script for the Station

echo "=== Stopping Station Production Services ==="

# --- Argument Parsing ---
FORCE_STOP=false
while [ $# -gt 0 ]; do
    case "$1" in
        --force)
            FORCE_STOP=true
            ;;
        -h|--help)
            echo "Usage: ./stop.sh [--force]"
            echo "  Default: pause the station and wait for queued/running Station jobs (including Research, Archive Surveyor, and External Counter requests) and web Surveyor requests to finish before stopping."
            echo "  --force: bypass pause/drain checks and run immediate cleanup."
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument '$1'"
            echo "Usage: ./stop.sh [--force]"
            exit 1
            ;;
    esac
    shift
done

# --- Configuration ---
CURRENT_DIR=$(pwd)
DEPLOYMENT_DIR="deployment"
ENV_FILE=".env"
PID_FILE="$DEPLOYMENT_DIR/gunicorn.pid"
NGINX_PID_FILE="$DEPLOYMENT_DIR/nginx.pid"

if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [ -z "$line" ] || [[ "$line" == \#* ]] || [[ "$line" != *=* ]]; then
            continue
        fi
        key="${line%%=*}"
        value="${line#*=}"
        if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
            continue
        fi
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        export "$key=$value"
    done < "$ENV_FILE"
fi

FLASK_PORT=${FLASK_PORT:-5000}
API_BASE_URL="http://127.0.0.1:$FLASK_PORT"
AUTH_ARGS=(-u "${FLASK_AUTH_USERNAME:-admin}:${FLASK_AUTH_PASSWORD:-changeme}")
STOP_API_TIMEOUT_SECONDS=${STOP_API_TIMEOUT_SECONDS:-10}
STOP_STATS_API_TIMEOUT_SECONDS=${STOP_STATS_API_TIMEOUT_SECONDS:-60}
STOP_STATS_RETRIES=${STOP_STATS_RETRIES:-3}
STOP_STATS_RETRY_DELAY_SECONDS=${STOP_STATS_RETRY_DELAY_SECONDS:-5}
MULTISTART_STOP_TIMEOUT_SECONDS=${MULTISTART_STOP_TIMEOUT_SECONDS:-7200}
SKIP_EXTERNAL_RETRY_DRAIN=false

api_curl() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy \
        curl --noproxy '*' "$@"
}

api_request() {
    api_request_with_timeout "$STOP_API_TIMEOUT_SECONDS" "$@"
}

api_request_with_timeout() {
    local timeout_seconds="$1"
    shift
    api_curl -sS --fail --max-time "$timeout_seconds" "${AUTH_ARGS[@]}" "$@"
}

api_stats_request() {
    api_request_with_timeout "$STOP_STATS_API_TIMEOUT_SECONDS" "$@"
}

read_station_statistics() {
    local attempt
    local stats_response

    for ((attempt = 1; attempt <= STOP_STATS_RETRIES; attempt++)); do
        if stats_response=$(api_stats_request "$API_BASE_URL/api/station/statistics"); then
            printf '%s\n' "$stats_response"
            return 0
        fi

        if [ "$attempt" -lt "$STOP_STATS_RETRIES" ]; then
            echo "Station statistics did not respond within ${STOP_STATS_API_TIMEOUT_SECONDS}s; retrying (${attempt}/${STOP_STATS_RETRIES})..." >&2
            sleep "$STOP_STATS_RETRY_DELAY_SECONDS"
        fi
    done

    return 1
}

pid_file_is_running() {
    local pid_file="$1"
    local pid
    if [ ! -f "$pid_file" ]; then
        return 1
    fi
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -z "$pid" ]; then
        return 1
    fi
    ps -p "$pid" >/dev/null 2>&1
}

find_station_gunicorn_pids() {
    local pid
    local cwd
    for pid in $(pgrep gunicorn 2>/dev/null); do
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
        if [ "$cwd" = "$CURRENT_DIR" ]; then
            printf '%s ' "$pid"
        fi
    done
}

find_station_nginx_pids() {
    local nginx_conf_path="$CURRENT_DIR/$DEPLOYMENT_DIR/nginx.conf"
    local pid
    local args
    if [ ! -f "$nginx_conf_path" ]; then
        return 0
    fi
    while read -r pid args; do
        if [[ "$args" == *"nginx: master process"* && "$args" == *"$nginx_conf_path"* ]]; then
            printf '%s ' "$pid"
        fi
    done < <(ps -eo pid=,args= 2>/dev/null)
}

wait_for_pid_to_exit() {
    local pid="$1"
    local timeout_seconds="${2:-10}"
    local waited=0
    while ps -p "$pid" >/dev/null 2>&1; do
        if [ "$waited" -ge "$timeout_seconds" ]; then
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 0
}

stop_station_nginx_processes() {
    local nginx_conf_path="$CURRENT_DIR/$DEPLOYMENT_DIR/nginx.conf"
    local pids
    local pid
    local remaining=()

    pids=$(find_station_nginx_pids)
    if [ -z "$pids" ] && [ -f "$NGINX_PID_FILE" ]; then
        pid=$(cat "$NGINX_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
            pids="$pid"
        fi
    fi

    if [ -z "$pids" ]; then
        rm -f "$NGINX_PID_FILE"
        echo "✓ No nginx found for this station."
        return 0
    fi

    echo "Found nginx process(es) for this station: $pids"
    if [ -f "$nginx_conf_path" ]; then
        if nginx -c "$nginx_conf_path" -s quit 2>/dev/null; then
            echo "✓ Nginx graceful stop requested."
        elif command -v sudo >/dev/null 2>&1 && sudo nginx -c "$nginx_conf_path" -s quit 2>/dev/null; then
            echo "✓ Nginx graceful stop requested."
        else
            echo "Graceful nginx stop command failed; falling back to direct process termination."
        fi
    fi

    for pid in $pids; do
        if wait_for_pid_to_exit "$pid" 10; then
            continue
        fi
        echo "Nginx PID $pid did not exit after graceful quit; terminating it."
        if command -v sudo >/dev/null 2>&1; then
            sudo kill "$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
        else
            kill "$pid" 2>/dev/null || true
        fi
        if ! wait_for_pid_to_exit "$pid" 5; then
            remaining+=("$pid")
        fi
    done

    rm -f "$NGINX_PID_FILE"
    if [ "${#remaining[@]}" -gt 0 ]; then
        echo "ERROR: Could not stop nginx process(es) for this station: ${remaining[*]}"
        return 1
    fi
    echo "✓ Nginx stopped."
    return 0
}

has_running_station_services() {
    if pid_file_is_running "$PID_FILE" || pid_file_is_running "$NGINX_PID_FILE"; then
        return 0
    fi
    if [ -n "$(find_station_gunicorn_pids)" ] || [ -n "$(find_station_nginx_pids)" ]; then
        return 0
    fi
    return 1
}

parse_orchestrator_status_field() {
    local response="$1"
    local field="$2"
    RESPONSE="$response" FIELD="$field" python3 - <<'PY'
import json
import os
import sys

try:
    payload = json.loads(os.environ.get("RESPONSE", ""))
except Exception:
    sys.exit(1)
status = payload.get("status") or {}
value = status.get(os.environ.get("FIELD", ""))
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

parse_station_job_counts() {
    local response="$1"
    RESPONSE="$response" SKIP_EXTERNAL_RETRY_DRAIN="$SKIP_EXTERNAL_RETRY_DRAIN" python3 - <<'PY'
import json
import os
import sys

try:
    payload = json.loads(os.environ.get("RESPONSE", ""))
except Exception:
    sys.exit(1)
stats = payload.get("statistics") or {}
skip_external = os.environ.get("SKIP_EXTERNAL_RETRY_DRAIN") == "true"

def count_jobs(list_key, drainable_count_key, count_key, legacy_count_key):
    jobs = stats.get(list_key)
    if isinstance(jobs, list):
        return sum(
            1
            for job in jobs
            if isinstance(job, dict)
            and job.get("drainable", True) is not False
            and not (skip_external and job.get("job_type") == "external_report")
        )
    return int(stats.get(
        drainable_count_key,
        stats.get(count_key, stats.get(legacy_count_key)),
    ) or 0)

running = count_jobs(
    "running_jobs",
    "drainable_running_jobs_count",
    "running_jobs_count",
    "running_experiments_count",
)
queued = count_jobs(
    "queued_jobs",
    "drainable_queued_jobs_count",
    "queued_jobs_count",
    "queued_experiments_count",
)
print(f"{running} {queued}")
PY
}

update_external_retry_drain_state() {
    local status_response="$1"
    local pause_reason
    pause_reason=$(parse_orchestrator_status_field "$status_response" "pause_reason") || return 0
    if [[ "$pause_reason" == *"External report"* && "$pause_reason" == *"failed and is being requeued"* ]]; then
        SKIP_EXTERNAL_RETRY_DRAIN=true
    fi
}

read_web_archive_survey_counts() {
    local index_path="$CURRENT_DIR/station_data/web_interface/archive_surveyor/index/web_archive_surveys.sqlite3"
    if [ ! -f "$index_path" ]; then
        echo "0 0"
        return 0
    fi

    python3 - "$index_path" <<'PY'
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

path = Path(sys.argv[1]).resolve()
uri = f"file:{quote(str(path))}?mode=ro"
try:
    with sqlite3.connect(uri, uri=True, timeout=30) as connection:
        rows = dict(connection.execute(
            "SELECT status, COUNT(*) FROM surveys WHERE status IN ('running', 'queued') GROUP BY status"
        ).fetchall())
except Exception as exc:
    print(f"ERROR: Could not read web Surveyor queue: {exc}", file=sys.stderr)
    sys.exit(1)
print(f"{int(rows.get('running', 0))} {int(rows.get('queued', 0))}")
PY
}

wait_for_station_to_pause() {
    local status_response
    local is_running
    local is_paused

    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: curl is required for safe stop checks. Use ./stop.sh --force to bypass checks."
        return 1
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 is required for safe stop checks. Use ./stop.sh --force to bypass checks."
        return 1
    fi

    if ! status_response=$(api_request "$API_BASE_URL/api/orchestrator/status"); then
        echo "ERROR: Station services are running, but the local API is not reachable."
        echo "       Use ./stop.sh --force to bypass pause/drain checks."
        return 1
    fi

    if ! is_running=$(parse_orchestrator_status_field "$status_response" "is_running"); then
        echo "ERROR: Could not parse orchestrator status. Use ./stop.sh --force to bypass checks."
        return 1
    fi
    if ! is_paused=$(parse_orchestrator_status_field "$status_response" "is_paused"); then
        echo "ERROR: Could not parse orchestrator status. Use ./stop.sh --force to bypass checks."
        return 1
    fi
    update_external_retry_drain_state "$status_response"

    if [ "$is_running" != "true" ]; then
        echo "✓ Orchestrator is not running; no pause needed."
        return 0
    fi
    if [ "$is_paused" = "true" ]; then
        echo "✓ Orchestrator is already paused."
        return 0
    fi

    echo "Requesting station pause before shutdown..."
    if ! api_request -X POST "$API_BASE_URL/api/orchestrator/pause" >/dev/null; then
        echo "ERROR: Failed to request station pause. Use ./stop.sh --force to bypass checks."
        return 1
    fi

    echo -n "Waiting for station to pause."
    while true; do
        sleep 5
        if ! status_response=$(api_request "$API_BASE_URL/api/orchestrator/status"); then
            echo ""
            echo "ERROR: Lost access to station API while waiting for pause."
            echo "       Use ./stop.sh --force to bypass checks."
            return 1
        fi
        if ! is_running=$(parse_orchestrator_status_field "$status_response" "is_running"); then
            echo ""
            echo "ERROR: Could not parse orchestrator status while waiting for pause."
            return 1
        fi
        if ! is_paused=$(parse_orchestrator_status_field "$status_response" "is_paused"); then
            echo ""
            echo "ERROR: Could not parse orchestrator status while waiting for pause."
            return 1
        fi
        update_external_retry_drain_state "$status_response"
        if [ "$is_running" != "true" ]; then
            echo ""
            echo "✓ Orchestrator stopped while pause was pending."
            return 0
        fi
        if [ "$is_paused" = "true" ]; then
            echo ""
            echo "✓ Station paused."
            return 0
        fi
        echo -n "."
    done
}

wait_for_station_jobs_to_finish() {
    local stats_response
    local counts
    local running_count
    local queued_count
    local web_counts
    local web_running_count
    local web_queued_count

    echo "Reading station statistics (timeout: ${STOP_STATS_API_TIMEOUT_SECONDS}s)..."
    if ! stats_response=$(read_station_statistics); then
        echo "ERROR: Could not read station statistics. Use ./stop.sh --force to bypass checks."
        return 1
    fi
    if ! counts=$(parse_station_job_counts "$stats_response"); then
        echo "ERROR: Could not parse station statistics. Use ./stop.sh --force to bypass checks."
        return 1
    fi
    read -r running_count queued_count <<< "$counts"
    if ! web_counts=$(read_web_archive_survey_counts); then
        echo "ERROR: Could not read web Surveyor state. Use ./stop.sh --force to bypass checks."
        return 1
    fi
    read -r web_running_count web_queued_count <<< "$web_counts"

    if [ "${running_count:-0}" -eq 0 ] && [ "${queued_count:-0}" -eq 0 ] && \
       [ "${web_running_count:-0}" -eq 0 ] && [ "${web_queued_count:-0}" -eq 0 ]; then
        echo "✓ No queued or running Station jobs, External Counter requests, or web Surveyor requests."
        return 0
    fi

    echo "Waiting for Station jobs, including Research, Archive Surveyor, and External Counter requests, plus web Surveyor requests to finish (Station running: ${running_count:-0}, Station queued: ${queued_count:-0}, web Surveyor running: ${web_running_count:-0}, web Surveyor queued: ${web_queued_count:-0})..."
    while true; do
        sleep 10
        if ! stats_response=$(read_station_statistics); then
            echo "ERROR: Lost access to station statistics while waiting for experiments."
            echo "       Use ./stop.sh --force to bypass checks."
            return 1
        fi
        if ! counts=$(parse_station_job_counts "$stats_response"); then
            echo "ERROR: Could not parse station statistics while waiting for jobs."
            return 1
        fi
        read -r running_count queued_count <<< "$counts"
        if ! web_counts=$(read_web_archive_survey_counts); then
            echo "ERROR: Could not read web Surveyor state while waiting."
            return 1
        fi
        read -r web_running_count web_queued_count <<< "$web_counts"
        if [ "${running_count:-0}" -eq 0 ] && [ "${queued_count:-0}" -eq 0 ] && \
           [ "${web_running_count:-0}" -eq 0 ] && [ "${web_queued_count:-0}" -eq 0 ]; then
            echo "✓ Station jobs, including External Counter requests, and web Surveyor requests finished."
            return 0
        fi
        echo "Still waiting (Station running: ${running_count:-0}, Station queued: ${queued_count:-0}, web Surveyor running: ${web_running_count:-0}, web Surveyor queued: ${web_queued_count:-0})..."
    done
}

CONDA_ENV_NAME=${CONDA_ENV_NAME:-station}
activate_station_conda() {
    if [ -n "${_STATION_CONDA_ACTIVE:-}" ]; then
        return 0
    fi

    if [ -z "${CONDA_BIN_PATH:-}" ]; then
        if command -v conda >/dev/null 2>&1; then
            CONDA_BIN_PATH=$(command -v conda)
        else
            return 0
        fi
    fi

    local conda_base_dir
    conda_base_dir=$(dirname "$(dirname "$CONDA_BIN_PATH")")
    local conda_sh_path="$conda_base_dir/etc/profile.d/conda.sh"
    if [ ! -f "$conda_sh_path" ]; then
        return 0
    fi

    # shellcheck source=/dev/null
    . "$conda_sh_path"
    conda activate "$CONDA_ENV_NAME" >/dev/null 2>&1 || return 0
    _STATION_CONDA_ACTIVE=1
    return 0
}

stop_multistart_controller() {
    if [ "${STATION_MULTISTART_SKIP_CONTROLLER_STOP:-}" = "1" ]; then
        return 0
    fi
    activate_station_conda || true
    if ! command -v python >/dev/null 2>&1; then
        echo "ERROR: Python executable not available; cannot verify that multistart processes stopped."
        return 1
    fi
    echo "Stopping multistart controller/branches (graceful timeout: ${MULTISTART_STOP_TIMEOUT_SECONDS}s)..."
    if [ "$FORCE_STOP" = true ]; then
        if ! python -m station.multistart.controller stop --repo "$CURRENT_DIR" --force; then
            echo "ERROR: Failed to force-stop multistart controller/branches."
            return 1
        fi
    else
        if ! python -m station.multistart.controller stop --repo "$CURRENT_DIR" --timeout-seconds "$MULTISTART_STOP_TIMEOUT_SECONDS"; then
            echo "ERROR: Multistart controller/branches did not stop cleanly."
            echo "       Wait longer by setting MULTISTART_STOP_TIMEOUT_SECONDS, or use ./stop.sh --force if you want to bypass the graceful wait."
            return 1
        fi
    fi
    echo "✓ Multistart controller/branches stopped."
}

stop_multistart_controller || exit 1

if [ "$FORCE_STOP" != true ]; then
    if [ -z "$(find_station_gunicorn_pids)" ] && ! pid_file_is_running "$PID_FILE" && [ -n "$(find_station_nginx_pids)" ]; then
        echo "Only stale nginx process(es) are running for this station; cleaning them before safe-stop checks."
        stop_station_nginx_processes || exit 1
    fi

    if ! has_running_station_services; then
        echo "✓ No running production services found for this station. Nothing to stop."
        exit 0
    fi

    wait_for_station_to_pause || exit 1
    wait_for_station_jobs_to_finish || exit 1
else
    echo "Force stop requested; bypassing pause and job drain checks. Active External Counter and web Surveyor requests may be requeued during shutdown."
fi

# --- Stop Nginx ---
echo "Stopping Nginx..."
stop_station_nginx_processes || exit 1

# --- Stop Gunicorn ---
echo "Stopping Gunicorn..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        # Try graceful station cleanup via API first
        if [ "$FORCE_STOP" != true ] && [ -f "$ENV_FILE" ]; then
            FLASK_PORT=${FLASK_PORT:-5000}
            echo "Requesting graceful station cleanup..."
            if command -v curl >/dev/null 2>&1; then
                curl --max-time 10 -X POST -u "${FLASK_AUTH_USERNAME:-admin}:${FLASK_AUTH_PASSWORD}" \
                     "http://127.0.0.1:$FLASK_PORT/api/shutdown" >/dev/null 2>&1 || true
            fi
        fi

        echo "Sending TERM signal to Gunicorn (PID: $PID)..."
        kill $PID
        
        # Wait for process to terminate
        echo -n "Waiting for Gunicorn to stop."
        for _ in {1..10}; do # Wait up to 10 seconds
            if ! ps -p $PID > /dev/null; then
                echo "" # Newline
                echo "✓ Gunicorn stopped."
                rm -f "$PID_FILE"
                break
            fi
            echo -n "."
            sleep 1
        done

        # Force kill if it's still running
        if ps -p $PID > /dev/null; then
            echo "" # Newline
            echo "Gunicorn did not stop gracefully. Sending KILL signal..."
            kill -9 $PID
            echo "✓ Gunicorn force-killed."
            rm -f "$PID_FILE"
        fi
    else
        echo "✓ Gunicorn was not running (stale PID file found)."
        rm -f "$PID_FILE"
    fi
else
    echo "✓ Gunicorn was not running (no PID file)."
fi

# Kill any orphaned gunicorn processes running from this station's directory
echo "Checking for orphaned Gunicorn processes in this station..."
ORPHANED_GUNICORN_PIDS=""
for pid in $(pgrep gunicorn 2>/dev/null); do
    cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
    if [ "$cwd" = "$CURRENT_DIR" ]; then
        ORPHANED_GUNICORN_PIDS="$ORPHANED_GUNICORN_PIDS $pid"
    fi
done

if [ -n "$ORPHANED_GUNICORN_PIDS" ]; then
    echo "Found orphaned Gunicorn processes for this station:$ORPHANED_GUNICORN_PIDS"
    for pid in $ORPHANED_GUNICORN_PIDS; do
        echo "  Killing PID $pid..."
        kill -9 $pid 2>/dev/null || true
    done
    echo "✓ Orphaned Gunicorn processes killed."
else
    echo "✓ No orphaned Gunicorn processes found."
fi

# --- Stop Research Evaluation Processes ---
echo "Stopping research evaluation processes for this station..."
# Kill wrapper.py processes specific to this station's directory
WRAPPER_PIDS=$(pgrep -f "python.*$CURRENT_DIR.*wrapper.py" | tr '\n' ' ')
if [ -n "$WRAPPER_PIDS" ]; then
    echo "Killing evaluation wrapper processes: $WRAPPER_PIDS"
    kill -9 $WRAPPER_PIDS 2>/dev/null || true
    echo "✓ Research evaluation processes stopped."
else
    echo "✓ No research evaluation processes running for this station."
fi

echo "Requeueing active Research Center instruction prompts for this station..."
activate_station_conda || true
if command -v python >/dev/null 2>&1; then
    if python scripts/restart_eval.py --shutdown-requeue-active; then
        echo "✓ Active Research Center instruction prompts requeued."
    else
        echo "⚠️  Failed to requeue active Research Center instruction prompts."
    fi
else
    echo "⚠️  Python executable not available; could not requeue active Research Center instruction prompts."
fi

# --- Final Check for Any Remaining Gunicorn Processes ---
echo ""
echo "Performing final check for gunicorn processes..."
REMAINING_GUNICORN=""
OTHER_GUNICORN_CWDS=$'\n'
OTHER_STATION_COUNT=0
for pid in $(pgrep gunicorn 2>/dev/null); do
    cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
    if [ "$cwd" = "$CURRENT_DIR" ]; then
        cmd=$(ps -p $pid -o args= 2>/dev/null | head -c 200)
        REMAINING_GUNICORN="${REMAINING_GUNICORN}PID $pid\n  Command: $cmd...\n"
    elif [ -n "$cwd" ]; then
        case "$OTHER_GUNICORN_CWDS" in
            *$'\n'"$cwd"$'\n'*)
                ;;
            *)
                OTHER_GUNICORN_CWDS="${OTHER_GUNICORN_CWDS}${cwd}"$'\n'
                OTHER_STATION_COUNT=$((OTHER_STATION_COUNT + 1))
                ;;
        esac
    fi
done

if [ -n "$REMAINING_GUNICORN" ]; then
    echo "⚠️  WARNING: Gunicorn processes for this station are still running:"
    echo ""
    echo -e "$REMAINING_GUNICORN"
    echo "    Please stop these processes manually if needed."
    if [ "$OTHER_STATION_COUNT" -gt 0 ]; then
        echo "    $OTHER_STATION_COUNT other station instance(s) also have gunicorn running."
    fi
elif [ "$OTHER_STATION_COUNT" -gt 0 ]; then
    echo "✓ No gunicorn processes remain for this station."
    echo "  $OTHER_STATION_COUNT other station instance(s) still have gunicorn running."
else
    echo "✓ No gunicorn processes found on the system."
fi

echo ""
echo "=== Production Services Stopped ==="
