#!/bin/bash
# Production startup script for the Station

echo "=== Starting Station Production Services ==="

# --- Argument Parsing ---
STATION_NAME_OVERRIDE=""
AUTO_START_STATION=false
FORCE_STOP=false
REBUILD_DB=false
TEST_MODE=false
NO_MULTISTART=false
REQUEUE_FAILED_EXTERNAL_REPORTS=false
while [ $# -gt 0 ]; do
    case "$1" in
        --name)
            shift
            STATION_NAME_OVERRIDE="$1"
            ;;
        --name=*)
            STATION_NAME_OVERRIDE="${1#*=}"
            ;;
        -s|--start)
            AUTO_START_STATION=true
            ;;
        --force)
            FORCE_STOP=true
            ;;
        --rebuild-db|--rebuild_db)
            REBUILD_DB=true
            ;;
        --test)
            TEST_MODE=true
            ;;
        --no-multistart|--no_multistart)
            NO_MULTISTART=true
            ;;
        --requeue-failed-external-reports)
            REQUEUE_FAILED_EXTERNAL_REPORTS=true
            ;;
        -h|--help)
            echo "Usage: ./start.sh [--name station_name] [--start|-s] [--force] [--rebuild-db] [--test|--no-multistart] [--requeue-failed-external-reports]"
            echo "  --force: pass --force to stop.sh before starting."
            echo "  --rebuild-db: rebuild derived SQLite station indexes from YAML before starting."
            echo "  --test: apply quick-test overrides before starting."
            echo "  --no-multistart: write runtime overrides disabling init and stagnation multistart."
            echo "  --requeue-failed-external-reports: one-time recovery of failed or failure-requeued External Counter reports whose authors are still active."
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument '$1'"
            echo "Usage: ./start.sh [--name station_name] [--start|-s] [--force] [--rebuild-db] [--test|--no-multistart] [--requeue-failed-external-reports]"
            exit 1
            ;;
    esac
    shift
done

# --- Configuration ---
DEPLOYMENT_DIR="deployment"
ENV_FILE=".env"

PID_FILE="$DEPLOYMENT_DIR/gunicorn.pid"
NGINX_CONF="$DEPLOYMENT_DIR/nginx.conf"
NGINX_PID_FILE="$DEPLOYMENT_DIR/nginx.pid"
ACCESS_LOG="$DEPLOYMENT_DIR/access.log"
ERROR_LOG="$DEPLOYMENT_DIR/error.log"

# --- Station Data Bootstrap ---
STATION_DATA_DIR="station_data"
DRAFT_STATION_DATA_DIR="draft_station_data"

if [ ! -e "$STATION_DATA_DIR" ] && [ -d "$DRAFT_STATION_DATA_DIR" ]; then
    echo "station_data not found; initializing from draft_station_data..."
    if ! cp -a "$DRAFT_STATION_DATA_DIR" "$STATION_DATA_DIR"; then
        echo "ERROR: Failed to initialize station_data from draft_station_data."
        exit 1
    fi
    echo "✓ station_data initialized from draft_station_data."
fi

# --- Source Environment Variables ---
load_env_file() {
    local line
    local key
    local value

    if [ ! -f "$ENV_FILE" ]; then
        return 1
    fi

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
}

if [ -f "$ENV_FILE" ]; then
    load_env_file
else
    echo "ERROR: $ENV_FILE not found. Run ./deploy.sh first."
    exit 1
fi

AUTH_ARGS=()
if [ -n "$FLASK_AUTH_USERNAME" ] || [ -n "$FLASK_AUTH_PASSWORD" ]; then
    AUTH_ARGS=(-u "${FLASK_AUTH_USERNAME:-admin}:${FLASK_AUTH_PASSWORD:-changeme}")
fi
FLASK_PORT=${FLASK_PORT:-5000}
NGINX_HTTP_PORT=${NGINX_HTTP_PORT:-80}
NGINX_HTTPS_PORT=${NGINX_HTTPS_PORT:-8443}
GUNICORN_THREADS=${GUNICORN_THREADS:-8}

# Always bypass proxies for local loopback API calls.
api_curl() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy \
        curl --noproxy '*' "$@"
}

resume_multistart_branches_if_autostart() {
    if [ "$AUTO_START_STATION" != true ]; then
        return 0
    fi
    python - <<'PY'
import sys
import time
from pathlib import Path
from station.multistart import ipc, state
from station.multistart.controller import find_running_controller_pid

repo = Path.cwd()
deadline = time.monotonic() + 30.0
last_error = "controller did not respond"

def resume_completed_server_side():
    current = state.load_current_job(repo)
    job_path = Path(str(current.get("job_dir") or ""))
    if not job_path.is_dir():
        return False
    payload = state.load_job_state(job_path)
    return (
        state.job_control(payload) == state.CONTROL_RUNNING
        and find_running_controller_pid(repo) is not None
    )

while time.monotonic() < deadline:
    try:
        response = ipc.request_resume_branches(repo=repo, timeout=30.0)
    except Exception as exc:
        if resume_completed_server_side():
            print("✓ Auto-start requested; multistart branch rolling resumed (confirmed after IPC timeout).")
            sys.exit(0)
        last_error = str(exc)
        time.sleep(1.0)
        continue
    if response.get("success") is True:
        print("✓ Auto-start requested; multistart branch rolling resumed.")
        sys.exit(0)
    if resume_completed_server_side():
        print("✓ Auto-start requested; multistart branch rolling resumed (confirmed from job state).")
        sys.exit(0)
    last_error = str(response.get("error") or "unknown error")
    time.sleep(1.0)

print(f"ERROR: Auto-start requested, but multistart resume failed: {last_error}")
sys.exit(1)
PY
}

# TEMPORARY: Remove after External Counter failures from the API 5 tool-capability
# incident have been recovered on all affected stations.
requeue_failed_external_reports_once() {
    python - "$STATION_DATA_DIR" <<'PY'
import os
import sys
from pathlib import Path

from station import agent as agent_module
from station import constants, file_io_utils
from station.eval_external import pending_queue
from station.multistart import state, waiting

repo = Path.cwd().resolve()
data_roots = []

def add_root(path):
    path = Path(path).resolve()
    if path.is_dir() and path not in data_roots:
        data_roots.append(path)

add_root(sys.argv[1])
active_job = waiting.active_job(repo)
if active_job:
    job_dir = Path(str(active_job.get("job_dir") or ""))
    if job_dir.is_dir():
        add_root(job_dir / state.ORIGIN_DIR_NAME)
        for branch_root in sorted(job_dir.glob("station_data_s*")):
            add_root(branch_root)

total_requeued = 0
for station_data in data_roots:
    constants.BASE_STATION_DATA_PATH = str(station_data)
    active_agents = set(agent_module.get_all_active_agent_names())
    reports_dir = station_data / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_EXTERNAL / constants.EXTERNAL_REPORTS_SUBDIR_NAME
    pending_path = station_data / constants.ROOMS_DIR_NAME / constants.SHORT_ROOM_NAME_EXTERNAL / constants.PENDING_EXTERNAL_REPORTS_FILENAME
    requeued = []
    skipped = []

    for filename in file_io_utils.list_files(str(reports_dir), constants.YAML_EXTENSION):
        if not filename.startswith("report_"):
            continue
        report_path = reports_dir / filename
        report = file_io_utils.load_yaml(str(report_path))
        if not isinstance(report, dict):
            continue
        status = report.get(constants.EXTERNAL_REPORT_STATUS_KEY)
        try:
            requeue_count = int(report.get(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY) or 0)
        except (TypeError, ValueError):
            requeue_count = 0
        failed_retry = (
            status in (
                constants.EXTERNAL_REPORT_STATUS_PENDING,
                constants.EXTERNAL_REPORT_STATUS_RUNNING,
            )
            and requeue_count > 0
        )
        if status != constants.EXTERNAL_REPORT_STATUS_FAILED and not failed_retry:
            continue

        report_id = str(report.get(constants.EXTERNAL_REPORT_ID_KEY) or "").strip()
        author = str(report.get(constants.EXTERNAL_REPORT_AUTHOR_KEY) or "").strip()
        if not report_id or author not in active_agents:
            skipped.append(report_id or filename)
            continue

        report[constants.EXTERNAL_REPORT_STATUS_KEY] = constants.EXTERNAL_REPORT_STATUS_PENDING
        report.pop(constants.EXTERNAL_REPORT_START_TICK_KEY, None)
        report.pop(constants.EXTERNAL_REPORT_COMPLETED_TICK_KEY, None)
        report.pop(constants.EXTERNAL_REPORT_NEXT_RETRY_AT_KEY, None)
        report.pop(constants.EXTERNAL_REPORT_REQUEUE_COUNT_KEY, None)
        report.pop(constants.EXTERNAL_REPORT_ERROR_KEY, None)
        file_io_utils.save_yaml(report, str(report_path))
        pending_queue.remove(str(pending_path), report_id, constants.EXTERNAL_REPORT_ID_KEY)
        pending_queue.append(str(pending_path), report, constants.EXTERNAL_REPORT_ID_KEY)
        requeued.append(report_id)

    total_requeued += len(requeued)
    if requeued:
        print(f"{station_data}: requeued External Counter report(s): {', '.join(requeued)}")
    if skipped:
        print(f"{station_data}: skipped inactive-author report(s): {', '.join(skipped)}")

if total_requeued == 0:
    print("No eligible failed or failure-requeued External Counter reports found.")
PY
}

start_loopback_gunicorn() {
    echo "Starting Gunicorn application server..."
    if [ "$REBUILD_DB" = true ]; then
        echo "StationIndex: rebuild requested for Gunicorn startup."
    fi
    mkdir -p "$DEPLOYMENT_DIR"
    touch "$ACCESS_LOG" "$ERROR_LOG"

    GUNICORN_ENV=(PYTHONUNBUFFERED=1)
    if [ "$REBUILD_DB" = true ]; then
        GUNICORN_ENV+=(STATION_REBUILD_DB=1)
    fi

    env "${GUNICORN_ENV[@]}" gunicorn \
        --bind 127.0.0.1:$FLASK_PORT \
        --timeout 600 \
        --workers 1 \
        --worker-class gthread \
        --threads "$GUNICORN_THREADS" \
        --error-logfile "$ERROR_LOG" \
        --capture-output \
        --daemon \
        --pid "$PID_FILE" \
        web_interface.app:app

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "✗ Gunicorn command failed with exit code $exit_code. Check logs:"
        echo "  - $ERROR_LOG"
        return 1
    fi

    sleep 2
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null; then
        echo "✓ Gunicorn started successfully (PID: $(cat $PID_FILE))."
        return 0
    fi

    echo "✗ Failed to start Gunicorn. Check logs for details:"
    echo "  - $ERROR_LOG"
    return 1
}

start_tls_gunicorn() {
    echo "Starting Gunicorn with TLS on port $NGINX_HTTPS_PORT..."
    if [ "$REBUILD_DB" = true ]; then
        echo "StationIndex: rebuild requested for Gunicorn startup."
    fi
    mkdir -p "$DEPLOYMENT_DIR"
    touch "$ACCESS_LOG" "$ERROR_LOG"

    if command -v lsof >/dev/null 2>&1; then
        PORT_PID=$(lsof -t -iTCP:"$NGINX_HTTPS_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1)
        if [ -n "$PORT_PID" ]; then
            echo "✗ Port $NGINX_HTTPS_PORT is already in use by PID $PORT_PID; cannot start TLS Gunicorn fallback."
            return 1
        fi
    fi

    GUNICORN_ENV=(PYTHONUNBUFFERED=1)
    if [ "$REBUILD_DB" = true ]; then
        GUNICORN_ENV+=(STATION_REBUILD_DB=1)
    fi

    env "${GUNICORN_ENV[@]}" gunicorn \
        --bind 0.0.0.0:$NGINX_HTTPS_PORT \
        --certfile "$DEPLOYMENT_DIR/cert.pem" \
        --keyfile "$DEPLOYMENT_DIR/key.pem" \
        --timeout 600 \
        --workers 1 \
        --worker-class gthread \
        --threads "$GUNICORN_THREADS" \
        --error-logfile "$ERROR_LOG" \
        --capture-output \
        --daemon \
        --pid "$PID_FILE" \
        web_interface.app:app

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "✗ Gunicorn TLS fallback failed with exit code $exit_code. Check logs:"
        echo "  - $ERROR_LOG"
        return 1
    fi

    sleep 2
    if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null; then
        echo "✓ Gunicorn TLS fallback started successfully (PID: $(cat $PID_FILE))."
        return 0
    fi

    echo "✗ Gunicorn TLS fallback did not start (PID file missing or process not running)."
    echo "  Check logs at: $ERROR_LOG"
    return 1
}

set_station_name_via_api() {
    local api_base_url="$1"
    shift
    if [ -z "$STATION_NAME_OVERRIDE" ]; then
        return 0
    fi

    echo "Setting station name via API to '$STATION_NAME_OVERRIDE'..."
    STATION_NAME_PAYLOAD=$(STATION_NAME_OVERRIDE="$STATION_NAME_OVERRIDE" python3 - <<'PY'
import json
import os
print(json.dumps({"station_name": os.environ.get("STATION_NAME_OVERRIDE", "")}))
PY
)
    if ! api_curl -sS --fail "$@" "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" \
        -X PUT \
        --data "$STATION_NAME_PAYLOAD" \
        "$api_base_url/api/station/config" >/dev/null; then
        echo "ERROR: Failed to set station name via API."
        return 1
    fi
    echo "✓ Station name updated via API."
}

# --- Conda Environment Setup ---
CONDA_ENV_NAME=${CONDA_ENV_NAME:-station} # Default to 'station' if not set

if [ -z "$CONDA_BIN_PATH" ]; then
    echo "WARNING: CONDA_BIN_PATH is not set in .env. Attempting to find conda in PATH."
    # Fallback to finding conda in PATH if not set
    if ! command -v conda &> /dev/null; then
        echo "ERROR: 'conda' command not found. Please run ./deploy.sh or ensure conda is in your PATH."
        exit 1
    fi
    CONDA_BIN_PATH=$(command -v conda)
fi

# Derive the path to conda.sh from the conda binary path
CONDA_BASE_DIR=$(dirname "$(dirname "$CONDA_BIN_PATH")")
CONDA_SH_PATH="$CONDA_BASE_DIR/etc/profile.d/conda.sh"

if [ -f "$CONDA_SH_PATH" ]; then
    echo "Activating conda environment '$CONDA_ENV_NAME'..."
    # Source the conda.sh script to make 'conda' command available
    # shellcheck source=/dev/null
    . "$CONDA_SH_PATH"
    # Activate the desired environment
    conda activate "$CONDA_ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate conda environment '$CONDA_ENV_NAME'. Please ensure it exists."
        exit 1
    fi
else
    echo "WARNING: conda.sh not found at '$CONDA_SH_PATH'. Ensure conda environment is activated manually if needed."
fi

if [ "$TEST_MODE" = true ]; then
    echo "Applying quick-test startup overrides in $STATION_DATA_DIR..."
    if ! python -m station.startup_overrides --station-data "$STATION_DATA_DIR" --test; then
        exit 1
    fi
    echo "✓ Quick-test startup overrides applied."
elif [ "$NO_MULTISTART" = true ]; then
    echo "Disabling init and stagnation multistart in $STATION_DATA_DIR/constant_config.yaml..."
    if ! python -m station.startup_overrides --station-data "$STATION_DATA_DIR" --no-multistart; then
        exit 1
    fi
    echo "✓ Multistart disabled for this station."
fi

# --- Pre-flight Checks ---
if [ ! -f "$DEPLOYMENT_DIR/cert.pem" ]; then
    echo "ERROR: Certificate files not found. Run ./deploy.sh first."
    exit 1
fi

if ! python -c "import gevent" &>/dev/null; then
    echo "Installing gevent for better performance..."
    pip install gevent
fi

# --- Stop Existing Services ---
MULTISTART_WAITING_PAGE=false
MULTISTART_BOOTSTRAP_PENDING=false
echo "Ensuring all services are stopped before starting..."

STOP_ARGS=()
if [ "$FORCE_STOP" = true ]; then
    STOP_ARGS=(--force)
fi
STOP_ENV=()
if [ "${STATION_MULTISTART_SKIP_CONTROLLER_START:-}" = "1" ] || [ "${STATION_MULTISTART_WAIT_ONLY:-}" = "1" ]; then
    STOP_ENV=(STATION_MULTISTART_SKIP_CONTROLLER_STOP=1)
fi
if ! env "${STOP_ENV[@]}" ./stop.sh "${STOP_ARGS[@]}"; then
    echo "ERROR: Could not stop existing services safely. Use ./start.sh -s --force to bypass pause/drain checks."
    exit 1
fi
echo ""

if [ "$REQUEUE_FAILED_EXTERNAL_REPORTS" = true ]; then
    echo "Running one-time External Counter failure recovery..."
    if ! requeue_failed_external_reports_once; then
        echo "ERROR: Legacy External Counter report recovery failed."
        exit 1
    fi
    echo "✓ Legacy External Counter report recovery complete."
    echo ""
fi

# --- Multistart hook ---
if [ "${STATION_MULTISTART_SKIP_HOOK:-}" != "1" ]; then
    MULTISTART_HOOK_OUTPUT=""
    MULTISTART_HOOK_STATUS=0
    MULTISTART_HOOK_OUTPUT=$(python -m station.multistart.start_hook --repo "$(pwd)" 2>&1) || MULTISTART_HOOK_STATUS=$?
    if [ "$MULTISTART_HOOK_STATUS" -eq 20 ]; then
        MULTISTART_WAITING_PAGE=true
        echo "Multistart job active; starting static waiting page."
        if [ -n "$MULTISTART_HOOK_OUTPUT" ]; then
            echo "$MULTISTART_HOOK_OUTPUT"
        fi
    elif [ "$MULTISTART_HOOK_STATUS" -eq 21 ]; then
        MULTISTART_BOOTSTRAP_PENDING=true
        AUTO_START_STATION=false
        echo "Pending stagnation multistart detected; starting the live API only so the controller can resume it."
        if [ -n "$MULTISTART_HOOK_OUTPUT" ]; then
            echo "$MULTISTART_HOOK_OUTPUT"
        fi
    elif [ "$MULTISTART_HOOK_STATUS" -ne 0 ]; then
        echo "$MULTISTART_HOOK_OUTPUT"
        echo "ERROR: multistart startup hook failed."
        exit 1
    fi
fi

if [ "$MULTISTART_WAITING_PAGE" = true ]; then
    resume_multistart_branches_if_autostart || exit 1
    AUTO_START_STATION=false
fi

# --- Start Nginx ---
echo "Starting Nginx reverse proxy..."

# Re-generate nginx.conf from current environment so port changes in .env take effect
cat > "$NGINX_CONF" << EOF
error_log $(pwd)/$DEPLOYMENT_DIR/nginx_error.log;
pid $(pwd)/$DEPLOYMENT_DIR/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_comp_level 5;
    gzip_types application/json text/plain text/css application/javascript;
    
    access_log off;

    # Redirect HTTP to HTTPS
    server {
        listen $NGINX_HTTP_PORT;
        server_name _;
        return 301 https://\$host:$NGINX_HTTPS_PORT\$request_uri;
    }

    # HTTPS Server
    server {
        listen $NGINX_HTTPS_PORT ssl;
        server_name _;

        ssl_certificate $(pwd)/$DEPLOYMENT_DIR/cert.pem;
        ssl_certificate_key $(pwd)/$DEPLOYMENT_DIR/key.pem;

        # Modern SSL settings
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        location /api/orchestrator/live_log_stream {
            proxy_pass http://127.0.0.1:$FLASK_PORT;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # SSE specific settings
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            proxy_buffering off;
            proxy_cache off;

            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
        }

        location / {
            proxy_pass http://127.0.0.1:$FLASK_PORT;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;

            # Disable buffering for large responses to prevent Content-Length issues
            proxy_buffering off;
            proxy_request_buffering off;

            # Don't modify response headers
            proxy_pass_request_headers on;
            proxy_http_version 1.1;

            proxy_connect_timeout 30s;
            proxy_read_timeout 600s;
            proxy_send_timeout 600s;
        }
    }
}
EOF

# Ensure nginx log files exist and are writable
touch "$DEPLOYMENT_DIR/nginx_error.log" "$DEPLOYMENT_DIR/nginx_access.log"
chmod 666 "$DEPLOYMENT_DIR/nginx_error.log" "$DEPLOYMENT_DIR/nginx_access.log" 2>/dev/null || true

# Check for port conflicts after stop.sh has cleaned this station's processes.
if command -v lsof >/dev/null 2>&1; then
    for checked_port in "$NGINX_HTTP_PORT" "$NGINX_HTTPS_PORT"; do
        PORT_PID=$(lsof -t -iTCP:"$checked_port" -sTCP:LISTEN 2>/dev/null | head -n 1)
        if [ -n "$PORT_PID" ]; then
            echo "✗ Port $checked_port is already in use by PID $PORT_PID."
            echo "  Choose a different port in .env, or stop the conflicting process."
            echo "  Stopping Gunicorn to prevent orphaned processes..."
            ./stop.sh --force
            exit 1
        fi
    done
fi

NGINX_ERROR_LOG_PATH="$(pwd)/$DEPLOYMENT_DIR/nginx_error.log"
NGINX_GLOBAL_DIRECTIVES="error_log $NGINX_ERROR_LOG_PATH;"
SERVER_MODE="nginx"
API_BASE_URL="http://127.0.0.1:$FLASK_PORT"
API_CURL_EXTRA=()

# Function to try nginx command with and without sudo
try_nginx_command() {
    local cmd="$1"
    shift
    
    # First try without sudo
    if $cmd "$@" 2>/dev/null; then
        return 0
    fi
    
    # If that fails and sudo exists, try with sudo
    if command -v sudo >/dev/null 2>&1; then
        # Only attempt sudo if it can actually escalate in this environment
        if sudo -n true 2>/dev/null; then
            sudo $cmd "$@"
            return $?
        fi
    fi
    
    return 1
}

# Check if OUR nginx instance is already running (using our specific PID file)
if [ -f "$NGINX_PID_FILE" ] && ps -p $(cat "$NGINX_PID_FILE" 2>/dev/null) > /dev/null 2>&1; then
    echo "✓ This station's Nginx is running. Reloading configuration..."
    if ! try_nginx_command nginx -g "$NGINX_GLOBAL_DIRECTIVES" -c "$(pwd)/$NGINX_CONF" -s reload; then
        echo "✗ Failed to reload Nginx. Attempting a full restart..."
        try_nginx_command nginx -g "$NGINX_GLOBAL_DIRECTIVES" -c "$(pwd)/$NGINX_CONF" -s quit
        sleep 1
        try_nginx_command nginx -g "$NGINX_GLOBAL_DIRECTIVES" -c "$(pwd)/$NGINX_CONF"
    fi
else
    echo "Attempting to start Nginx for this station..."
    if ! try_nginx_command nginx -g "$NGINX_GLOBAL_DIRECTIVES" -c "$(pwd)/$NGINX_CONF"; then
        echo "✗ Failed to start Nginx with standard permissions."
        
        # Try to fix log file permissions if possible
        if [ -f "$DEPLOYMENT_DIR/nginx_error.log" ]; then
            chmod 666 "$DEPLOYMENT_DIR/nginx_error.log" 2>/dev/null || \
                (command -v sudo >/dev/null 2>&1 && sudo chmod 666 "$DEPLOYMENT_DIR/nginx_error.log" 2>/dev/null)
        fi
        if [ -f "$DEPLOYMENT_DIR/nginx_access.log" ]; then
            chmod 666 "$DEPLOYMENT_DIR/nginx_access.log" 2>/dev/null || \
                (command -v sudo >/dev/null 2>&1 && sudo chmod 666 "$DEPLOYMENT_DIR/nginx_access.log" 2>/dev/null)
        fi
        
        # Try one more time
        if ! try_nginx_command nginx -g "$NGINX_GLOBAL_DIRECTIVES" -c "$(pwd)/$NGINX_CONF"; then
            echo "✗ Still unable to start Nginx."
        fi
    fi
fi

sleep 1
if [ -f "$NGINX_PID_FILE" ] && ps -p $(cat "$NGINX_PID_FILE" 2>/dev/null) > /dev/null 2>&1; then
    echo "✓ This station's Nginx is running (PID: $(cat $NGINX_PID_FILE))."
else
    echo "✗ Failed to start or reload Nginx for this station."
    echo "  Check logs at: $DEPLOYMENT_DIR/nginx_error.log"
    echo "  Falling back to direct TLS Gunicorn startup without restarting the station."
    SERVER_MODE="tls_gunicorn"
    API_BASE_URL="https://127.0.0.1:$NGINX_HTTPS_PORT"
    API_CURL_EXTRA=(-k)
fi

if [ "$SERVER_MODE" = "nginx" ]; then
    if ! start_loopback_gunicorn; then
        ./stop.sh --force
        exit 1
    fi
else
    if ! start_tls_gunicorn; then
        ./stop.sh --force
        exit 1
    fi
fi

# --- Final Status ---
echo ""
if [ "$SERVER_MODE" = "nginx" ]; then
    echo "=== Production Services Started Successfully ==="
else
    echo "=== Production Services Started Successfully (Gunicorn TLS, no Nginx) ==="
fi
echo "🌐 Access your station at: https://localhost:$NGINX_HTTPS_PORT (or your server's IP)"
echo "👤 Username: $FLASK_AUTH_USERNAME"
echo "🔑 Password: [hidden]"
echo ""
echo "Logs:"
echo "  Application: $ERROR_LOG"
echo "  HTTP Access: disabled"
echo ""
echo "To stop services, run: ./stop.sh"

if [ "$MULTISTART_WAITING_PAGE" = true ]; then
    echo "Multistart is running; normal Station controls are disabled until selection completes."
    exit 0
fi

if [ "$MULTISTART_BOOTSTRAP_PENDING" = true ]; then
    echo "Verifying pending stagnation multistart recovery..."
    if ! python - <<'PY'
from pathlib import Path
import sys

from station.multistart import paths, waiting
from station.multistart.controller import find_running_controller_pid

repo = Path.cwd().resolve()
pid = find_running_controller_pid(repo)
active = waiting.active_job(repo)
pending = paths.pending_stagnation_path(repo).is_file()

if pid is None:
    print("ERROR: Pending stagnation multistart was not resumed because its controller is not running.")
    sys.exit(1)
if active:
    print(f"✓ Stagnation multistart controller is running (PID {pid}); active job: {active.get('job_id') or 'preparing'}.")
    sys.exit(0)
if pending:
    print(f"✓ Stagnation multistart controller is running (PID {pid}) and is actively processing the pending request.")
    sys.exit(0)

print("ERROR: The pending stagnation request disappeared without an active multistart job.")
sys.exit(1)
PY
    then
        echo "ERROR: Refusing to leave the ordinary station running after multistart recovery failed."
        STATION_MULTISTART_SKIP_CONTROLLER_START=1 ./stop.sh --force
        exit 1
    fi
    echo "Normal orchestrator auto-start is suppressed until the pending multistart request is converted into a branch job."
    exit 0
fi

if ! set_station_name_via_api "$API_BASE_URL" "${API_CURL_EXTRA[@]}"; then
    exit 1
fi

if [ "$AUTO_START_STATION" = true ]; then
    echo ""
    echo "Auto-start requested. Preparing orchestrator..."
    PREPARE_RESPONSE=$(api_curl -sS "${API_CURL_EXTRA[@]}" "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" \
        -X POST \
        "$API_BASE_URL/api/orchestrator/prepare")

    if ! PREPARE_RESPONSE="$PREPARE_RESPONSE" python3 - <<'PY'
import json
import os
import sys
raw = os.environ.get("PREPARE_RESPONSE", "")
try:
    data = json.loads(raw)
except Exception:
    print(f"ERROR: Invalid prepare response: {raw}")
    sys.exit(1)
if data.get("success") is True:
    sys.exit(0)
print(f"ERROR: Prepare failed: {data.get('message', 'Unknown error')}")
sys.exit(1)
PY
    then
        exit 1
    fi

    echo "Starting orchestrator loop..."
    START_RESPONSE=$(api_curl -sS "${API_CURL_EXTRA[@]}" "${AUTH_ARGS[@]}" \
        -H "Content-Type: application/json" \
        -X POST \
        "$API_BASE_URL/api/orchestrator/start_loop")

    if ! START_RESPONSE="$START_RESPONSE" python3 - <<'PY'
import json
import os
import sys
raw = os.environ.get("START_RESPONSE", "")
try:
    data = json.loads(raw)
except Exception:
    print(f"ERROR: Invalid start response: {raw}")
    sys.exit(1)
if data.get("success") is True:
    print("✓ Station orchestrator loop started.")
    sys.exit(0)
print(f"ERROR: Start loop failed: {data.get('message', 'Unknown error')}")
sys.exit(1)
PY
    then
        exit 1
    fi
fi
