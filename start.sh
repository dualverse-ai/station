#!/bin/bash
# Production startup script for the Station

echo "=== Starting Station Production Services ==="

# --- Argument Parsing ---
STATION_NAME_OVERRIDE=""
AUTO_START_STATION=false
FORCE_STOP=false
REBUILD_DB=false
RUN_MIGRATIONS=false
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
        --migrate)
            RUN_MIGRATIONS=true
            ;;
        -h|--help)
            echo "Usage: ./start.sh [--name station_name] [--start|-s] [--force] [--rebuild-db] [--migrate]"
            echo "  --force: pass --force to stop.sh before starting."
            echo "  --rebuild-db: rebuild derived SQLite station indexes from YAML before starting."
            echo "  --migrate: check and run station data migration scripts before starting."
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument '$1'"
            echo "Usage: ./start.sh [--name station_name] [--start|-s] [--force] [--rebuild-db] [--migrate]"
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

# Always bypass proxies for local loopback API calls.
api_curl() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy \
        curl --noproxy '*' "$@"
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
        --threads 4 \
        --access-logfile "$ACCESS_LOG" \
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
        --threads 4 \
        --access-logfile "$ACCESS_LOG" \
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

migrate_research_eval_artifacts_if_needed() {
    local needs_migration
    local needs_migration_output
    if ! needs_migration_output=$(python scripts/migrate/migrate_research_eval_artifacts.py --check 2>&1); then
        printf '%s\n' "$needs_migration_output"
        echo "ERROR: Could not inspect Research eval metadata schema."
        return 1
    fi
    printf '%s\n' "$needs_migration_output" | sed '$d'
    needs_migration="${needs_migration_output##*$'\n'}"

    if [ "$needs_migration" = "yes" ]; then
        echo "Research eval metadata is legacy; migrating inline blobs to artifacts..."
        if ! python scripts/migrate/migrate_research_eval_artifacts.py; then
            echo "ERROR: Research eval artifact migration failed. Station startup aborted."
            return 1
        fi
        echo "✓ Research eval artifact migration complete."
    elif [ "$needs_migration" = "no" ]; then
        echo "✓ Research eval metadata is artifact-backed."
    else
        echo "ERROR: Unexpected Research eval migration check result: $needs_migration"
        return 1
    fi
}

migrate_protected_dialogue_ticks_if_needed() {
    local needs_migration
    if ! needs_migration=$(python scripts/migrate/migrate_protected_dialogue_ticks.py --check); then
        echo "ERROR: Could not inspect agent dialogue protection schema."
        return 1
    fi

    if [ "$needs_migration" = "yes" ]; then
        echo "Agent dialogue protection schema is legacy; migrating protected ticks into agent YAML..."
        if ! python scripts/migrate/migrate_protected_dialogue_ticks.py; then
            echo "ERROR: Protected dialogue tick migration failed. Station startup aborted."
            return 1
        fi
        echo "✓ Protected dialogue tick migration complete."
    else
        echo "✓ Agent dialogue protection schema is current."
    fi
}

migrate_lobby_codex_help_if_needed() {
    local needs_migration
    if ! needs_migration=$(python scripts/migrate/migrate_lobby_codex_help.py --check); then
        echo "ERROR: Could not inspect first-turn Lobby Codex help history."
        return 1
    fi

    if [ "$needs_migration" = "yes" ]; then
        echo "First-turn Lobby help is missing Codex text for active agents; migrating dialogue history..."
        if ! python scripts/migrate/migrate_lobby_codex_help.py; then
            echo "ERROR: Lobby Codex help migration failed. Station startup aborted."
            return 1
        fi
        echo "✓ Lobby Codex help migration complete."
    else
        echo "✓ First-turn Lobby Codex help history is current."
    fi
}

index_schema_version_bump_requires_migration() {
    local result
    if ! result=$(python scripts/migrate/check_index_schema_migration.py); then
        printf '%s\n' "$result"
        echo "ERROR: Could not inspect SQLite index schema versions."
        return 2
    fi

    case "$result" in
        yes)
            return 0
            ;;
        no)
            return 1
            ;;
        *)
            echo "ERROR: Unexpected SQLite index schema check result: $result"
            return 2
            ;;
    esac
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

# --- Theory Room cache refresh (guarded) ---
# If Theory Room is enabled, wipe only this repo's theory cache and rebuild.
THEORY_CACHE_DIR=$(python3 - <<'PY'
from pathlib import Path
import hashlib
import sys
import contextlib
import os

try:
    # Suppress noisy prints (e.g., proxy overrides) when importing constants
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        from station import constants
except Exception:
    print("")
    sys.exit(0)

enabled = getattr(constants, "THEORY_ROOM_ENABLED", False)
if enabled:
    repo_root = Path.cwd()
    repo_hash = hashlib.md5(str(repo_root).encode()).hexdigest()[:8]
    cache_dir = Path.home() / f".cache/station_theory_{repo_hash}"
    print(cache_dir)
else:
    print("")
PY
)

if [ -n "$THEORY_CACHE_DIR" ]; then
    if [ -d "$THEORY_CACHE_DIR" ]; then
        echo "Theory Room enabled; cache exists at $THEORY_CACHE_DIR (no wipe)"
        echo "Rebuilding Theory Room cache via scripts/setup_theory.sh..."
        if ! REBUILD_ONLY=true /bin/bash scripts/setup_theory.sh; then
            echo "✗ Theory Room cache rebuild failed. Check permissions or rerun scripts/setup_theory.sh manually."
            exit 1
        fi
        echo "✓ Theory Room cache rebuild complete for $THEORY_CACHE_DIR"
    else
        echo "Theory Room enabled; cache missing, creating fresh at $THEORY_CACHE_DIR"
        echo "Building Theory Room cache via scripts/setup_theory.sh..."
        if ! /bin/bash scripts/setup_theory.sh; then
            echo "✗ Theory Room cache build failed. Check permissions or rerun scripts/setup_theory.sh manually."
            exit 1
        fi
        echo "✓ Theory Room cache build complete for $THEORY_CACHE_DIR"
    fi
else
    echo "Theory Room disabled; skipping theory cache refresh."
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
echo "Ensuring all services are stopped before starting..."
STOP_ARGS=()
if [ "$FORCE_STOP" = true ]; then
    STOP_ARGS=(--force)
fi
if ! ./stop.sh "${STOP_ARGS[@]}"; then
    echo "ERROR: Could not stop existing services safely. Use ./start.sh -s --force to bypass pause/drain checks."
    exit 1
fi
echo ""

# --- Optional station data migrations ---
if [ "$RUN_MIGRATIONS" != true ]; then
    index_schema_version_bump_requires_migration
    INDEX_SCHEMA_CHECK_STATUS=$?
    if [ $INDEX_SCHEMA_CHECK_STATUS -eq 0 ]; then
        echo "SQLite index schema version changed; running station data migrations before startup."
        RUN_MIGRATIONS=true
    elif [ $INDEX_SCHEMA_CHECK_STATUS -eq 2 ]; then
        exit 1
    fi
fi

if [ "$RUN_MIGRATIONS" = true ]; then
    # --- Research Center artifact migration ---
    if ! migrate_research_eval_artifacts_if_needed; then
        exit 1
    fi
    echo ""

    # --- Dialogue prune protection migration ---
    if ! migrate_protected_dialogue_ticks_if_needed; then
        exit 1
    fi
    echo ""

    # --- First-turn Lobby Codex help migration ---
    if ! migrate_lobby_codex_help_if_needed; then
        exit 1
    fi
    echo ""
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
    
    access_log $(pwd)/$DEPLOYMENT_DIR/nginx_access.log;

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

# Check for port conflicts (best-effort; don't attempt to kill processes automatically)
if command -v lsof >/dev/null 2>&1; then
    PORT_PID=$(lsof -t -iTCP:"$NGINX_HTTP_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1)
    if [ -n "$PORT_PID" ]; then
        echo "✗ Port $NGINX_HTTP_PORT is already in use by PID $PORT_PID."
        echo "  Choose a different port via NGINX_HTTP_PORT in .env, or stop the conflicting process."
        echo "  Stopping Gunicorn to prevent orphaned processes..."
        ./stop.sh --force
        exit 1
    fi
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
echo "  HTTP Access: $ACCESS_LOG"
echo ""
echo "To stop services, run: ./stop.sh"

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
