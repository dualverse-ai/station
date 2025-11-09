#!/bin/bash
# Production stop script for the Station

echo "=== Stopping Station Production Services ==="

# --- Configuration ---
CURRENT_DIR=$(pwd)
DEPLOYMENT_DIR="deployment"
PID_FILE="$DEPLOYMENT_DIR/gunicorn.pid"
NGINX_PID_FILE="$DEPLOYMENT_DIR/nginx.pid"

# --- Stop Nginx ---
echo "Stopping Nginx..."
# Use local PID file
if [ -f "$NGINX_PID_FILE" ]; then
    NGINX_PID=$(cat "$NGINX_PID_FILE")
    if ps -p $NGINX_PID > /dev/null 2>&1; then
        echo "Found nginx process (PID: $NGINX_PID)"
        # Try to stop using our config file which has the correct pid path
        NGINX_CONF="$DEPLOYMENT_DIR/nginx.conf"
        if [ -f "$NGINX_CONF" ]; then
            # Try without sudo first, then with sudo if available
            if nginx -c "$(pwd)/$NGINX_CONF" -s quit 2>/dev/null; then
                echo "✓ Nginx stopped gracefully."
            elif command -v sudo >/dev/null 2>&1 && sudo nginx -c "$(pwd)/$NGINX_CONF" -s quit 2>/dev/null; then
                echo "✓ Nginx stopped gracefully."
            else
                # Fallback: kill the specific process
                echo "Graceful stop failed, killing nginx process directly..."
                if command -v sudo >/dev/null 2>&1; then
                    sudo kill $NGINX_PID 2>/dev/null || kill $NGINX_PID 2>/dev/null
                else
                    kill $NGINX_PID 2>/dev/null
                fi
                echo "✓ Nginx process killed."
            fi
        else
            # No config file, kill the process directly
            echo "No config file found, killing nginx process directly..."
            if command -v sudo >/dev/null 2>&1; then
                sudo kill $NGINX_PID 2>/dev/null || kill $NGINX_PID 2>/dev/null
            else
                kill $NGINX_PID 2>/dev/null
            fi
            echo "✓ Nginx process killed."
        fi
        rm -f "$NGINX_PID_FILE"
    else
        echo "✓ Nginx not running (stale PID file found)."
        rm -f "$NGINX_PID_FILE"
    fi
else
    echo "✓ No nginx found for this station (no PID file)."
fi

# Also check for orphaned nginx processes using this station's config file
NGINX_CONF_PATH="$(pwd)/$DEPLOYMENT_DIR/nginx.conf"
if [ -f "$NGINX_CONF_PATH" ]; then
    # Find nginx master processes using our config file
    ORPHANED_NGINX_PIDS=$(ps aux | grep -E "nginx.*master.*$NGINX_CONF_PATH" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    if [ -n "$ORPHANED_NGINX_PIDS" ]; then
        echo "Found orphaned nginx processes using this station's config: $ORPHANED_NGINX_PIDS"
        echo "Attempting to stop orphaned nginx processes..."
        for PID in $ORPHANED_NGINX_PIDS; do
            if command -v sudo >/dev/null 2>&1; then
                # Try graceful stop first with sudo
                if sudo nginx -c "$NGINX_CONF_PATH" -s quit 2>/dev/null; then
                    echo "✓ Orphaned nginx stopped gracefully."
                else
                    # Force kill if graceful stop fails
                    sudo kill $PID 2>/dev/null && echo "✓ Orphaned nginx process $PID killed."
                fi
            else
                # Try without sudo
                if nginx -c "$NGINX_CONF_PATH" -s quit 2>/dev/null; then
                    echo "✓ Orphaned nginx stopped gracefully."
                else
                    kill $PID 2>/dev/null && echo "✓ Orphaned nginx process $PID killed."
                fi
            fi
        done
    fi
fi

# --- Stop Gunicorn ---
echo "Stopping Gunicorn..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        # Try graceful station cleanup via API first
        if [ -f "$ENV_FILE" ]; then
            source "$ENV_FILE" 2>/dev/null || true
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

# --- Stop Docker Containers ---
echo "Stopping research Docker containers for this station..."
if command -v docker &>/dev/null; then
    # Check if we have permission to access Docker
    if docker ps >/dev/null 2>&1; then
        # Check for containers with volumes from this station
        CONTAINERS=$(docker ps -q --filter "volume=$CURRENT_DIR/station_data" 2>/dev/null)
        if [ -n "$CONTAINERS" ]; then
            docker stop $CONTAINERS
            echo "✓ Stopped research containers with volumes from this station."
        else
            echo "✓ No research containers running for this station."
        fi
    else
        echo "✓ No Docker access (may need to be in docker group)."
    fi
else
    echo "✓ Docker not installed."
fi

# --- Final Check for Any Remaining Gunicorn Processes ---
echo ""
echo "Performing final check for gunicorn processes..."
REMAINING_GUNICORN=""
for pid in $(pgrep gunicorn 2>/dev/null); do
    cwd=$(readlink /proc/$pid/cwd 2>/dev/null)
    cmd=$(ps -p $pid -o args= 2>/dev/null | head -c 200)
    REMAINING_GUNICORN="${REMAINING_GUNICORN}PID $pid (CWD: $cwd)\n  Command: $cmd...\n"
done

if [ -n "$REMAINING_GUNICORN" ]; then
    echo "⚠️  WARNING: The following gunicorn processes are still running:"
    echo "    (These may belong to other station instances or applications)"
    echo ""
    echo -e "$REMAINING_GUNICORN"
    echo "    Please verify these processes and stop them manually if needed."
else
    echo "✓ No gunicorn processes found on the system."
fi

echo ""
echo "=== Production Services Stopped ==="