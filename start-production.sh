#!/bin/bash
# Production startup script for the Station

echo "=== Starting Station Production Services ==="

# --- Configuration ---
DEPLOYMENT_DIR="deployment"
ENV_FILE=".env"

PID_FILE="$DEPLOYMENT_DIR/gunicorn.pid"
NGINX_CONF="$DEPLOYMENT_DIR/nginx.conf"
NGINX_PID_FILE="$DEPLOYMENT_DIR/nginx.pid"
ACCESS_LOG="$DEPLOYMENT_DIR/access.log"
ERROR_LOG="$DEPLOYMENT_DIR/error.log"

# --- Source Environment Variables ---
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "ERROR: $ENV_FILE not found. Run ./deploy.sh first."
    exit 1
fi

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
./stop-production.sh
echo ""

# --- Start Gunicorn ---
echo "Starting Gunicorn application server..."
if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null; then
    echo "✓ Gunicorn is already running (PID: $(cat $PID_FILE))."
else
    mkdir -p "$DEPLOYMENT_DIR"
    # Ensure log files exist
    touch "$ACCESS_LOG" "$ERROR_LOG"
    
    # Use port from .env or default
    FLASK_PORT=${FLASK_PORT:-5000}

    PYTHONUNBUFFERED=1 gunicorn \
        --bind 127.0.0.1:$FLASK_PORT \
        --timeout 120 \
        --workers 1 \
        --worker-class gthread \
        --threads 4 \
        --access-logfile "$ACCESS_LOG" \
        --error-logfile "$ERROR_LOG" \
        --capture-output \
        --daemon \
        --pid "$PID_FILE" \
        web_interface.app:app

    GUNICORN_EXIT_CODE=$?
    if [ $GUNICORN_EXIT_CODE -eq 0 ]; then
        sleep 2 # Give it a moment to start and create the PID file
        if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null; then
            echo "✓ Gunicorn started successfully (PID: $(cat $PID_FILE))."
        else
            echo "✗ Failed to start Gunicorn. Check logs for details:"
            echo "  - $ERROR_LOG"
            exit 1
        fi
    else
        echo "✗ Gunicorn command failed with exit code $GUNICORN_EXIT_CODE. Check logs:"
        echo "  - $ERROR_LOG"
        exit 1
    fi
fi

# --- Start Nginx ---
echo "Starting Nginx reverse proxy..."

# Check and kill processes on default HTTP port if using port 80
NGINX_HTTP_PORT=${NGINX_HTTP_PORT:-80}
if [ "$NGINX_HTTP_PORT" = "80" ]; then
    echo "Checking for processes using port 80..."
    # Find process using port 80
    PORT_PID=$(lsof -t -i:80 2>/dev/null)
    if [ ! -z "$PORT_PID" ]; then
        echo "Found process $PORT_PID using port 80. Attempting to kill it..."
        # Try to kill gracefully first
        if kill $PORT_PID 2>/dev/null; then
            echo "✓ Successfully killed process on port 80"
        else
            # If that fails, try with sudo
            if command -v sudo >/dev/null 2>&1; then
                if sudo kill $PORT_PID 2>/dev/null; then
                    echo "✓ Successfully killed process on port 80 (with sudo)"
                else
                    # Force kill if regular kill didn't work
                    if sudo kill -9 $PORT_PID 2>/dev/null; then
                        echo "✓ Force killed process on port 80 (with sudo)"
                    else
                        echo "✗ Failed to kill process on port 80. You may need to stop it manually."
                    fi
                fi
            else
                echo "✗ Failed to kill process on port 80. No sudo access available."
            fi
        fi
        # Brief pause to allow port to be freed
        sleep 1
    else
        echo "✓ Port 80 is available"
    fi
fi

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
        sudo $cmd "$@"
        return $?
    fi
    
    return 1
}

# Check if OUR nginx instance is already running (using our specific PID file)
if [ -f "$NGINX_PID_FILE" ] && ps -p $(cat "$NGINX_PID_FILE" 2>/dev/null) > /dev/null 2>&1; then
    echo "✓ This station's Nginx is running. Reloading configuration..."
    if ! try_nginx_command nginx -c "$(pwd)/$NGINX_CONF" -s reload; then
        echo "✗ Failed to reload Nginx. Attempting a full restart..."
        try_nginx_command nginx -c "$(pwd)/$NGINX_CONF" -s quit
        sleep 1
        try_nginx_command nginx -c "$(pwd)/$NGINX_CONF"
    fi
else
    echo "Attempting to start Nginx for this station..."
    if ! try_nginx_command nginx -c "$(pwd)/$NGINX_CONF"; then
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
        if ! try_nginx_command nginx -c "$(pwd)/$NGINX_CONF"; then
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
    if command -v sudo >/dev/null 2>&1; then
        echo "  Try manually: sudo nginx -c $(pwd)/$NGINX_CONF"
    else
        echo "  Try manually: nginx -c $(pwd)/$NGINX_CONF"
    fi
    echo "  Stopping Gunicorn to prevent orphaned processes..."
    ./stop-production.sh
    exit 1
fi

# --- Final Status ---
echo ""
echo "=== Production Services Started Successfully ==="
NGINX_HTTPS_PORT=${NGINX_HTTPS_PORT:-8443}
echo "🌐 Access your station at: https://localhost:$NGINX_HTTPS_PORT (or your server's IP)"
echo "👤 Username: $FLASK_AUTH_USERNAME"
echo "🔑 Password: [hidden]"
echo ""
echo "Logs:"
echo "  Application: $ERROR_LOG"
echo "  HTTP Access: $ACCESS_LOG"
echo ""
echo "To stop services, run: ./stop-production.sh"