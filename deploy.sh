#!/bin/bash
# Deployment script for the Station

echo "=== Station Deployment ==="
if [ -n "$1" ] && [ "$1" = "-h" -o "$1" = "--help" ]; then
    echo "Usage: ./deploy.sh [web-auth-password]"
    echo ""
    echo "If .env already contains FLASK_AUTH_PASSWORD, that value is kept."
    echo "If a password argument is provided, it is saved to .env."
    echo "If neither exists, a strong password is generated and printed once."
    exit 0
fi

# --- Configuration ---
DEPLOYMENT_DIR="deployment"
ENV_FILE=".env"
CERT_FILE="$DEPLOYMENT_DIR/cert.pem"
KEY_FILE="$DEPLOYMENT_DIR/key.pem"
AUTH_PASSWORD_ARG="${1:-}"

# Get absolute path of current directory for unique identification
STATION_DIR="$(cd "$(dirname "$0")" && pwd)"
STATION_ID="$(echo "$STATION_DIR" | md5sum | cut -c1-8)"
NGINX_CONF="$DEPLOYMENT_DIR/nginx.conf"

# --- Helper Functions ---
install_package() {
    local package=$1
    echo "Installing $package..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y "$package"
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y "$package"
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y "$package"
    else
        echo "WARNING: Could not detect package manager. Please install $package manually."
        return 1
    fi
}

get_env_value() {
    local key=$1
    if [ ! -f "$ENV_FILE" ]; then
        return 0
    fi
    grep -E "^${key}=" "$ENV_FILE" | tail -n 1 | cut -d'=' -f2-
}

set_env_value() {
    local key=$1
    local value=$2
    python - "$ENV_FILE" "$key" "$value" <<'PY'
import pathlib
import sys

env_file = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]

lines = env_file.read_text().splitlines() if env_file.exists() else []
updated = False
for i, line in enumerate(lines):
    if line.startswith(f"{key}="):
        lines[i] = f"{key}={value}"
        updated = True
if not updated:
    lines.append(f"{key}={value}")
env_file.write_text("\n".join(lines) + "\n")
PY
}

# --- Main Script ---

# 1. Create deployment directory
echo "Creating deployment directory..."
mkdir -p "$DEPLOYMENT_DIR"

# 2. Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
install_package "nginx"

# 3. Environment Configuration
echo "Setting up environment file..."
if [ ! -f "$ENV_FILE" ]; then
    touch "$ENV_FILE"
    echo "Created $ENV_FILE."
fi

# Check for FLASK_SECRET_KEY
if [ -z "$(get_env_value FLASK_SECRET_KEY)" ]; then
    echo "Generating and saving FLASK_SECRET_KEY..."
    SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    set_env_value FLASK_SECRET_KEY "$SECRET_KEY"
fi

# Check for FLASK_AUTH_USERNAME and FLASK_AUTH_PASSWORD
AUTH_USERNAME="$(get_env_value FLASK_AUTH_USERNAME)"
if [ -z "$AUTH_USERNAME" ]; then
    AUTH_USERNAME="admin"
    set_env_value FLASK_AUTH_USERNAME "$AUTH_USERNAME"
    echo "WARNING: FLASK_AUTH_USERNAME set to 'admin'. Change it in the .env file."
fi

AUTH_PASSWORD="$(get_env_value FLASK_AUTH_PASSWORD)"
GENERATED_AUTH_PASSWORD=0
if [ -n "$AUTH_PASSWORD" ]; then
    echo "✓ Using existing FLASK_AUTH_PASSWORD from $ENV_FILE."
elif [ -n "$AUTH_PASSWORD_ARG" ]; then
    AUTH_PASSWORD="$AUTH_PASSWORD_ARG"
    set_env_value FLASK_AUTH_PASSWORD "$AUTH_PASSWORD"
    echo "✓ Saved provided FLASK_AUTH_PASSWORD to $ENV_FILE."
else
    AUTH_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    set_env_value FLASK_AUTH_PASSWORD "$AUTH_PASSWORD"
    GENERATED_AUTH_PASSWORD=1
    echo "Generated and saved a strong FLASK_AUTH_PASSWORD."
fi

# 4. Find and save paths for Conda and Claude
echo "Detecting and saving paths for Conda and Claude..."

# Find Conda path
CONDA_PATH=$(which conda)
if [ -z "$CONDA_PATH" ]; then
    echo "WARNING: 'conda' command not found. Python sandbox evaluation might not work. Please ensure Conda is installed and in your PATH."
else
    if ! grep -q "CONDA_BIN_PATH" "$ENV_FILE"; then
        echo "CONDA_BIN_PATH=$CONDA_PATH" >> "$ENV_FILE"
        echo "✓ Conda path saved to .env: $CONDA_PATH"
    else
        echo "✓ Conda path already in .env."
    fi
fi

# Find Claude path
CLAUDE_PATH=$(which claude)
if [ -z "$CLAUDE_PATH" ]; then
    echo "WARNING: 'claude' command not found. Claude Code Debugger might not work. Please ensure Claude is installed and in your PATH."
else
    if ! grep -q "CLAUDE_BIN_PATH" "$ENV_FILE"; then
        echo "CLAUDE_BIN_PATH=$CLAUDE_PATH" >> "$ENV_FILE"
        echo "✓ Claude path saved to .env: $CLAUDE_PATH"
    else
        echo "✓ Claude path already in .env."
    fi
fi

# 5. Generate SSL Certificate (Self-signed)
echo "Generating self-signed SSL certificate..."
if [ ! -f "$CERT_FILE" ]; then
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo "Self-signed certificate created."
    echo "NOTE: For production, replace with a certificate from a trusted authority (e.g., Let's Encrypt)."
fi

# 5. Create Nginx Configuration
echo "Creating Nginx configuration..."
CURRENT_DIR=$(pwd)

# Determine ports (can be overridden in .env)
if [ -z "$FLASK_PORT" ]; then
    FLASK_PORT=5000
    echo "FLASK_PORT=$FLASK_PORT" >> "$ENV_FILE"
fi
if [ -z "$NGINX_HTTP_PORT" ]; then
    NGINX_HTTP_PORT=80
    echo "NGINX_HTTP_PORT=$NGINX_HTTP_PORT" >> "$ENV_FILE"
fi
if [ -z "$NGINX_HTTPS_PORT" ]; then
    NGINX_HTTPS_PORT=8443
    echo "NGINX_HTTPS_PORT=$NGINX_HTTPS_PORT" >> "$ENV_FILE"
fi

# Check for port conflicts
echo "Checking for port conflicts..."
check_port() {
    local port=$1
    local service=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "ERROR: Port $port ($service) is already in use!"
        echo "Process using port $port:"
        lsof -Pi :$port -sTCP:LISTEN 2>/dev/null || ss -tlnp | grep ":$port"
        return 1
    else
        echo "✓ Port $port ($service) is available"
        return 0
    fi
}

PORT_CONFLICT=0
check_port $FLASK_PORT "Flask/Gunicorn" || PORT_CONFLICT=1
check_port $NGINX_HTTP_PORT "Nginx HTTP" || PORT_CONFLICT=1
check_port $NGINX_HTTPS_PORT "Nginx HTTPS" || PORT_CONFLICT=1

if [ $PORT_CONFLICT -eq 1 ]; then
    echo ""
    echo "ERROR: Port conflicts detected!"
    echo "Please either:"
    echo "  1. Stop the conflicting services"
    echo "  2. Choose different ports by setting in .env:"
    echo "     FLASK_PORT=<different_port>"
    echo "     NGINX_HTTP_PORT=<different_port>"
    echo "     NGINX_HTTPS_PORT=<different_port>"
    echo ""
    echo "For multiple stations, consider using:"
    echo "  Station 1: FLASK_PORT=5001, NGINX_HTTP_PORT=8080, NGINX_HTTPS_PORT=8443"
    echo "  Station 2: FLASK_PORT=5002, NGINX_HTTP_PORT=8081, NGINX_HTTPS_PORT=8444"
    echo "  Station 3: FLASK_PORT=5003, NGINX_HTTP_PORT=8082, NGINX_HTTPS_PORT=8445"
    exit 1
fi

cat > "$NGINX_CONF" << EOF
error_log $CURRENT_DIR/$DEPLOYMENT_DIR/nginx_error.log;
pid $CURRENT_DIR/$DEPLOYMENT_DIR/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    access_log $CURRENT_DIR/$DEPLOYMENT_DIR/nginx_access.log;

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

        ssl_certificate $CURRENT_DIR/$CERT_FILE;
        ssl_certificate_key $CURRENT_DIR/$KEY_FILE;

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
        }
    }
}
EOF

# Create nginx log files with proper permissions
touch "$DEPLOYMENT_DIR/nginx_error.log" "$DEPLOYMENT_DIR/nginx_access.log"
chmod 666 "$DEPLOYMENT_DIR/nginx_error.log" "$DEPLOYMENT_DIR/nginx_access.log"

echo "---"
echo "Deployment setup complete!"
echo "Please review the configuration in the '$ENV_FILE' file."
echo ""
echo "Web login:"
echo "  Username: $AUTH_USERNAME"
if [ "$GENERATED_AUTH_PASSWORD" -eq 1 ]; then
    echo "  Password: $AUTH_PASSWORD"
    echo "  Record this generated password now. It is also saved in $ENV_FILE."
elif [ -n "$AUTH_PASSWORD_ARG" ]; then
    echo "  Password: the password provided to ./deploy.sh"
else
    echo "  Password: existing FLASK_AUTH_PASSWORD from $ENV_FILE"
fi
echo ""
echo "Next steps:"
echo "1. Start production services: ./start-production.sh"
echo "2. Access your station at: https://your-server-ip:$NGINX_HTTPS_PORT"
echo "3. Stop services when done: ./stop-production.sh"
echo ""
echo "For development: python -m web_interface.app"
