# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# web_interface/app.py
import os
import sys
import json
import argparse
import hashlib
import threading
import time
from datetime import timedelta
from functools import wraps
from queue import Queue, Empty as QueueEmpty # Renamed to avoid conflict
from flask import Flask, request, jsonify, render_template, Response, session, redirect, url_for
from flask_httpauth import HTTPBasicAuth
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env file to ensure environment variables persist across gunicorn worker restarts
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# Adjust path to import the 'station' package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from station.station import Station
from station.station_runner import Orchestrator 
from station.base_room import InternalActionHandler 
from station.rooms.common import CommonRoom
from station import constants
from station import file_io_utils
from station import runtime_api_config
from station import capsule as capsule_module
from station import __version__ 
from station.llm_connectors.presets import load_model_presets
from station.system_messages import build_station_level_system_prompt
from web_interface.archive_utils import (
    build_archive_detail_payload,
    build_archive_list_payload,
)
from web_interface.input_utils import normalize_optional_role_definition
from web_interface.stream_utils import sanitize_stream_event_payload as _sanitize_stream_event_payload

# --- Global Variables ---
OPERATION_MODE: str = "api" 
station_instance: Optional[Station] = None
orchestrator_instance: Optional[Orchestrator] = None
orchestrator_log_queue = Queue() 



app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

def _cookie_hash_suffix() -> str:
    secret = app.secret_key
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    return hashlib.sha256(secret).hexdigest()[:8]

def _build_session_cookie_name(station_id: Optional[str] = None) -> str:
    base = "station_session"
    if station_id:
        base = f"{base}_{station_id}"
    return f"{base}_{_cookie_hash_suffix()}"

# Initial cookie name (may be refined once station instance is available)
_env_cookie_name = os.environ.get("FLASK_SESSION_COOKIE_NAME")
app.config["SESSION_COOKIE_NAME"] = _env_cookie_name or _build_session_cookie_name()

# ProxyFix for handling X-Forwarded headers correctly
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Import Flask utilities after app creation
# --- Authentication Setup ---
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    if not constants.WEB_AUTH_ENABLED:
        return True  # No auth required when disabled
    
    # Get credentials from environment variables
    auth_username = os.environ.get('FLASK_AUTH_USERNAME', 'admin')
    auth_password = os.environ.get('FLASK_AUTH_PASSWORD', 'changeme')
    
    return username == auth_username and password == auth_password

def _get_auth_credentials():
    auth_username = os.environ.get('FLASK_AUTH_USERNAME', 'admin')
    auth_password = os.environ.get('FLASK_AUTH_PASSWORD', 'changeme')
    return auth_username, auth_password

def _session_authenticated():
    if not constants.WEB_AUTH_ENABLED:
        return True
    auth_username, _ = _get_auth_credentials()
    return session.get("user") == auth_username

# Configure cookie-based sessions for "remember me"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=14)
if os.environ.get("FLASK_SESSION_COOKIE_SECURE", "true").lower() in ("1", "true", "yes"):
    app.config["SESSION_COOKIE_SECURE"] = True

# Auth decorator that respects the enable/disable setting
def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not constants.WEB_AUTH_ENABLED or _session_authenticated():
            return f(*args, **kwargs)
        # Fall back to HTTP Basic for API/CLI clients
        return auth.login_required(f)(*args, **kwargs)
    return wrapper

def _is_public_route():
    if request.endpoint in ("login_page", "login_submit", "static"):
        return True
    if request.path.startswith("/static/"):
        return True
    return False

# Add auth to all routes by default using before_request
@app.before_request
def require_auth():
    if not constants.WEB_AUTH_ENABLED or _is_public_route():
        return None

    # Session cookie keeps you signed in
    if _session_authenticated():
        return None

    # Let API clients continue using HTTP Basic
    if request.authorization or request.path.startswith("/api/"):
        return auth.login_required(lambda: None)()

    # Default: send to login form
    return redirect(url_for("login_page", next=request.path))

@app.route("/login", methods=["GET"])
def login_page():
    if not constants.WEB_AUTH_ENABLED:
        return redirect(url_for("dashboard_page"))
    if _session_authenticated():
        return redirect(url_for("dashboard_page"))
    error = request.args.get("error")
    return render_template("login.html", error=error)

@app.route("/login", methods=["POST"])
def login_submit():
    if not constants.WEB_AUTH_ENABLED:
        return redirect(url_for("dashboard_page"))
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    remember_me = bool(request.form.get("remember_me"))
    auth_username, auth_password = _get_auth_credentials()

    if username == auth_username and password == auth_password:
        session["user"] = username
        session.permanent = remember_me
        next_url = request.args.get("next") or url_for("dashboard_page")
        return redirect(next_url)

    return redirect(url_for("login_page", error="Invalid credentials"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

def _get_parallel_tick_status() -> Optional[Dict[str, Any]]:
    if getattr(constants, "SYNC_MODE", constants.SYNC_MODE_PARALLEL) != constants.SYNC_MODE_PARALLEL:
        return None

    try:
        from station.sync.parallel_status import load_parallel_tick_status

        return load_parallel_tick_status(orchestrator_instance)
    except Exception as exc:
        return {"active": False, "error": str(exc)}


# --- Initialization ---
def initialize_station_and_orchestrator():
    global station_instance, orchestrator_instance, OPERATION_MODE
    OPERATION_MODE = "api"
    print(f"Initializing application in '{OPERATION_MODE}' mode.")
    runtime_api_config.validate_provider_backup_env_config()
    try:
        station_instance = Station()
        # Once station is available, scope the session cookie to its unique ID (unless overridden)
        if "FLASK_SESSION_COOKIE_NAME" not in os.environ:
            station_cookie_name = _build_session_cookie_name(
                getattr(station_instance, "station_id", "unknown")
            )
            app.config["SESSION_COOKIE_NAME"] = station_cookie_name
        orchestrator_log_queue.put({"event": "status_update", "data": {"message": "Station instance initialized."}, "timestamp": time.time()})

        orchestrator_instance = Orchestrator(
            station_instance, 
            auto_prepare_on_init=True, # Will be prepared by UI action
            log_event_queue=orchestrator_log_queue
        )
        orchestrator_log_queue.put({"event": "status_update", "data": {"message": "Orchestrator instance created for API mode (idle)."}, "timestamp": time.time()})

    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        station_instance = None
        orchestrator_instance = None
        orchestrator_log_queue.put({"event": "error", "data": {"message": f"Station/Orchestrator initialization failed: {str(e)}"}, "timestamp": time.time()})

# --- HTML Serving Routes ---
@app.route('/')
@auth_required
def root_redirect_route():
    return redirect(url_for('dashboard_page'))

@app.route('/dashboard')
@auth_required
def dashboard_page():
    model_presets = load_model_presets()
    return render_template('dashboard.html', operation_mode=OPERATION_MODE, model_presets=model_presets)


@app.route('/api/archive/papers', methods=['GET'])
def archive_papers_list_route():
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503

    try:
        payload = build_archive_list_payload(capsule_module)
        return jsonify({"success": True, **payload})
    except Exception as e:
        app.logger.error(f"Error loading archive papers: {e}")
        return jsonify({"success": False, "error": f"Failed to load archive papers: {str(e)}"}), 500


@app.route('/api/archive/papers/<int:numeric_id>', methods=['GET'])
def archive_paper_detail_route(numeric_id: int):
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503

    try:
        payload = build_archive_detail_payload(capsule_module, numeric_id)
        if not payload:
            return jsonify({"success": False, "error": f"Archive paper #{numeric_id} not found."}), 404
        return jsonify({"success": True, **payload})
    except Exception as e:
        app.logger.error(f"Error loading archive paper #{numeric_id}: {e}")
        return jsonify({"success": False, "error": f"Failed to load archive paper #{numeric_id}: {str(e)}"}), 500

# --- API Endpoints - Orchestrator Control (API Mode) ---
@app.route('/api/orchestrator/status', methods=['GET'])
def get_orchestrator_status_route():
    if OPERATION_MODE != "api" or not orchestrator_instance or not station_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode.", 
                        "status": {"is_running": False, "is_prepared": False, "is_paused": False, "current_tick": -1, "turn_order":[], "agents_awaiting_human": []}}), 200
    
    agents_awaiting_human_list = station_instance.get_agents_awaiting_human_intervention() if hasattr(station_instance, 'get_agents_awaiting_human_intervention') else []
    status_data = {
        "is_prepared": orchestrator_instance.is_prepared, # ADDED
        "is_running": orchestrator_instance.is_running,
        "is_paused": orchestrator_instance.is_paused,
        "sync_mode": getattr(constants, "SYNC_MODE", constants.SYNC_MODE_PARALLEL),
        "pause_requested": orchestrator_instance.pause_requested,
        "pause_condition_met": orchestrator_instance.pause_condition_met,
        "pause_reason": orchestrator_instance.get_pause_reason(),
        "is_waiting": orchestrator_instance.is_waiting,
        "waiting_reasons": orchestrator_instance.waiting_reasons,
        "current_tick": station_instance._get_current_tick(),
        "station_status": station_instance.config.get(constants.STATION_CONFIG_STATION_STATUS, "Unknown"),
        "turn_order": list(orchestrator_instance.agent_turn_order),
        "next_agent_index": orchestrator_instance.current_agent_index_in_turn_order,
        "next_agent_to_act": (orchestrator_instance.agent_turn_order[orchestrator_instance.current_agent_index_in_turn_order]
                              if orchestrator_instance.agent_turn_order and
                                 0 <= orchestrator_instance.current_agent_index_in_turn_order < len(orchestrator_instance.agent_turn_order)
                              else "N/A"),
        "parallel_tick_status": _get_parallel_tick_status(),
        "agents_awaiting_human": agents_awaiting_human_list
    }
    return jsonify({"success": True, "status": status_data})

@app.route('/api/orchestrator/prepare', methods=['POST'])
def prepare_orchestrator_route():
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    if orchestrator_instance.is_running:
        return jsonify({"success": False, "message": "Cannot prepare while Orchestrator is running. Pause or Stop first."}), 400

    success = orchestrator_instance.prepare_for_run()
    msg = "Orchestrator prepared successfully." if success else "Orchestrator preparation failed."
    if success and not orchestrator_instance.agent_turn_order:
        msg += " No agents currently in turn order. Add agents before starting loop."
    return jsonify({"success": success, "message": msg, "is_prepared": orchestrator_instance.is_prepared})

# NEW: Endpoint to start the processing loop
@app.route('/api/orchestrator/start_loop', methods=['POST'])
def start_orchestrator_loop_route():
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    if not orchestrator_instance.is_prepared:
        return jsonify({"success": False, "message": "Orchestrator is not prepared. Please prepare first."}), 400
    if orchestrator_instance.is_running and not orchestrator_instance.is_paused:
         return jsonify({"success": False, "message": "Orchestrator loop is already running."}), 400
    if not orchestrator_instance.agent_turn_order:
        return jsonify({"success": False, "message": "No agents to process. Add agents before starting loop."}), 400

    success = orchestrator_instance.start_processing_loop() # This starts the thread
    msg = "Orchestrator processing loop started." if success else "Failed to start orchestrator processing loop."
    return jsonify({"success": success, "message": msg, "is_running": orchestrator_instance.is_running})

@app.route('/api/orchestrator/pause', methods=['POST'])
def pause_orchestrator_route_ep():
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    message = orchestrator_instance.request_manual_pause()
    return jsonify({"success": True, "message": message})

@app.route('/api/orchestrator/cancel_pause', methods=['POST'])
def cancel_pause_route_ep():
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    message = orchestrator_instance.cancel_pause_request()
    return jsonify({"success": True, "message": message})

@app.route('/api/orchestrator/resume', methods=['POST'])
def resume_orchestrator_route_ep():
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    message = orchestrator_instance.resume_orchestration()
    return jsonify({"success": True, "message": message})

@app.route('/api/orchestrator/stop', methods=['POST'])
def stop_orchestrator_route_ep():
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    success = orchestrator_instance.stop_orchestration()
    msg = "Orchestrator stopped." if success else "Failed to stop orchestrator or already stopped."
    return jsonify({"success": success, "message": msg})

# --- API Endpoints - Agent Management (API Mode Specific via Orchestrator) ---
@app.route('/api/orchestrator/add_agent', methods=['POST'])
def orchestrator_add_agent_route_ep(): # Renamed
    if OPERATION_MODE != "api" or not orchestrator_instance or not station_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    
    # Orchestrator's method should check if it's paused/stopped
    # if orchestrator_instance.is_running and not orchestrator_instance.is_paused:
    #     return jsonify({"success": False, "message": "Orchestrator must be paused or stopped to add agents."}), 400

    data = request.get_json()
    if not data: return jsonify({"success": False, "message": "Invalid JSON payload"}), 400

    agent_type = data.get('agent_type', constants.AGENT_STATUS_GUEST)
    model_provider_class = data.get('model_provider_class')
    model_name = data.get('model_name') # Specific LLM model for connector
    
    agent_name_override = data.get('agent_name')
    lineage = data.get('lineage')
    generation_str = data.get('generation')
    generation = int(generation_str) if generation_str and generation_str.isdigit() else None
    
    initial_tokens_max_str = data.get('initial_tokens_max')
    initial_tokens_max = int(initial_tokens_max_str) if initial_tokens_max_str and str(initial_tokens_max_str).isdigit() else None
    internal_note = data.get('internal_note', "")
    assigned_ancestor = data.get('assigned_ancestor', "")
    
    llm_system_prompt = normalize_optional_role_definition(data.get('llm_system_prompt'))
    llm_temperature_str = data.get('llm_temperature')
    llm_temperature = float(llm_temperature_str) if llm_temperature_str else None
    llm_max_tokens_str = data.get('llm_max_tokens')
    llm_max_tokens = int(llm_max_tokens_str) if llm_max_tokens_str and llm_max_tokens_str.isdigit() else None
    llm_custom_api_params = data.get('llm_custom_api_params', {})

    if not model_provider_class or not model_name:
        return jsonify({"success": False, "message": "model_provider_class and model_name are required for API agents."}), 400

    success, msg = orchestrator_instance.dynamic_add_agent_to_station(
        agent_type=agent_type,
        model_provider_class=model_provider_class,
        model_name=model_name,
        agent_name_override=agent_name_override,
        lineage=lineage,
        generation=generation,
        initial_tokens_max=initial_tokens_max,
        internal_note=internal_note,
        assigned_ancestor=assigned_ancestor,
        role_definition=llm_system_prompt,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        llm_custom_api_params=llm_custom_api_params
    )
    return jsonify({"success": success, "message": msg})

@app.route('/api/orchestrator/end_agent', methods=['POST'])
def orchestrator_end_agent_route_ep(): # Renamed
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Orchestrator not active or not in API mode."}), 403
    
    data = request.get_json()
    agent_name = data.get('agent_name')
    if not agent_name: return jsonify({"success": False, "message": "agent_name is required"}), 400

    success, msg = orchestrator_instance.dynamic_end_agent_session_manually(agent_name)
    return jsonify({"success": success, "message": msg})

# --- API Endpoints - Manual Takeover & Human Intervention (API Mode) ---
@app.route('/api/orchestrator/manual_message', methods=['POST'])
def orchestrator_manual_message_route_ep_v2(): # Renamed
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode."}), 403
    
    data = request.get_json()
    agent_name = data.get('agent_name')
    message_text = data.get('message_text')
    end_chat_after = data.get('end_chat_after_send', False) 

    if not agent_name or not message_text:
        return jsonify({"success": False, "error": "agent_name and message_text are required."}), 400

    # Orchestrator method will check if it's paused
    success, response_data = orchestrator_instance.send_manual_message_to_agent_llm(
        agent_name, message_text, end_chat_after
    )
    return jsonify({"success": success, **response_data})

@app.route('/api/reviewer/manual_message', methods=['POST'])
def reviewer_manual_message_route():
    """Manual direct message to the Reviewer (archive evaluator)"""
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 403

    evaluator = getattr(station_instance, "auto_archive_evaluator", None)
    if not evaluator:
        return jsonify({"success": False, "error": "Reviewer system is not available."}), 503

    data = request.get_json() or {}
    message_text = data.get('message_text')
    if not message_text or not isinstance(message_text, str) or not message_text.strip():
        return jsonify({"success": False, "error": "message_text is required and cannot be empty."}), 400

    success, response_data = evaluator.send_manual_message_to_reviewer(message_text)
    return jsonify({"success": success, **response_data})

@app.route('/api/orchestrator/get_human_request', methods=['GET'])
def get_human_request_details():
    """Get details of a pending human request for a specific agent"""
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode."}), 403

    agent_name = request.args.get('agent_name')
    if not agent_name:
        return jsonify({"success": False, "error": "agent_name is required."}), 400
    request_id_param = request.args.get('request_id')
    request_id = None
    if request_id_param is not None:
        try:
            request_id = int(request_id_param)
        except (TypeError, ValueError):
            request_id = request_id_param

    # Get the request details from the administrative counter
    try:
        from station import constants
        admin_counter = orchestrator_instance.station.rooms.get(constants.ROOM_ADMIN)
        if not admin_counter:
            return jsonify({"success": False, "error": "Administrative Counter not available"}), 404

        # Check if agent has a pending request
        if agent_name not in admin_counter.pending_requests:
            return jsonify({"success": False, "error": f"No pending request for agent {agent_name}"}), 404

        pending_request_ids = admin_counter.pending_requests[agent_name]
        if isinstance(pending_request_ids, list):
            pending_request_ids = list(pending_request_ids)
        else:
            pending_request_ids = [pending_request_ids]

        # Load the human requests log to get details
        import os
        log_path = admin_counter.log_file_path
        if not os.path.exists(log_path):
            return jsonify({"success": False, "error": "Human requests log not found"}), 404

        # Load all requests and find the matching one
        from station import file_io_utils
        requests = file_io_utils.load_yaml_lines(log_path)

        matching_requests = []
        for req in requests:
            if req.get('request_id') in pending_request_ids and not req.get('resolved', False):
                matching_requests.append({
                    "request_id": req.get('request_id'),
                    "tick": req.get('tick'),
                    "agent_name": req.get('agent_name'),
                    "agent_model": req.get('agent_model'),
                    "title": req.get('title'),
                    "content": req.get('content'),
                    "timestamp": req.get('timestamp')
                })

        if request_id is not None:
            matching_requests = [req for req in matching_requests if req.get('request_id') == request_id]

        if not matching_requests:
            if request_id is not None:
                return jsonify({"success": False, "error": f"Request {request_id} not found in log"}), 404
            return jsonify({"success": False, "error": f"No pending requests found for agent {agent_name}"}), 404

        return jsonify({
            "success": True,
            "requests": matching_requests,
            "request": matching_requests[0],
            "selected_request_id": request_id
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Error fetching request: {str(e)}"}), 500

@app.route('/api/orchestrator/resolve_human_intervention', methods=['POST'])
def orchestrator_resolve_human_intervention_route_ep_v2(): # Renamed
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode."}), 403

    data = request.get_json()
    agent_name = data.get('agent_name')
    request_id = data.get('request_id')
    if not agent_name:
        return jsonify({"success": False, "error": "agent_name is required."}), 400

    resolution_reason = data.get("reason", "Intervention resolved by UI action.")
    response_text = data.get("response_text", None)  # Optional human response

    success, message = orchestrator_instance.resolve_human_intervention(
        agent_name,
        resolution_reason,
        human_response=response_text,
        request_id=request_id
    )
    return jsonify({"success": success, "message": message})

def _transform_reviewer_history_to_agent_format(reviewer_entries):
    """Transform reviewer LLM chat history format to agent dialogue format"""
    transformed_entries = []
    
    for entry in reviewer_entries:
        if not isinstance(entry, dict):
            continue
            
        tick = entry.get('tick')
        role = entry.get('role')  # 'user', 'model', or 'assistant'
        thinking_content = entry.get('thinking_content')
        
        # Extract text content from parts array or direct content field
        parts = entry.get('parts', [])
        text_content = ""
        if parts and isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and 'text' in part:
                    text_content += part.get('text', '')
        
        # If no parts array, try direct content field (used by OpenAI/Grok connectors)
        if not text_content:
            text_content = entry.get('content', '')
        
        # Create entries in agent format
        if role == 'user':
            # Human/system prompt to reviewer (this is like Station giving a prompt)
            transformed_entries.append({
                'tick': tick,
                'speaker': 'Station',
                'type': 'observation',
                'content': text_content,
                'text_content': text_content,
                'agent_name': 'Reviewer'
            })
        elif role in ['model', 'assistant']:
            # Reviewer's thinking (if exists)
            if thinking_content:
                transformed_entries.append({
                    'tick': tick,
                    'speaker': 'ReviewerLLM',
                    'type': 'thinking_block',
                    'content': thinking_content,
                    'text_content': thinking_content,
                    'agent_name': 'Reviewer'
                })
            
            # Reviewer's response (this is like an Agent submission)
            if text_content:
                transformed_entries.append({
                    'tick': tick,
                    'speaker': 'ReviewerLLM',
                    'type': 'submission',
                    'content': text_content,
                    'text_content': text_content,
                    'agent_name': 'Reviewer'
                })
    
    return transformed_entries

@app.route('/api/agent_dialogue_history/<agent_name>', methods=['GET'])
def get_agent_dialogue_history_route(agent_name: str):
    if not station_instance or not station_instance.agent_module:
        return jsonify({"success": False, "error": "Station not properly initialized."}), 500

    load_full = request.args.get('full', 'false').lower() == 'true'
    window = request.args.get('window', 'recent').lower()
    try:
        tick_limit = int(request.args.get('ticks', '50'))
    except (TypeError, ValueError):
        tick_limit = 50
    tick_limit = max(1, min(tick_limit, 1000))

    # Handle special case for Reviewer
    if agent_name == "Reviewer":
        # Use the archive room's LLM chat history file
        log_file_path = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_ARCHIVE,
            "llm_chat_history.yamll"
        )
    else:
        # Regular agent dialogue history
        dialogue_logs_base_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.DIALOGUE_LOGS_DIR_NAME)
        safe_agent_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in agent_name)
        log_filename = f"{safe_agent_name}{constants.DIALOGUE_LOG_FILENAME_SUFFIX}"
        log_file_path = os.path.join(dialogue_logs_base_path, log_filename)

    if not os.path.exists(log_file_path):
        return jsonify({"success": True, "history": [], "message": f"No dialogue history found for {agent_name}."})

    try:
        range_meta: Dict[str, Any] = {
            "mode": "full" if load_full else window,
            "ticks": tick_limit,
            "min_tick": None,
            "max_tick": None,
            "is_partial": False,
            "has_older": False,
            "has_newer": False,
        }
        if load_full or agent_name == "Reviewer":
            history_entries = file_io_utils.load_yaml_lines(log_file_path)
        elif window in ("recent", "earliest"):
            history_entries, range_meta = file_io_utils.load_yaml_lines_tick_window(
                log_file_path,
                window=window,
                tick_limit=tick_limit,
            )
        else:
            return jsonify({"success": False, "error": f"Unsupported history window: {window}"}), 400
        is_truncated = False

        # Transform reviewer format to agent format if this is the reviewer
        if agent_name == "Reviewer":
            history_entries = _transform_reviewer_history_to_agent_format(history_entries)

        if not load_full and history_entries:
            # First try tick-based truncation for backwards compatibility
            tick_entries = [entry for entry in history_entries if 'tick' in entry and entry['tick'] is not None]
            if tick_entries:
                unique_ticks = sorted(list(set(entry['tick'] for entry in tick_entries)), reverse=True)
                if len(unique_ticks) > 100:
                    is_truncated = True
                    cutoff_tick = unique_ticks[99] # Get the 100th most recent tick
                    history_entries = [entry for entry in history_entries if not 'tick' in entry or entry.get('tick') is None or entry.get('tick') >= cutoff_tick]
            # Also check if we have too many total entries (more than 500)
            elif len(history_entries) > 500:
                is_truncated = True
                # Keep only the most recent 500 entries
                history_entries = history_entries[-500:]

        # Use json.dumps with Response instead of jsonify to avoid Content-Length mismatch
        response_data = {"success": True, "history": history_entries, "is_truncated": is_truncated, "range": range_meta}
        json_str = json.dumps(response_data)
        return Response(json_str, mimetype='application/json')
    except Exception as e:
        app.logger.error(f"Error reading dialogue history for {agent_name} from {log_file_path}: {e}")
        return jsonify({"success": False, "error": f"Could not read dialogue history: {str(e)}"}), 500

# --- General Station Info (Shared) ---
@app.route('/api/station_tick', methods=['GET'])
def get_station_tick_ep_v2(): # Renamed
    if not station_instance: return jsonify({"success": False, "error": "Station not initialized"}), 500
    return jsonify({"success": True, "current_tick": station_instance._get_current_tick()})

@app.route('/api/agents', methods=['GET'])
def get_agents_ep_v2(): # Renamed
    if not station_instance: return jsonify({"success": False, "error": "Station not initialized"}), 500
    return jsonify({"success": True, "agents": station_instance.get_all_agents_summary()})


def _build_reviewer_system_prompt_text() -> str:
    """Return the actual runtime system prompt for the archive reviewer."""
    evaluator = getattr(station_instance, "auto_archive_evaluator", None) if station_instance else None

    connector_system_prompt = constants.ARCHIVE_REVIEWER_SYSTEM_PROMPT
    if evaluator and getattr(evaluator, "llm_connector", None):
        connector_system_prompt = getattr(
            evaluator.llm_connector,
            "system_prompt",
            connector_system_prompt,
        ) or connector_system_prompt

    return connector_system_prompt


@app.route('/api/agent/<agent_name>/system_prompt', methods=['GET'])
def get_agent_system_prompt_ep(agent_name: str):
    if not station_instance:
        return jsonify({"success": False, "error": "Station not initialized"}), 500

    if agent_name == "Reviewer":
        return jsonify({
            "success": True,
            "agent_name": agent_name,
            "system_prompt": _build_reviewer_system_prompt_text()
        })

    if not getattr(station_instance, "agent_module", None):
        return jsonify({"success": False, "error": "Station agent module is not available"}), 500

    try:
        agent_data = station_instance.agent_module.load_agent_data(
            agent_name,
            include_ended=True,
            include_ascended=True,
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to load agent data: {str(e)}"}), 500

    if not agent_data:
        return jsonify({"success": False, "error": f"Agent '{agent_name}' not found."}), 404

    raw_role_definition = station_instance.agent_module.get_agent_role_definition(agent_data)
    system_prompt = build_station_level_system_prompt(agent_name, raw_role_definition)
    return jsonify({
        "success": True,
        "agent_name": agent_name,
        "system_prompt": system_prompt or ""
    })

@app.route('/api/station/statistics', methods=['GET'])
def get_station_statistics():
    """Get station-wide statistics including pending human requests and top research submission"""
    if not station_instance: 
        return jsonify({"success": False, "error": "Station not initialized"}), 500
    
    try:
        stats = station_instance.get_station_statistics()
        return jsonify({
            "success": True,
            "statistics": stats
        })
    except Exception as e:
        app.logger.error(f"Error getting station statistics: {e}")
        return jsonify({
            "success": False,
            "error": f"Error getting station statistics: {str(e)}"
        }), 500

@app.route('/api/station/version', methods=['GET'])
def get_station_version():
    """Get station version information"""
    return jsonify({
        "success": True,
        "version": __version__
    })

@app.route('/api/station/config', methods=['GET'])
def get_station_config():
    """Get station configuration information"""
    if not station_instance:
        return jsonify({"success": False, "error": "Station not initialized"}), 500
        
    try:
        config = {
            "station_status": station_instance.config.get(constants.STATION_CONFIG_STATION_STATUS, "Unknown"),
            "station_name": station_instance.config.get(constants.STATION_CONFIG_NAME, ""),
            "station_description": station_instance.config.get(constants.STATION_CONFIG_DESCRIPTION, ""),
            "station_id": station_instance.config.get(constants.STATION_ID_KEY, "Unknown"),
            "sync_mode": getattr(constants, "SYNC_MODE", "parallel"),
        }
        return jsonify({"success": True, "config": config})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/station/config', methods=['PUT'])
def update_station_config():
    """Update station configuration"""
    if not station_instance:
        return jsonify({"success": False, "error": "Station not initialized"}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        status = data.get('station_status')
        name = data.get('station_name')
        description = data.get('station_description')
        
        result = station_instance.update_station_config(status=status, name=name, description=description)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/station/api_runtime_config', methods=['GET'])
def get_api_runtime_config():
    """Return sanitized runtime API/proxy configuration for the dashboard."""
    try:
        return jsonify({
            "success": True,
            "config": runtime_api_config.build_public_config(),
        })
    except Exception as e:
        app.logger.error(f"Error loading runtime API config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/station/api_runtime_config', methods=['PUT'])
def update_api_runtime_config():
    """Apply an in-memory runtime API/proxy update for this Station process."""
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid JSON payload"}), 400

        config = runtime_api_config.apply_update(data)
        if orchestrator_instance:
            try:
                orchestrator_instance.handle_runtime_api_config_updated(config.get("generation", 0))
            except Exception as e:
                app.logger.warning(f"Failed to notify orchestrator of runtime API config update: {e}")

        return jsonify({
            "success": True,
            "message": "Runtime API settings updated for this running Station process.",
            "config": config,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error updating runtime API config: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# --- SSE Stream ---
@app.route('/api/orchestrator/live_log_stream') # Or your actual SSE route path
def live_log_stream_route_ep_v2(): # Ensure this is the correct function name
    if OPERATION_MODE != "api":
        return Response("SSE only available in API mode.", status=403, mimetype='text/event-stream')
    selected_agent_name = request.args.get("agent_name") or None
    
    def event_stream():
        # ... (connect_event yield as before) ...
        app.logger.info("SSE client connected for live log stream.") # Use app.logger
        connect_event = {"event": "stream_status", "data": {"message": "SSE client connected to live log."}, "timestamp": time.time()}
        yield f"data: {json.dumps(connect_event)}\n\n"
        
        try:
            while True:
                try:
                    log_entry = orchestrator_log_queue.get(timeout=1) # Waits up to 1 second
                    log_entry = _sanitize_stream_event_payload(log_entry, selected_agent_name)
                    yield f"data: {json.dumps(log_entry)}\n\n"
                except QueueEmpty:
                    # Send a comment as a keep-alive signal
                    yield ": keepalive\n\n"
                except Exception as e_inner: 
                    app.logger.error(f"Error during SSE event generation: {e_inner}")
                    error_event = {"event": "stream_error", "data": {"message": f"SSE internal error: {str(e_inner)}"}}
                    try:
                        yield f"data: {json.dumps(error_event)}\n\n"
                    except Exception as e_yield:
                        app.logger.error(f"Error yielding SSE error event: {e_yield}")
                    time.sleep(0.1) # Small delay before retrying get
        except GeneratorExit: 
            app.logger.info("SSE client disconnected (GeneratorExit).")
        except Exception as e_outer:
            app.logger.error(f"Critical error in SSE event_stream: {e_outer}")
        finally:
            app.logger.info("SSE event_stream closing for this client.")
            
    return Response(event_stream(), content_type='text/event-stream')

# --- Polling API (fallback for when SSE doesn't work) ---
@app.route('/api/orchestrator/recent_events')
def recent_events_route():
    """Get recent events from the log queue for polling mode (fallback)"""
    if OPERATION_MODE != "api":
        return jsonify({"success": False, "message": "Recent events only available in API mode"}), 403
    selected_agent_name = request.args.get("agent_name") or None
    
    events = []
    try:
        # Drain up to 50 events from the queue without blocking
        for _ in range(50):
            try:
                event = orchestrator_log_queue.get_nowait()
                event = _sanitize_stream_event_payload(event, selected_agent_name)
                events.append(event)
            except QueueEmpty:
                break
        
        return jsonify({
            "success": True,
            "events": events,
            "count": len(events)
        })
    except Exception as e:
        app.logger.error(f"Error getting recent events: {e}")
        return jsonify({
            "success": False,
            "message": f"Error retrieving events: {str(e)}"
        }), 500


@app.route('/api/agent/<agent_name>/final_chat', methods=['POST'])
def final_chat_with_agent_route(agent_name: str):
    # Ensure orchestrator_instance is used, as it now holds the logic
    if OPERATION_MODE != "api" or not orchestrator_instance: # Check if orchestrator is available
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode, cannot perform final chat."}), 503 # Service Unavailable

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON payload."}), 400
        
    human_message = data.get('human_message')
    if not human_message or not isinstance(human_message, str) or not human_message.strip():
        return jsonify({"success": False, "error": "Field 'human_message' is required and cannot be empty."}), 400

    # Call the method on the orchestrator instance
    llm_response, thinking_text, error_msg = orchestrator_instance.perform_final_chat_with_ended_agent(agent_name, human_message)

    if error_msg:
        return jsonify({"success": False, "error": error_msg}), 500 
    
    return jsonify({"success": True, "agent_response": llm_response})

@app.route('/api/agent/<agent_name>/temporal_chat', methods=['GET'])
def temporal_chat_state_route(agent_name: str):
    """Load the persisted temporal chat transcript for an agent."""
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode."}), 503

    chat_state, error_msg = orchestrator_instance.get_temporal_chat_state(agent_name)
    if error_msg:
        return jsonify({"success": False, "error": error_msg}), 500
    return jsonify({"success": True, "chat": chat_state})


@app.route('/api/agent/<agent_name>/temporal_chat/refresh', methods=['POST'])
def temporal_chat_refresh_route(agent_name: str):
    """Discard an agent's temporal fork and freeze a new one from a branch tick."""
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode."}), 503

    data = request.get_json(silent=True) or {}
    base_tick = data.get("base_tick") if isinstance(data, dict) else None

    chat_state, error_msg = orchestrator_instance.refresh_temporal_chat(agent_name, base_tick=base_tick)
    if error_msg:
        status_code = 400 if "branch tick" in error_msg.lower() else 500
        return jsonify({"success": False, "error": error_msg}), status_code
    return jsonify({"success": True, "chat": chat_state})

@app.route('/api/agent/<agent_name>/temporal_chat', methods=['POST'])
def temporal_chat_with_agent_route(agent_name: str):
    """
    Persistent temporal chat with an agent.

    The first send freezes the agent's current LLM history into a temporal fork.
    Further sends continue from that fork until the user branches again.
    """
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode, cannot perform temporal chat."}), 503

    data = request.get_json() or {}
    user_message = data.get("user_message")
    base_tick = data.get("base_tick")

    if not user_message or not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"success": False, "error": "Field 'user_message' is required and cannot be empty."}), 400

    llm_response, thinking_text, chat_state, error_msg = orchestrator_instance.perform_temporal_chat_with_agent(
        agent_name=agent_name,
        user_message=user_message,
        base_tick=base_tick,
    )

    if error_msg:
        status_code = 400 if "branch tick" in error_msg.lower() else 500
        return jsonify({"success": False, "error": error_msg, "chat": chat_state}), status_code

    return jsonify({
        "success": True,
        "agent_response": llm_response,
        "thinking_text": thinking_text,
        "chat": chat_state,
    })

@app.route('/api/station/send_system_message', methods=['POST'])
def station_send_system_message_route():
    global station_instance # Ensure access to global station_instance
    if OPERATION_MODE != "api" or not station_instance or not station_instance.agent_module:
        return jsonify({"success": False, "message": "Station/Orchestrator not active or not in API mode for this function."}), 403

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid JSON payload"}), 400

    target_agents = data.get('target_agents')
    message_content = data.get('message_content')

    if not isinstance(target_agents, list) or not target_agents:
        return jsonify({"success": False, "message": "target_agents must be a non-empty list."}), 400
    if not message_content or not isinstance(message_content, str) or not message_content.strip():
        return jsonify({"success": False, "message": "message_content is required and cannot be empty."}), 400

    successful_sends = []
    failed_sends = []
    
    # Make sure app.logger is available or use print for server-side logging
    # app.logger.info(f"Attempting to send system message to: {target_agents}")

    for agent_name in target_agents:
        # Load agent data, ensuring it's an active agent (default behavior of load_agent_data)
        agent_data = station_instance.agent_module.load_agent_data(agent_name)
        if agent_data:
            try:
                station_instance.agent_module.add_pending_notification(agent_data, message_content)
                if station_instance.agent_module.save_agent_data(agent_name, agent_data):
                    successful_sends.append(agent_name)
                    # Optionally, push an SSE event here for each successful send
                    if orchestrator_instance and orchestrator_instance.log_event_queue: # Check if orchestrator_instance exists
                        orchestrator_instance.log_event_queue.put({
                            "event": "system_message", # Use this type for agent-specific log
                            "data": {
                                "agent_name": agent_name, # Critical for routing in JS
                                "message": message_content,
                                "source": "manual_system_message_tool"
                            },
                            "timestamp": time.time()
                        })
                else:
                    failed_sends.append({"name": agent_name, "reason": "Failed to save agent data after adding notification."})
            except Exception as e:
                app.logger.error(f"Error adding system message for agent {agent_name}: {e}")
                failed_sends.append({"name": agent_name, "reason": str(e)})
        else:
            failed_sends.append({"name": agent_name, "reason": "Agent not found or not active."})
    
    if not failed_sends:
        return jsonify({"success": True, "message": f"System message successfully sent to {len(successful_sends)} agent(s)."})
    else:
        # Log the overall outcome
        app.logger.warning(f"System message sending: Successes: {len(successful_sends)}, Failures: {len(failed_sends)}. Details: {failed_sends}")
        return jsonify({
            "success": len(successful_sends) > 0, # Overall success if at least one worked
            "message": f"System message sent to {len(successful_sends)} agent(s). Failed for {len(failed_sends)} agent(s). See details.",
            "details": {
                "successful_sends": successful_sends,
                "failed_sends": failed_sends
            }
        }), 207 # Multi-Status

@app.route('/api/room/common/speak', methods=['POST'])
def common_room_speak_as_route():
    global station_instance # Ensure access to global station_instance
    global orchestrator_instance # For SSE logging

    if not station_instance or not station_instance.rooms.get(constants.ROOM_COMMON):
        return jsonify({"success": False, "message": "Station or Common Room not initialized."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Invalid JSON payload"}), 400

    speaker_name = data.get('speaker_name')
    message_content = data.get('message_content')

    if not speaker_name or not isinstance(speaker_name, str) or not speaker_name.strip():
        return jsonify({"success": False, "message": "speaker_name is required and cannot be empty."}), 400
    if not message_content or not isinstance(message_content, str) or not message_content.strip():
        return jsonify({"success": False, "message": "message_content is required and cannot be empty."}), 400

    common_room_instance = station_instance.rooms.get(constants.ROOM_COMMON)
    current_tick = station_instance._get_current_tick()
    
    # Ensure room_context is available; it's an attribute of station_instance
    if not hasattr(station_instance, 'room_context'):
        app.logger.error("Station instance is missing room_context.")
        return jsonify({"success": False, "message": "Internal server error: Room context not found."}), 500
    
    room_context = station_instance.room_context
    
    if isinstance(common_room_instance, CommonRoom):
        success = common_room_instance.add_message_as_speaker(
            speaker_name=speaker_name,
            message_content=message_content,
            current_tick=current_tick,
            room_context=room_context
        )
        if success:
            # Optionally, push an SSE event if you want real-time notification of this
            # Ensure orchestrator_instance might be None if in manual mode, so check it.
            if orchestrator_instance and hasattr(orchestrator_instance, 'log_event_queue') and orchestrator_instance.log_event_queue:
                try:
                    orchestrator_instance.log_event_queue.put_nowait({ # Use put_nowait
                        "event": "common_room_message", # New SSE event type
                        "data": {
                            "speaker": speaker_name,
                            "message": message_content,
                            "tick": current_tick,
                            "source": "external_ui_tool"
                        },
                        "timestamp": time.time()
                    })
                except Exception as e:
                    app.logger.error(f"Failed to put common room message to SSE queue: {e}")

            return jsonify({"success": True, "message": f"Message from '{speaker_name}' posted to Common Room."})
        else:
            return jsonify({"success": False, "message": "Failed to post message to Common Room (check server logs)."}), 500
    else:
        app.logger.error(f"Common Room instance type mismatch: {type(common_room_instance)}")
        return jsonify({"success": False, "message": "Common Room instance is not of the correct type."}), 500

@app.route('/api/backup/create', methods=['POST'])
def create_backup_route():
    """Create a manual backup of station data."""
    if OPERATION_MODE != "api" or not orchestrator_instance:
        return jsonify({"success": False, "message": "Backup only available in API mode with active orchestrator."}), 403
    
    try:
        # This will now raise an exception on failure instead of returning (False, error)
        success, backup_path = orchestrator_instance.create_manual_backup()
        
        # Manual backup errors should NOT halt the orchestrator, only return error to user
        return jsonify({
            "success": True, 
            "message": f"Backup created successfully",
            "backup_path": backup_path
        })
            
    except Exception as e:
        app.logger.error(f"Error creating backup: {e}")
        # Re-raise to halt the orchestrator as requested
        raise


# --- Initialize for Gunicorn (when module is imported) ---
# For Gunicorn, initialize with default API mode since __name__ != '__main__'
if station_instance is None:
    initialize_station_and_orchestrator()

# --- Shutdown API ---
@app.route('/api/shutdown', methods=['POST'])
@auth.login_required
def api_shutdown():
    """Gracefully shutdown research evaluations"""
    try:
        if station_instance:
            print("API: Received shutdown request, cleaning up station...")
            # Stop all evaluation loops
            if hasattr(station_instance, 'auto_research_evaluator') and station_instance.auto_research_evaluator:
                station_instance.stop_auto_research_evaluator()
            if hasattr(station_instance, 'auto_archive_evaluator') and station_instance.auto_archive_evaluator:
                station_instance.stop_auto_archive_evaluator()
            if hasattr(station_instance, 'auto_archive_surveyor') and station_instance.auto_archive_surveyor:
                station_instance.stop_auto_archive_surveyor()
            print("API: Station cleanup completed")
            return jsonify({"status": "success", "message": "Station cleanup completed"})
        else:
            return jsonify({"status": "error", "message": "No station instance available"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Cleanup failed: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Station Web Interface')
    # Check environment variable first, then fall back to command line argument
    default_port = int(os.environ.get('FLASK_PORT', 5000))
    parser.add_argument('--port', type=int, default=default_port,
                        help=f'Port to run the web interface on (default: {default_port})')
    parser.add_argument('--rebuild-db', '--rebuild_db', action='store_true',
                        help='Rebuild the derived SQLite index from authoritative YAML before startup.')
    args = parser.parse_args()

    # Suppress Flask/Werkzeug access logs
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Only show errors, not every request
    
    # Re-initialize if instance is missing (e.g. if run directly after import)
    if station_instance is None:
        initialize_station_and_orchestrator()

    if not station_instance or not orchestrator_instance:
        print("FATAL: Station or Orchestrator instance could not be initialized. Exiting.")
        sys.exit(1)

    templates_dir = os.path.join(current_dir, 'templates')
    if not os.path.exists(templates_dir): os.makedirs(templates_dir)
    
    static_js_dir = os.path.join(current_dir, 'static', 'js')
    if not os.path.exists(static_js_dir): os.makedirs(static_js_dir, exist_ok=True)
    static_css_dir = os.path.join(current_dir, 'static', 'css')
    if not os.path.exists(static_css_dir): os.makedirs(static_css_dir, exist_ok=True)

    # Print startup information
    print("\n" + "=" * 60)
    print("STATION WEB INTERFACE STARTING")
    print("=" * 60)
    print(f"Running in standard mode")
    print(f"\nAccess your station at:")
    print(f"  http://localhost:{args.port}/")
    print("=" * 60 + "\n")

    # Run with standard settings
    app.run(debug=True, host='0.0.0.0', port=args.port, use_reloader=False)
