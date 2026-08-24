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
import html
import threading
import time
from datetime import timedelta
from functools import wraps
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
from station.multistart import ipc as multistart_ipc
from station.multistart import waiting as multistart_waiting
from web_interface import multistart_preview
from station.llm_connectors.presets import load_model_presets
from station.system_messages import build_station_level_system_prompt
from web_interface.archive_utils import (
    build_archive_detail_payload,
    build_archive_list_payload,
)
from web_interface.archive_survey_service import (
    WebArchiveSurveyBusyError,
    WebArchiveSurveyNotFoundError,
    WebArchiveSurveyService,
    build_web_archive_survey_templates,
)
from web_interface.question_utils import (
    build_question_detail_payload,
    build_question_list_payload,
    build_question_survey_preview,
)
from web_interface.task_spec_utils import (
    TaskSpecConflictError,
    get_task_spec_snapshot,
    save_task_spec_snapshot,
)
from web_interface.input_utils import normalize_optional_role_definition
from web_interface.live_event_broker import DashboardEventBroker
from web_interface.stream_utils import sanitize_stream_event_payload as _sanitize_stream_event_payload

# --- Global Variables ---
OPERATION_MODE: str = "api" 
station_instance: Optional[Station] = None
orchestrator_instance: Optional[Orchestrator] = None
orchestrator_event_broker = DashboardEventBroker()
web_archive_survey_service: Optional[WebArchiveSurveyService] = None
web_archive_survey_service_lock = threading.Lock()
station_statistics_lock = threading.Lock()
station_statistics_cache: Optional[Dict[str, Any]] = None
research_evaluator_refresh_lock = threading.Lock()
research_evaluator_refresh_state: Dict[str, Any] = {"status": "idle"}



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

def _multistart_waiting_response():
    status = multistart_waiting.public_status()
    initial_status_json = json.dumps(status, separators=(",", ":")).replace("</", "<\\/")
    page_title = html.escape(str(status.get("station_name") or "Station"), quote=True)
    response_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title}</title>
  <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d7dde8;
      --text: #18212f;
      --muted: #5f6f86;
      --accent: #2563eb;
      --good: #0f766e;
      --warn: #9a5b00;
      --bad: #b42318;
      --chip: #eef2f8;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 42px; }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .title-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }}
    h1 {{ margin: 0; font-size: 26px; line-height: 1.2; letter-spacing: 0; }}
    p {{ margin: 0; line-height: 1.55; color: var(--muted); }}
    .updated {{ font-size: 13px; color: var(--muted); white-space: nowrap; padding-top: 6px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .metric .label {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .metric .value {{ font-size: 18px; font-weight: 650; overflow-wrap: anywhere; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .panel-head {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 16px;
      align-items: center;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }}
    .panel-head h2 {{ margin: 0; font-size: 16px; }}
    .hint {{ color: var(--muted); font-size: 13px; }}
    .controls {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: nowrap;
      white-space: nowrap;
    }}
    .control-button {{
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
      font-weight: 650;
      line-height: 1;
      padding: 8px 11px;
      cursor: pointer;
    }}
    .control-button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .control-button:disabled {{
      cursor: not-allowed;
      opacity: 0.45;
    }}
    .control-state {{
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
      text-align: right;
    }}
    .action-message {{
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
      text-align: right;
      white-space: nowrap;
    }}
    .action-message.error {{ color: var(--bad); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf0f5; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; background: #fafbfe; }}
    tr:last-child td {{ border-bottom: 0; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 3px 8px;
      background: var(--chip);
      color: var(--text);
      font-weight: 600;
      white-space: nowrap;
    }}
    .status.running, .status.interviewing {{ color: var(--accent); background: #eaf1ff; }}
    .status.paused {{ color: var(--warn); background: #fff7e8; }}
    .status.completed {{ color: var(--good); background: #e7f6f2; }}
    .status.failed {{ color: var(--bad); background: #fff0ed; }}
    .status.waiting_quiescent {{ color: var(--warn); background: #fff7e8; }}
    .progress {{
      height: 8px;
      border-radius: 999px;
      background: #e7ebf2;
      overflow: hidden;
      width: 100%;
      min-width: 86px;
      margin-top: 5px;
    }}
    .bar {{ height: 100%; width: 0; background: var(--accent); }}
    .small {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .tick-value {{ font-size: 15px; font-weight: 650; }}
    .progress-cell {{ min-width: 130px; }}
    .note {{ max-width: 270px; overflow-wrap: anywhere; }}
    .empty {{ padding: 24px 16px; color: var(--muted); }}
    .preview-link {{
      display: inline-flex;
      align-items: center;
      margin-top: 12px;
      padding: 8px 12px;
      border-radius: 7px;
      background: var(--accent);
      color: white;
      font-weight: 650;
      text-decoration: none;
    }}
    @media (max-width: 900px) {{
      header {{ display: block; }}
      .updated {{ margin-top: 8px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .panel-head {{ grid-template-columns: 1fr; }}
      .controls {{ justify-content: flex-start; overflow-x: auto; }}
      .control-state, .action-message {{ text-align: left; }}
    }}
    @media (max-width: 520px) {{
      main {{ width: min(100vw - 20px, 1180px); padding-top: 18px; }}
      .summary {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="title-row">
          <h1>Multistart is working</h1>
        </div>
        <p>Candidate starts are running independently. Seed 1 is available as a read-only Station preview and may not be the branch ultimately selected.</p>
        <a class="preview-link" href="/dashboard">View Seed 1 dashboard</a>
      </div>
      <div class="updated" id="updated">Loading...</div>
    </header>
    <section class="summary" aria-label="Multistart summary">
      <div class="metric"><div class="label">Mode</div><div class="value" id="mode">-</div></div>
      <div class="metric"><div class="label">Stage</div><div class="value" id="stage">-</div></div>
      <div class="metric"><div class="label">Job</div><div class="value mono" id="job">-</div></div>
      <div class="metric"><div class="label">Branch span tick</div><div class="value" id="branchTick">-</div></div>
      <div class="metric"><div class="label">Seeds</div><div class="value" id="seeds">-</div></div>
      <div class="metric"><div class="label">Parallel</div><div class="value" id="parallel">-</div></div>
      <div class="metric"><div class="label">Active coders</div><div class="value" id="activeCoders">-</div></div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <h2>Candidate Branches</h2>
        <div class="controls">
          <span class="control-state" id="controlState">Control: -</span>
          <button class="control-button" id="pauseButton" type="button">Pause</button>
          <button class="control-button primary" id="resumeButton" type="button">Resume</button>
          <div class="action-message" id="actionMessage"></div>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Seed</th>
              <th>Status</th>
              <th>Station</th>
              <th>Tick</th>
              <th>Progress</th>
              <th>Top eval</th>
              <th>Top score</th>
              <th>Coders</th>
              <th>PID</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody id="branches"></tbody>
        </table>
      </div>
      <div class="empty" id="empty" hidden>No branch information is available yet.</div>
    </section>
  </main>
  <script>
    const initialStatus = {initial_status_json};
    const text = (value) => value === null || value === undefined || value === "" ? "-" : String(value);
    const esc = (value) => text(value).replace(/[&<>"']/g, (ch) => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#39;"}}[ch]));
    const cls = (value) => text(value).toLowerCase().replace(/[^a-z0-9_]+/g, "_");

    function setText(id, value) {{
      document.getElementById(id).textContent = text(value);
    }}

    function tickCell(branch) {{
      const current = branch.current_tick ?? "-";
      const age = branch.last_tick_timestamp ? formatAge((Date.now() / 1000) - branch.last_tick_timestamp) : text(branch.last_tick_age_display);
      return `
        <div class="mono tick-value">${{esc(current)}}</div>
        <div class="small">last tick ${{esc(age)}} ago</div>
      `;
    }}

    function progressCell(branch) {{
      const percent = branch.progress_percent ?? 0;
      const done = Number(branch.progress_done_ticks);
      const total = Number(branch.progress_total_ticks);
      const branchProgress = Number.isFinite(done) && Number.isFinite(total) && total > 0
        ? `<div class="small mono">${{done}} / ${{total}} branch ticks</div>`
        : "";
      return `
        ${{branchProgress}}
        <div class="progress" aria-hidden="true"><div class="bar" style="width:${{percent}}%"></div></div>
      `;
    }}

    function formatAge(seconds) {{
      const value = Number(seconds);
      if (!Number.isFinite(value)) return "-";
      const clamped = Math.max(0, value);
      if (clamped < 60) return `${{Math.floor(clamped)}}s`;
      if (clamped < 3600) return `${{Math.floor(clamped / 60)}}m`;
      return `${{Math.floor(clamped / 3600)}}h ${{Math.floor((clamped % 3600) / 60)}}m`;
    }}

    function coderCell(branch) {{
      const ev = branch.evaluations || {{}};
      const count = ev.active_coders ?? 0;
      const ids = ev.active_evaluation_ids || [];
      return `<div>${{esc(count)}}</div><div class="small mono">${{ids.length ? "eval " + ids.map(esc).join(", ") : ""}}</div>`;
    }}

    function render(status) {{
      setText("mode", status.mode || "multistart");
      setText("stage", status.stage || status.status || "-");
      setText("job", status.job_id || "pending");
      document.title = status.station_name || "Station";
      const branchStart = Number(status.branch_tick);
      const rollTicks = Number(status.roll_ticks);
      const branchSpan = Number.isFinite(branchStart) && Number.isFinite(rollTicks)
        ? `${{branchStart}}/${{branchStart + rollTicks}}`
        : status.branch_tick;
      setText("branchTick", branchSpan);
      setText("seeds", `${{status.counts?.completed ?? 0}}/${{status.seed_count ?? "?"}} done`);
      setText("parallel", status.max_parallel);
      setText("activeCoders", status.active_coders ?? 0);
      document.getElementById("updated").textContent = `Last refresh: ${{new Date().toLocaleString()}} (Refresh every 5s)`;
      renderControls(status);

      const tbody = document.getElementById("branches");
      const empty = document.getElementById("empty");
      const branches = status.branches || [];
      empty.hidden = branches.length > 0;
      tbody.innerHTML = branches.map((branch) => `
        <tr>
          <td class="mono">s${{branch.seed}}${{branch.selected ? " *" : ""}}</td>
          <td><span class="status ${{cls(branch.status)}}">${{esc(branch.status)}}</span></td>
          <td>
            <div>${{esc(branch.station_label || branch.station_name || ("Seed " + branch.seed))}}</div>
            <div class="small mono">${{esc(branch.data_dir)}}</div>
          </td>
          <td>${{tickCell(branch)}}</td>
          <td class="progress-cell">${{progressCell(branch)}}</td>
          <td class="mono">${{esc(branch.top_evaluation_id)}}<div class="small">tick ${{esc(branch.top_tick)}}</div></td>
          <td class="mono">${{esc(branch.top_score_display)}}</td>
          <td>${{coderCell(branch)}}</td>
          <td class="mono">${{esc(branch.pid)}}</td>
          <td class="note">${{esc(branch.note)}}</td>
        </tr>
      `).join("");
    }}

    function hasPauseTarget(status) {{
      return (status.branches || []).some((branch) => {{
        const statusText = text(branch.status).toLowerCase();
        const current = Number(branch.current_tick);
        const target = Number(branch.target_tick);
        if (["completed", "failed", "interviewing", "waiting_quiescent"].includes(statusText)) return false;
        if (Number.isFinite(current) && Number.isFinite(target) && current >= target) return false;
        return ["pending", "running", "paused"].includes(statusText);
      }});
    }}

    function hasResumeTarget(status) {{
      return (status.branches || []).some((branch) => {{
        const statusText = text(branch.status).toLowerCase();
        if (["completed", "interviewing", "waiting_quiescent"].includes(statusText)) return false;
        return ["pending", "running", "paused", "failed"].includes(statusText);
      }});
    }}

    function renderControls(status) {{
      const control = text(status.control || "running").toLowerCase();
      const pauseButton = document.getElementById("pauseButton");
      const resumeButton = document.getElementById("resumeButton");
      const active = Boolean(status.active);
      const canPause = hasPauseTarget(status);
      const jobFailed = text(status.status).toLowerCase() === "failed";
      const canResume = jobFailed || hasResumeTarget(status);
      pauseButton.disabled = !active || control === "paused" || !canPause;
      resumeButton.disabled = !active || !canResume;
      document.getElementById("controlState").textContent = `Control: ${{control}}`;
    }}

    async function sendControl(action) {{
      const pauseButton = document.getElementById("pauseButton");
      const resumeButton = document.getElementById("resumeButton");
      const message = document.getElementById("actionMessage");
      pauseButton.disabled = true;
      resumeButton.disabled = true;
      message.classList.remove("error");
      message.textContent = `${{action === "pause" ? "Pause" : "Resume"}} requested...`;
      try {{
        const response = await fetch(`/api/multistart/${{action}}`, {{method: "POST", cache: "no-store"}});
        const payload = await response.json();
        if (!payload.success) {{
          message.classList.add("error");
          message.textContent = payload.message || payload.error || "Control request failed.";
        }} else {{
          message.textContent = payload.message || "Control request accepted.";
          if (payload.status) render(payload.status);
        }}
      }} catch (_error) {{
        message.classList.add("error");
        message.textContent = "Control request failed; retry after the controller is reachable.";
      }} finally {{
        refresh();
      }}
    }}

    async function refresh() {{
      try {{
        const response = await fetch("/api/multistart/status", {{cache: "no-store"}});
        const payload = await response.json();
        if (payload.success && !payload.multistart?.active) {{
          window.location.replace("/dashboard");
          return;
        }}
        if (payload.success) render(payload.multistart);
      }} catch (_error) {{
        document.getElementById("updated").textContent = "Refresh failed; retrying...";
      }}
    }}

    document.getElementById("pauseButton").addEventListener("click", () => sendControl("pause"));
    document.getElementById("resumeButton").addEventListener("click", () => sendControl("resume"));
    render(initialStatus);
    window.setInterval(refresh, 5000);
  </script>
</body>
</html>"""
    return Response(response_html, mimetype="text/html")

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


@app.before_request
def lock_multistart_preview_mutations():
    """Keep the seed preview observational even if a client bypasses disabled controls."""
    if OPERATION_MODE != "multistart_preview" or multistart_preview.request_allowed(request.method, request.path):
        return None
    return jsonify({
        "success": False,
        "error": "multistart_preview_read_only",
        "message": "Seed 1 is a read-only multistart preview. Use the multistart page for job controls.",
    }), 423

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
    try:
        from station.sync.parallel_status import load_parallel_tick_status

        return load_parallel_tick_status(orchestrator_instance)
    except Exception as exc:
        return {"active": False, "error": str(exc)}


def _web_archive_survey_owner() -> str:
    if session.get("user"):
        return str(session["user"])
    if request.authorization and request.authorization.username:
        return str(request.authorization.username)
    return "dashboard"


def _get_web_archive_survey_service() -> WebArchiveSurveyService:
    global web_archive_survey_service
    if web_archive_survey_service is None:
        with web_archive_survey_service_lock:
            if web_archive_survey_service is None:
                web_archive_survey_service = WebArchiveSurveyService(constants.BASE_STATION_DATA_PATH)
    web_archive_survey_service.ensure_worker_started()
    return web_archive_survey_service


# --- Initialization ---
def initialize_station_and_orchestrator():
    global station_instance, orchestrator_instance, OPERATION_MODE
    if multistart_waiting.waiting_mode_active():
        OPERATION_MODE = "multistart_preview"
        station_instance = None
        orchestrator_instance = None
        print("Initializing application in read-only multistart seed preview mode.")
        return
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
        orchestrator_event_broker.put({"event": "status_update", "data": {"message": "Station instance initialized."}, "timestamp": time.time()})

        orchestrator_instance = Orchestrator(
            station_instance, 
            auto_prepare_on_init=True, # Will be prepared by UI action
            log_event_queue=orchestrator_event_broker
        )
        orchestrator_event_broker.put({"event": "status_update", "data": {"message": "Orchestrator instance created for API mode (idle)."}, "timestamp": time.time()})

    except Exception as e:
        print(f"CRITICAL ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        station_instance = None
        orchestrator_instance = None
        orchestrator_event_broker.put({"event": "error", "data": {"message": f"Station/Orchestrator initialization failed: {str(e)}"}, "timestamp": time.time()})

# --- HTML Serving Routes ---
@app.route('/')
@auth_required
def root_redirect_route():
    return redirect(url_for('dashboard_page'))

@app.route('/dashboard')
@auth_required
def dashboard_page():
    preview = None
    if OPERATION_MODE == "multistart_preview":
        preview = multistart_preview.dashboard_context()
        if preview is None:
            return _multistart_waiting_response()
    model_presets = load_model_presets()
    return render_template(
        'dashboard.html',
        operation_mode=OPERATION_MODE,
        model_presets=model_presets,
        multistart_preview=preview,
    )


@app.route('/multistart')
@auth_required
def multistart_page():
    if not multistart_waiting.waiting_mode_active():
        return redirect(url_for('dashboard_page'))
    return _multistart_waiting_response()


@app.route('/api/multistart/status', methods=['GET'])
@auth_required
def multistart_status_route():
    return jsonify({
        "success": True,
        "multistart": multistart_waiting.public_status(),
    })


@app.route('/api/multistart/pause', methods=['POST'])
@auth_required
def multistart_pause_route():
    response = multistart_ipc.request_pause_branches()
    status = response.get("status") if isinstance(response, dict) else None
    return jsonify({
        "success": bool(response.get("success")) if isinstance(response, dict) else False,
        "message": response.get("message") if isinstance(response, dict) else None,
        "error": response.get("error") if isinstance(response, dict) else "invalid controller response",
        "status": status,
    })


@app.route('/api/multistart/resume', methods=['POST'])
@auth_required
def multistart_resume_route():
    response = multistart_ipc.request_resume_branches()
    status = response.get("status") if isinstance(response, dict) else None
    return jsonify({
        "success": bool(response.get("success")) if isinstance(response, dict) else False,
        "message": response.get("message") if isinstance(response, dict) else None,
        "error": response.get("error") if isinstance(response, dict) else "invalid controller response",
        "status": status,
    })


@app.route('/api/archive/papers', methods=['GET'])
def archive_papers_list_route():
    if OPERATION_MODE == "multistart_preview":
        view = multistart_preview.capsule_view()
        if view is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        return jsonify({"success": True, **build_archive_list_payload(view)})
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
    if OPERATION_MODE == "multistart_preview":
        view = multistart_preview.capsule_view()
        payload = build_archive_detail_payload(view, numeric_id) if view else None
        if not payload:
            return jsonify({"success": False, "error": f"Archive paper #{numeric_id} not found."}), 404
        return jsonify({"success": True, **payload})
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


@app.route('/api/web/archive-surveys/templates', methods=['GET'])
@auth_required
def web_archive_survey_templates_route():
    return jsonify({"success": True, "templates": build_web_archive_survey_templates()})


@app.route('/api/web/archive-surveys', methods=['GET'])
@auth_required
def web_archive_surveys_list_route():
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503
    try:
        service = _get_web_archive_survey_service()
        return jsonify({
            "success": True,
            "surveys": service.store.list(_web_archive_survey_owner()),
        })
    except Exception as exc:
        app.logger.error(f"Error loading web Archive surveys: {exc}")
        return jsonify({"success": False, "error": f"Failed to load web surveys: {exc}"}), 500


@app.route('/api/web/archive-surveys', methods=['POST'])
@auth_required
def web_archive_survey_submit_route():
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503
    if not constants.ARCHIVE_SURVEY_ENABLED:
        return jsonify({"success": False, "error": "Archive Surveyor is disabled."}), 503
    data = request.get_json(silent=True) or {}
    try:
        archive_payload = build_archive_list_payload(capsule_module)
        question_preview = build_question_survey_preview(capsule_module)
        task_snapshot = get_task_spec_snapshot(constants)
        current_tick = station_instance._get_current_tick() if hasattr(station_instance, "_get_current_tick") else None
        service = _get_web_archive_survey_service()
        record = service.submit(
            owner=_web_archive_survey_owner(),
            prompt=data.get("prompt"),
            selected_archive_ids=data.get("selected_archive_ids"),
            source_tick=current_tick,
            task_spec_snapshot=task_snapshot.get("raw_markdown") or "",
            archive_preview_snapshot=archive_payload.get("all_abstracts_markdown") or "",
            question_preview_snapshot=question_preview,
        )
        summaries = service.store.list(_web_archive_survey_owner())
        summary = next((item for item in summaries if int(item.get("id") or 0) == int(record["id"])), None)
        return jsonify({
            "success": True,
            "survey": summary,
            "message": f"Archive Survey #{record['id']} queued.",
        })
    except (ValueError, WebArchiveSurveyBusyError) as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.error(f"Error queueing web Archive survey: {exc}")
        return jsonify({"success": False, "error": f"Failed to queue web survey: {exc}"}), 500


@app.route('/api/web/archive-surveys/<int:survey_id>', methods=['GET'])
@auth_required
def web_archive_survey_detail_route(survey_id: int):
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503
    try:
        record = _get_web_archive_survey_service().store.get(
            survey_id,
            owner=_web_archive_survey_owner(),
            include_report=True,
        )
        session_data = record.get("session") if isinstance(record.get("session"), dict) else {}
        return jsonify({
            "success": True,
            "survey": {
                "id": record.get("id"),
                "status": record.get("status"),
                "prompt": record.get("prompt") or "",
                "selected_archive_ids": record.get("selected_archive_ids") or [],
                "source_tick": record.get("source_tick"),
                "submitted_timestamp": record.get("submitted_timestamp"),
                "started_timestamp": session_data.get("started_timestamp"),
                "completed_timestamp": record.get("completed_timestamp"),
                "error": record.get("error") or session_data.get("last_error"),
                "report_markdown": record.get("report_markdown") or "",
            },
        })
    except WebArchiveSurveyNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        app.logger.error(f"Error loading web Archive survey #{survey_id}: {exc}")
        return jsonify({"success": False, "error": f"Failed to load web survey: {exc}"}), 500


@app.route('/api/web/archive-surveys/<int:survey_id>', methods=['DELETE'])
@auth_required
def web_archive_survey_delete_route(survey_id: int):
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503
    try:
        _get_web_archive_survey_service().store.delete(survey_id, _web_archive_survey_owner())
        return jsonify({"success": True, "message": f"Archive Survey #{survey_id} removed."})
    except WebArchiveSurveyNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except WebArchiveSurveyBusyError as exc:
        return jsonify({"success": False, "error": str(exc)}), 409
    except Exception as exc:
        app.logger.error(f"Error deleting web Archive survey #{survey_id}: {exc}")
        return jsonify({"success": False, "error": f"Failed to remove web survey: {exc}"}), 500


@app.route('/api/questions', methods=['GET'])
def question_room_list_route():
    if OPERATION_MODE == "multistart_preview":
        view = multistart_preview.capsule_view()
        if view is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        payload = build_question_list_payload(
            view,
            page=request.args.get("page", default=1, type=int) or 1,
            page_size=request.args.get("page_size", default=100, type=int) or 100,
            sort_by=request.args.get("sort_by", default="authored_tick", type=str),
            sort_direction=request.args.get("sort_direction", default="desc", type=str),
        )
        return jsonify({"success": True, **payload})
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503

    try:
        payload = build_question_list_payload(
            capsule_module,
            page=request.args.get("page", default=1, type=int) or 1,
            page_size=request.args.get("page_size", default=100, type=int) or 100,
            sort_by=request.args.get("sort_by", default="authored_tick", type=str),
            sort_direction=request.args.get("sort_direction", default="desc", type=str),
        )
        return jsonify({"success": True, **payload})
    except Exception as e:
        app.logger.error(f"Error loading Question Room activity: {e}")
        return jsonify({"success": False, "error": f"Could not load questions: {str(e)}"}), 500


@app.route('/api/questions/<int:numeric_id>', methods=['GET'])
def question_room_detail_route(numeric_id: int):
    if OPERATION_MODE == "multistart_preview":
        view = multistart_preview.capsule_view()
        payload = build_question_detail_payload(view, numeric_id) if view else None
        if not payload:
            return jsonify({"success": False, "error": f"Question #{numeric_id} not found."}), 404
        return jsonify({"success": True, **payload})
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503

    try:
        payload = build_question_detail_payload(capsule_module, numeric_id)
        if not payload:
            return jsonify({"success": False, "error": f"Question #{numeric_id} not found."}), 404
        return jsonify({"success": True, **payload})
    except Exception as e:
        app.logger.error(f"Error loading Question Room thread #{numeric_id}: {e}")
        return jsonify({"success": False, "error": f"Could not load Question #{numeric_id}: {str(e)}"}), 500

# --- API Endpoints - Orchestrator Control (API Mode) ---
@app.route('/api/orchestrator/status', methods=['GET'])
def get_orchestrator_status_route():
    if OPERATION_MODE == "multistart_preview":
        status = multistart_preview.orchestrator_status()
        if status is None:
            return jsonify({
                "success": False,
                "error": "Seed 1 preview is unavailable during multistart finalization.",
                "status": {
                    "is_running": False,
                    "is_prepared": False,
                    "is_paused": True,
                    "current_tick": -1,
                    "turn_order": [],
                    "agents_awaiting_human": [],
                    "read_only": True,
                },
            }), 200
        return jsonify({
            "success": True,
            "status": status,
        })
    if OPERATION_MODE != "api" or not orchestrator_instance or not station_instance:
        return jsonify({"success": False, "error": "Orchestrator not active or not in API mode.", 
                        "status": {"is_running": False, "is_prepared": False, "is_paused": False, "current_tick": -1, "turn_order":[], "agents_awaiting_human": []}}), 200
    
    agents_awaiting_human_list = station_instance.get_agents_awaiting_human_intervention() if hasattr(station_instance, 'get_agents_awaiting_human_intervention') else []
    status_data = {
        "is_prepared": orchestrator_instance.is_prepared, # ADDED
        "is_running": orchestrator_instance.is_running,
        "is_paused": orchestrator_instance.is_paused,
        "pause_requested": orchestrator_instance.pause_requested,
        "pause_condition_met": orchestrator_instance.pause_condition_met,
        "pause_reason": orchestrator_instance.get_pause_reason(),
        "is_waiting": orchestrator_instance.is_waiting,
        "waiting_reasons": orchestrator_instance.waiting_reasons,
        "current_tick": station_instance._get_current_tick(),
        "station_status": station_instance.config.get(constants.STATION_CONFIG_STATION_STATUS, "Unknown"),
        "turn_order": list(orchestrator_instance.agent_turn_order),
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
    if OPERATION_MODE != "multistart_preview" and (not station_instance or not station_instance.agent_module):
        return jsonify({"success": False, "error": "Station not properly initialized."}), 500

    load_full = request.args.get('full', 'false').lower() == 'true'
    window = request.args.get('window', 'recent').lower()
    try:
        tick_limit = int(request.args.get('ticks', '50'))
    except (TypeError, ValueError):
        tick_limit = 50
    tick_limit = max(1, min(tick_limit, 1000))

    preview_source_modified_at_ns = None
    if OPERATION_MODE == "multistart_preview":
        preview_path = multistart_preview.dialogue_log_path(agent_name)
        if preview_path is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        log_file_path = str(preview_path)
        try:
            preview_source_modified_at_ns = str(preview_path.stat().st_mtime_ns)
        except OSError:
            preview_source_modified_at_ns = None
        requested_modified_at_ns = request.args.get("modified_at_ns")
        if requested_modified_at_ns and requested_modified_at_ns == preview_source_modified_at_ns:
            return jsonify({
                "success": True,
                "unchanged": True,
                "source_modified_at_ns": preview_source_modified_at_ns,
            })
    # Handle special case for Reviewer
    elif agent_name == "Reviewer":
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
        if OPERATION_MODE == "multistart_preview":
            response_data["source_modified_at_ns"] = preview_source_modified_at_ns
        json_str = json.dumps(response_data)
        return Response(json_str, mimetype='application/json')
    except Exception as e:
        app.logger.error(f"Error reading dialogue history for {agent_name} from {log_file_path}: {e}")
        return jsonify({"success": False, "error": f"Could not read dialogue history: {str(e)}"}), 500

# --- General Station Info (Shared) ---
@app.route('/api/station_tick', methods=['GET'])
def get_station_tick_ep_v2(): # Renamed
    if OPERATION_MODE == "multistart_preview":
        status = multistart_preview.orchestrator_status()
        if status is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        return jsonify({"success": True, "current_tick": status.get("current_tick", -1), "read_only": True})
    if not station_instance: return jsonify({"success": False, "error": "Station not initialized"}), 500
    return jsonify({"success": True, "current_tick": station_instance._get_current_tick()})

@app.route('/api/agents', methods=['GET'])
def get_agents_ep_v2(): # Renamed
    if OPERATION_MODE == "multistart_preview":
        preview_agents = multistart_preview.agents()
        if preview_agents is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        return jsonify({"success": True, "agents": preview_agents, "read_only": True})
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
    global station_statistics_cache
    if OPERATION_MODE == "multistart_preview":
        preview_stats = multistart_preview.statistics()
        if preview_stats is None:
            # Finalization clears current_job.yaml before start.sh stops the old
            # preview process. Keep the safe-stop drain endpoint available for
            # that short handoff window.
            preview_stats = multistart_preview.handoff_statistics()
        return jsonify({"success": True, "statistics": preview_stats})
    if not station_instance: 
        return jsonify({"success": False, "error": "Station not initialized"}), 500
    
    # At most one request may refresh statistics. Polling clients receive the
    # last complete snapshot instead of occupying every Gunicorn request thread
    # behind a slow control-plane refresh.
    if not station_statistics_lock.acquire(timeout=0.05):
        if station_statistics_cache is not None:
            return jsonify({
                "success": True,
                "statistics": station_statistics_cache,
                "stale": True,
            })
        return jsonify({
            "success": False,
            "error": "Station statistics refresh is already in progress.",
        }), 503

    started_at = time.perf_counter()
    try:
        stats = station_instance.get_station_statistics()
        station_statistics_cache = stats
        elapsed = time.perf_counter() - started_at
        if elapsed >= 1.0:
            app.logger.warning("Station statistics refresh took %.3fs", elapsed)
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
    finally:
        station_statistics_lock.release()

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
    if OPERATION_MODE == "multistart_preview":
        config = multistart_preview.station_config()
        if config is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        return jsonify({"success": True, "config": config})
    if not station_instance:
        return jsonify({"success": False, "error": "Station not initialized"}), 500
        
    try:
        config = {
            "station_status": station_instance.config.get(constants.STATION_CONFIG_STATION_STATUS, "Unknown"),
            "station_name": station_instance.config.get(constants.STATION_CONFIG_NAME, ""),
            "station_description": station_instance.config.get(constants.STATION_CONFIG_DESCRIPTION, ""),
            "station_id": station_instance.config.get(constants.STATION_ID_KEY, "Unknown"),
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


@app.route('/api/station/research_task_spec', methods=['GET'])
@auth_required
def get_research_task_spec():
    """Return the complete raw Research Task Markdown for the admin editor."""
    if OPERATION_MODE == "multistart_preview":
        snapshot = multistart_preview.task_spec_snapshot()
        if snapshot is None:
            return jsonify({"success": False, "error": "Seed 1 preview is unavailable."}), 503
        return jsonify({"success": True, **snapshot})
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503
    try:
        return jsonify({"success": True, **get_task_spec_snapshot(constants)})
    except Exception as e:
        app.logger.error(f"Error loading Research Task specification: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/station/research_task_spec', methods=['PUT'])
@auth_required
def update_research_task_spec():
    """Atomically replace the active Research Task Markdown."""
    if OPERATION_MODE != "api" or not station_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON payload."}), 400
    try:
        snapshot = save_task_spec_snapshot(
            data.get("raw_markdown"),
            expected_revision=str(data.get("expected_revision") or ""),
            consts_module=constants,
        )
        return jsonify({
            "success": True,
            "message": "Research Task specification saved atomically.",
            **snapshot,
        })
    except TaskSpecConflictError as e:
        return jsonify({"success": False, "error": str(e), "conflict": True}), 409
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error updating Research Task specification: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def _set_research_evaluator_refresh_state(**values):
    with research_evaluator_refresh_lock:
        research_evaluator_refresh_state.clear()
        research_evaluator_refresh_state.update(values)


def _run_research_evaluator_refresh(evaluator_service, orchestrator, current_job_path):
    try:
        while orchestrator.is_running and not orchestrator.is_paused:
            time.sleep(0.5)
        if not orchestrator.is_running:
            raise RuntimeError("Orchestrator stopped before evaluator refresh.")

        _set_research_evaluator_refresh_state(status="draining")
        while evaluator_service.has_pending_or_running():
            time.sleep(1.0)
        if file_io_utils.load_yaml(current_job_path):
            raise RuntimeError("Evaluator refresh became invalid because multistart started.")

        evaluator = evaluator_service.refresh_task_registry()
        _set_research_evaluator_refresh_state(
            status="completed",
            evaluator_class=type(evaluator).__name__,
            task_description=evaluator.get_task_description(),
        )
    except Exception as e:
        app.logger.error(f"Error refreshing Research evaluator: {e}")
        _set_research_evaluator_refresh_state(status="failed", error=str(e))


@app.route('/api/station/research_evaluator/refresh', methods=['GET', 'POST'])
@auth_required
def refresh_research_evaluator():
    """Request or inspect an asynchronous Research evaluator refresh."""
    if OPERATION_MODE != "api" or not station_instance or not orchestrator_instance:
        return jsonify({"success": False, "error": "Station not active or not in API mode."}), 503

    if request.method == "GET":
        with research_evaluator_refresh_lock:
            refresh = dict(research_evaluator_refresh_state)
        return jsonify({
            "success": True,
            "refresh": refresh,
            "is_paused": bool(orchestrator_instance.is_paused),
        })

    current_job_path = os.path.join(project_root, "station_multistart", "current_job.yaml")
    if file_io_utils.load_yaml(current_job_path):
        return jsonify({"success": False, "error": "Evaluator refresh is unavailable during multistart."}), 409

    evaluator_service = getattr(station_instance, "auto_research_evaluator", None)
    if evaluator_service is None:
        return jsonify({"success": False, "error": "Research evaluator is not running."}), 503
    if not orchestrator_instance.is_running:
        return jsonify({"success": False, "error": "Orchestrator is not running."}), 409

    with research_evaluator_refresh_lock:
        current_status = str(research_evaluator_refresh_state.get("status") or "idle")
        if current_status not in {"requested", "draining"}:
            research_evaluator_refresh_state.clear()
            research_evaluator_refresh_state.update(status="requested")
    if current_status in {"requested", "draining"}:
        return jsonify({"success": True, "status": current_status}), 202

    orchestrator_instance.request_manual_pause()
    threading.Thread(
        target=_run_research_evaluator_refresh,
        args=(evaluator_service, orchestrator_instance, current_job_path),
        daemon=True,
    ).start()
    return jsonify({
        "success": True,
        "status": "requested",
        "message": "Evaluator refresh requested. Poll this endpoint for completion.",
    }), 202


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
        try:
            multistart_ipc.notify_runtime_api_update(data)
        except Exception as e:
            app.logger.warning(f"Failed to notify multistart controller of runtime API config update: {e}")

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


# --- Dashboard live-event transport ---
@app.route('/api/orchestrator/live_log_stream')
def live_log_stream_route():
    if OPERATION_MODE != "api":
        return Response("SSE only available in API mode.", status=403, mimetype='text/event-stream')
    selected_agent_name = request.args.get("agent_name") or None
    requested_cursor = request.args.get("cursor")
    
    def event_stream():
        app.logger.info("SSE client connected for live log stream.")
        cursor_state = orchestrator_event_broker.open_cursor(requested_cursor)
        cursor = cursor_state.cursor
        connect_event = {
            "event": "stream_cursor",
            "data": {
                "cursor_reset": cursor_state.reset,
                "dropped_count": cursor_state.dropped_count,
            },
            "timestamp": time.time(),
            "stream_sequence": cursor,
            "stream_epoch": orchestrator_event_broker.epoch,
            "stream_control": True,
        }
        yield f"id: {cursor}\ndata: {json.dumps(connect_event)}\n\n"

        try:
            while True:
                try:
                    batch = orchestrator_event_broker.read_after(cursor, limit=50, wait_timeout=5)
                    cursor = batch.cursor
                    if not batch.events:
                        yield ": keepalive\n\n"
                        continue
                    for sequence, raw_event in batch.events:
                        log_entry = _sanitize_stream_event_payload(raw_event, selected_agent_name)
                        log_entry = dict(log_entry)
                        log_entry["stream_sequence"] = sequence
                        log_entry["stream_epoch"] = orchestrator_event_broker.epoch
                        yield f"id: {sequence}\ndata: {json.dumps(log_entry)}\n\n"
                        cursor = sequence
                except Exception as e_inner:
                    app.logger.error(f"Error during SSE event generation: {e_inner}")
                    return
        except GeneratorExit:
            app.logger.info("SSE client disconnected (GeneratorExit).")
        except Exception as e_outer:
            app.logger.error(f"Critical error in SSE event_stream: {e_outer}")
        finally:
            app.logger.info("SSE event_stream closing for this client.")
            
    return Response(
        event_stream(),
        content_type='text/event-stream',
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )

@app.route('/api/orchestrator/recent_events')
def recent_events_route():
    """Cursor-based fallback for clients that cannot keep an SSE connection."""
    if OPERATION_MODE != "api":
        return jsonify({"success": False, "message": "Recent events only available in API mode"}), 403
    selected_agent_name = request.args.get("agent_name") or None
    requested_cursor = request.args.get("cursor")
    requested_limit = request.args.get("limit", default=50, type=int) or 50

    try:
        batch = orchestrator_event_broker.read_after(requested_cursor, limit=requested_limit)
        events = []
        for sequence, raw_event in batch.events:
            event = _sanitize_stream_event_payload(raw_event, selected_agent_name)
            event = dict(event)
            event["stream_sequence"] = sequence
            event["stream_epoch"] = orchestrator_event_broker.epoch
            events.append(event)
        return jsonify({
            "success": True,
            "events": events,
            "count": len(events),
            "cursor": batch.cursor,
            "cursor_reset": batch.reset,
            "dropped_count": batch.dropped_count,
            "stream_epoch": orchestrator_event_broker.epoch,
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
if station_instance is not None and constants.ARCHIVE_SURVEY_ENABLED:
    try:
        _get_web_archive_survey_service()
    except Exception as exc:
        app.logger.error(f"Failed to start or recover web Archive Surveyor: {exc}")

# --- Shutdown API ---
@app.route('/api/shutdown', methods=['POST'])
@auth.login_required
def api_shutdown():
    """Gracefully shutdown research evaluations"""
    try:
        if web_archive_survey_service:
            web_archive_survey_service.stop()
        if station_instance:
            print("API: Received shutdown request, cleaning up station...")
            # Stop all evaluation loops
            if hasattr(station_instance, 'auto_research_evaluator') and station_instance.auto_research_evaluator:
                station_instance.stop_auto_research_evaluator()
            if hasattr(station_instance, 'auto_archive_evaluator') and station_instance.auto_archive_evaluator:
                station_instance.stop_auto_archive_evaluator()
            if hasattr(station_instance, 'auto_archive_surveyor') and station_instance.auto_archive_surveyor:
                station_instance.stop_auto_archive_surveyor()
            if hasattr(station_instance, 'auto_external_reporter') and station_instance.auto_external_reporter:
                station_instance.stop_auto_external_reporter()
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

    if OPERATION_MODE == "multistart_preview":
        pass
    elif not station_instance or not orchestrator_instance:
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
