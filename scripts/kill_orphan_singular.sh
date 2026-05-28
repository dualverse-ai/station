#!/usr/bin/env bash
set -euo pipefail

PROCESS_NAME="${SINGULAR_PROCESS_NAME:-Singular}"
MIN_AGE_SECONDS="${SINGULAR_ORPHAN_MIN_AGE_SECONDS:-3600}"
INTERVAL_SECONDS="${SINGULAR_ORPHAN_SCAN_INTERVAL_SECONDS:-300}"
GRACE_SECONDS="${SINGULAR_ORPHAN_GRACE_SECONDS:-10}"
DRY_RUN=0
LOOP=1

usage() {
    cat <<'EOF'
Usage: scripts/kill_orphan_singular.sh [options]

Scan for orphaned Singular processes and terminate only those with:
  - process name: Singular
  - parent PID: 1
  - elapsed runtime >= 1 hour by default

Options:
  --dry-run                 Print matching processes without killing them.
  --once                    Scan once, then exit.
  --loop                    Keep scanning until interrupted. This is the default.
  --interval-seconds N      Sleep N seconds between scans.
  --min-age-seconds N       Kill only processes older than N seconds.
  --grace-seconds N         Seconds to wait after SIGTERM before SIGKILL.
  -h, --help                Show this help text.

Environment overrides:
  SINGULAR_ORPHAN_MIN_AGE_SECONDS
  SINGULAR_ORPHAN_SCAN_INTERVAL_SECONDS
  SINGULAR_ORPHAN_GRACE_SECONDS
  SINGULAR_PROCESS_NAME
EOF
}

is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

require_option_value() {
    local option="$1"
    local value="${2:-}"

    if [[ -z "$value" || "$value" == --* ]]; then
        echo "$option requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --loop)
            LOOP=1
            shift
            ;;
        --once)
            LOOP=0
            shift
            ;;
        --interval-seconds)
            require_option_value "$1" "${2:-}"
            INTERVAL_SECONDS="${2:-}"
            shift 2
            ;;
        --min-age-seconds)
            require_option_value "$1" "${2:-}"
            MIN_AGE_SECONDS="${2:-}"
            shift 2
            ;;
        --grace-seconds)
            require_option_value "$1" "${2:-}"
            GRACE_SECONDS="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for value_name in MIN_AGE_SECONDS INTERVAL_SECONDS GRACE_SECONDS; do
    value="${!value_name}"
    if ! is_nonnegative_int "$value"; then
        echo "$value_name must be a nonnegative integer, got: $value" >&2
        exit 2
    fi
done

if [[ "$LOOP" -eq 1 && "$INTERVAL_SECONDS" -eq 0 ]]; then
    echo "INTERVAL_SECONDS must be greater than zero when looping" >&2
    exit 2
fi

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

find_candidates() {
    ps -eo pid=,ppid=,etimes=,comm=,args= -ww |
        awk -v min_age="$MIN_AGE_SECONDS" -v proc_name="$PROCESS_NAME" '
            $2 == 1 && $3 >= min_age && $4 == proc_name {
                pid = $1
                age = $3
                $1 = $2 = $3 = $4 = ""
                sub(/^[[:space:]]+/, "")
                print pid "\t" age "\t" $0
            }
        '
}

still_same_orphan() {
    local pid="$1"
    local info

    info="$(ps -o ppid=,comm= -p "$pid" 2>/dev/null || true)"
    [[ -n "$info" ]] || return 1

    # shellcheck disable=SC2086
    set -- $info
    [[ "${1:-}" == "1" && "${2:-}" == "$PROCESS_NAME" ]]
}

scan_once() {
    local candidates=()
    local killed_pids=()
    local line pid age args

    mapfile -t candidates < <(find_candidates)
    if [[ "${#candidates[@]}" -eq 0 ]]; then
        echo "[$(timestamp)] no orphaned $PROCESS_NAME processes older than ${MIN_AGE_SECONDS}s"
        return 0
    fi

    for line in "${candidates[@]}"; do
        IFS=$'\t' read -r pid age args <<<"$line"
        echo "[$(timestamp)] matched orphaned $PROCESS_NAME pid=$pid age=${age}s args=$args"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            continue
        fi

        if kill -TERM "$pid" 2>/dev/null; then
            killed_pids+=("$pid")
            echo "[$(timestamp)] sent SIGTERM to pid=$pid"
        else
            echo "[$(timestamp)] failed to send SIGTERM to pid=$pid" >&2
        fi
    done

    if [[ "$DRY_RUN" -eq 1 || "${#killed_pids[@]}" -eq 0 ]]; then
        return 0
    fi

    sleep "$GRACE_SECONDS"

    for pid in "${killed_pids[@]}"; do
        if still_same_orphan "$pid"; then
            if kill -KILL "$pid" 2>/dev/null; then
                echo "[$(timestamp)] sent SIGKILL to still-running pid=$pid"
            else
                echo "[$(timestamp)] failed to send SIGKILL to pid=$pid" >&2
            fi
        else
            echo "[$(timestamp)] pid=$pid exited after SIGTERM"
        fi
    done
}

if [[ "$LOOP" -eq 1 ]]; then
    while true; do
        scan_once
        sleep "$INTERVAL_SECONDS"
    done
else
    scan_once
fi
