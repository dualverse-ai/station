#!/bin/bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [--output <dir>] [--multistart-job <job> | --multistart-snapshot <job>] <station_id_or_zip> [tick_number]"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message and exit"
    echo "  -o, --output <dir>         Restore into <dir> instead of the default target"
    echo "  --multistart-job <job>     Restore only station_data/multistart/<job>"
    echo "  --multistart-snapshot <job> Restore the full Station from that job's named snapshot"
    echo
    echo "Examples:"
    echo "  $0 6d8bc 1200"
    echo "  $0 6d8bc"
    echo "  $0 --output station_data_tmp 6d8bc 1200"
    echo "  $0 6d8bc --output station_data_tmp"
    echo "  $0 --multistart-job 501_abcd1234 6d8bc 542"
    echo "  $0 --multistart-snapshot 501_abcd1234 6d8bc 542"
    echo "  $0 /path/to/any_station_archive.zip"
    echo
    echo "With no tick, an archived active multistart job is restored only when its"
    echo "recorded station tick is newer than the latest ordinary snapshot tick."
    echo "A zip passed directly is removed only after the full restore succeeds."
}

BACKUP_DIR="./backup"
TARGET_DIR="./station_data"
OUTPUT_SET=0
MULTISTART_JOB=""
MULTISTART_SNAPSHOT=""
POSITIONAL_ARGS=()
ACTIVE_MULTISTART_RESTORE=0
ACTIVE_MULTISTART_MANIFEST=""

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -o|--output)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}Error: --output requires a directory argument${NC}"
                echo
                usage
                exit 1
            fi
            TARGET_DIR="$2"
            OUTPUT_SET=1
            shift 2
            ;;
        --output=*)
            TARGET_DIR="${1#*=}"
            if [ -z "$TARGET_DIR" ]; then
                echo -e "${RED}Error: --output requires a directory argument${NC}"
                echo
                usage
                exit 1
            fi
            OUTPUT_SET=1
            shift
            ;;
        --multistart-job)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}Error: --multistart-job requires a job folder name${NC}"
                echo
                usage
                exit 1
            fi
            MULTISTART_JOB="$2"
            shift 2
            ;;
        --multistart-job=*)
            MULTISTART_JOB="${1#*=}"
            if [ -z "$MULTISTART_JOB" ]; then
                echo -e "${RED}Error: --multistart-job requires a job folder name${NC}"
                echo
                usage
                exit 1
            fi
            shift
            ;;
        --multistart-snapshot)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}Error: --multistart-snapshot requires a job folder name${NC}"
                echo
                usage
                exit 1
            fi
            MULTISTART_SNAPSHOT="$2"
            shift 2
            ;;
        --multistart-snapshot=*)
            MULTISTART_SNAPSHOT="${1#*=}"
            if [ -z "$MULTISTART_SNAPSHOT" ]; then
                echo -e "${RED}Error: --multistart-snapshot requires a job folder name${NC}"
                echo
                usage
                exit 1
            fi
            shift
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do
                POSITIONAL_ARGS+=("$1")
                shift
            done
            ;;
        -*)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo
            usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#POSITIONAL_ARGS[@]} -lt 1 ] || [ ${#POSITIONAL_ARGS[@]} -gt 2 ]; then
    echo -e "${RED}Error: Expected <partial_station_id> and optional [tick_number]${NC}"
    echo
    usage
    exit 1
fi

PARTIAL_ID="${POSITIONAL_ARGS[0]}"
TICK="${POSITIONAL_ARGS[1]}"
SOURCE_ZIP=""

# A zip filename is accepted directly, from either the current directory or
# this checkout's backup directory. Extraction is performed by the safe Python
# helper, which refuses path traversal and never overwrites an existing backup.
if [[ "$PARTIAL_ID" == *.zip ]]; then
    if [ -f "$PARTIAL_ID" ]; then
        SOURCE_ZIP="$(cd "$(dirname "$PARTIAL_ID")" && pwd)/$(basename "$PARTIAL_ID")"
    elif [ -f "$BACKUP_DIR/$PARTIAL_ID" ]; then
        SOURCE_ZIP="$(cd "$BACKUP_DIR" && pwd)/$PARTIAL_ID"
    else
        echo -e "${RED}Error: archive zip not found: $PARTIAL_ID${NC}"
        exit 1
    fi
    echo "Extracting station archive $SOURCE_ZIP..."
    EXTRACTED_DIR=$("$PYTHON_BIN" -c "import sys; from pathlib import Path; from station_tools.multistart_backup import extract_station_archive_zip; print(extract_station_archive_zip(Path(sys.argv[1]), Path(sys.argv[2])))" "$SOURCE_ZIP" "$BACKUP_DIR")
    PARTIAL_ID="$(basename "$EXTRACTED_DIR")"
fi

if [ -n "$MULTISTART_JOB" ]; then
    case "$MULTISTART_JOB" in
        */*|*..*)
            echo -e "${RED}Error: --multistart-job must be a job folder name, not a path${NC}"
            exit 1
            ;;
    esac
    if [ "$OUTPUT_SET" -eq 0 ]; then
        TARGET_DIR="multistart_${MULTISTART_JOB}"
    fi
fi

if [ -n "$MULTISTART_SNAPSHOT" ]; then
    case "$MULTISTART_SNAPSHOT" in
        */*|*..*)
            echo -e "${RED}Error: --multistart-snapshot must be a job folder name, not a path${NC}"
            exit 1
            ;;
    esac
fi

if [ -n "$MULTISTART_JOB" ] && [ -n "$MULTISTART_SNAPSHOT" ]; then
    echo -e "${RED}Error: --multistart-job and --multistart-snapshot are mutually exclusive${NC}"
    exit 1
fi

# Find matching station IDs in backup directory. Multi-init stores rejected
# seeds under backup/<chosen_station_id>/unchosen_seeds/<seed_station_id>.
echo "Searching for station IDs starting with '$PARTIAL_ID'..."
MATCHING_DIRS=()
if [ -d "$BACKUP_DIR" ]; then
    while IFS= read -r dir; do
        if [ -d "$dir/snapshots" ] || [ -d "$dir/multistart_archives" ]; then
            MATCHING_DIRS+=("$dir")
        fi
    done < <(find "$BACKUP_DIR" -maxdepth 1 -type d -name "$PARTIAL_ID*")
    while IFS= read -r dir; do
        if [ -d "$dir/snapshots" ]; then
            MATCHING_DIRS+=("$dir")
        fi
    done < <(find "$BACKUP_DIR" -mindepth 3 -maxdepth 3 -type d -path "$BACKUP_DIR/*/unchosen_seeds/$PARTIAL_ID*")
fi

# Locate portable Station archive zips by inspecting their embedded station ID,
# so UUID lookup does not depend on the archive filename convention.
if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
    MATCHING_ZIPS=()
    while IFS= read -r archive_zip; do
        if [ -n "$archive_zip" ]; then
            MATCHING_ZIPS+=("$archive_zip")
        fi
    done < <("$PYTHON_BIN" -c "import sys; from pathlib import Path; from station_tools.multistart_backup import find_archive_zips; [print(path) for path in find_archive_zips([Path(sys.argv[1]), Path(sys.argv[2])], sys.argv[3])]" "$BACKUP_DIR" "." "$PARTIAL_ID")
    if [ ${#MATCHING_ZIPS[@]} -eq 1 ]; then
        echo "Extracting archived backup ${MATCHING_ZIPS[0]}..."
        EXTRACTED_DIR=$("$PYTHON_BIN" -c "import sys; from pathlib import Path; from station_tools.multistart_backup import extract_station_archive_zip; print(extract_station_archive_zip(Path(sys.argv[1]), Path(sys.argv[2])))" "${MATCHING_ZIPS[0]}" "$BACKUP_DIR")
        MATCHING_DIRS+=("$EXTRACTED_DIR")
    elif [ ${#MATCHING_ZIPS[@]} -gt 1 ]; then
        echo -e "${RED}Error: Multiple multistart archive zips match '$PARTIAL_ID':${NC}"
        for archive_zip in "${MATCHING_ZIPS[@]}"; do
            echo "  - $archive_zip"
        done
        exit 1
    fi
fi

# Check number of matches
if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No station ID found starting with '$PARTIAL_ID' in $BACKUP_DIR${NC}"
    exit 1
elif [ ${#MATCHING_DIRS[@]} -gt 1 ]; then
    echo -e "${RED}Error: Multiple station IDs found starting with '$PARTIAL_ID':${NC}"
    for dir in "${MATCHING_DIRS[@]}"; do
        echo "  - $(basename "$dir") ($dir)"
    done
    echo -e "${YELLOW}Please provide a more specific partial ID${NC}"
    exit 1
fi

STATION_DIR="${MATCHING_DIRS[0]}"
STATION_ID=$(basename "$STATION_DIR")
RESTORE_STATION_KEY="${STATION_DIR#"$BACKUP_DIR"/}"
echo -e "${GREEN}Found station ID: $STATION_ID ($STATION_DIR)${NC}"

# If tick is omitted, compare the newest ordinary snapshot with the newest
# archived active multistart job. Explicit ticks always select normal restore.
if [ -z "$TICK" ]; then
    echo "Finding latest available station state..."
    SNAPSHOTS_DIR="$STATION_DIR/snapshots"
    LATEST_TICK=0
    if [ -d "$SNAPSHOTS_DIR" ]; then
        for tick_file in "$SNAPSHOTS_DIR"/tick_*.json; do
            if [ -f "$tick_file" ]; then
                filename=$(basename "$tick_file")
                if [[ "$filename" =~ ^tick_([0-9]+)\.json$ ]]; then
                    tick_num="${BASH_REMATCH[1]}"
                else
                    continue
                fi
                if [ "$tick_num" -gt "$LATEST_TICK" ]; then
                    LATEST_TICK=$tick_num
                fi
            fi
        done
    fi

    ACTIVE_INFO=""
    if [ -z "$MULTISTART_JOB" ] && [ -z "$MULTISTART_SNAPSHOT" ]; then
        ACTIVE_INFO=$("$PYTHON_BIN" -c "import sys; from pathlib import Path; from station_tools.multistart_backup import cli_active_info; print(cli_active_info(Path(sys.argv[1])))" "$STATION_DIR")
    fi
    ACTIVE_TICK=""
    if [ -n "$ACTIVE_INFO" ]; then
        IFS=$'\t' read -r ACTIVE_TICK ACTIVE_MULTISTART_MANIFEST <<< "$ACTIVE_INFO"
    fi

    if [[ "$ACTIVE_TICK" =~ ^[0-9]+$ ]] && { [ "$LATEST_TICK" -eq 0 ] || [ "$ACTIVE_TICK" -gt "$LATEST_TICK" ]; }; then
        ACTIVE_MULTISTART_RESTORE=1
        if [ "$OUTPUT_SET" -eq 0 ]; then
            TARGET_DIR="./station_multistart"
        fi
        echo -e "${GREEN}Using active multistart archive at tick $ACTIVE_TICK (ordinary latest: $LATEST_TICK)${NC}"
    elif [ "$LATEST_TICK" -gt 0 ]; then
        TICK=$LATEST_TICK
        if [[ "$ACTIVE_TICK" =~ ^[0-9]+$ ]]; then
            echo -e "${GREEN}Using ordinary tick $TICK; archived multistart tick $ACTIVE_TICK is not newer${NC}"
        else
            echo -e "${GREEN}Using latest ordinary tick: $TICK${NC}"
        fi
    else
        echo -e "${RED}Error: No restorable ordinary snapshot or active multistart archive found in $STATION_DIR${NC}"
        exit 1
    fi
fi

# Check if station_data exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Warning: $TARGET_DIR directory exists.${NC}"
    read -p "Do you want to remove it before restoring? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing $TARGET_DIR..."
        "$PYTHON_BIN" -c "import sys; from pathlib import Path; from station_tools.multistart_backup import remove_restore_target; remove_restore_target(Path(sys.argv[1]))" "$TARGET_DIR"
        echo -e "${GREEN}Removed $TARGET_DIR${NC}"
    else
        echo -e "${RED}Restore cancelled by user${NC}"
        exit 1
    fi
fi

# Perform the restore
if [ "$ACTIVE_MULTISTART_RESTORE" -eq 1 ]; then
    echo -e "${GREEN}Restoring active multistart station $STATION_ID at tick $ACTIVE_TICK to $TARGET_DIR...${NC}"
elif [ -n "$MULTISTART_JOB" ]; then
    echo -e "${GREEN}Restoring multistart job $MULTISTART_JOB from station $STATION_ID at tick $TICK to $TARGET_DIR...${NC}"
elif [ -n "$MULTISTART_SNAPSHOT" ]; then
    echo -e "${GREEN}Restoring station $STATION_ID from multistart snapshot $MULTISTART_SNAPSHOT at tick $TICK to $TARGET_DIR...${NC}"
else
    echo -e "${GREEN}Restoring station $STATION_ID at tick $TICK to $TARGET_DIR...${NC}"
fi
if [ "$ACTIVE_MULTISTART_RESTORE" -eq 1 ]; then
    "$PYTHON_BIN" -c "import sys; from pathlib import Path; from station_tools.multistart_backup import restore_active_multistart_archive; restore_active_multistart_archive(Path(sys.argv[1]), Path(sys.argv[2]))" "$ACTIVE_MULTISTART_MANIFEST" "$TARGET_DIR"
elif [ -n "$MULTISTART_JOB" ]; then
    "$PYTHON_BIN" -c "import sys; from station.backup_utils import restore_backup_subtree; raise SystemExit(0 if restore_backup_subtree(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]) else 1)" "$RESTORE_STATION_KEY" "$TICK" "multistart/$MULTISTART_JOB" "$TARGET_DIR"
else
    SNAPSHOT_SUFFIX=""
    if [ -n "$MULTISTART_SNAPSHOT" ]; then
        SNAPSHOT_SUFFIX="multistart_$MULTISTART_SNAPSHOT"
    fi
    "$PYTHON_BIN" -c "import sys; from station.backup_utils import restore_backup; suffix = sys.argv[4] or None; raise SystemExit(0 if restore_backup(sys.argv[1], int(sys.argv[2]), sys.argv[3], snapshot_suffix=suffix) else 1)" "$RESTORE_STATION_KEY" "$TICK" "$TARGET_DIR" "$SNAPSHOT_SUFFIX"
fi

if [ $? -eq 0 ]; then
    if [ -n "$SOURCE_ZIP" ]; then
        echo "Removing successfully restored station archive $SOURCE_ZIP..."
        rm -- "$SOURCE_ZIP"
        echo -e "${GREEN}Removed station archive $SOURCE_ZIP${NC}"
    fi
    echo -e "${GREEN}✓ Restore completed successfully!${NC}"
    echo
    if [ "$ACTIVE_MULTISTART_RESTORE" -eq 1 ]; then
        echo "Resume the restored multistart job with:"
        echo "  ./start.sh -s"
    elif [ -z "$MULTISTART_JOB" ]; then
        echo "If there were any research evaluations in progress, run:"
        echo "  python scripts/restart_eval.py"
    fi
else
    echo -e "${RED}✗ Restore failed${NC}"
    exit 1
fi
