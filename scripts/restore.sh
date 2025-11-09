#!/bin/bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo -e "${RED}Usage: $0 <partial_station_id> [tick_number]${NC}"
    echo "Examples:"
    echo "  $0 6d8bc 1200       # Restore station starting with 6d8bc at tick 1200"
    echo "  $0 6d8bc            # Restore station starting with 6d8bc at latest available tick"
    exit 1
fi

PARTIAL_ID=$1
TICK=$2
BACKUP_DIR="./backup"
TARGET_DIR="./station_data"

# Find matching station IDs in backup directory
echo "Searching for station IDs starting with '$PARTIAL_ID'..."
MATCHING_DIRS=()
if [ -d "$BACKUP_DIR" ]; then
    while IFS= read -r dir; do
        MATCHING_DIRS+=("$dir")
    done < <(find "$BACKUP_DIR" -maxdepth 1 -type d -name "$PARTIAL_ID*")
fi

# Check number of matches
if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No station ID found starting with '$PARTIAL_ID' in $BACKUP_DIR${NC}"
    exit 1
elif [ ${#MATCHING_DIRS[@]} -gt 1 ]; then
    echo -e "${RED}Error: Multiple station IDs found starting with '$PARTIAL_ID':${NC}"
    for dir in "${MATCHING_DIRS[@]}"; do
        echo "  - $(basename "$dir")"
    done
    echo -e "${YELLOW}Please provide a more specific partial ID${NC}"
    exit 1
fi

STATION_DIR="${MATCHING_DIRS[0]}"
STATION_ID=$(basename "$STATION_DIR")
echo -e "${GREEN}Found station ID: $STATION_ID${NC}"

# If tick not provided, find the latest tick
if [ -z "$TICK" ]; then
    echo "Finding latest available tick..."
    SNAPSHOTS_DIR="$STATION_DIR/snapshots"

    if [ ! -d "$SNAPSHOTS_DIR" ]; then
        echo -e "${RED}Error: No snapshots directory found in $STATION_DIR${NC}"
        exit 1
    fi

    LATEST_TICK=0
    for tick_file in "$SNAPSHOTS_DIR"/tick_*.json; do
        if [ -f "$tick_file" ]; then
            # Extract tick number from filename: tick_1200.json -> 1200
            filename=$(basename "$tick_file")
            tick_num=${filename#tick_}
            tick_num=${tick_num%.json}

            if [ "$tick_num" -gt "$LATEST_TICK" ]; then
                LATEST_TICK=$tick_num
            fi
        fi
    done

    if [ "$LATEST_TICK" -eq 0 ]; then
        echo -e "${RED}Error: No valid tick files found in $SNAPSHOTS_DIR${NC}"
        exit 1
    fi

    TICK=$LATEST_TICK
    echo -e "${GREEN}Using latest tick: $TICK${NC}"
fi

# Check if station_data exists
if [ -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Warning: $TARGET_DIR directory exists.${NC}"
    read -p "Do you want to remove it before restoring? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing $TARGET_DIR..."
        rm -rf "$TARGET_DIR"
        echo -e "${GREEN}Removed $TARGET_DIR${NC}"
    else
        echo -e "${RED}Restore cancelled by user${NC}"
        exit 1
    fi
fi

# Perform the restore
echo -e "${GREEN}Restoring station $STATION_ID at tick $TICK...${NC}"
python -c "from station.backup_utils import restore_backup; restore_backup('$STATION_ID', $TICK, '$TARGET_DIR')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Restore completed successfully!${NC}"
    echo
    echo "If there were any research evaluations in progress, run:"
    echo "  python scripts/restart_eval.py"
else
    echo -e "${RED}✗ Restore failed${NC}"
    exit 1
fi
