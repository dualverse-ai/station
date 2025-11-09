#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: bash eval.sh <submission_file> <test_data_path>"
    echo "Example: bash eval.sh eval/eval_123.py /mnt/data/test_data"
    exit 1
fi

SUBMISSION_FILE=$1
TEST_DATA_PATH=$2

# Verify test data path exists
if [ ! -d "$TEST_DATA_PATH" ]; then
    echo "Error: Test data path does not exist: $TEST_DATA_PATH"
    exit 1
fi

# Verify submission file exists
if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "Error: Submission file not found: $SUBMISSION_FILE"
    exit 1
fi

# Extract base name for log files (e.g., eval/eval_123.py -> eval_123)
SUBMISSION_NAME=$(basename "$SUBMISSION_FILE" .py)

echo "Evaluating submission ${SUBMISSION_FILE} on test datasets in ${TEST_DATA_PATH}"
echo ""

# Run evaluation on all 6 test datasets
OPENBLAS_NUM_THREADS=32 python -u eval.py --submission "$SUBMISSION_FILE" --dataset "${TEST_DATA_PATH}/dkd_processed.h5ad" 2>&1 | tee "eval/${SUBMISSION_NAME}_dkd.log" && \
OPENBLAS_NUM_THREADS=32 python -u eval.py --submission "$SUBMISSION_FILE" --dataset "${TEST_DATA_PATH}/gtex_v9_processed.h5ad" 2>&1 | tee "eval/${SUBMISSION_NAME}_gtex_v9.log" && \
OPENBLAS_NUM_THREADS=32 python -u eval.py --submission "$SUBMISSION_FILE" --dataset "${TEST_DATA_PATH}/hypomap_processed.h5ad" 2>&1 | tee "eval/${SUBMISSION_NAME}_hypomap.log" && \
OPENBLAS_NUM_THREADS=32 python -u eval.py --submission "$SUBMISSION_FILE" --dataset "${TEST_DATA_PATH}/immune_cell_atlas_processed.h5ad" 2>&1 | tee "eval/${SUBMISSION_NAME}_immune_cell_atlas.log" && \
OPENBLAS_NUM_THREADS=32 python -u eval.py --submission "$SUBMISSION_FILE" --dataset "${TEST_DATA_PATH}/mouse_pancreas_atlas_processed.h5ad" 2>&1 | tee "eval/${SUBMISSION_NAME}_mouse_pancreas_atlas.log" && \
OPENBLAS_NUM_THREADS=32 python -u eval.py --submission "$SUBMISSION_FILE" --dataset "${TEST_DATA_PATH}/tabula_sapiens_processed.h5ad" 2>&1 | tee "eval/${SUBMISSION_NAME}_tabula_sapiens.log"

# Print final scores
echo ""
echo "=== Final Scores ==="
echo ""
echo 'dkd:' && tail "eval/${SUBMISSION_NAME}_dkd.log" | grep 'Final score'
echo 'gtex_v9:' && tail "eval/${SUBMISSION_NAME}_gtex_v9.log" | grep 'Final score'
echo 'hypomap:' && tail "eval/${SUBMISSION_NAME}_hypomap.log" | grep 'Final score'
echo 'immune_cell_atlas:' && tail "eval/${SUBMISSION_NAME}_immune_cell_atlas.log" | grep 'Final score'
echo 'mouse_pancreas_atlas:' && tail "eval/${SUBMISSION_NAME}_mouse_pancreas_atlas.log" | grep 'Final score'
echo 'tabula_sapiens:' && tail "eval/${SUBMISSION_NAME}_tabula_sapiens.log" | grep 'Final score'