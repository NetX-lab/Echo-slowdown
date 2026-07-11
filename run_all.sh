#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

skip_kernel_metric=${SKIP_KERNEL_METRIC:-0}

if [ "$skip_kernel_metric" = "1" ]; then
    echo "Skipping kernel_metric module and reusing merge/input/kernel_metric_output.csv"
    if [ ! -f "$SCRIPT_DIR/merge/input/kernel_metric_output.csv" ]; then
        echo "Error: merge/input/kernel_metric_output.csv is required when SKIP_KERNEL_METRIC=1."
        exit 1
    fi
else
    echo "Running kernel_metric module..."
    cd "$SCRIPT_DIR/kernel_metric"
    bash run.sh
    cd "$SCRIPT_DIR"
    cp kernel_metric/output/* merge/input/
fi

echo "Running slowdown_collection module..."
cd "$SCRIPT_DIR/slowdown_collection"
bash run.sh
cd "$SCRIPT_DIR"
cp slowdown_collection/output/* merge/input/

echo "Running merge module..."
cd "$SCRIPT_DIR/merge"
bash run.sh
cd "$SCRIPT_DIR"
cp merge/output/* training_testing/input/test_csv/
cp merge/output/* training_testing/input/train_csv/

echo "Running training_testing module..."
cd "$SCRIPT_DIR/training_testing"
bash run.sh
cd "$SCRIPT_DIR"

echo "All modules ran successfully."
