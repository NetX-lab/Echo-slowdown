#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ECHO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$ECHO_ROOT:${PYTHONPATH:-}"

json_get() {
  python - "$1" "$2" <<'PYCFG'
import json
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
value = data[sys.argv[2]]
print(value)
PYCFG
}

mkdir -p output
mkdir -p output/prediction

global_config="input/global_config.json"
python_path=$(json_get "$global_config" "python_path")

echo "Running create_dataset.py..."
${python_path} create_dataset.py

echo "Running train.py..."
${python_path} train.py

echo "Running predict.py..."
${python_path} predict.py
