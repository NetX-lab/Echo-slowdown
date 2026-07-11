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

global_config="input/global_config.json"
python_path=$(json_get "$global_config" "python_path")

mkdir -p "output"

${python_path} merge_script.py --output output/merged_features.csv
