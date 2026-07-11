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

# Collect slowdown cases
echo "Collecting slowdown cases..."

mkdir -p temp
mkdir -p output

global_config="input/global_config.json"
python_path=$(json_get "$global_config" "python_path")

bash run-nsys.sh 1
bash run-nsys.sh 2

${python_path} analyse.py
