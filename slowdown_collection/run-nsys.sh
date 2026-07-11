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

if [ -z "$1" ]; then
    echo "Error: No parameter provided."
    echo "Usage: $0 <world_size>"
    exit 1
fi

world_size=$1
global_config="input/global_config.json"
local_config="input/local_config.json"
python_script="input/train_script.py"

cuda_visible_devices=$(json_get "$global_config" "cuda_visible_devices")
nsys_path=$(json_get "$global_config" "nsys_path")
python_path=$(json_get "$global_config" "python_path")

output_name="temp/output_ws$world_size"
stats_output="temp/stats_output_ws$world_size"

set +e
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} ${nsys_path} profile --trace=cuda,nvtx,osrt --sample=none --gpuctxsw=true --wait=all     --output=${output_name} --export=none --force-overwrite true --cuda-graph-trace=node     --capture-range=cudaProfilerApi     ${python_path} ${python_script} --world_size=${world_size} --local_config_file=${local_config} --global_config_file=${global_config}
profile_rc=$?
set -e

if [ ${profile_rc} -ne 0 ] && [ ${profile_rc} -ne 143 ]; then
    echo "Error: nsys profile failed with exit code ${profile_rc}."
    exit ${profile_rc}
fi

if [ ! -f "${output_name}.nsys-rep" ]; then
    echo "Error: expected report ${output_name}.nsys-rep was not generated."
    exit 1
fi

${nsys_path} export -t sqlite --force-overwrite true -o ${stats_output}.sqlite ${output_name}.nsys-rep
echo "Profiling and analysis complete. Output saved to ${stats_output}.sqlite"
