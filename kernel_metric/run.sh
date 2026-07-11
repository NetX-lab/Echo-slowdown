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

# Collect kernel metrics
echo "Collecting kernel metrics..."

global_config="input/global_config.json"
local_config="input/local_config.json"
python_script="input/train_script.py"

cuda_visible_devices=$(json_get "$global_config" "cuda_visible_devices")
ncu_path=$(json_get "$global_config" "ncu_path")
python_path=$(json_get "$global_config" "python_path")

mkdir -p "temp"
mkdir -p "output"
output_name="temp/output"

CUDA_VISIBLE_DEVICES=${cuda_visible_devices} ${ncu_path} -o ${output_name}   --profile-from-start off   -f --replay-mode application   --target-processes all   --app-replay-buffer file   --device 0   --nvtx   --section "SpeedOfLight" --section "LaunchStats" --section "Occupancy" --section "MemoryWorkloadAnalysis"   ${python_path} ${python_script} --world_size=1 --local_config_file=${local_config} --global_config_file=${global_config}

${ncu_path} -i "${output_name}.ncu-rep" --page details --csv --log-file "${output_name}_details.csv"
${ncu_path} -i "${output_name}.ncu-rep" --page raw --csv --log-file "${output_name}_raw.csv"
${ncu_path} -i "${output_name}.ncu-rep" --print-kernel-base function --csv > "${output_name}_kshortname.csv"

${python_path} ncu_report_process.py
