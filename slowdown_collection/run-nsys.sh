#!/bin/bash

# Check if a parameter is provided
if [ -z "$1" ]; then
    echo "Error: No parameter provided."
    echo "Usage: $0 <string>"
    exit 1
fi

world_size=$1

# Filepath to the JSON file
global_config="input/global_config.json"
local_config="input/local_config.json"
python_script="input/train_script.py"

cuda_visible_devices=$(jq -r '.cuda_visible_devices' "$global_config")
nsys_path=$(jq -r '.nsys_path' "$global_config")
python_path=$(jq -r '.python_path' "$global_config")

output_name="temp/output_ws$world_size"
stats_output="temp/stats_output_ws$world_size"

# Step 1: Profile the model training process using the configuration file
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} ${nsys_path} profile --trace=cuda,nvtx,osrt,python-gil --sample=cpu --python-sampling=true --python-backtrace=cuda --gpuctxsw=true \
    --output=${output_name} --export=none --force-overwrite true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    ${python_path} ${python_script} --world_size=${world_size} --local_config_file=${local_config} --global_config_file=${global_config}

# Step 2: Generate statistics and export to sqlite
${nsys_path} export -t sqlite --force-overwrite true -o ${stats_output}.sqlite ${output_name}.nsys-rep
echo "Profiling and analysis complete. Output saved to ${stats_output}.sqlite"