#!/bin/bash

rm -rf output
mkdir -p output
mkdir -p output/prediction

# Filepath to the JSON file
global_config="input/global_config.json"

python_path=$(jq -r '.python_path' "$global_config")

# Execute the Python scripts
echo "Running create_dataset.py..."
${python_path} create_dataset.py

echo "Running train.py..."
${python_path} train.py

echo "Running predict.py..."
${python_path} predict.py
