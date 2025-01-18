#!/bin/bash

# Collect slowdown cases
echo "Collecting slowdown cases..."

rm -rf temp
rm -rf output
mkdir -p temp
mkdir -p output

# Filepath to the JSON file
global_config="input/global_config.json"

python_path=$(jq -r '.python_path' "$global_config")

sh run-nsys.sh 1
sh run-nsys.sh 2

${python_path} analyse.py