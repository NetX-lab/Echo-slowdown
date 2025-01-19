# Filepath to the JSON file
global_config="input/global_config.json"

python_path=$(jq -r '.python_path' "$global_config")

mkdir -p "output"

${python_path} merge_script.py --output output/merged_features.csv