# echo_slowdown

Author: 

This script will collect kernel features and slowdown data using Nvidia Nsight Compute and Nvidia Nsight Systems and predict the slowdown rate.

## Installation

Clone git repository
```
git clone https://github.com/ericskh2/echo_slowdown.git
cd echo_slowdown
```

Setup Conda environment
```
conda create --name simulator-echo --file requirements.txt
conda activate simulator-echo
```

## Configuration
Update the configuration by running the Python file
```
python update_configs.py
```

This script will automatically detect the paths for `nsys`, `python`, and `ncu` using the `which` command and update the `global_config.json` files in the following directories:
- `kernel_metric/input/global_config.json`
- `merge/input/global_config.json`
- `slowdown_collection/input/global_config.json`

## Usage
Execute `run_all.sh`
```
sh ./run_all.sh
```

