# echo_slowdown

This repository contains the slowdown prediction module of [Echo: Simulating Distributed Training At Scale](https://arxiv.org/abs/2412.12487).

## Installation

Clone git repository
```
git clone https://github.com/ericskh2/echo_slowdown.git
cd echo_slowdown
```

Setup Conda environment
```
conda env create -f environment.yml
conda activate simulator_echo
```

## Configuration
Update the configuration by running the Python file:
```
python update_configs.py
```

This script will automatically detect the paths for `nsys`, `python`, and `ncu` using the `which` command and update the `global_config.json` files in the following directories:
- `kernel_metric/input/global_config.json`
- `merge/input/global_config.json`
- `slowdown_collection/input/global_config.json`

Additionally, it will check the installed CUDA version using PyTorch and update the `cuda_version_check` field in the configuration files.

Our script is tested on NVIDIA Nsight Compute CLI 2024.3.0.0 and NVIDIA Nsight Systems 2024.4.2.133.

## Usage
After configuration, execute `run_all.sh`
```
sh ./run_all.sh
```

## Output
The trained model and prediction results are stored in `training_testing/output`, and intermediate results are stored in `output` folder of each submodule.
