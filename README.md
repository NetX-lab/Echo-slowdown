# Echo Slowdown Prediction Module

This repository contains the slowdown prediction module of [Echo: Simulating Distributed Training at Scale](https://arxiv.org/abs/2412.12487). The module predicts the slowdown of GPU kernels when they overlap with other kernels during distributed training.

## Project Overview

The Echo Slowdown Prediction Module is designed to predict the performance impact of kernel overlaps in distributed training scenarios. The system consists of three core components:

1. **Kernel Metric Collection**
   - Utilizes NVIDIA Nsight Compute and Nsight Systems to profile GPU kernels
   - Captures detailed execution metrics including:
     * Kernel duration
     * Memory bandwidth utilization
     * Compute throughput
     * Instruction mix statistics
   - Generates baseline performance profiles for isolated kernel execution
   - Outputs structured JSON files containing raw kernel metrics

2. **Slowdown Collection**
   - Analyzes kernel behavior under various overlap scenarios
   - Measures actual slowdown factors through controlled experiments
   - Collects data on:
     * Resource contention patterns
     * Memory access interference
     * Compute unit saturation
   - Generates ground truth data for model training and validation

3. **Training & Testing**
   - Implements machine learning models for slowdown prediction
   - Features include:
     * Multi-layer perceptron (MLP) regression
     * Gradient boosting decision trees
     * Feature importance analysis
   - Provides comprehensive evaluation metrics:
     * Mean absolute percentage error (MAPE)
     * R-squared scores
     * Prediction error distribution
   - Outputs trained models and prediction results for integration with the Echo simulator

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- NVIDIA Nsight Compute CLI 2024.3.0.0 or later
- NVIDIA Nsight Systems 2024.4.2.133 or later
- Conda package manager

### Setup Instructions

1. Clone git repository
    ```bash
    git clone https://github.com/ericskh2/echo_slowdown.git
    cd echo_slowdown
    ```

2. Setup Conda environment
    ```bash
    conda env create -f environment.yml
    conda activate simulator_echo
    ```

## Configuration

Update the configuration by running the Python file:
  ```bash
  python update_configs.py
  ```


This script will automatically detect the paths for `nsys`, `python`, and `ncu` using the `which` command and update the `global_config.json` files in the following directories:
- `kernel_metric/input/global_config.json`
- `merge/input/global_config.json`
- `slowdown_collection/input/global_config.json`

Additionally, it will check the installed CUDA version using PyTorch and update the `cuda_version_check` field in the configuration files.

Our script is tested on NVIDIA Nsight Compute CLI 2024.3.0.0 and NVIDIA Nsight Systems 2024.4.2.133.

## Usage

### Basic Execution

Run the complete pipeline:

```bash
sh ./run_all.sh
```

### Advanced Usage

You can run individual modules separately:

1. Collect kernel metrics:
    ```bash
    cd kernel_metric
    python main.py
    ```

2. Merge and preprocess data:
    ```bash
    cd merge
    python main.py
    ```

3. Train and evaluate slowdown prediction:

    ```bash
    cd slowdown_collection
    python main.py  
    ```




## Output
The pipeline generates the following outputs:

```plaintext
echo_slowdown/
├── training_testing/
│   └── output/
│       ├── train_dataset.csv         # Processed training dataset that suits the input format of the XGBoost model
│       ├── xgb_model.json    # Trained XGBoost model
│       └── prediction/     # Model predictions
│           ├── output_df_merged_features.csv         # Simplified output of slowdown predictions
│           ├── output_full_df_merged_features.csv        # Detailed output of slowdown predictions
│           ├── output_metrics.txt         # Evaluation metrics such as MAE, MSE and RMSE
│           └── feature_importance_merged_features.png         # Plot of feature importance
├── kernel_metric/
│   └── output/
│       └── kernel_metric_output.csv    # Profiled kernel metrics
├── slowdown_collection/
│   └── output/
│       └── slowdown_stats_output_device_0.xlsx    # Profiled slowdown statistics
├── merge/
│   └── output/
│       └── merged_features.csv       # Combined kernel metrics and slowdown statistics, aligned by kernel name
```


## Expected Results

After running the pipeline, you should expect:

- Trained machine learning models for slowdown prediction
- Prediction accuracy reports for different hardware configurations
- Detailed performance metrics for overlapping kernel scenarios
- Processed datasets ready for simulation integration

## Troubleshooting

1. **Nsight Tools Not Found**
   - Ensure Nsight Compute and Systems are properly installed
   - Verify they are in your system PATH

2. **CUDA Version Mismatch**
   - Check installed CUDA version matches your GPU driver
   - Update PyTorch to compatible version if needed

3. **Permission Errors**
   - Run scripts with appropriate permissions
   - Ensure output directories are writable


## Citation

If you use this module in your research, please cite our paper:

```bibtex
@article{echo2024,
  title={Echo: Simulating Distributed Training At Scale},
  author={Yicheng Feng, Yuetao Chen, Kaiwen Chen, Jingzong Li, Tianyuan Wu, Peng Cheng, Chuan Wu, Wei Wang, Tsung-Yi Ho, Hong Xu},
  journal={arXiv preprint arXiv:2412.12487},
  year={2024}
}
```


## Contact

Please email Yicheng Feng (<yichengfeng@link.cuhk.edu.hk>) or Kin Hang Sew (<ericskh@link.cuhk.edu.hk>) if you have any questions.


## License

This project is licensed under the BSD-3-Clause license - see the [LICENSE](LICENSE) file for details.

