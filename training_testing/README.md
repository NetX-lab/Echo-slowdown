# Training and Testing Module
This module trains and tests an XGBoost model.

## Usage
1. Prepare all training csvs in `input/train_csv/`, `merged_features.csv` in `input/test_csv/`, `training_testing_config.json` in `input/`.

2. Execute shell script:
```
./run.sh
```

3. The combined training dataset will be saved at `output/train_dataset.csv`, trained model will be stored in `output/xgb_model.json`, 

## Example Configurations

All the configurations should be placed into the input folder.

## Example `training_testing_config.json`
```
{
    "test_data_file": "input/test_csv/merged_features.csv",
    "test_baseline_ratio": 1.3,
    "predict_only_include_overlapped_kernel": true
}
```

|Key|Value|
|-|-|
|test_data_file|Path to the test csv|
|test_baseline_ratio|Baseline overlap factor, the baseline prediction will be calculated by non_overlap_duration*baseline_overlap_factor|
|predict_only_include_overlapped_kernel|If true, the evaluation metrics will only consider kernels with overlap|
