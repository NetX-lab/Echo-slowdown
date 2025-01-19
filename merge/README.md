# Merge Module
This module merges collected kernel metric and slowdown data into a single csv.

## Usage
1. Prepare `kernel_metric_output.csv` (obtain from kernel metric module), `slowdown_stats_output_device_0.xlsx` (obtain from slowdown collection module), `global_config.json` in `input` folder.

2. Execute shell script
```
./run.sh
```

3. The output will be saved at `output/merged_features.csv`.


## Example Configurations
All the configurations should be placed into the `input` folder.

### Example `global_config.json`
```
{
	"python_path": "/home/eric/miniconda3/envs/torchgraph/bin/python"
}
```

|Key|Value|
|-|-|
|python_path|Path to python executable, can check with "which python"|