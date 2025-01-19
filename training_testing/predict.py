import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sys
import json
import math

# Load configuration from config.json
with open('input/training_testing_config.json', 'r') as config_file:
    config = json.load(config_file)

# Set the model file name and CSV file name as variables
model_file = 'output/xgb_model.json'  # Replace with your actual model file
test_data_file = config['test_data_file']  # Read from config.json
test_baseline_ratio = config['test_baseline_ratio']

output_base_directory = 'output/prediction'

# Load the saved XGBoost model
model = xgb.Booster()
model.load_model(model_file)

# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model)
plt.tight_layout()
plt.savefig(f'{output_base_directory}/feature_importance_{test_data_file.split("/")[-1].split(".")[0]}.png')  # Save the plot as a PNG file

# Load test data from the CSV file
test_data = pd.read_csv(test_data_file)

# If "predict_only_include_overlapped_kernel" is true, drop kernels without overlap
if config['predict_only_include_overlapped_kernel']:
    test_data = test_data[test_data['overlap_ratio'] != 0].reset_index(drop=True)

X_test = test_data.drop(columns=['id', 'kShortName', 'kDemangledName', 'duration', 'overlap_ratio', 'slowdown', 'ID', 'Kernel Name', 'SM'])

# Convert test data to DMatrix (XGBoost's required data format)
dtest = xgb.DMatrix(X_test)

# Perform prediction
slowdown_predictions = model.predict(dtest)

slowdown_predictions = pd.Series(slowdown_predictions)

slowdown_predictions_clipped = slowdown_predictions.clip(lower=0) # set negative prediction to zero

print('slowdown_predictions.shape', slowdown_predictions.shape)
print('test_data.shape', test_data.shape)

calculated_duration = (1-test_data['overlap_ratio']) * test_data['ground_truth'] + test_data['overlap_ratio'] * test_data['ground_truth'] * (1+slowdown_predictions_clipped)

calculated_1_3_duration_comparison = (1-test_data['overlap_ratio']) * test_data['ground_truth'] + test_data['overlap_ratio'] * test_data['ground_truth'] * (test_baseline_ratio) # compare with slowdown=test_baseline_ratio

print('calculated_duration.shape', calculated_duration.shape)

print('type(slowdown_predictions), type(calculated_duration)', type(slowdown_predictions), type(calculated_duration))


output_df = pd.concat([test_data['id'], test_data['kShortName'], (1-test_data['overlap_ratio']), test_data['ground_truth'], test_data['overlap_ratio'], test_data['ground_truth'], slowdown_predictions, slowdown_predictions_clipped, test_data['slowdown'], calculated_duration, calculated_1_3_duration_comparison, test_data['duration']], axis=1)

output_df.columns = ['id', 'kShortName', '1-overlap_ratio', 'ground_truth duration without overlap', 'overlap_ratio', 'ground_truth duration without overlap', 'slowdown predictions', 'slowdown predictions clipped', 'ground truth slowdown', 'calculated duration with overlap', 'comparison using slowdown=test_baseline_ratio', 'ground truth duration with overlap']

left = output_df
right = test_data

print(left.head())
print(right.head())
out_full_df = pd.merge(left, right, on='id', how='inner')

out_full_df['slowdown diff'] = out_full_df['slowdown predictions'] - out_full_df['ground truth slowdown']
out_full_df['slowdown abs diff'] = out_full_df['slowdown diff'].abs()

out_full_df['our duration diff'] = out_full_df['calculated duration with overlap'] - out_full_df['ground truth duration with overlap']
out_full_df['our duration abs diff'] = out_full_df['our duration diff'].abs()
out_full_df['our kernel error rate'] = out_full_df['our duration abs diff'] / out_full_df['ground truth duration with overlap']

out_full_df['baseline duration diff'] = out_full_df['comparison using slowdown=test_baseline_ratio'] - out_full_df['ground truth duration with overlap']
out_full_df['baseline duration abs diff'] = out_full_df['baseline duration diff'].abs()
out_full_df['baseline kernel error rate'] = out_full_df['baseline duration abs diff'] / out_full_df['ground truth duration with overlap']



# calculate metrics

our_slowdown_mae = mean_absolute_error(test_data['slowdown'], slowdown_predictions)
our_slowdown_mse = mean_squared_error(test_data['slowdown'], slowdown_predictions)
our_slowdown_rmse = math.sqrt(our_slowdown_mse)

our_slowdown_clipped_mae = mean_absolute_error(test_data['slowdown'], slowdown_predictions_clipped)
our_slowdown_clipped_mse = mean_squared_error(test_data['slowdown'], slowdown_predictions_clipped)
our_slowdown_clipped_rmse = math.sqrt(our_slowdown_clipped_mse)

our_duration_mae = mean_absolute_error(test_data['duration'], calculated_duration) # ground truth duration with overlap, calculated duration with overlap
our_duration_mse = mean_squared_error(test_data['duration'], calculated_duration) # ground truth duration with overlap, calculated duration with overlap
our_duration_rmse = math.sqrt(our_duration_mse)

baseline_slowdown_mae = mean_absolute_error(test_data['slowdown'], [test_baseline_ratio for x in range(len(test_data['slowdown']))])
baseline_slowdown_mse = mean_squared_error(test_data['slowdown'], [test_baseline_ratio for x in range(len(test_data['slowdown']))])
baseline_slowdown_rmse = math.sqrt(baseline_slowdown_mse)

baseline_duration_mae = mean_absolute_error(test_data['duration'], calculated_1_3_duration_comparison) # ground truth duration with overlap, calculated duration with overlap
baseline_duration_mse = mean_squared_error(test_data['duration'], calculated_1_3_duration_comparison) # ground truth duration with overlap, calculated duration with overlap
baseline_duration_rmse = math.sqrt(baseline_duration_mse)

sum_calculated_duration_with_overlap = output_df['calculated duration with overlap'].sum()
sum_comparison_using_slowdown_1_3 = output_df['comparison using slowdown=test_baseline_ratio'].sum()
sum_ground_truth_duration_with_overlap = output_df['ground truth duration with overlap'].sum()

sum_calcualted_duration_ground_truth_error_rate = (sum_calculated_duration_with_overlap - sum_ground_truth_duration_with_overlap) / sum_ground_truth_duration_with_overlap * 100
sum_comparison_using_slowdown_1_3_ground_truth_error_rate = (sum_comparison_using_slowdown_1_3 - sum_ground_truth_duration_with_overlap) / sum_ground_truth_duration_with_overlap * 100

kernel_error_rate_bounds = [0.05, 0.1, 0.15]
our_kernel_error_rate_bounds_count = [0 for x in kernel_error_rate_bounds]
baseline_kernel_error_rate_bounds_count = [0 for x in kernel_error_rate_bounds]
kernel_count = 0

for index, row in out_full_df.iterrows():
    kernel_count += 1
    for i in range(len(kernel_error_rate_bounds)):
        kernel_error_bound = kernel_error_rate_bounds[i]
        if row['our kernel error rate'] < kernel_error_bound:
            our_kernel_error_rate_bounds_count[i] += 1
        if row['baseline kernel error rate'] < kernel_error_bound:
            baseline_kernel_error_rate_bounds_count[i] += 1

kernel_error_rate_bounds_count_rates = [{
    'kernel_error_rate_bound': kernel_error_rate_bounds[i], 
    'our_kernel_error_rate_bounds_count': our_kernel_error_rate_bounds_count[i], 
    'baseline_kernel_error_rate_bounds_count': baseline_kernel_error_rate_bounds_count[i],
    'kernel_count': kernel_count, 
    'our_kernel_error_rate_bounds_count_rate': our_kernel_error_rate_bounds_count[i]/kernel_count,
    'baseline_kernel_error_rate_bounds_count_rate': baseline_kernel_error_rate_bounds_count[i]/kernel_count,
    } for i in range(len(kernel_error_rate_bounds))]

# Save the original standard output
original_stdout = sys.stdout

# Open a file for writing
with open(f'{output_base_directory}/output_metrics.txt', 'w') as f:
    # Redirect standard output to the file
    sys.stdout = f

    # Print statements will now write to the file
    print('Our MAE of slowdown (original)', our_slowdown_mae)
    print('Our MSE of slowdown (original)', our_slowdown_mse)
    print('Our RMSE of slowdown (original)', our_slowdown_rmse)

    print('Our MAE of slowdown (clipped)', our_slowdown_clipped_mae)
    print('Our MSE of slowdown (clipped)', our_slowdown_clipped_mse)
    print('Our RMSE of slowdown (clipped)', our_slowdown_clipped_rmse)

    print('Our MAE of duration', our_duration_mae)
    print('Our MSE of duration', our_duration_mse)
    print('Our RMSE of duration', our_duration_rmse)

    print('Baseline MAE of slowdown', baseline_slowdown_mae)
    print('Baseline MSE of slowdown', baseline_slowdown_mse)
    print('Baseline RMSE of slowdown', baseline_slowdown_rmse)

    print('Baseline MAE of duration', baseline_duration_mae)
    print('Baseline MSE of duration', baseline_duration_mse)
    print('Baseline RMSE of duration', baseline_duration_rmse)

    print(f'Sum of calculated duration with overlap: {sum_calculated_duration_with_overlap} (Error rate: {sum_calcualted_duration_ground_truth_error_rate}%)')
    print(f'Sum of comparison using slowdown=test_baseline_ratio: {sum_comparison_using_slowdown_1_3} (Error rate: {sum_comparison_using_slowdown_1_3_ground_truth_error_rate}%)')
    print(f'Sum of ground truth duration with overlap: {sum_ground_truth_duration_with_overlap}')

    for x in kernel_error_rate_bounds_count_rates:
        print(f'Our {x['kernel_error_rate_bound']} error: {x['our_kernel_error_rate_bounds_count_rate']}')
        print(f'Baseline {x['kernel_error_rate_bound']} error: {x['baseline_kernel_error_rate_bounds_count_rate']}')

# Restore the original standard output
sys.stdout = original_stdout


# save output df to file
test_data_file_suffix = test_data_file.split('/')[-1].split('.')[0]
output_df.to_csv(f'{output_base_directory}/output_df_{test_data_file_suffix}.csv', index=False)
out_full_df.to_csv(f'{output_base_directory}/output_full_df_{test_data_file_suffix}.csv', index=False)
