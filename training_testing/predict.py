import pandas as pd

from prediction_api import SlowdownPredictor

input_data = {
    'ground_truth': [1.0],
    'Compute throughput': [50.0],
    'Memory throughput': [60.0],
    'DRAM throughput': [70.0],
    'Achieved occupancy': [80.0],
    'Maximum occupancy': [90.0],
    'L1 hit rate': [95.0],
    'L2 hit rate': [85.0],
}
input_overlap_ratio = 0.5

input_df = pd.DataFrame(input_data)
predictor = SlowdownPredictor(model_path='output/xgb_model.json', scaler_path='output/standard_scaler.json')
result = predictor.predict_slowdown(input_df.iloc[0].to_dict(), input_overlap_ratio)
print(result)
