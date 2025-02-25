import xgboost as xgb
import numpy as np
import pandas as pd

class SlowdownPredictor:
    def __init__(self, model_path: str):
        """
        Initialize the predictor by loading the XGBoost model.
        
        :param model_path: Path to the pre-trained XGBoost model.
        """
        self.model = xgb.Booster()
        self.model.load_model(model_path)
    
    def predict_slowdown(self, input_features: dict, input_overlap_ratio: float):
        """
        Predicts execution time and slowdown factor given input features and overlap ratio.
        
        :param input_features: Dictionary containing input variables.
            input_features:
            ground_truth (duration without slowdown)
            Compute throughput       
            Memory throughput       
            DRAM throughput         
            Achieved occupancy      
            Maximum occupancy      
            L1 hit rate              
            L2 hit rate             

        :param input_overlap_ratio: Ratio of overlap affecting execution time.
        :return: Dictionary with original execution time, predicted execution time, and slowdown factor.
        """

        # print('input_features:', input_features, sep='\n')
        df_single_row = pd.DataFrame([input_features])
        dtest = xgb.DMatrix(df_single_row)

        # Perform prediction
        slowdown_predictions = self.model.predict(dtest)
        slowdown_predictions = pd.Series(slowdown_predictions)
        slowdown_predictions_clipped = slowdown_predictions.clip(lower=0) # set negative prediction to zero

        # print('slowdown_predictions.shape', slowdown_predictions.shape)
        # print('slowdown_predictions', slowdown_predictions)

        predicted_slowdown_factor = slowdown_predictions.iloc[0]
        predicted_slowdown_factor_clipped = slowdown_predictions_clipped.iloc[0]

        predicted_execution_time = (1-input_overlap_ratio) * input_features['ground_truth'] + input_overlap_ratio * input_features['ground_truth'] * (1+predicted_slowdown_factor)
        predicted_execution_time_clipped = (1-input_overlap_ratio) * input_features['ground_truth'] + input_overlap_ratio * input_features['ground_truth'] * (1+predicted_slowdown_factor_clipped)

        res = {
            "original_execution_time": input_features['ground_truth'],
            "predicted_execution_time": predicted_execution_time,
            "predicted_execution_time_clipped": predicted_execution_time_clipped,
            "predicted_slowdown_factor": predicted_slowdown_factor,
            "predicted_slowdown_factor_clipped": predicted_slowdown_factor_clipped
        }
        # print('res', res)

        return res
