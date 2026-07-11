import json
from pathlib import Path

import pandas as pd
import xgboost as xgb


class SlowdownPredictor:
    def __init__(self, model_path: str, scaler_path: str | None = None):
        """
        Initialize the predictor by loading the XGBoost model and the fitted scaler spec.

        :param model_path: Path to the pre-trained XGBoost model.
        :param scaler_path: Path to the persisted scaler JSON. Defaults to a sibling file
            named ``standard_scaler.json`` next to ``model_path``.
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path is not None else self.model_path.with_name(
            'standard_scaler.json'
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Model file does not exist: {self.model_path}')
        if not self.scaler_path.is_file():
            raise FileNotFoundError(f'Scaler spec does not exist: {self.scaler_path}')

        self.model = xgb.Booster()
        self.model.load_model(str(self.model_path))
        self.scaler_spec = json.loads(self.scaler_path.read_text(encoding='utf-8'))
        self.feature_names = list(self.scaler_spec['feature_names'])
        self.feature_means = list(self.scaler_spec['mean'])
        self.feature_scales = list(self.scaler_spec['scale'])
        if not (
            len(self.feature_names) == len(self.feature_means) == len(self.feature_scales)
        ):
            raise ValueError('Scaler spec feature_names/mean/scale length mismatch')

    def _normalize_features(self, input_features: dict) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in input_features]
        if missing:
            raise ValueError(f'Missing slowdown features: {missing}')
        normalized_row = {}
        for index, feature_name in enumerate(self.feature_names):
            raw_value = float(input_features[feature_name])
            scale = float(self.feature_scales[index])
            if scale == 0:
                raise ValueError(f'Scaler scale for feature {feature_name!r} must be non-zero')
            normalized_row[feature_name] = (raw_value - float(self.feature_means[index])) / scale
        return pd.DataFrame([normalized_row], columns=self.feature_names)

    def predict_slowdown(self, input_features: dict, input_overlap_ratio: float):
        """
        Predict execution time and slowdown factor given input features and overlap ratio.
        """
        df_single_row = self._normalize_features(input_features)
        dtest = xgb.DMatrix(df_single_row)

        slowdown_predictions = self.model.predict(dtest)
        slowdown_predictions = pd.Series(slowdown_predictions)
        slowdown_predictions_clipped = slowdown_predictions.clip(lower=0)

        predicted_slowdown_factor = slowdown_predictions.iloc[0]
        predicted_slowdown_factor_clipped = slowdown_predictions_clipped.iloc[0]

        predicted_execution_time = (1 - input_overlap_ratio) * input_features['ground_truth'] + input_overlap_ratio * input_features['ground_truth'] * (1 + predicted_slowdown_factor)
        predicted_execution_time_clipped = (1 - input_overlap_ratio) * input_features['ground_truth'] + input_overlap_ratio * input_features['ground_truth'] * (1 + predicted_slowdown_factor_clipped)

        res = {
            'original_execution_time': input_features['ground_truth'],
            'predicted_execution_time': predicted_execution_time,
            'predicted_execution_time_clipped': predicted_execution_time_clipped,
            'predicted_slowdown_factor': predicted_slowdown_factor,
            'predicted_slowdown_factor_clipped': predicted_slowdown_factor_clipped,
        }

        return res
