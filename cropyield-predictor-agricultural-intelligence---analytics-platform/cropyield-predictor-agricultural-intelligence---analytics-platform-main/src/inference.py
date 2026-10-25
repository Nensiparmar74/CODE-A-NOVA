"""
CropYield Predictor - Inference Module
Defines the deployable End-to-End Inference Pipeline class for Joblib serialization.
"""

import pandas as pd


class FullInferencePipeline:
    """
    End-to-End inference pipeline combining agricultural feature engineering
    and Scikit-Learn ColumnTransformer + Estimator.
    Accepts raw dictionaries, lists of dictionaries, or pandas DataFrames.
    """
    def __init__(self, feature_engineer, model_pipeline):
        self.feature_engineer = feature_engineer
        self.model_pipeline = model_pipeline

    def predict(self, X_input):
        if isinstance(X_input, dict):
            X_df = pd.DataFrame([X_input])
        elif isinstance(X_input, list):
            X_df = pd.DataFrame(X_input)
        else:
            X_df = X_input.copy()
        X_trans = self.feature_engineer.transform(X_df)
        return self.model_pipeline.predict(X_trans)
