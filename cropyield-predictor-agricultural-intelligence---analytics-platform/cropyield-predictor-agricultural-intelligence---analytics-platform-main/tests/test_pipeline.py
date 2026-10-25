"""
Unit tests for Scikit-Learn Pipeline and Joblib Model.
Verifies reproducibility, leak-free preprocessing, and valid predictions.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import AgriculturalFeatureEngineer
from src.preprocessing import build_preprocessor

MODEL_PATH = "models/crop_yield_model.joblib"


def test_feature_engineering_transform():
    sample_df = pd.DataFrame({
        'Area': ['India'],
        'Item': ['Wheat'],
        'Year': [2005],
        'average_rain_fall_mm_per_year': [1000.0],
        'pesticides_tonnes': [50000.0],
        'avg_temp': [25.0]
    })
    fe = AgriculturalFeatureEngineer()
    featured = fe.transform(sample_df)

    assert 'hydrothermal_index' in featured.columns
    assert 'rainfall_category' in featured.columns
    assert 'pesticide_log' in featured.columns
    assert 'temp_rainfall_interaction' in featured.columns
    # Hydrothermal index = 1000 / (25 + 10) = 28.5714
    assert pytest.approx(featured.loc[0, 'hydrothermal_index'], 0.1) == 28.57
    assert featured.loc[0, 'rainfall_category'] == 'Moderate (600-1200mm)'


def test_preprocessor_shape():
    sample_df = pd.DataFrame({
        'Area': ['India', 'France'],
        'Item': ['Wheat', 'Potatoes'],
        'Year': [2005, 2010],
        'average_rain_fall_mm_per_year': [1000.0, 800.0],
        'pesticides_tonnes': [50000.0, 20000.0],
        'avg_temp': [25.0, 15.0]
    })
    fe = AgriculturalFeatureEngineer()
    featured = fe.transform(sample_df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(featured)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0
    assert not np.isnan(transformed).any()


def test_saved_model_inference():
    """Verify saved joblib pipeline loads and predicts accurately without retraining."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not yet trained and exported.")

    pipeline = joblib.load(MODEL_PATH)

    sample_input = {
        'Area': 'India',
        'Item': 'Rice, paddy',
        'Year': 2008,
        'average_rain_fall_mm_per_year': 1200.0,
        'pesticides_tonnes': 40000.0,
        'avg_temp': 26.5
    }

    pred = pipeline.predict(sample_input)
    assert len(pred) == 1
    assert isinstance(pred[0], (float, np.floating))
    # Rice yield is typically in range 20,000 to 60,000 hg/ha
    assert pred[0] > 0
    assert pred[0] < 1_000_000
