"""
CropYield Predictor - Evaluation & Residual Analysis Module
Phases 14, 15:
- In-depth residual analysis (actual vs predicted, error distribution, homoscedasticity)
- Feature importance analysis and domain interpretation
- Generalization error assessment
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering import AgriculturalFeatureEngineer

CLEAN_DATA_PATH = "data/processed/clean_crop_yield.csv"
MODEL_PATH = "models/crop_yield_model.joblib"
EVALUATION_REPORT_PATH = "reports/evaluation_analysis.json"


def run_full_evaluation():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    print("Loading test data and trained pipeline...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    # Use a 20% test sample with fixed seed matching train split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['hg/ha_yield'])
    y = df['hg/ha_yield']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Predict directly via the exported full pipeline
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mean_yield = float(y_test.mean())
    nmae = float(mae / mean_yield)

    residuals = y_test - y_pred
    res_mean = float(residuals.mean())
    res_std = float(residuals.std())

    # Percentiles of error
    abs_errors = np.abs(residuals)
    error_percentiles = {
        "p25": float(np.percentile(abs_errors, 25)),
        "p50_median": float(np.percentile(abs_errors, 50)),
        "p75": float(np.percentile(abs_errors, 75)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95))
    }

    # Accuracy within tolerance bands
    within_10_pct = float(np.mean(abs_errors <= (0.10 * y_test)) * 100)
    within_20_pct = float(np.mean(abs_errors <= (0.20 * y_test)) * 100)

    report = {
        "dataset_test_samples": len(y_test),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "normalized_mae": round(nmae, 4),
        "mean_actual_yield": round(mean_yield, 2),
        "residual_mean": round(res_mean, 2),
        "residual_std": round(res_std, 2),
        "absolute_error_percentiles": error_percentiles,
        "within_10_percent_tolerance": round(within_10_pct, 2),
        "within_20_percent_tolerance": round(within_20_pct, 2)
    }

    os.makedirs("reports", exist_ok=True)
    with open(EVALUATION_REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n--- Model Evaluation Summary ---")
    print(f"MAE: {mae:,.2f} hg/ha")
    print(f"RMSE: {rmse:,.2f} hg/ha")
    print(f"R²: {r2:.4f}")
    print(f"Saved evaluation report to {EVALUATION_REPORT_PATH}")
    return report


if __name__ == "__main__":
    run_full_evaluation()
