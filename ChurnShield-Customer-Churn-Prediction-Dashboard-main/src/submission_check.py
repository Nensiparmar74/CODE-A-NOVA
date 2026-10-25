"""Final submission audit for ChurnShield."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "app.py", "README.md", "FINAL_INSIGHTS.md", "requirements.txt", ".gitignore",
    "notebooks/01_EDA.ipynb", "notebooks/02_Feature_Engineering.ipynb", "notebooks/03_Model_Training.ipynb",
    "models/churn_model.joblib", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "data/processed/telco_clean.csv", "outputs/reports/model_metrics.json",
    "outputs/reports/dataset_metadata.json", "outputs/shap/global_shap_importance.png",
    "outputs/shap/global_shap_importance.csv",
]

def main():
    print("CHURNSHIELD SUBMISSION AUDIT")
    failures = []
    for rel in REQUIRED:
        ok = (ROOT / rel).exists()
        print(f"{'PASS' if ok else 'FAIL'}  {rel}")
        if not ok: failures.append(rel)

    meta = json.loads((ROOT/'outputs/reports/dataset_metadata.json').read_text())
    print(f"Dataset type: {meta.get('dataset_type')}")
    print(f"Dataset rows: {meta.get('rows')}")
    if meta.get('dataset_type') != 'ibm_kaggle':
        failures.append('dataset_type')
        print("FAIL  Dataset metadata is not marked as ibm_kaggle.")

    df = pd.read_csv(ROOT/'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"CSV schema: {df.shape[0]} rows x {df.shape[1]} columns")
    if df.shape != (7043, 21):
        failures.append('dataset_shape')

    artifact = joblib.load(ROOT/'models/churn_model.joblib')
    print(f"Model: {artifact.get('model_name')}")
    print(f"XGBoost available at training: {json.loads((ROOT/'outputs/reports/model_metrics.json').read_text()).get('xgboost_available_in_this_environment')}")
    print(f"SHAP background stored: {'shap_background' in artifact}")

    metrics = json.loads((ROOT/'outputs/reports/model_metrics.json').read_text())
    final = metrics['final_test_metrics_tuned_threshold']
    print(f"Final test Accuracy: {final['accuracy']:.3%}")
    print(f"Final test Recall:   {final['recall']:.3%}")
    print(f"Final test ROC-AUC:   {final['roc_auc']:.4f}")
    print(f"Latency: {metrics['single_prediction_latency_seconds']*1000:.2f} ms")
    print("Targets met:", metrics['targets_met'])

    if failures:
        print("AUDIT RESULT: FAIL", failures)
        raise SystemExit(1)
    print("AUDIT RESULT: PASS (with dataset-type warning if shown above)")

if __name__ == '__main__':
    main()
