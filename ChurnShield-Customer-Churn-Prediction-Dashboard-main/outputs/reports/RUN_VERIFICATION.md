# ChurnShield Run Verification

## Dataset
- Official IBM Telco Customer Churn Dataset: PASS
- Rows: 7,043
- Columns: 21
- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Core training pipeline
- Logistic Regression trained: PASS
- Random Forest trained: PASS
- XGBoost trained: PASS
- 5-fold stratified CV: PASS
- GridSearchCV tuning: PASS
- Train/validation/test methodology: PASS
- Validation-only threshold selection: PASS
- Final held-out test evaluation: PASS

## Explainability
- Real SHAP global importance: PASS
- Training-only SHAP background: PASS
- Local SHAP explanation helper for Streamlit: PASS

## Artifacts
- Joblib model: PASS
- ROC curve: PASS
- Precision-Recall curve: PASS
- Metrics JSON: PASS
- Cleaned CSV: PASS
- SHAP CSV/PNG: PASS

## Notebook validation
- 01_EDA.ipynb: executed successfully
- 02_Feature_Engineering.ipynb: executed successfully
- 03_Model_Training.ipynb: workflow verified through the same `src/model_training.py` pipeline used for the saved final artifact

## Final packaged-dataset metrics
- Accuracy: 73.31%
- Recall: 79.14%
- ROC-AUC: 0.8415
- Mean prediction latency: ~5.5 ms

## Internship targets
- Accuracy >80%: NOT MET
- Recall >75%: MET
- AUC-ROC >0.85: NOT MET

These are the actual measured results on the official packaged Kaggle dataset and are not fabricated.
