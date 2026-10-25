# ChurnShield — Submission Status

## Completed in the supplied project and corrected in this build

- Leakage-safe preprocessing with Scikit-Learn Pipeline / ColumnTransformer
- EDA notebook and saved charts
- Feature-engineering notebook
- Logistic Regression baseline
- Random Forest comparison
- XGBoost comparison (no fallback model)
- 5-fold stratified CV
- GridSearchCV tuning
- Train/validation/test methodology where the test set is used only for final evaluation
- Recall-oriented validation threshold selection
- Final evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and PR curves
- Real SHAP global feature importance
- Real SHAP local explanation for the Streamlit prediction page
- Joblib model artifact with training-only SHAP background sample
- Streamlit dashboard
- requirements.txt
- README.md
- FINAL_INSIGHTS.md
- automated submission audit

## Final dataset compliance

The packaged CSV is the official IBM Telco Customer Churn Dataset from the specified Kaggle source (7,043 rows × 21 columns), and the training/evaluation artifacts in this archive were generated from that dataset.

For a fresh local rebuild, run:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_dataset.py
python src/model_training.py
python src/explainability.py
python src/submission_check.py
streamlit run app.py
```
