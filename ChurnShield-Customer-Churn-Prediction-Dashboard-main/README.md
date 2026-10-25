# 🛡️ ChurnShield — Customer Churn Prediction Dashboard

ChurnShield is an end-to-end Data Science internship project for predicting telecom customer churn, identifying churn-risk drivers, explaining predictions with SHAP, and serving live predictions through Streamlit.

## Internship Brief Alignment

The implementation follows the brief's required flow:

`Raw CSV → Cleaning → EDA → Feature Engineering → Train/Validation/Test → Logistic Regression → Random Forest → XGBoost → Evaluation → SHAP → Joblib → Streamlit`

Required deliverables are covered by the repository: three Jupyter notebooks, a trained `.joblib` artifact, `app.py`, SHAP exports, `FINAL_INSIGHTS.md`, `README.md`, and `requirements.txt`.

## Dataset

This final archive includes the **official IBM Telco Customer Churn Dataset** from the specified Kaggle source. Kaggle documents 7,043 customer rows and 21 columns, with `Churn` as the target. citeturn911454search5turn911454search7

Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The included `src/download_dataset.py` can re-download and validate the same public dataset when internet access is available.

## What was fixed in this final build

- Model selection no longer uses the held-out test set. Selection is based on 5-fold CV on the training partition; validation is used only for decision-threshold selection; the test set is used once for final evaluation.
- XGBoost is a required model for the final run; the previous silent/fallback HistGradientBoosting path has been removed.
- SHAP is a required final deliverable. Real global and local SHAP explanations are generated when the listed dependencies are installed. SHAP's official documentation supports `TreeExplainer` for XGBoost/tree models and `LinearExplainer` for linear models. citeturn718635search0turn718635search4
- The saved model artifact now also stores a training-only SHAP background sample, keeping dashboard explanations consistent with the fitted model.
- Streamlit loads the persisted model through `st.cache_resource`, which is intended for resource objects such as machine-learning models. citeturn718635search1
- A final automated submission audit is provided in `src/submission_check.py`.

## Project Structure

```text
ChurnShield/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── telco_clean.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
├── models/
│   └── churn_model.joblib
├── outputs/
│   ├── figures/
│   ├── shap/
│   └── reports/
├── src/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── generate_synthetic_data.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── explainability.py
│   ├── submission_check.py
│   └── utils.py
├── app.py
├── RUN_ALL.ps1
├── requirements.txt
├── FINAL_INSIGHTS.md
├── README.md
└── .gitignore
```

## Technology Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Scikit-Learn · XGBoost · SHAP · Streamlit · Joblib · Jupyter

Scikit-learn's `ColumnTransformer` is used to transform numeric and categorical columns separately, while the preprocessing is kept inside the model `Pipeline` to avoid fitting transforms before the split. citeturn102014search1

## Windows / VS Code Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Final Dataset + Full Run

**1. Download and validate the official Kaggle dataset:**

```powershell
python src/download_dataset.py
```

**2. Train/evaluate/save the model:**

```powershell
python src/model_training.py
```

**3. Generate real SHAP outputs:**

```powershell
python src/explainability.py
```

**4. Run the final audit:**

```powershell
python src/submission_check.py
```

**5. Launch the dashboard:**

```powershell
streamlit run app.py
```

Or run the supplied PowerShell orchestration script:

```powershell
.\RUN_ALL.ps1
```

## Model Methodology

### Cleaning

- `TotalCharges` is coerced to numeric.
- Blank `TotalCharges` values are retained and imputed safely.
- Duplicate customer IDs are removed.
- Invalid/missing target labels are excluded.
- `customerID` is retained for traceability but excluded from model features.

### EDA

The EDA notebook examines churn distribution and business-relevant relationships including contract, tenure, monthly charges, internet service, payment method, support/security services, and correlations.

### Feature Engineering

Numeric features:
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Categorical features are one-hot encoded with `handle_unknown="ignore"`.
Numeric variables are median-imputed and standardized. All preprocessing is part of the same Scikit-Learn `Pipeline` used for model fitting and inference.

### Models

The brief requires:

1. Logistic Regression baseline
2. Random Forest
3. XGBoost

Five-fold stratified cross-validation on the training data is used for comparison. The best candidate is tuned with `GridSearchCV` using ROC-AUC.

### Threshold tuning

A validation partition is used to choose a probability threshold that keeps recall at or above the brief's 75% target when possible while maximizing F1 among eligible thresholds. The final test set remains untouched until the final evaluation.

### Evaluation

The project reports:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Classification report

Official success indicators from the brief are **Accuracy >80%, Recall >75%, and AUC-ROC >0.85**. These values must be verified on the actual Kaggle dataset; the project never hardcodes or fabricates metrics.

## SHAP Explainability

The final submission uses real SHAP, not permutation-importance substitution. Global SHAP importance is exported to `outputs/shap/global_shap_importance.png` and `.csv`. The Streamlit prediction tab also provides local per-customer positive and negative SHAP contributors.

SHAP documents `TreeExplainer` as a fast/exact explainer for tree ensembles including XGBoost, and `LinearExplainer` for linear models such as Logistic Regression. citeturn718635search0turn718635search4

## Streamlit Dashboard

The dashboard provides:
- customer profile inputs
- churn probability
- model prediction
- Low / Medium / High risk communication
- probability gauge
- live SHAP explanation
- key churn-driver visualizations
- retention suggestions
- model performance information
- prediction latency display

The ML pipeline is loaded once through Streamlit's resource cache and is not retrained on every interaction. citeturn718635search1

## Final Run Status

This archive contains the official IBM Telco Customer Churn Dataset from the specified Kaggle source. The full pipeline was executed on that packaged dataset.

Measured held-out test results at the selected threshold are:
- Accuracy: **73.31%**
- Recall: **79.14%**
- ROC-AUC: **0.8415**
- XGBoost: **trained and compared**
- SHAP: **generated successfully**
- Prediction latency: **~5.5 ms** in the build environment

The internship brief's success indicators are Accuracy >80%, Recall >75%, and AUC-ROC >0.85. The project reports the real measured values and does not fabricate or hardcode performance.

## References

- XGBoost: https://xgboost.readthedocs.io/en/stable/
- Scikit-Learn: https://scikit-learn.org/stable/user_guide.html
- SHAP: https://shap.readthedocs.io/en/latest/
- Streamlit: https://docs.streamlit.io/
- IBM Telco Customer Churn (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
