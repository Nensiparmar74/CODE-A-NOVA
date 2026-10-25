# ChurnShield — Final Insights Report

## 1. Business Problem

Telecom companies need to identify customers who are likely to leave so retention teams can prioritize proactive engagement. ChurnShield frames this as a supervised binary classification problem where `Churn=1` represents a customer who left and `Churn=0` represents a retained customer.

## 2. Dataset

This final archive uses the official IBM Telco Customer Churn Dataset downloaded from the specified Kaggle source. It contains 7,043 customer records and 21 columns; the `Churn` column is the binary target. citeturn911454search5turn911454search7

## 3. Data Cleaning

The cleaning workflow:

- converts `TotalCharges` to numeric;
- converts blank/non-numeric `TotalCharges` entries to missing and imputes them safely;
- checks and removes duplicate customer IDs;
- removes rows with invalid target labels;
- strips incidental string whitespace;
- excludes the high-cardinality `customerID` from the model feature matrix while retaining it in the cleaned file for traceability.

## 4. Exploratory Data Analysis

The EDA workflow studies churn distribution and churn-rate differences across important customer groups. In the official Kaggle dataset used for this run, the strongest visible patterns include contract type, tenure, monthly charges, internet service, payment method, and support/security services.

These are **associations in the observed data, not causal claims**.

## 5. Feature Engineering

The model separates:

**Numeric:** `tenure`, `MonthlyCharges`, `TotalCharges`

**Categorical:** customer/service/contract/payment attributes

Numeric variables are median-imputed and standardized. Categorical variables are imputed with the most frequent value and one-hot encoded. The transformations are fitted inside a Scikit-Learn `Pipeline` / `ColumnTransformer`, preventing preprocessing leakage into the held-out evaluation data. `ColumnTransformer` is specifically designed for applying different transformations to different column subsets. citeturn102014search1

## 6. Model Comparison

The required models are:

1. Logistic Regression
2. Random Forest
3. XGBoost

Model comparison is based on five-fold stratified cross-validation over the training partition. The final test set is **not** used for model selection.

The final tuned model is selected from training/CV performance and the validation set is reserved for decision-threshold tuning.

## 7. Evaluation Targets

The internship brief specifies:

- Accuracy > 80%
- Recall > 75%
- AUC-ROC > 0.85

The code measures these values honestly and records whether each target is met. No performance value is hardcoded.

### Final official-dataset run

The current archive was executed on the official Kaggle dataset and produced:

| Metric | Result |
|---|---:|
| Accuracy | 73.31% |
| Precision | 49.83% |
| Recall | 79.14% |
| F1 | 61.16% |
| ROC-AUC | 0.8415 |
| Prediction latency | ~5.5 ms |

Therefore, in this official-dataset run:

- Recall target: **met**
- Accuracy target: **not met**
- ROC-AUC target: **not met**

These metrics are from the official Kaggle dataset packaged in this archive and are recorded in `outputs/reports/model_metrics.json`.

## 8. Explainability

The final implementation uses real SHAP.

For tree models such as XGBoost, SHAP's `TreeExplainer` provides fast and exact SHAP values under the documented assumptions. For linear models such as Logistic Regression, SHAP provides `LinearExplainer`. citeturn718635search0turn718635search4

The final official-dataset SHAP output ranks transformed features with the strongest mean absolute contributions around:

1. tenure
2. MonthlyCharges
3. Month-to-month contract
4. Two-year contract
5. Electronic check payment
6. Fiber optic internet
7. TotalCharges
8. OnlineSecurity
9. Partner
10. Dependents

These are model-attribution results for the fitted final model included in this archive.

## 9. Dashboard

The Streamlit app loads a saved model artifact rather than retraining at runtime. Streamlit's `st.cache_resource` is intended for caching resource objects such as machine-learning models. citeturn718635search1

The dashboard provides:

- customer profile controls
- churn probability
- prediction and risk category
- probability gauge
- local SHAP explanation
- churn-driver charts
- retention suggestions
- model metrics and threshold information
- prediction latency

## 10. Business Interpretation

A high predicted churn probability should be treated as a prioritization signal for retention workflows. Potential actions can include contract incentives, onboarding support, service support offers, and review of price-to-value positioning. The exact intervention should depend on the customer's situation and should not be treated as guaranteed causality from the model.

## 11. Limitations

- Model performance depends on the train/test split and model configuration; the metrics reported here are the measured results for the packaged official dataset and held-out test split.
- SHAP explains model behavior; it does not establish causal business effects.
- A classification threshold is a business decision layer and can trade recall against precision.

## 12. Future Improvements

- threshold optimization against measured retention-team capacity;
- calibrated probabilities;
- stronger cross-validation reporting and confidence intervals;
- monitoring for data drift;
- periodic retraining on newly labeled churn outcomes;
- segment-specific retention strategies.

## 13. Conclusion

ChurnShield implements the requested end-to-end architecture: data cleaning, EDA, leakage-safe feature engineering, model comparison, tuning, final evaluation, SHAP explainability, model persistence, and an interactive Streamlit dashboard.

The implementation and official-dataset run are complete. The final test results are reported honestly: recall meets the internship target, while accuracy and ROC-AUC remain below the brief thresholds on this held-out test split.
