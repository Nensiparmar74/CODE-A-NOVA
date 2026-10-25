"""
utils.py
--------
Small shared helpers used by both the training scripts and the Streamlit
dashboard: risk-category bucketing and business-insight text generation.
Keeping this logic in one place guarantees notebooks and the dashboard
apply identical rules.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs" / "reports" / "model_metrics.json"


def load_metrics_report():
    with open(METRICS_PATH) as f:
        return json.load(f)


def risk_category(probability: float) -> str:
    """
    Documented threshold strategy for the dashboard's risk label.
    These bands are a business-communication convenience layered ON TOP of
    the model's raw probability output and decision threshold — they do not
    change the underlying prediction, only how it's presented to a
    non-technical user:
      < 0.30            -> Low Risk
      0.30 - 0.60        -> Medium Risk
      > 0.60            -> High Risk
    """
    if probability < 0.30:
        return "Low Risk"
    elif probability <= 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def retention_suggestions(customer_dict: dict) -> list:
    """
    Rule-based retention suggestions grounded in the EDA/SHAP churn drivers.
    Phrased as association, never causation, per project constraints.
    """
    tips = []
    if customer_dict.get("Contract") == "Month-to-month":
        tips.append("Month-to-month contracts are associated with higher predicted churn risk — "
                     "consider offering an incentive to move to a 1- or 2-year contract.")
    if customer_dict.get("tenure", 99) < 12:
        tips.append("Low tenure (under 12 months) is associated with higher churn risk — "
                     "early-lifecycle engagement or onboarding check-ins may help.")
    try:
        if float(customer_dict.get("MonthlyCharges", 0)) > 80:
            tips.append("High monthly charges are associated with higher churn risk — "
                         "a loyalty discount or bundled-service review could reduce price sensitivity.")
    except (TypeError, ValueError):
        pass
    if customer_dict.get("TechSupport") == "No" and customer_dict.get("InternetService") != "No":
        tips.append("Customers without Tech Support show higher churn risk — "
                     "proactively offering a free trial of Tech Support may help retention.")
    if customer_dict.get("OnlineSecurity") == "No" and customer_dict.get("InternetService") != "No":
        tips.append("Lack of Online Security add-on is associated with elevated churn risk — "
                     "bundling basic security could increase perceived value.")
    if customer_dict.get("InternetService") == "Fiber optic":
        tips.append("Fiber optic customers show a higher churn rate in this data than DSL customers — "
                     "worth investigating service reliability or price-to-value perception.")
    if customer_dict.get("PaymentMethod") == "Electronic check":
        tips.append("Electronic check payers are associated with higher churn — "
                     "encouraging automatic payment methods (bank transfer/credit card) may correlate with retention.")
    if not tips:
        tips.append("No major risk drivers detected for this customer profile based on the "
                     "modeled churn factors — maintain standard engagement.")
    return tips
