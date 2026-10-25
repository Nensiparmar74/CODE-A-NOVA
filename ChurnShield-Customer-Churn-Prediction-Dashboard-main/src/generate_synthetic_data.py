"""
generate_synthetic_data.py
---------------------------
Generates a synthetic dataset that exactly matches the column schema and
realistic statistical patterns of the IBM Telco Customer Churn dataset
(Kaggle: blastchar/telco-customer-churn).

WHY THIS EXISTS:
This project was built in an offline sandbox with no internet access, so the
real Kaggle CSV could not be downloaded here. This script produces a
same-schema, same-style dataset with realistic churn drivers (month-to-month
contracts, low tenure, high monthly charges, fiber optic + no tech support,
electronic check payment -> higher churn probability) so the ENTIRE pipeline
(cleaning, EDA, feature engineering, modeling, SHAP, dashboard) is fully
functional end-to-end.

TO USE THE REAL DATASET INSTEAD:
Download WA_Fn-UseC_-Telco-Customer-Churn.csv from
https://www.kaggle.com/datasets/blastchar/telco-customer-churn and place it at
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv, overwriting the synthetic file.
No other code changes are required — every script reads that same path and
that same column schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_CUSTOMERS = 7043  # matches real dataset size

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def generate(n=N_CUSTOMERS, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"{rng.integers(1000,9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)]

    gender = rng.choice(["Male", "Female"], n)
    senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.30, 0.70])

    tenure = rng.integers(0, 73, n)

    phone_service = rng.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def dependent_internet_feature(p_yes_if_dsl_fiber=0.35):
        vals = np.empty(n, dtype=object)
        for i in range(n):
            if internet_service[i] == "No":
                vals[i] = "No internet service"
            else:
                vals[i] = rng.choice(["Yes", "No"], p=[p_yes_if_dsl_fiber, 1 - p_yes_if_dsl_fiber])
        return vals

    online_security = dependent_internet_feature(0.35)
    online_backup = dependent_internet_feature(0.40)
    device_protection = dependent_internet_feature(0.40)
    tech_support = dependent_internet_feature(0.35)
    streaming_tv = dependent_internet_feature(0.45)
    streaming_movies = dependent_internet_feature(0.45)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless_billing = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # ---- Monthly charges: built from service add-ons (realistic, not random) ----
    base = np.where(internet_service == "No", 20.0,
             np.where(internet_service == "DSL", 45.0, 70.0))
    addon_cost = 5.0
    for addon in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        base = base + np.where(addon == "Yes", addon_cost, 0.0)
    phone_cost = np.where(phone_service == "Yes", 5.0, 0.0) + np.where(multiple_lines == "Yes", 5.0, 0.0)
    monthly_charges = np.round(base + phone_cost + rng.normal(0, 3, n), 2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 15, n), 2)
    total_charges = np.clip(total_charges, 0, None)
    total_charges_str = total_charges.astype(str)
    # Real dataset quirk: brand-new customers (tenure=0) sometimes have blank TotalCharges
    blank_mask = tenure == 0
    total_charges_str = np.where(blank_mask, " ", total_charges_str)

    # ---- Churn probability model (the "ground truth" driving churn) ----
    logit = (
        -1.6
        + 1.55 * (contract == "Month-to-month")
        + 0.35 * (contract == "One year")
        + 0.0 * (contract == "Two year")
        - 0.045 * tenure
        + 0.018 * (monthly_charges - 60)
        + 0.55 * (internet_service == "Fiber optic")
        - 0.55 * (internet_service == "No")
        + 0.45 * (tech_support == "No") * (internet_service != "No")
        + 0.35 * (online_security == "No") * (internet_service != "No")
        + 0.40 * (payment_method == "Electronic check")
        + 0.25 * (paperless_billing == "Yes")
        + 0.20 * (senior_citizen == 1)
        - 0.20 * (partner == "Yes")
        - 0.15 * (dependents == "Yes")
        + rng.normal(0, 0.55, n)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob_churn, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges_str,
        "Churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    meta_path = OUT_PATH.parents[2] / "outputs" / "reports" / "dataset_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    meta_path.write_text(json.dumps({
        "dataset_type": "synthetic_demo",
        "source": "Generated locally for offline development only",
        "filename": OUT_PATH.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }, indent=2))
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df["Churn"].value_counts(normalize=True))
