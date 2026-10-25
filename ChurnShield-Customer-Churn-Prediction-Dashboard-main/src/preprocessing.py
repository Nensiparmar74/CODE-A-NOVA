"""
preprocessing.py
-----------------
Data loading, cleaning, and the Scikit-Learn preprocessing pipeline
(ColumnTransformer) shared by training (notebooks/scripts) and the
Streamlit dashboard. Keeping this in one module guarantees the dashboard
transforms inputs EXACTLY the way training data was transformed.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "telco_clean.csv"
METADATA_PATH = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "dataset_metadata.json"

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = [ID_COL] + ALL_FEATURES + [TARGET_COL]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load and validate the raw Telco CSV required by the brief."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python src/download_dataset.py` "
            "to download the IBM Telco Customer Churn dataset from Kaggle."
        )
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco dataframe.

    Cleaning decisions (documented, per project brief requirement):
    1. `TotalCharges` is loaded as an object dtype because ~11 rows in the
       real dataset contain a blank string " " (new customers with tenure=0
       who haven't been billed yet). We coerce to numeric with errors='coerce',
       which turns those blanks into NaN, then impute them with 0 — a new
       customer legitimately has $0 total charges billed to date. This is a
       principled imputation, not a silent drop: the tenure=0 rows are kept.
    2. `SeniorCitizen` is loaded as int (0/1) in the source data; we keep it
       as a categorical-like binary flag rather than scaling it, since it is
       not a continuous quantity.
    3. Duplicate `customerID` rows (if any) are dropped, keeping the first
       occurrence, since a customer should appear once.
    4. Rows with a missing/invalid `Churn` label are dropped since the target
       must be known for supervised training. Any such removals are logged.
    5. `customerID` is retained in the cleaned file for traceability but is
       EXCLUDED from the model's feature set (it is a random identifier and
       would leak nothing useful and only harm generalization if a
       high-cardinality unique string were passed to OneHotEncoder).
    """
    df = df.copy()
    n_start = len(df)

    # --- Fix TotalCharges ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank_total_charges = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # --- Duplicates ---
    n_dupes = df.duplicated(subset=[ID_COL]).sum()
    df = df.drop_duplicates(subset=[ID_COL], keep="first")

    # --- Missing / invalid target ---
    before = len(df)
    df = df[df[TARGET_COL].isin(["Yes", "No"])]
    n_target_dropped = before - len(df)

    # --- Strip whitespace from all string/object columns (common CSV artifact) ---
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    n_end = len(df)
    print("=== Cleaning summary ===")
    print(f"Rows before cleaning: {n_start}")
    print(f"Blank TotalCharges coerced to NaN then imputed with 0: {n_blank_total_charges}")
    print(f"Duplicate customerID rows dropped: {n_dupes}")
    print(f"Rows dropped for missing/invalid Churn label: {n_target_dropped}")
    print(f"Rows after cleaning: {n_end}")

    return df.reset_index(drop=True)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn: No -> 0, Yes -> 1."""
    df = df.copy()
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used inside the full modeling Pipeline.
    - Numeric features: median-impute (safety net) + StandardScaler.
    - Categorical features: most-frequent-impute (safety net) + OneHotEncoder.
    This is fit ONLY on training data inside the Pipeline (never on the full
    dataset before the train/test split), preventing data leakage.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_target(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    # SeniorCitizen arrives as int 0/1 in the source data; treat as categorical string
    X["SeniorCitizen"] = X["SeniorCitizen"].astype(str)
    y = df[TARGET_COL].copy()
    return X, y


def run_cleaning_pipeline(save: bool = True) -> pd.DataFrame:
    df = load_raw()
    df = clean_data(df)
    df = encode_target(df)
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"Saved cleaned data to {PROCESSED_PATH}")
    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
