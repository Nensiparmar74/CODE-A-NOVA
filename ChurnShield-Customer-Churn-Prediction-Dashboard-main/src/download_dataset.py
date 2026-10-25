"""Download the public IBM Telco Customer Churn dataset used by the internship brief."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
}


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    req = Request(DATASET_URL, headers={"User-Agent": "ChurnShield-Internship/1.0"})
    with urlopen(req, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("Kaggle download did not contain a CSV file.")
        target = next((n for n in csv_names if Path(n).name == RAW_FILE.name), csv_names[0])
        RAW_FILE.write_bytes(zf.read(target))

    import json
    import pandas as pd
    df = pd.read_csv(RAW_FILE)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing or len(df) != 7043:
        raise RuntimeError(
            f"Downloaded file failed validation: rows={len(df)}, missing_columns={sorted(missing)}"
        )
    meta_path = ROOT / "outputs" / "reports" / "dataset_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "dataset_type": "ibm_kaggle",
        "source": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "filename": RAW_FILE.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }, indent=2))
    print(f"Downloaded and validated {len(df):,} rows x {len(df.columns)} columns.")
    print(f"Saved to: {RAW_FILE}")
    return RAW_FILE


if __name__ == "__main__":
    download()
