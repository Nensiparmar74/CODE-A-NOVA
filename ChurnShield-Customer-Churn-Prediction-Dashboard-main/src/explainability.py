"""
Explainability utilities for ChurnShield.

The internship brief specifically requires SHAP. This module therefore
REQUIRES SHAP for the final submission. It supports the most likely final
estimators (Logistic Regression, Random Forest, XGBoost) and produces:
- global SHAP importance CSV + PNG
- a reusable local explanation helper for Streamlit
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
SHAP_DIR = ROOT / "outputs" / "shap"


def _require_shap() -> None:
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required for the internship deliverable. Run `pip install shap`.")


def get_transformed_feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def transform_dense(pipeline, X: pd.DataFrame) -> np.ndarray:
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=float)


def build_explainer(pipeline, background_transformed: np.ndarray):
    _require_shap()
    classifier = pipeline.named_steps["classifier"]
    name = type(classifier).__name__
    if name == "LogisticRegression":
        return shap.LinearExplainer(classifier, background_transformed)
    if name in {"RandomForestClassifier", "XGBClassifier"}:
        return shap.TreeExplainer(classifier, background_transformed)
    return shap.Explainer(classifier.predict_proba, background_transformed)


def _positive_class_values(shap_values):
    values = shap_values
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def global_shap(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    _require_shap()
    bg = transform_dense(pipeline, X_background)
    explain = transform_dense(pipeline, X_explain)
    explainer = build_explainer(pipeline, bg)
    raw_values = explainer.shap_values(explain) if hasattr(explainer, "shap_values") else explainer(explain).values
    values = _positive_class_values(raw_values)
    feature_names = get_transformed_feature_names(pipeline)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    importance.head(max_display).to_csv(SHAP_DIR / "global_shap_importance.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = importance.head(max_display).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_title("Global SHAP Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(SHAP_DIR / "global_shap_importance.png", dpi=150)
    plt.close(fig)
    return importance


def explain_local(pipeline, X_customer: pd.DataFrame, X_background: pd.DataFrame, top_n: int = 8):
    _require_shap()
    bg = transform_dense(pipeline, X_background)
    customer = transform_dense(pipeline, X_customer)
    explainer = build_explainer(pipeline, bg)
    raw_values = explainer.shap_values(customer) if hasattr(explainer, "shap_values") else explainer(customer).values
    values = _positive_class_values(raw_values)[0]
    names = get_transformed_feature_names(pipeline)
    contributions = pd.DataFrame({"feature": names, "shap_value": values})
    positive = contributions[contributions["shap_value"] > 0].sort_values("shap_value", ascending=False).head(top_n)
    negative = contributions[contributions["shap_value"] < 0].sort_values("shap_value", ascending=True).head(top_n)
    return positive, negative, contributions


def generate_global_outputs() -> pd.DataFrame:
    _require_shap()
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    from src.preprocessing import get_feature_target, run_cleaning_pipeline

    df = run_cleaning_pipeline(save=False)
    X, _ = get_feature_target(df)
    X_trainval = X
    # Recreate the same 80/20 train+validation vs held-out test split used during training.
    from sklearn.model_selection import train_test_split
    X_trainval, _ = train_test_split(X, test_size=0.20, stratify=df["Churn"], random_state=42)
    X_background = X_trainval.sample(min(100, len(X_trainval)), random_state=42)
    X_explain = X_trainval.sample(min(500, len(X_trainval)), random_state=123)
    importance = global_shap(pipeline, X_background, X_explain)
    print(importance.head(20).to_string(index=False))
    return importance


if __name__ == "__main__":
    generate_global_outputs()
