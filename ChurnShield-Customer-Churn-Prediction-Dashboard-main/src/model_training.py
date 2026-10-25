"""
model_training.py
-----------------
Leakage-safe training and evaluation pipeline for ChurnShield.

Workflow:
    raw data -> cleaning -> train/validation/test split -> 5-fold CV model
    comparison -> GridSearchCV on training data -> threshold tuning on
    validation data -> refit final pipeline on train+validation -> final
    held-out test evaluation -> save model artifact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, get_feature_target, run_cleaning_pipeline

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "outputs" / "reports" / "model_metrics.json"
FIG_DIR = ROOT / "outputs" / "figures"
RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def get_models() -> dict[str, object]:
    """Return the exact three models required by the internship brief."""
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is required for the internship submission. Install it with "
            "`pip install xgboost` and rerun training."
        )

    # scale_pos_weight is estimated after seeing the training split in main().
    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return models


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def cross_validate_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    report = {}
    for metric in scoring:
        values = results[f"test_{metric}"]
        report[f"cv_{metric}_mean"] = float(np.mean(values))
        report[f"cv_{metric}_std"] = float(np.std(values))
    return report


def tune_model(name: str, estimator: object, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict, float]:
    pipeline = build_pipeline(estimator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if name == "Logistic Regression":
        param_grid = {
            "classifier__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        }
    elif name == "Random Forest":
        param_grid = {
            "classifier__n_estimators": [250, 400],
            "classifier__max_depth": [8, 14, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        }
    else:
        param_grid = {
            "classifier__n_estimators": [300, 500],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.03, 0.05, 0.08],
            "classifier__subsample": [0.85, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def find_best_threshold(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, min_recall: float = 0.75) -> dict:
    """Select threshold on validation only; the test set remains untouched."""
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    rows = []
    for threshold in np.round(np.arange(0.20, 0.801, 0.01), 2):
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        })
    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates["recall"] >= min_recall]
    if eligible.empty:
        best = candidates.iloc[(candidates["recall"] - min_recall).abs().argsort()[:1]].iloc[0]
    else:
        best = eligible.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return best.to_dict()


def main() -> dict:
    print("Loading and cleaning data...")
    df = run_cleaning_pipeline(save=True)
    X, y = get_feature_target(df)

    # 60% train, 20% validation, 20% final held-out test.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
    )

    print(f"Train={len(X_train)}  Validation={len(X_val)}  Test={len(X_test)}")
    print(f"Train churn rate={y_train.mean():.3f}  Validation={y_val.mean():.3f}  Test={y_test.mean():.3f}")

    models = get_models()
    if "XGBoost" in models:
        positive = max(float(y_train.sum()), 1.0)
        negative = max(float((1 - y_train).sum()), 1.0)
        models["XGBoost"].set_params(scale_pos_weight=negative / positive)

    comparison = {}
    for name, estimator in models.items():
        print(f"\n--- Baseline: {name} ---")
        pipeline = build_pipeline(estimator)
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        cv_scores = cross_validate_model(build_pipeline(estimator), X_train, y_train)
        val_metrics = evaluate(pipeline, X_val, y_val, threshold=0.5)
        comparison[name] = {
            "fit_time_sec": float(fit_time),
            **cv_scores,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
        }
        print(
            f"CV ROC-AUC={cv_scores['cv_roc_auc_mean']:.4f}; "
            f"Validation ROC-AUC={val_metrics['roc_auc']:.4f}; "
            f"Validation Recall={val_metrics['recall']:.4f}"
        )

    # Model selection uses TRAINING CV only — never the held-out test set.
    best_baseline_name = max(
        comparison,
        key=lambda n: (comparison[n]["cv_roc_auc_mean"], comparison[n]["cv_recall_mean"]),
    )
    print(f"\n>>> Best baseline selected from CV only: {best_baseline_name}")

    tuned_pipeline, best_params, best_cv_auc = tune_model(
        best_baseline_name, models[best_baseline_name], X_train, y_train
    )
    print(f"Best params: {best_params}")
    print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")

    # Threshold is selected only on validation data.
    tuned_pipeline.fit(X_train, y_train)
    threshold_report = find_best_threshold(tuned_pipeline, X_val, y_val, min_recall=0.75)
    best_threshold = float(threshold_report["threshold"])
    print(f"Selected validation threshold: {best_threshold:.2f}")

    # Final model is refit on the full non-test partition.
    tuned_pipeline.fit(X_trainval, y_trainval)
    final_default = evaluate(tuned_pipeline, X_test, y_test, threshold=0.5)
    final_tuned = evaluate(tuned_pipeline, X_test, y_test, threshold=best_threshold)

    # Prediction latency requirement.
    sample = X_test.iloc[[0]]
    start = time.perf_counter()
    for _ in range(20):
        tuned_pipeline.predict_proba(sample)
    latency = (time.perf_counter() - start) / 20

    # Curves/figures.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

        y_test_proba = tuned_pipeline.predict_proba(X_test)[:, 1]
        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=ax)
        ax.set_title("Final Model ROC Curve")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "09_final_roc_curve.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        PrecisionRecallDisplay.from_predictions(y_test, y_test_proba, ax=ax)
        ax.set_title("Final Model Precision-Recall Curve")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "10_final_precision_recall_curve.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"Plot generation warning: {exc}")

    metadata_path = ROOT / "outputs" / "reports" / "dataset_metadata.json"
    dataset_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"dataset_type": "unknown"}

    report = {
        "dataset_name": "IBM Telco Customer Churn Dataset",
        "dataset_metadata": dataset_metadata,
        "dataset_expected_rows": 7043,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test), "random_state": RANDOM_STATE},
        "best_model_name": best_baseline_name,
        "xgboost_available_in_this_environment": XGBOOST_AVAILABLE,
        "model_selection_basis": "5-fold CV ROC-AUC on training split; validation used for threshold only; test used once for final evaluation.",
        "model_comparison": comparison,
        "best_hyperparameters": best_params,
        "best_cv_roc_auc_during_search": best_cv_auc,
        "validation_threshold_selection": threshold_report,
        "final_test_metrics_default_threshold_0.5": final_default,
        "final_test_metrics_tuned_threshold": final_tuned,
        "single_prediction_latency_seconds": float(latency),
        "targets": {"accuracy_target": 0.80, "recall_target": 0.75, "roc_auc_target": 0.85},
        "targets_met": {
            "accuracy": final_tuned["accuracy"] > 0.80,
            "recall": final_tuned["recall"] > 0.75,
            "roc_auc": final_tuned["roc_auc"] > 0.85,
        },
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))

    artifact = {
        "pipeline": tuned_pipeline,
        "threshold": best_threshold,
        "feature_columns": list(X.columns),
        "model_name": best_baseline_name,
        "dataset_name": "IBM Telco Customer Churn Dataset",
        "training_rows": len(X_trainval),
        "random_state": RANDOM_STATE,
        "shap_background": X_trainval.sample(min(100, len(X_trainval)), random_state=RANDOM_STATE).copy(),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved model:   {MODEL_PATH}")
    print("Final test metrics:")
    print(json.dumps(final_tuned, indent=2))
    return report


if __name__ == "__main__":
    main()
