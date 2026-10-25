"""
CropYield Predictor - MLflow Experiment Tracking Module
Phase 18: Tracking model training runs, hyperparameters, CV metrics, and artifacts locally.
"""

import os
import json
import time

MLRUNS_DIR = "mlruns"
MODEL_COMPARISON_PATH = "models/model_comparison.json"
EVALUATION_METRICS_PATH = "models/evaluation_metrics.json"


def record_mlflow_runs():
    """
    Implements a robust local file-based experiment tracker adhering to the MLflow logging schema.
    Creates structured run directories under mlruns/ with params, metrics, and tags.
    """
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    experiment_id = "0"
    exp_dir = os.path.join(MLRUNS_DIR, experiment_id)
    os.makedirs(exp_dir, exist_ok=True)

    # Write experiment meta.yaml
    with open(os.path.join(exp_dir, "meta.yaml"), "w") as f:
        f.write(f"artifact_location: {os.path.abspath(exp_dir)}\n")
        f.write(f"creation_time: {int(time.time() * 1000)}\n")
        f.write(f"experiment_id: '{experiment_id}'\n")
        f.write("lifecycle_stage: active\n")
        f.write("name: CropYield-Predictor-Experiments\n")

    if not os.path.exists(MODEL_COMPARISON_PATH):
        print("Comparison metrics not found. Please run src/train.py first.")
        return

    with open(MODEL_COMPARISON_PATH) as f:
        comparison_runs = json.load(f)

    with open(EVALUATION_METRICS_PATH) as f:
        eval_metrics = json.load(f)

    print(f"Logging {len(comparison_runs)} model experiments to local MLflow store ({exp_dir})...")

    timestamp_ms = int(time.time() * 1000)

    for i, run in enumerate(comparison_runs):
        run_uuid = f"run_{i+1:03d}_{run['model'].lower().replace(' ', '_')}"
        run_dir = os.path.join(exp_dir, run_uuid)
        params_dir = os.path.join(run_dir, "params")
        metrics_dir = os.path.join(run_dir, "metrics")
        tags_dir = os.path.join(run_dir, "tags")

        os.makedirs(params_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(tags_dir, exist_ok=True)

        # Log parameters
        with open(os.path.join(params_dir, "model_architecture"), "w") as pf:
            pf.write(str(run['model']))
        with open(os.path.join(params_dir, "cv_folds"), "w") as pf:
            pf.write("5")

        # Log metrics
        for metric_name, val in [
            ("cv_r2_mean", run['cv_r2_mean']),
            ("cv_r2_std", run['cv_r2_std']),
            ("cv_mae", run['cv_mae']),
            ("cv_rmse", run['cv_rmse']),
            ("test_r2", run['test_r2']),
            ("test_mae", run['test_mae']),
            ("test_rmse", run['test_rmse']),
            ("training_time_sec", run['training_time_sec'])
        ]:
            with open(os.path.join(metrics_dir, metric_name), "w") as mf:
                mf.write(f"{timestamp_ms} {val} 0\n")

        # Log tags
        with open(os.path.join(tags_dir, "mlflow.runName"), "w") as tf:
            tf.write(f"{run['model']}_5Fold_CV")
        with open(os.path.join(tags_dir, "mlflow.user"), "w") as tf:
            tf.write("crop-data-scientist")

    # Also log the tuned final model run
    tuned_run_uuid = "run_tuned_final_random_forest"
    tuned_dir = os.path.join(exp_dir, tuned_run_uuid)
    os.makedirs(os.path.join(tuned_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(tuned_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(tuned_dir, "tags"), exist_ok=True)

    for p_name, p_val in eval_metrics.get("best_parameters", {}).items():
        with open(os.path.join(tuned_dir, "params", p_name), "w") as pf:
            pf.write(str(p_val))

    for m_name in ["test_mae", "test_rmse", "test_r2", "post_tuning_cv_r2", "improvement_pct"]:
        if m_name in eval_metrics:
            with open(os.path.join(tuned_dir, "metrics", m_name), "w") as mf:
                mf.write(f"{timestamp_ms} {eval_metrics[m_name]} 0\n")

    with open(os.path.join(tuned_dir, "tags", "mlflow.runName"), "w") as tf:
        tf.write("Tuned_Random_Forest_Production")

    print("MLflow logging completed successfully.")


if __name__ == "__main__":
    record_mlflow_runs()
