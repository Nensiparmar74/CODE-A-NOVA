"""Evaluate the trained model on the held-out test split.  Usage: python -m src.evaluate"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    split = config.OUTPUT_DIR / "test_split.npz"
    if not split.exists() or not config.MODEL_PATH.exists():
        raise SystemExit("Run `python -m src.train` first.")

    data = np.load(split)
    X, y = data["X"], data["y"].astype(int)
    model = tf.keras.models.load_model(config.MODEL_PATH)

    prob = model.predict(X, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)

    report = classification_report(y, pred, target_names=config.CLASS_NAMES, digits=4)
    print(report)
    if len(set(y)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y, prob):.4f}")
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    print("Saved outputs/confusion_matrix.png and outputs/classification_report.txt")


if __name__ == "__main__":
    main()
