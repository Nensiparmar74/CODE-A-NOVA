"""Train the CNN.  Usage:  python -m src.train [--epochs 25] [--no-segment]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.model import build_model  # noqa: E402


def plot_history(history, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--no-segment", action="store_true",
                    help="skip leaf segmentation during preprocessing")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(segment=not args.no_segment)
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # save the test split so evaluate.py uses exactly the same held-out data
    np.savez_compressed(config.OUTPUT_DIR / "test_split.npz", X=X_test, y=y_test)

    weights = compute_class_weight("balanced", classes=np.array([0.0, 1.0]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATH), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    model.save(config.MODEL_PATH)
    plot_history(history, config.OUTPUT_DIR / "training_curves.png")
    with open(config.OUTPUT_DIR / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
