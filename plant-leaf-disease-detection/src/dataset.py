"""Load the dataset from data/healthy and data/infected and split it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocessing import preprocess_path  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(data_dir: Path = config.DATA_DIR):
    paths, labels = [], []
    for label, name in enumerate(config.CLASS_NAMES):
        folder = data_dir / name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder {folder}. See data/README.md for how to get the dataset."
            )
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                paths.append(p)
                labels.append(label)
    if not paths:
        raise RuntimeError("No images found in data/healthy or data/infected.")
    return paths, np.array(labels)


def load_dataset(data_dir: Path = config.DATA_DIR, segment: bool = True):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays."""
    paths, labels = list_images(data_dir)
    print(f"Found {len(paths)} images "
          f"({int((labels == 0).sum())} healthy, {int((labels == 1).sum())} infected)")

    X = np.stack([preprocess_path(p, segment=segment) for p in paths])
    y = labels.astype("float32")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT, stratify=y, random_state=config.SEED
    )
    val_rel = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=config.SEED
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
