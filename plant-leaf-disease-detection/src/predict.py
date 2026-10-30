"""Predict on one or more images.  Usage: python -m src.predict path/to/leaf.jpg [more.jpg ...]"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocessing import preprocess_for_model, load_image  # noqa: E402

_model = None


def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("No trained model found. Run `python -m src.train` first.")
        _model = tf.keras.models.load_model(config.MODEL_PATH)
    return _model


def predict_array(img_rgb: np.ndarray, segment: bool = True) -> tuple[str, float]:
    """img_rgb: uint8 RGB of size IMG_SIZE. Returns (label, confidence)."""
    x = preprocess_for_model(img_rgb, segment=segment)[None, ...]
    p_infected = float(get_model().predict(x, verbose=0)[0, 0])
    label = config.CLASS_NAMES[int(p_infected >= 0.5)]
    confidence = p_infected if label == "infected" else 1.0 - p_infected
    return label, confidence


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        label, conf = predict_array(load_image(path))
        print(f"{path}: {label} ({conf:.1%})")


if __name__ == "__main__":
    main()
