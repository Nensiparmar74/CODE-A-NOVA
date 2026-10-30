"""Self-designed CNN for binary leaf classification (healthy vs infected)."""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return x


def build_model(img_size: int = config.IMG_SIZE) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # light on-the-fly augmentation (active only during training)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    for f in (32, 64, 128, 256):
        x = conv_block(x, f)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)   # P(infected)

    model = models.Model(inputs, outputs, name="leaf_disease_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
