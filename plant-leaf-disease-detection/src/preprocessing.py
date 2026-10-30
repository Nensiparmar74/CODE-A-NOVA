"""Image-processing pipeline (classical computer vision) for leaf images.

Steps:
  1. Resize to a fixed size
  2. Denoise with a Gaussian blur
  3. Improve contrast with CLAHE on the L channel (LAB colour space)
  4. Segment the leaf from the background using HSV thresholding + morphology
  5. (Optional) compute simple hand-crafted features such as the fraction of
     "diseased-looking" pixels (brown / yellow spots) inside the leaf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_image(path: str | Path, size: int = config.IMG_SIZE) -> np.ndarray:
    """Read an image from disk as RGB uint8, resized to size x size."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def denoise(img_rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)


def enhance_contrast(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel so colour information is preserved."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def segment_leaf(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (segmented_rgb, mask). Mask is uint8 {0, 255}.

    A leaf is assumed to be green / yellow / brown foliage on a background that is
    comparatively unsaturated (white / grey / black). We keep pixels with enough
    saturation and value, clean the mask with morphology and keep the largest blob.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    mask = ((s > 40) & (v > 40)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # keep the largest connected component (the leaf)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8) * 255
    else:
        mask = np.full_like(mask, 255)  # fallback: keep everything

    # fill holes so that spots inside the leaf stay part of it
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=filled)
    return segmented, filled


def diseased_area_ratio(img_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Estimate the share of the leaf that looks diseased (non-green foliage).

    Healthy tissue is green (hue ~ 35-85 in OpenCV's 0-179 scale). Within the leaf
    mask, saturated pixels outside that band (yellow, brown, dark necrotic spots)
    are counted as potentially diseased. This is a heuristic used for visual
    explanation in the app, not for the CNN decision itself.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    leaf = mask > 0
    if leaf.sum() == 0:
        return 0.0
    green = (h >= 35) & (h <= 85) & (s > 40)
    dark = v < 60
    diseased = leaf & (~green | dark)
    return float(diseased.sum() / leaf.sum())


def diseased_overlay(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the image with suspicious regions highlighted in red."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 35) & (h <= 85) & (s > 40)
    diseased = (mask > 0) & (~green | (v < 60))
    out = img_rgb.copy()
    out[diseased] = (0.4 * out[diseased] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    return out


def preprocess_for_model(img_rgb: np.ndarray, segment: bool = True) -> np.ndarray:
    """Full pipeline -> float32 array in [0, 1] of shape (H, W, 3)."""
    img = denoise(img_rgb)
    img = enhance_contrast(img)
    if segment:
        img, _ = segment_leaf(img)
    return img.astype("float32") / 255.0


def preprocess_path(path: str | Path, segment: bool = True) -> np.ndarray:
    return preprocess_for_model(load_image(path), segment=segment)
