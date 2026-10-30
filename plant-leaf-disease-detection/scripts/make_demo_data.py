"""Generate a small SYNTHETIC leaf dataset so you can smoke-test the whole pipeline
(train -> evaluate -> predict -> app) without downloading anything.

This is NOT real plant data - do not report its accuracy. Use PlantVillage for real results.

    python scripts/make_demo_data.py --n 200
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_leaf(rng: np.random.Generator, infected: bool, size: int = 256) -> np.ndarray:
    img = np.full((size, size, 3), 245, np.uint8)           # light background (BGR)
    centre = (size // 2 + int(rng.integers(-10, 10)), size // 2 + int(rng.integers(-10, 10)))
    axes = (int(rng.integers(70, 100)), int(rng.integers(100, 118)))
    angle = float(rng.integers(0, 180))
    green = (int(rng.integers(30, 60)), int(rng.integers(120, 170)), int(rng.integers(30, 70)))
    cv2.ellipse(img, centre, axes, angle, 0, 360, green, -1)
    cv2.ellipse(img, centre, (axes[0] // 8, axes[1]), angle, 0, 360,
                tuple(int(c * 0.7) for c in green), -1)        # midrib
    if infected:
        for _ in range(int(rng.integers(6, 18))):
            dx, dy = rng.integers(-45, 45, 2)
            r = int(rng.integers(4, 14))
            colour = [(20, 60, 110), (30, 140, 190), (15, 30, 60)][int(rng.integers(0, 3))]
            cv2.circle(img, (centre[0] + int(dx), centre[1] + int(dy)), r, colour, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="images per class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for name, flag in (("healthy", False), ("infected", True)):
        out = ROOT / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            cv2.imwrite(str(out / f"demo_{i:04d}.jpg"), make_leaf(rng, flag))
        print(f"wrote {args.n} images to {out}")


if __name__ == "__main__":
    main()
