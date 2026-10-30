"""Convert the PlantVillage dataset into the binary layout used by this project.

Download PlantVillage (e.g. from Kaggle: "emmarex/plantdisease" or
"abdallahalidev/plantvillage-dataset"), unzip it, then run:

    python scripts/prepare_plantvillage.py /path/to/PlantVillage --per-class 2000

Any class folder whose name contains "healthy" -> data/healthy
All other class folders                          -> data/infected
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="PlantVillage root folder (contains class sub-folders)")
    ap.add_argument("--per-class", type=int, default=2000,
                    help="max images copied per output class (keeps classes balanced)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    healthy, infected = [], []
    for folder in sorted(p for p in args.src.rglob("*") if p.is_dir()):
        imgs = [f for f in folder.iterdir() if f.suffix.lower() in EXTS]
        if not imgs:
            continue
        (healthy if "healthy" in folder.name.lower() else infected).extend(imgs)

    print(f"Found {len(healthy)} healthy and {len(infected)} infected images")
    rng = random.Random(args.seed)
    for name, files in (("healthy", healthy), ("infected", infected)):
        rng.shuffle(files)
        out = ROOT / "data" / name
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files[: args.per_class]):
            shutil.copy2(f, out / f"{i:05d}_{f.name}")
        print(f"  copied {min(len(files), args.per_class)} -> {out}")


if __name__ == "__main__":
    main()
