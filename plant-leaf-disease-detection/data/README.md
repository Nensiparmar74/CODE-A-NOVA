# Dataset

Expected layout:

```
data/
├── healthy/    *.jpg
└── infected/   *.jpg
```

## Option A – real data (recommended): PlantVillage

1. Download PlantVillage from Kaggle (search "PlantVillage dataset") and unzip it.
2. Convert it to the binary layout:

   ```bash
   python scripts/prepare_plantvillage.py /path/to/PlantVillage --per-class 2000
   ```

## Option B – quick smoke test with synthetic images

```bash
python scripts/make_demo_data.py --n 200
```

These images are artificially drawn. They only prove the code runs; the accuracy
means nothing for real leaves.
