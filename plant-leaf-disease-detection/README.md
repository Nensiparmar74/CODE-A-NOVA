# Plant Leaf Disease Detection

An image-based automatic inspection system that classifies plant leaves as **healthy** or **infected**.
It combines a classical image-processing pipeline (OpenCV) with a self-designed Convolutional Neural Network (TensorFlow/Keras) and a Streamlit web interface.

## How it works

```
Leaf photo
   │
   ├─ Resize (128×128)
   ├─ Gaussian blur            (noise removal)
   ├─ CLAHE on LAB L-channel   (contrast enhancement)
   ├─ HSV threshold + morphology + largest blob  (leaf segmentation)
   │
   └─ Custom CNN (4 conv blocks → GAP → Dense → sigmoid)  →  healthy / infected
```

The app also shows a heuristic "non-green area" overlay that highlights suspicious (yellow/brown/dark) regions.

## Project structure

```
app.py                       Streamlit interface
config.py                    Paths and hyper-parameters
requirements.txt
src/preprocessing.py         Image-processing pipeline
src/model.py                 CNN architecture
src/dataset.py               Data loading and train/val/test split
src/train.py                 Training script
src/evaluate.py              Metrics, confusion matrix, ROC-AUC
src/predict.py               Command-line prediction
scripts/prepare_plantvillage.py   Convert PlantVillage → binary layout
scripts/make_demo_data.py         Synthetic images for a smoke test
data/README.md               Dataset instructions
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. **Get data** – see `data/README.md` (PlantVillage recommended).
2. **Train**
   ```bash
   python -m src.train --epochs 25
   ```
   Saves `models/leaf_cnn.keras`, `outputs/training_curves.png` and the held-out test split.
3. **Evaluate**
   ```bash
   python -m src.evaluate
   ```
   Prints precision/recall/F1 and ROC-AUC, saves `outputs/confusion_matrix.png`.
4. **Predict one image**
   ```bash
   python -m src.predict path/to/leaf.jpg
   ```
5. **Launch the web app**
   ```bash
   streamlit run app.py
   ```

## Notes

- Class imbalance is handled with balanced class weights.
- Use `--no-segment` in training if your photos have no clean background; if you do, also set `segment=False` in `predict_array` calls.
- PlantVillage images have uniform backgrounds, so real-world field photos may score lower. Add field images to `data/` for a more robust model.
- Possible extensions: multi-class disease identification, Grad-CAM explanations, transfer learning (MobileNetV2), mobile deployment with TFLite.
