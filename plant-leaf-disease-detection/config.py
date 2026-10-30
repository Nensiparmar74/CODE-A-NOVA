"""Central configuration for the Plant Leaf Disease Detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"          # expects data/healthy/*.jpg and data/infected/*.jpg
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "leaf_cnn.keras"

CLASS_NAMES = ["healthy", "infected"]   # index 0 = healthy, 1 = infected

IMG_SIZE = 128          # images are resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
