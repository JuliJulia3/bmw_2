# bikes_config.py
from pathlib import Path

# ----- YOUR FOLDERS -----
DATASET_DIR = Path("dataset")
CLEAN_DIR = DATASET_DIR / "clean"
PREDICT_DIR = Path("predict")

# classes must match folder names exactly
CLASSES = ["r1300gs", "r1300gs_adventure", "other"]

# run dirs
RUNS_DIR = Path("runs/bikes_ensemble")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# training (keep as you had it; CPU is fine)
IMG_SIZE = 224
BATCH = 16
EPOCHS = 18
LR = 3e-4
WEIGHT_DECAY = 0.05
SEED = 42

# splits
TEST_SIZE = 0.15
VAL_SIZE_FROM_TRAIN = 0.18

# ---------- INFERENCE / ENSEMBLE MANAGER ----------
# your existing unknown logic
MIN_CONF_FINAL = 0.60
MIN_MARGIN_FINAL = 0.08
MIN_CONF_PER_MODEL = 0.45

# TTA / multi-crop
USE_MULTI_CROP = True
NUM_CROPS = 5  # CPU ok; set 3 if slow

# ---------- OOD / UNKNOWN DETECTION (NEW) ----------
# Binary max entropy is ~0.693. In-distribution usually has lower entropy.
# Start here; tune after seeing CSV values for "other bikes".
MAX_ENTROPY_FINAL = 0.55

# Energy needs calibration. We'll log it in the CSV.
# Start conservative; if too many OOD still slip through, increase (less negative) or tighten.
MAX_ENERGY_FINAL = -2.5

# output
SORT_OUT = True
SORT_DIR = Path("predicted_ensemble")
OUT_CSV = Path("predictions_ensemble.csv")
