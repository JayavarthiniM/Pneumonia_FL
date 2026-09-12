import os
import torch


# =========================
# Reproducibility
# =========================

SEED = 42


# =========================
# Dataset
# =========================

# Default path for local VS Code execution.
# Can be overridden with the CSV_FILE environment variable.
CSV_FILE = os.environ.get(
    "CSV_FILE",
    r"data\rsna\mapping\rsna_splits.csv"
)

# Column containing the DICOM image path.
# Local Windows CSV uses "dicom_path".
# Colab can use "colab_path".
PATH_COLUMN = os.environ.get(
    "PATH_COLUMN",
    "dicom_path"
)

IMAGE_SIZE = 224


# =========================
# Training
# =========================

BATCH_SIZE = 16

NUM_EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# =========================
# Classes
# =========================

NUM_CLASSES = 2


# =========================
# Device
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# Model
# =========================

MODEL_NAME = "densenet121"


# =========================
# Output directories
# =========================

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "models"
)

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "results"
)


# =========================
# Configuration summary
# =========================

print("===== CONFIGURATION =====")
print("Device:", DEVICE)
print("Model:", MODEL_NAME)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("CSV file:", CSV_FILE)
print("Path column:", PATH_COLUMN)
print("Model directory:", MODEL_DIR)
print("Results directory:", RESULTS_DIR)