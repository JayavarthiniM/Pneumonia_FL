import torch


# Reproducibility
SEED = 42


# Dataset
CSV_FILE = r"data\rsna\mapping\rsna_splits.csv"

IMAGE_SIZE = 224


# Training
BATCH_SIZE = 8
NUM_EPOCHS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# Classes
NUM_CLASSES = 2


# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Model
MODEL_NAME = "densenet121"


# Output directories
MODEL_DIR = r"models"
RESULTS_DIR = r"results"


print("===== CONFIGURATION =====")
print("Device:", DEVICE)
print("Model:", MODEL_NAME)
print("Batch size:", BATCH_SIZE)
print("Epochs:", NUM_EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)