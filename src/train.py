import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rsna_dataset import RSNADataset, get_transforms
from model import create_model

from config import (
    CSV_FILE,
    PATH_COLUMN,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
    SEED,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_class_weights(labels, device):
    """Calculate inverse-frequency class weights."""

    class_counts = np.bincount(labels)

    class_weights = len(labels) / (
        2 * class_counts
    )

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    """Train model for one epoch."""

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(
    model,
    loader,
    criterion,
    device
):
    """Evaluate model on validation data."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():

    # =========================
    # Reproducibility
    # =========================

    set_seed(SEED)

    # =========================
    # Directories
    # =========================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("===== CENTRALIZED TRAINING =====")

    print("Device:", DEVICE)
    print("Model: DenseNet121")
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("CSV file:", CSV_FILE)
    print("Path column:", PATH_COLUMN)

    # =========================
    # Load datasets
    # =========================

    print("\n===== LOADING DATA =====")

    train_dataset = RSNADataset(
        CSV_FILE,
        "train",
        get_transforms(train=True),
        path_column=PATH_COLUMN
    )

    val_dataset = RSNADataset(
        CSV_FILE,
        "val",
        get_transforms(train=False),
        path_column=PATH_COLUMN
    )

    # =========================
    # DataLoaders
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        "Train images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # =========================
    # Create model
    # =========================

    print("\n===== CREATING MODEL =====")

    model = create_model()

    model = model.to(DEVICE)

    print(
        "Classifier:",
        model.classifier
    )

    # =========================
    # Class weights
    # =========================

    train_labels = (
        train_dataset.df["pneumonia"]
        .values
    )

    class_counts = np.bincount(
        train_labels
    )

    class_weights = calculate_class_weights(
        train_labels,
        DEVICE
    )

    print(
        "Class counts:",
        class_counts
    )

    print(
        "Class weights:",
        class_weights
    )

    # =========================
    # Loss function
    # =========================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # =========================
    # Optimizer
    # =========================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # =========================
    # Training history
    # =========================

    history = []

    # =========================
    # Best model tracking
    # =========================

    best_val_loss = float("inf")

    best_model_path = os.path.join(
        MODEL_DIR,
        "densenet121_centralized_best.pth"
    )

    history_path = os.path.join(
        RESULTS_DIR,
        "centralized_training_history.csv"
    )

    print(
        "\nBest model will be saved to:",
        best_model_path
    )

    print(
        "Training history will be saved to:",
        history_path
    )

    # =========================
    # Training
    # =========================

    print("\n===== STARTING TRAINING =====")

    for epoch in range(NUM_EPOCHS):

        start_time = time.time()

        # -------------------------
        # Training
        # -------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        # -------------------------
        # Validation
        # -------------------------

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        elapsed_time = (
            time.time() - start_time
        )

        # -------------------------
        # Store history
        # -------------------------

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "time_minutes": elapsed_time / 60
        }

        history.append(
            epoch_record
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            history_path,
            index=False
        )

        # -------------------------
        # Epoch results
        # -------------------------

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Time: "
            f"{elapsed_time / 60:.2f} minutes"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch + 1,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "model_name":
                    "densenet121",

                "seed":
                    SEED,
            }

            torch.save(
                checkpoint,
                best_model_path
            )

            print(
                "Best model saved:",
                best_model_path
            )

    # =========================
    # Training complete
    # =========================

    print(
        "\n===== TRAINING COMPLETE ====="
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Best model:",
        best_model_path
    )

    print(
        "Training history:",
        history_path
    )


if __name__ == "__main__":
    main()