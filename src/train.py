import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rsna_dataset import RSNADataset, get_transforms
from model import create_model
from config import (
    CSV_FILE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    MODEL_DIR,
    SEED,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    set_seed(SEED)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("===== LOADING DATA =====")

    train_dataset = RSNADataset(
        CSV_FILE,
        "train",
        get_transforms(train=True)
    )

    val_dataset = RSNADataset(
        CSV_FILE,
        "val",
        get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    print("\n===== CREATING MODEL =====")

    model = create_model()
    model = model.to(DEVICE)

    # Automatically calculate class weights
    train_labels = train_dataset.df["pneumonia"].values

    class_counts = np.bincount(train_labels)

    class_weights = len(train_labels) / (
        2 * class_counts
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(DEVICE)

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    print("\n===== STARTING TRAINING =====")

    for epoch in range(NUM_EPOCHS):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        train_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            model_path = os.path.join(
                MODEL_DIR,
                "densenet121_centralized_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                "Saved best model:",
                model_path
            )


if __name__ == "__main__":
    main()