import os
import random
import time
import copy

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
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
)


# ============================================================
# FEDPROX CONFIGURATION
# ============================================================

NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 5))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", 5))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", 1))

# FedProx proximal coefficient
MU = float(os.environ.get("MU", 0.01))

CLIENT_MAP_FILE = os.environ.get(
    "CLIENT_MAP_FILE",
    os.path.join(
        os.path.dirname(CSV_FILE),
        "federated_client_map.csv"
    )
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "densenet121_fedprox_best.pth"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "fedprox_training_history.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def get_class_weights(df):
    counts = df["pneumonia"].value_counts().sort_index()

    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)

    total = count_0 + count_1

    weight_0 = total / (2.0 * count_0)
    weight_1 = total / (2.0 * count_1)

    return torch.tensor(
        [weight_0, weight_1],
        dtype=torch.float32,
        device=DEVICE
    )


# ============================================================
# LOCAL FEDPROX TRAINING
# ============================================================

def train_local_model(
    global_model,
    client_model,
    loader,
    criterion,
    local_epochs,
    mu,
):
    """
    Train one client using the FedProx objective:

        Local Loss =
            Classification Loss
            + (mu / 2) * ||w - w_global||^2

    The global model is kept fixed while the client trains.
    """

    client_model.train()

    optimizer = torch.optim.AdamW(
        client_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Store the global parameters once.
    global_params = {
        name: param.detach().clone()
        for name, param in global_model.named_parameters()
        if param.requires_grad
    }

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(local_epochs):

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for images, labels in loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = client_model(images)

            classification_loss = criterion(
                outputs,
                labels
            )

            # ------------------------------------------------
            # FedProx proximal term
            # ------------------------------------------------

            proximal_term = torch.zeros(
                1,
                device=DEVICE
            )

            for name, param in client_model.named_parameters():

                if param.requires_grad:
                    proximal_term += torch.sum(
                        (param - global_params[name]) ** 2
                    )

            proximal_term *= mu / 2.0

            loss = classification_loss + proximal_term

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)

            epoch_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            epoch_correct += (
                predictions == labels
            ).sum().item()

            epoch_samples += batch_size

        epoch_loss /= epoch_samples
        epoch_accuracy = epoch_correct / epoch_samples

        print(
            f"      Local epoch {epoch + 1}/{local_epochs} "
            f"- loss: {epoch_loss:.4f} "
            f"- acc: {epoch_accuracy:.4f}"
        )

        total_loss = epoch_loss
        total_correct = epoch_correct
        total_samples = epoch_samples

    return (
        total_loss,
        total_correct / total_samples,
        total_samples
    )


# ============================================================
# FEDERATED AGGREGATION
# ============================================================

def fedprox_aggregate(global_model, client_models, client_sizes):
    """
    Sample-weighted federated averaging.

    FedProx changes the LOCAL optimization objective.
    Aggregation remains weighted averaging.
    """

    total_samples = sum(client_sizes)

    global_state = global_model.state_dict()

    for key in global_state.keys():

        # Floating-point tensors can be averaged.
        if torch.is_floating_point(global_state[key]):

            aggregated = torch.zeros_like(
                global_state[key]
            )

            for client_model, client_size in zip(
                client_models,
                client_sizes
            ):

                weight = client_size / total_samples

                aggregated += (
                    client_model.state_dict()[key]
                    * weight
                )

            global_state[key] = aggregated

        else:
            # Non-floating buffers cannot be averaged.
            global_state[key] = (
                client_models[0]
                .state_dict()[key]
                .clone()
            )

    global_model.load_state_dict(global_state)

    return global_model


# ============================================================
# GLOBAL VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_global(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(42)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n==========================================")
    print("FEDPROX TRAINING")
    print("==========================================")
    print("Device:", DEVICE)
    print("Clients:", NUM_CLIENTS)
    print("Rounds:", NUM_ROUNDS)
    print("Local epochs:", LOCAL_EPOCHS)
    print("FedProx MU:", MU)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Client map:", CLIENT_MAP_FILE)
    print("==========================================\n")

    # --------------------------------------------------------
    # Load federated client map
    # --------------------------------------------------------

    if not os.path.exists(CLIENT_MAP_FILE):
        raise FileNotFoundError(
            f"Client map not found:\n{CLIENT_MAP_FILE}"
        )

    client_map = pd.read_csv(CLIENT_MAP_FILE)

    required_columns = {
        "img_id",
        "client_id"
    }

    missing = required_columns - set(
        client_map.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns in client map: {missing}"
        )

    client_ids = sorted(
        client_map["client_id"].unique()
    )

    if client_ids != list(range(1, NUM_CLIENTS + 1)):
        raise ValueError(
            f"Expected client IDs "
            f"1..{NUM_CLIENTS}, got {client_ids}"
        )

    print("Client distribution:")

    for client_id in client_ids:

        subset = client_map[
            client_map["client_id"] == client_id
        ]

        pneumonia_count = (
            subset["pneumonia"].sum()
            if "pneumonia" in subset.columns
            else "N/A"
        )

        print(
            f"  Client {client_id}: "
            f"{len(subset)} images | "
            f"Pneumonia: {pneumonia_count}"
        )

    # --------------------------------------------------------
    # Build client datasets
    # --------------------------------------------------------

    client_datasets = []
    client_loaders = []

    for client_id in client_ids:

        client_df = client_map[
            client_map["client_id"] == client_id
        ].copy()

        client_datasets.append(client_df)

        # Merge client assignments with master metadata.
        master_df = pd.read_csv(CSV_FILE)

        client_metadata = master_df[
            master_df["img_id"].isin(
                client_df["img_id"]
            )
        ].copy()

        temp_file = os.path.join(
            RESULTS_DIR,
            f"_fedprox_client_{client_id}.csv"
        )

        client_metadata.to_csv(
            temp_file,
            index=False
        )

        dataset = RSNADataset(
            csv_file=temp_file,
            split="train",
            transform=get_transforms(train=True),
            path_column=PATH_COLUMN,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        client_loaders.append(loader)

    # --------------------------------------------------------
    # Global validation dataset
    # --------------------------------------------------------

    val_dataset = RSNADataset(
        csv_file=CSV_FILE,
        split="val",
        transform=get_transforms(train=False),
        path_column=PATH_COLUMN,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Global class weights
    # --------------------------------------------------------

    train_df = pd.read_csv(CSV_FILE)

    train_df = train_df[
        train_df["split"] == "train"
    ].copy()

    class_weights = get_class_weights(train_df)

    print(
        "\nClass weights:",
        class_weights.detach().cpu().numpy()
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Create global model
    # --------------------------------------------------------

    global_model = create_model().to(DEVICE)

    best_val_loss = float("inf")
    history = []

    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    for round_num in range(1, NUM_ROUNDS + 1):

        round_start = time.time()

        print("\n" + "=" * 60)
        print(f"FEDPROX ROUND {round_num}/{NUM_ROUNDS}")
        print("=" * 60)

        client_models = []
        client_sizes = []
        client_losses = []
        client_accuracies = []

        # ----------------------------------------------------
        # Train every client
        # ----------------------------------------------------

        for client_id, loader in zip(
            client_ids,
            client_loaders
        ):

            print(
                f"\n  Client {client_id} training..."
            )

            client_model = copy.deepcopy(
                global_model
            ).to(DEVICE)

            loss, accuracy, samples = (
                train_local_model(
                    global_model=global_model,
                    client_model=client_model,
                    loader=loader,
                    criterion=criterion,
                    local_epochs=LOCAL_EPOCHS,
                    mu=MU,
                )
            )

            client_models.append(client_model)
            client_sizes.append(samples)
            client_losses.append(loss)
            client_accuracies.append(accuracy)

            print(
                f"  Client {client_id} finished: "
                f"loss={loss:.4f}, "
                f"accuracy={accuracy:.4f}, "
                f"samples={samples}"
            )

        # ----------------------------------------------------
        # Aggregate clients
        # ----------------------------------------------------

        global_model = fedprox_aggregate(
            global_model,
            client_models,
            client_sizes
        )

        # ----------------------------------------------------
        # Global validation
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate_global(
            global_model,
            val_loader,
            criterion
        )

        total_client_samples = sum(client_sizes)

        weighted_client_loss = sum(
            loss * size
            for loss, size in zip(
                client_losses,
                client_sizes
            )
        ) / total_client_samples

        weighted_client_accuracy = sum(
            accuracy * size
            for accuracy, size in zip(
                client_accuracies,
                client_sizes
            )
        ) / total_client_samples

        elapsed_minutes = (
            time.time() - round_start
        ) / 60.0

        print("\n  Round summary:")
        print(
            f"    Weighted client loss: "
            f"{weighted_client_loss:.4f}"
        )
        print(
            f"    Weighted client accuracy: "
            f"{weighted_client_accuracy:.4f}"
        )
        print(
            f"    Validation loss: "
            f"{val_loss:.4f}"
        )
        print(
            f"    Validation accuracy: "
            f"{val_accuracy:.4f}"
        )
        print(
            f"    Time: "
            f"{elapsed_minutes:.2f} minutes"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        global_model.state_dict(),
                    "round": round_num,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "mu": MU,
                },
                BEST_MODEL_PATH
            )

            print(
                f"    ✓ Best model saved "
                f"(validation loss={val_loss:.4f})"
            )

        history.append(
            {
                "round": round_num,
                "weighted_client_loss":
                    weighted_client_loss,
                "weighted_client_accuracy":
                    weighted_client_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "mu": MU,
                "time_minutes": elapsed_minutes,
            }
        )

        pd.DataFrame(history).to_csv(
            HISTORY_PATH,
            index=False
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 60)
    print("FEDPROX TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best validation loss:",
        best_val_loss
    )

    print(
        "Best model:",
        BEST_MODEL_PATH
    )

    print(
        "Training history:",
        HISTORY_PATH
    )

    # --------------------------------------------------------
    # Remove temporary client CSVs
    # --------------------------------------------------------

    for client_id in client_ids:

        temp_file = os.path.join(
            RESULTS_DIR,
            f"_fedprox_client_{client_id}.csv"
        )

        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    main()