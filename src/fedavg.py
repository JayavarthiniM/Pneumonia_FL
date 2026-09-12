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
    SEED,
)


# ============================================================
# FEDERATED CONFIGURATION
# ============================================================

NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "5"))

# Default = 5 communication rounds.
# Can temporarily override with:
# os.environ["NUM_ROUNDS"] = "1"
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "5"))

# Number of local epochs performed by each client.
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "1"))

# Client partition file.
CLIENT_MAP_FILE = os.environ.get(
    "CLIENT_MAP_FILE",
    os.path.join(
        "data",
        "rsna",
        "mapping",
        "federated_client_map.csv",
    ),
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "densenet121_fedavg_best.pth",
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "fedavg_training_history.csv",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_local_model(
    model,
    loader,
    criterion,
    optimizer,
    device,
    local_epochs,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(local_epochs):

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return (
        copy.deepcopy(model.state_dict()),
        average_loss,
        accuracy,
        total_samples,
    )


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

def fedavg(
    client_states,
    client_sizes,
):
    """
    Standard sample-weighted FedAvg.

    Floating-point parameters/buffers are averaged
    using the number of training samples on each client.

    Non-floating buffers are copied from the first client
    instead of attempting invalid floating-point averaging.
    """

    total_samples = sum(client_sizes)

    global_state = {}

    for key in client_states[0]:

        first_tensor = client_states[0][key]

        if torch.is_floating_point(first_tensor):
            aggregated = torch.zeros_like(
                first_tensor
            )

            for client_index in range(
                len(client_states)
            ):
                weight = (
                    client_sizes[client_index]
                    / total_samples
                )

                aggregated += (
                    client_states[client_index][key]
                    * weight
                )

            global_state[key] = aggregated

        else:
            # Non-floating buffers such as
            # num_batches_tracked cannot be
            # meaningfully averaged as floats.
            global_state[key] = (
                first_tensor.clone()
            )

    return global_state


# ============================================================
# GLOBAL VALIDATION
# ============================================================

def evaluate_global_model(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    print("\n===== FEDAVG TRAINING =====")

    print("Device:", DEVICE)
    print("Model: DenseNet121")
    print("Clients:", NUM_CLIENTS)
    print("Communication rounds:", NUM_ROUNDS)
    print("Local epochs:", LOCAL_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("CSV file:", CSV_FILE)
    print("Client map:", CLIENT_MAP_FILE)

    # --------------------------------------------------------
    # LOAD CLIENT MAP
    # --------------------------------------------------------

    print("\n===== LOADING CLIENT DATA =====")

    if not os.path.exists(CLIENT_MAP_FILE):
        raise FileNotFoundError(
            f"Client map not found:\n{CLIENT_MAP_FILE}\n\n"
            "Generate the federated client map first."
        )

    client_df = pd.read_csv(
        CLIENT_MAP_FILE
    )

    required_columns = {
        "client_id",
        "patient_key",
        "pneumonia",
    }

    missing_columns = (
        required_columns
        - set(client_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Client map is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(
        "Federated training images:",
        len(client_df),
    )

    print(
        "Federated patients:",
        client_df["patient_key"].nunique(),
    )

    # Verify all expected clients exist.
    actual_clients = sorted(
        client_df["client_id"]
        .unique()
        .tolist()
    )

    expected_clients = list(
        range(1, NUM_CLIENTS + 1)
    )

    if actual_clients != expected_clients:
        raise ValueError(
            f"Expected clients {expected_clients}, "
            f"but found {actual_clients}"
        )

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    val_dataset = RSNADataset(
        CSV_FILE,
        "val",
        get_transforms(train=False),
        path_column=PATH_COLUMN,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "Validation images:",
        len(val_dataset),
    )

    # --------------------------------------------------------
    # CREATE CLIENT DATASETS
    # --------------------------------------------------------

    client_datasets = []
    client_loaders = []

    print("\n===== CLIENT DISTRIBUTION =====")

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):

        client_subset = client_df[
            client_df["client_id"]
            == client_id
        ].copy()

        if len(client_subset) == 0:
            raise ValueError(
                f"Client {client_id} has no images."
            )

        client_dataset = RSNADataset(
            CSV_FILE,
            "train",
            get_transforms(train=True),
            path_column=PATH_COLUMN,
        )

        # Replace the full training dataframe
        # with this client's patient-level subset.
        client_dataset.df = (
            client_subset.reset_index(
                drop=True
            )
        )

        client_loader = DataLoader(
            client_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        client_datasets.append(
            client_dataset
        )

        client_loaders.append(
            client_loader
        )

        pneumonia_count = (
            client_subset["pneumonia"] == 1
        ).sum()

        no_pneumonia_count = (
            client_subset["pneumonia"] == 0
        ).sum()

        pneumonia_ratio = (
            pneumonia_count / len(client_subset)
        )

        print(
            f"Client {client_id}: "
            f"{len(client_subset)} images | "
            f"{client_subset['patient_key'].nunique()} patients | "
            f"Pneumonia: {pneumonia_count} | "
            f"No pneumonia: {no_pneumonia_count} | "
            f"Pneumonia ratio: {pneumonia_ratio:.3f}"
        )

    # --------------------------------------------------------
    # GLOBAL MODEL
    # --------------------------------------------------------

    print("\n===== CREATING GLOBAL MODEL =====")

    global_model = create_model()

    global_model = global_model.to(
        DEVICE
    )

    print(
        "Classifier:",
        global_model.classifier,
    )

    # --------------------------------------------------------
    # GLOBAL TRAINING LOSS
    # --------------------------------------------------------

    global_labels = (
        client_df["pneumonia"].values
    )

    class_counts = np.bincount(
        global_labels,
        minlength=2,
    )

    class_weights = (
        len(global_labels)
        / (2 * class_counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print(
        "Class counts:",
        class_counts,
    )

    print(
        "Class weights:",
        class_weights,
    )

    # --------------------------------------------------------
    # FEDERATED TRAINING
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_round = None

    print("\n===== STARTING FEDAVG =====")

    for round_number in range(
        1,
        NUM_ROUNDS + 1,
    ):

        round_start = time.time()

        print(
            f"\n========== ROUND "
            f"{round_number}/{NUM_ROUNDS} =========="
        )

        client_states = []
        client_sizes = []

        client_losses = []
        client_accuracies = []

        # ----------------------------------------------------
        # CLIENT TRAINING
        # ----------------------------------------------------

        for client_index in range(
            NUM_CLIENTS
        ):

            client_id = (
                client_index + 1
            )

            print(
                f"\nClient {client_id} "
                f"local training..."
            )

            # Each client starts from the
            # current global model.
            local_model = create_model()

            local_model.load_state_dict(
                global_model.state_dict()
            )

            local_model = local_model.to(
                DEVICE
            )

            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            )

            (
                state_dict,
                local_loss,
                local_accuracy,
                client_size,
            ) = train_local_model(
                local_model,
                client_loaders[client_index],
                criterion,
                optimizer,
                DEVICE,
                LOCAL_EPOCHS,
            )

            client_states.append(
                state_dict
            )

            client_sizes.append(
                client_size
            )

            client_losses.append(
                local_loss
            )

            client_accuracies.append(
                local_accuracy
            )

            print(
                f"Client {client_id} "
                f"Loss: {local_loss:.4f} | "
                f"Accuracy: {local_accuracy:.4f}"
            )

            del local_model
            del optimizer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # FEDAVG AGGREGATION
        # ----------------------------------------------------

        print(
            "\nAggregating client models..."
        )

        global_state = fedavg(
            client_states,
            client_sizes,
        )

        global_model.load_state_dict(
            global_state
        )

        # ----------------------------------------------------
        # GLOBAL VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            evaluate_global_model(
                global_model,
                val_loader,
                criterion,
                DEVICE,
            )
        )

        round_time = (
            time.time() - round_start
        )

        average_client_loss = (
            np.average(
                client_losses,
                weights=client_sizes,
            )
        )

        average_client_accuracy = (
            np.average(
                client_accuracies,
                weights=client_sizes,
            )
        )

        print(
            f"\nRound {round_number} results:"
        )

        print(
            f"Weighted client loss: "
            f"{average_client_loss:.4f}"
        )

        print(
            f"Weighted client accuracy: "
            f"{average_client_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Round time: "
            f"{round_time / 60:.2f} minutes"
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history.append(
            {
                "round": round_number,
                "weighted_client_loss":
                    average_client_loss,
                "weighted_client_accuracy":
                    average_client_accuracy,
                "validation_loss":
                    val_loss,
                "validation_accuracy":
                    val_accuracy,
                "round_time_minutes":
                    round_time / 60,
            }
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            HISTORY_PATH,
            index=False,
        )

        # ----------------------------------------------------
        # SAVE BEST GLOBAL MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_round = round_number

            checkpoint = {
                "model_state_dict":
                    copy.deepcopy(
                        global_model.state_dict()
                    ),

                "round":
                    round_number,

                "validation_loss":
                    val_loss,

                "validation_accuracy":
                    val_accuracy,

                "model_name":
                    "densenet121",

                "federated_algorithm":
                    "FedAvg",

                "num_clients":
                    NUM_CLIENTS,

                "local_epochs":
                    LOCAL_EPOCHS,

                "seed":
                    SEED,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                "\nBest global model saved:"
            )

            print(
                BEST_MODEL_PATH
            )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n===== FEDAVG COMPLETE ====="
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}",
    )

    print(
        "Best round:",
        best_round,
    )

    print(
        "Best global model:",
        BEST_MODEL_PATH,
    )

    print(
        "Training history:",
        HISTORY_PATH,
    )


if __name__ == "__main__":
    main()