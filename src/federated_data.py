import os
import random

import numpy as np
import pandas as pd


# =========================
# CONFIGURATION
# =========================

SEED = 42
NUM_CLIENTS = 5

CSV_FILE = os.environ.get(
    "CSV_FILE",
    r"data\rsna\mapping\rsna_splits.csv"
)

OUTPUT_FILE = os.environ.get(
    "CLIENT_MAP_FILE",
    r"data\rsna\mapping\federated_client_map.csv"
)


# =========================
# REPRODUCIBILITY
# =========================

random.seed(SEED)
np.random.seed(SEED)


# =========================
# PATIENT-LEVEL NON-IID
# =========================

def create_non_iid_partition(df, num_clients):

    # -------------------------------------------------
    # Each patient must belong to exactly ONE client.
    # -------------------------------------------------

    patient_stats = (
        df.groupby("patient_key")
        .agg(
            total_images=("img_id", "count"),
            pneumonia_images=("pneumonia", "sum")
        )
        .reset_index()
    )

    patient_stats["no_pneumonia_images"] = (
        patient_stats["total_images"]
        - patient_stats["pneumonia_images"]
    )

    # A patient's pneumonia proportion.
    patient_stats["pneumonia_ratio"] = (
        patient_stats["pneumonia_images"]
        / patient_stats["total_images"]
    )

    # Shuffle patients reproducibly.
    patient_stats = patient_stats.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # Target pneumonia proportions for clients.
    target_ratios = np.array([
        0.80,
        0.65,
        0.50,
        0.35,
        0.20
    ])

    # Current client statistics.
    client_stats = []

    for client_id in range(num_clients):

        client_stats.append({
            "client_id": client_id + 1,
            "total_images": 0,
            "pneumonia_images": 0
        })

    # -------------------------------------------------
    # Greedy patient assignment.
    #
    # Each complete patient is assigned to one client.
    # We choose the client that is currently closest
    # to its target pneumonia distribution.
    # -------------------------------------------------

    # Process larger patients first so they don't
    # dominate a client unexpectedly near the end.
    patient_stats = patient_stats.sort_values(
        by="total_images",
        ascending=False
    ).reset_index(drop=True)

    patient_to_client = {}

    for _, patient in patient_stats.iterrows():

        best_client = None
        best_score = float("inf")

        for client_index in range(num_clients):

            current = client_stats[client_index]

            new_total = (
                current["total_images"]
                + patient["total_images"]
            )

            new_pneumonia = (
                current["pneumonia_images"]
                + patient["pneumonia_images"]
            )

            new_ratio = (
                new_pneumonia / new_total
                if new_total > 0
                else 0
            )

            # Penalize deviation from desired
            # pneumonia ratio.
            ratio_error = abs(
                new_ratio
                - target_ratios[client_index]
            )

            # Also encourage reasonably balanced
            # client sizes.
            size_error = abs(
                new_total
                - len(df) / num_clients
            ) / len(df)

            score = (
                ratio_error
                + 0.15 * size_error
            )

            if score < best_score:

                best_score = score
                best_client = client_index

        client_stats[best_client]["total_images"] += (
            patient["total_images"]
        )

        client_stats[best_client]["pneumonia_images"] += (
            patient["pneumonia_images"]
        )

        patient_to_client[
            patient["patient_key"]
        ] = best_client + 1

    # Map every image to its patient's client.
    result = df.copy()

    result["client_id"] = (
        result["patient_key"]
        .map(patient_to_client)
    )

    return result


# =========================
# MAIN
# =========================

def main():

    print(
        "===== PATIENT-LEVEL FEDERATED PARTITION ====="
    )

    print("CSV file:", CSV_FILE)
    print("Number of clients:", NUM_CLIENTS)
    print("Seed:", SEED)

    df = pd.read_csv(CSV_FILE)

    # Only training data are partitioned.
    train_df = df[
        df["split"] == "train"
    ].copy()

    print(
        "\nTraining images:",
        len(train_df)
    )

    print(
        "Training patients:",
        train_df["patient_key"].nunique()
    )

    client_df = create_non_iid_partition(
        train_df,
        NUM_CLIENTS
    )

    # =========================
    # PATIENT LEAKAGE CHECK
    # =========================

    patient_client_counts = (
        client_df
        .groupby("patient_key")["client_id"]
        .nunique()
    )

    leaked_patients = (
        patient_client_counts[
            patient_client_counts > 1
        ]
    )

    print(
        "\nPatient leakage:",
        len(leaked_patients)
    )

    if len(leaked_patients) > 0:

        raise RuntimeError(
            "Patient leakage detected!"
        )

    # =========================
    # CLIENT COVERAGE CHECK
    # =========================

    assigned_images = len(client_df)

    if assigned_images != len(train_df):

        raise RuntimeError(
            "Some training images were not assigned "
            "to a federated client!"
        )

    missing_clients = set(
        range(1, NUM_CLIENTS + 1)
    ) - set(
        client_df["client_id"].unique()
    )

    if missing_clients:

        raise RuntimeError(
            f"Missing clients: {missing_clients}"
        )

    # =========================
    # SAVE
    # =========================

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    client_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # =========================
    # CLIENT STATISTICS
    # =========================

    print(
        "\n===== CLIENT DISTRIBUTION ====="
    )

    for client_id in range(
        1,
        NUM_CLIENTS + 1
    ):

        subset = client_df[
            client_df["client_id"] ==
            client_id
        ]

        pneumonia_count = (
            subset["pneumonia"] == 1
        ).sum()

        no_pneumonia_count = (
            subset["pneumonia"] == 0
        ).sum()

        total_images = len(subset)

        patient_count = (
            subset["patient_key"]
            .nunique()
        )

        pneumonia_ratio = (
            pneumonia_count / total_images
            if total_images > 0
            else 0
        )

        print(
            f"Client {client_id}: "
            f"{total_images} images | "
            f"{patient_count} patients | "
            f"Pneumonia: {pneumonia_count} | "
            f"No pneumonia: {no_pneumonia_count} | "
            f"Pneumonia ratio: "
            f"{pneumonia_ratio:.3f}"
        )

    # =========================
    # FINAL CHECK
    # =========================

    print(
        "\n===== FINAL CHECKS ====="
    )

    print(
        "Original training images:",
        len(train_df)
    )

    print(
        "Assigned training images:",
        len(client_df)
    )

    print(
        "Unique patients:",
        client_df["patient_key"].nunique()
    )

    print(
        "Patient leakage:",
        len(leaked_patients)
    )

    print(
        "\nSaved client map:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()