import os
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = os.environ.get(
    "CSV_FILE",
    os.path.join(
        "data",
        "rsna",
        "mapping",
        "rsna_splits.csv",
    ),
)

NUM_CLIENTS = int(
    os.environ.get("NUM_CLIENTS", "5")
)

SEED = int(
    os.environ.get("SEED", "42")
)

OUTPUT_FILE = os.environ.get(
    "CLIENT_MAP_FILE",
    os.path.join(
        "data",
        "rsna",
        "mapping",
        "federated_client_map.csv",
    ),
)

# Controls how strongly client pneumonia proportions differ.
# Lower value = stronger heterogeneity.
DIRICHLET_ALPHA = 0.5

# Minimum desired number of images per client.
MIN_CLIENT_IMAGES = 2000


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(
            f"CSV file not found:\n{CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = {
        "patient_key",
        "pneumonia",
        "split",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    train_df = (
        df[df["split"] == "train"]
        .copy()
        .reset_index(drop=True)
    )

    if len(train_df) == 0:
        raise ValueError(
            "No training images found."
        )

    return train_df


# ============================================================
# BUILD PATIENT STATISTICS
# ============================================================

def build_patient_statistics(train_df):

    patient_stats = (
        train_df
        .groupby("patient_key")
        .agg(
            total_images=("pneumonia", "size"),
            pneumonia_images=("pneumonia", "sum"),
        )
        .reset_index()
    )

    patient_stats["no_pneumonia_images"] = (
        patient_stats["total_images"]
        - patient_stats["pneumonia_images"]
    )

    patient_stats["pneumonia_ratio"] = (
        patient_stats["pneumonia_images"]
        / patient_stats["total_images"]
    )

    # Classify patients according to their image labels.
    patient_stats["patient_type"] = np.where(
        patient_stats["pneumonia_images"]
        == patient_stats["total_images"],
        "pneumonia_only",
        np.where(
            patient_stats["pneumonia_images"] == 0,
            "no_pneumonia_only",
            "mixed",
        ),
    )

    return patient_stats


# ============================================================
# TARGET CLIENT DISTRIBUTIONS
# ============================================================

def create_client_targets(
    patient_stats,
    total_images,
):
    """
    Create approximately equal image targets while
    introducing controlled label heterogeneity.

    This is a patient-level Dirichlet-inspired
    partitioning strategy rather than a textbook
    per-image Dirichlet partition.
    """

    target_images = np.full(
        NUM_CLIENTS,
        total_images / NUM_CLIENTS,
        dtype=float,
    )

    global_pneumonia_ratio = (
        patient_stats["pneumonia_images"].sum()
        / total_images
    )

    # Generate Dirichlet proportions.
    pneumonia_noise = np.random.dirichlet(
        np.full(
            NUM_CLIENTS,
            DIRICHLET_ALPHA,
        )
    )

    # Center the random proportions around the
    # actual global pneumonia prevalence.
    noise_mean = pneumonia_noise.mean()

    target_ratios = (
        global_pneumonia_ratio
        + (
            pneumonia_noise
            - noise_mean
        )
        * 0.75
    )

    # Keep targets in a realistic range.
    target_ratios = np.clip(
        target_ratios,
        0.20,
        0.85,
    )

    # Re-center after clipping.
    target_ratios = (
        target_ratios
        * (
            global_pneumonia_ratio
            / target_ratios.mean()
        )
    )

    target_ratios = np.clip(
        target_ratios,
        0.20,
        0.85,
    )

    return (
        target_images,
        target_ratios,
    )


# ============================================================
# PATIENT-LEVEL ASSIGNMENT
# ============================================================

def assign_patients(
    patient_stats,
    target_images,
    target_ratios,
):
    """
    Assign complete patients to clients.

    No patient is split across clients.

    Assignment considers:
      1. Client image-count target
      2. Client pneumonia-ratio target
      3. Patient label composition

    Larger patient groups are assigned first to
    reduce final client-size imbalance.
    """

    client_images = np.zeros(
        NUM_CLIENTS,
        dtype=int,
    )

    client_pneumonia = np.zeros(
        NUM_CLIENTS,
        dtype=int,
    )

    patient_to_client = {}

    # Randomize patients reproducibly first.
    shuffled = patient_stats.sample(
        frac=1.0,
        random_state=SEED,
    ).copy()

    # Larger patients first.
    shuffled = shuffled.sort_values(
        by="total_images",
        ascending=False,
    )

    for _, patient in shuffled.iterrows():

        patient_images = int(
            patient["total_images"]
        )

        patient_pneumonia = int(
            patient["pneumonia_images"]
        )

        patient_ratio = float(
            patient["pneumonia_ratio"]
        )

        candidate_scores = []

        for client_id in range(
            NUM_CLIENTS
        ):

            current_images = (
                client_images[client_id]
            )

            current_pneumonia = (
                client_pneumonia[client_id]
            )

            new_images = (
                current_images
                + patient_images
            )

            new_pneumonia = (
                current_pneumonia
                + patient_pneumonia
            )

            new_ratio = (
                new_pneumonia / new_images
            )

            # ------------------------------------------------
            # Size error
            # ------------------------------------------------

            size_error = (
                abs(
                    new_images
                    - target_images[client_id]
                )
                / target_images[client_id]
            )

            # ------------------------------------------------
            # Label-ratio error
            # ------------------------------------------------

            ratio_error = abs(
                new_ratio
                - target_ratios[client_id]
            )

            # ------------------------------------------------
            # Current-size penalty
            # ------------------------------------------------

            # Prevent one client from becoming
            # excessively large early.
            overflow = max(
                0,
                new_images
                - target_images[client_id]
            )

            overflow_penalty = (
                overflow
                / target_images[client_id]
            )

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------

            score = (
                3.0 * size_error
                + 2.0 * ratio_error
                + 4.0 * overflow_penalty
            )

            # Small random tie breaker.
            score += np.random.random() * 1e-6

            candidate_scores.append(
                score
            )

        best_client = int(
            np.argmin(candidate_scores)
        )

        patient_to_client[
            patient["patient_key"]
        ] = best_client + 1

        client_images[
            best_client
        ] += patient_images

        client_pneumonia[
            best_client
        ] += patient_pneumonia

    return (
        patient_to_client,
        client_images,
        client_pneumonia,
    )


# ============================================================
# BUILD CLIENT MAP
# ============================================================

def build_client_map(
    train_df,
    patient_to_client,
):

    result = train_df.copy()

    result["client_id"] = (
        result["patient_key"]
        .map(patient_to_client)
    )

    if result["client_id"].isna().any():
        raise ValueError(
            "Some training images were not assigned "
            "to a client."
        )

    result["client_id"] = (
        result["client_id"]
        .astype(int)
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_partition(
    train_df,
    client_df,
):

    print(
        "\n===== FINAL CHECKS ====="
    )

    # --------------------------------------------------------
    # Image coverage
    # --------------------------------------------------------

    original_images = len(
        train_df
    )

    assigned_images = len(
        client_df
    )

    print(
        "Original training images:",
        original_images,
    )

    print(
        "Assigned training images:",
        assigned_images,
    )

    if original_images != assigned_images:
        raise ValueError(
            "Image count mismatch."
        )

    # --------------------------------------------------------
    # Client IDs
    # --------------------------------------------------------

    expected_clients = set(
        range(1, NUM_CLIENTS + 1)
    )

    actual_clients = set(
        client_df["client_id"]
        .unique()
    )

    if actual_clients != expected_clients:
        raise ValueError(
            "Incorrect client IDs."
        )

    # --------------------------------------------------------
    # Patient leakage
    # --------------------------------------------------------

    patient_client_counts = (
        client_df
        .groupby("patient_key")[
            "client_id"
        ]
        .nunique()
    )

    leaked_patients = (
        patient_client_counts[
            patient_client_counts > 1
        ]
    )

    print(
        "Unique patients:",
        client_df["patient_key"].nunique(),
    )

    print(
        "Patient leakage:",
        len(leaked_patients),
    )

    if len(leaked_patients) > 0:
        raise ValueError(
            "Patient leakage detected."
        )

    # --------------------------------------------------------
    # Duplicate assignment check
    # --------------------------------------------------------

    if "img_id" in client_df.columns:

        duplicate_images = (
            client_df["img_id"]
            .duplicated()
            .sum()
        )

        if duplicate_images > 0:
            raise ValueError(
                f"Duplicate images detected: "
                f"{duplicate_images}"
            )

    # --------------------------------------------------------
    # Minimum client size
    # --------------------------------------------------------

    client_sizes = (
        client_df
        .groupby("client_id")
        .size()
    )

    smallest_client = (
        client_sizes.min()
    )

    print(
        "Smallest client:",
        smallest_client,
        "images",
    )

    if smallest_client < MIN_CLIENT_IMAGES:
        raise ValueError(
            "A client is below the minimum "
            f"size of {MIN_CLIENT_IMAGES} images."
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "===== PATIENT-LEVEL FEDERATED PARTITION ====="
    )

    print(
        "CSV file:",
        CSV_FILE,
    )

    print(
        "Number of clients:",
        NUM_CLIENTS,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "Dirichlet alpha:",
        DIRICHLET_ALPHA,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df = load_training_data()

    print(
        "\nTraining images:",
        len(train_df),
    )

    print(
        "Training patients:",
        train_df["patient_key"].nunique(),
    )

    # --------------------------------------------------------
    # Patient statistics
    # --------------------------------------------------------

    patient_stats = (
        build_patient_statistics(
            train_df
        )
    )

    # --------------------------------------------------------
    # Client targets
    # --------------------------------------------------------

    (
        target_images,
        target_ratios,
    ) = create_client_targets(
        patient_stats,
        len(train_df),
    )

    print(
        "\n===== TARGET DISTRIBUTION ====="
    )

    for client_id in range(
        NUM_CLIENTS
    ):

        print(
            f"Client {client_id + 1}: "
            f"target images ≈ "
            f"{target_images[client_id]:.0f} | "
            f"target pneumonia ratio ≈ "
            f"{target_ratios[client_id]:.3f}"
        )

    # --------------------------------------------------------
    # Assign patients
    # --------------------------------------------------------

    (
        patient_to_client,
        client_images,
        client_pneumonia,
    ) = assign_patients(
        patient_stats,
        target_images,
        target_ratios,
    )

    # --------------------------------------------------------
    # Build image-level client map
    # --------------------------------------------------------

    client_df = build_client_map(
        train_df,
        patient_to_client,
    )

    # --------------------------------------------------------
    # Display distribution
    # --------------------------------------------------------

    print(
        "\n===== CLIENT DISTRIBUTION ====="
    )

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):

        subset = client_df[
            client_df["client_id"]
            == client_id
        ]

        total = len(subset)

        pneumonia = (
            subset["pneumonia"] == 1
        ).sum()

        no_pneumonia = (
            subset["pneumonia"] == 0
        ).sum()

        patient_count = (
            subset["patient_key"]
            .nunique()
        )

        ratio = (
            pneumonia / total
        )

        print(
            f"Client {client_id}: "
            f"{total} images | "
            f"{patient_count} patients | "
            f"Pneumonia: {pneumonia} | "
            f"No pneumonia: {no_pneumonia} | "
            f"Pneumonia ratio: {ratio:.3f}"
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_partition(
        train_df,
        client_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_directory = (
        os.path.dirname(OUTPUT_FILE)
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    client_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved client map:",
        OUTPUT_FILE,
    )

    print(
        "\n===== PARTITION COMPLETE ====="
    )


if __name__ == "__main__":
    main()