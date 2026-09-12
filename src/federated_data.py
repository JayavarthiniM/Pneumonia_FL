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

NUM_CLIENTS = 5
SEED = 42

OUTPUT_FILE = os.environ.get(
    "CLIENT_MAP_FILE",
    os.path.join(
        "data",
        "rsna",
        "mapping",
        "federated_client_map.csv",
    ),
)


# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    df = pd.read_csv(CSV_FILE)

    required = {
        "patient_key",
        "pneumonia",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    train_df = (
        df[df["split"] == "train"]
        .copy()
        .reset_index(drop=True)
    )

    return train_df


# ============================================================
# PATIENT STATISTICS
# ============================================================

def make_patient_table(train_df):

    patients = (
        train_df
        .groupby("patient_key")
        .agg(
            total_images=("pneumonia", "size"),
            pneumonia_images=("pneumonia", "sum"),
        )
        .reset_index()
    )

    patients["pneumonia_ratio"] = (
        patients["pneumonia_images"]
        / patients["total_images"]
    )

    patients["no_pneumonia_images"] = (
        patients["total_images"]
        - patients["pneumonia_images"]
    )

    return patients


# ============================================================
# BALANCED PATIENT ASSIGNMENT
# ============================================================

def create_partition(patients):

    rng = random.Random(SEED)

    total_images = int(
        patients["total_images"].sum()
    )

    target = total_images / NUM_CLIENTS

    # --------------------------------------------------------
    # Start with empty clients.
    # --------------------------------------------------------

    clients = [
        {
            "patients": [],
            "images": 0,
            "pneumonia": 0,
        }
        for _ in range(NUM_CLIENTS)
    ]

    # --------------------------------------------------------
    # Separate patients by label composition.
    #
    # This allows controlled non-IID behaviour while
    # keeping complete patients together.
    # --------------------------------------------------------

    pneumonia_only = patients[
        patients["pneumonia_ratio"] == 1.0
    ].copy()

    no_pneumonia_only = patients[
        patients["pneumonia_ratio"] == 0.0
    ].copy()

    mixed = patients[
        (patients["pneumonia_ratio"] > 0.0)
        & (patients["pneumonia_ratio"] < 1.0)
    ].copy()

    # Reproducible shuffling.
    pneumonia_only = pneumonia_only.sample(
        frac=1,
        random_state=SEED,
    )

    no_pneumonia_only = no_pneumonia_only.sample(
        frac=1,
        random_state=SEED + 1,
    )

    mixed = mixed.sample(
        frac=1,
        random_state=SEED + 2,
    )

    # --------------------------------------------------------
    # Assign pure-class patients using rotating clients.
    # This creates label heterogeneity.
    # --------------------------------------------------------

    def assign_group(group, start_client):

        client_order = [
            (start_client + i) % NUM_CLIENTS
            for i in range(NUM_CLIENTS)
        ]

        for _, patient in group.iterrows():

            # Only consider clients that still have
            # reasonable capacity.
            available = [
                c for c in client_order
                if clients[c]["images"]
                + int(patient["total_images"])
                <= target
            ]

            if not available:
                available = sorted(
                    range(NUM_CLIENTS),
                    key=lambda c: clients[c]["images"]
                )[:2]

            # Prefer the least-filled available client.
            best = min(
                available,
                key=lambda c: clients[c]["images"]
            )

            clients[best]["patients"].append(
                patient["patient_key"]
            )

            clients[best]["images"] += int(
                patient["total_images"]
            )

            clients[best]["pneumonia"] += int(
                patient["pneumonia_images"]
            )

    assign_group(
        pneumonia_only,
        0,
    )

    assign_group(
        no_pneumonia_only,
        2,
    )

    # --------------------------------------------------------
    # Mixed patients go to the currently smallest client.
    # --------------------------------------------------------

    for _, patient in mixed.iterrows():

        size = int(
            patient["total_images"]
        )

        candidates = sorted(
            range(NUM_CLIENTS),
            key=lambda c: clients[c]["images"]
        )

        # Prefer a client that remains close to target.
        best = None

        for c in candidates:

            if (
                clients[c]["images"]
                + size
                <= target + 50
            ):
                best = c
                break

        if best is None:
            best = candidates[0]

        clients[best]["patients"].append(
            patient["patient_key"]
        )

        clients[best]["images"] += size

        clients[best]["pneumonia"] += int(
            patient["pneumonia_images"]
        )

    # --------------------------------------------------------
    # Rebalance clients using whole-patient moves.
    # --------------------------------------------------------

    for _ in range(10000):

        sizes = [
            c["images"]
            for c in clients
        ]

        largest = int(
            np.argmax(sizes)
        )

        smallest = int(
            np.argmin(sizes)
        )

        difference = (
            sizes[largest]
            - sizes[smallest]
        )

        if difference <= 10:
            break

        moved = False

        # Find a patient whose movement reduces
        # the size difference.
        for patient_key in list(
            clients[largest]["patients"]
        ):

            row = patients[
                patients["patient_key"]
                == patient_key
            ].iloc[0]

            size = int(
                row["total_images"]
            )

            if (
                clients[smallest]["images"]
                + size
                <= target + 50
            ):

                clients[largest]["patients"].remove(
                    patient_key
                )

                clients[smallest]["patients"].append(
                    patient_key
                )

                clients[largest]["images"] -= size
                clients[smallest]["images"] += size

                pneumonia = int(
                    row["pneumonia_images"]
                )

                clients[largest]["pneumonia"] -= (
                    pneumonia
                )

                clients[smallest]["pneumonia"] += (
                    pneumonia
                )

                moved = True
                break

        if not moved:
            break

    return clients


# ============================================================
# BUILD IMAGE-LEVEL MAP
# ============================================================

def build_client_map(
    train_df,
    clients,
):

    patient_to_client = {}

    for client_id, client in enumerate(
        clients,
        start=1,
    ):

        for patient in client["patients"]:

            if patient in patient_to_client:
                raise ValueError(
                    f"Patient assigned twice: {patient}"
                )

            patient_to_client[patient] = client_id

    result = train_df.copy()

    result["client_id"] = (
        result["patient_key"]
        .map(patient_to_client)
    )

    if result["client_id"].isna().any():

        missing = result[
            result["client_id"].isna()
        ]["patient_key"].unique()

        raise ValueError(
            f"Unassigned patients: {missing[:10]}"
        )

    result["client_id"] = (
        result["client_id"].astype(int)
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate(
    train_df,
    client_df,
):

    print("\n===== FINAL CHECKS =====")

    # Image count
    print(
        "Original training images:",
        len(train_df),
    )

    print(
        "Assigned training images:",
        len(client_df),
    )

    assert (
        len(train_df)
        == len(client_df)
    )

    # Clients
    client_ids = sorted(
        client_df["client_id"]
        .unique()
        .tolist()
    )

    print(
        "Client IDs:",
        client_ids,
    )

    assert client_ids == [
        1,
        2,
        3,
        4,
        5,
    ]

    # Patient leakage
    leakage = (
        client_df
        .groupby("patient_key")["client_id"]
        .nunique()
    )

    leaked = leakage[
        leakage > 1
    ]

    print(
        "Unique patients:",
        client_df["patient_key"].nunique(),
    )

    print(
        "Patient leakage:",
        len(leaked),
    )

    assert len(leaked) == 0

    # Client sizes
    sizes = (
        client_df
        .groupby("client_id")
        .size()
    )

    print(
        "Client size range:",
        int(sizes.min()),
        "-",
        int(sizes.max()),
    )

    # Every client must have data.
    assert (sizes > 0).all()


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

    train_df = load_training_data()

    print(
        "\nTraining images:",
        len(train_df),
    )

    print(
        "Training patients:",
        train_df["patient_key"].nunique(),
    )

    patients = make_patient_table(
        train_df
    )

    clients = create_partition(
        patients
    )

    client_df = build_client_map(
        train_df,
        clients,
    )

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

        pneumonia = int(
            subset["pneumonia"].sum()
        )

        total = len(subset)

        print(
            f"Client {client_id}: "
            f"{total} images | "
            f"{subset['patient_key'].nunique()} patients | "
            f"Pneumonia: {pneumonia} | "
            f"No pneumonia: {total - pneumonia} | "
            f"Pneumonia ratio: "
            f"{pneumonia / total:.3f}"
        )

    validate(
        train_df,
        client_df,
    )

    output_dir = os.path.dirname(
        OUTPUT_FILE
    )

    if output_dir:
        os.makedirs(
            output_dir,
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