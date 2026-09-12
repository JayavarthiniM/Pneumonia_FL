import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from rsna_dataset import RSNADataset, get_transforms
from model import create_model

from config import (
    CSV_FILE,
    PATH_COLUMN,
    BATCH_SIZE,
    DEVICE,
    MODEL_DIR,
    RESULTS_DIR,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "densenet121_fedavg_best.pth",
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "fedavg_test_metrics.csv",
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("===== FEDAVG TEST EVALUATION =====")

    print("Device:", DEVICE)
    print("Model: DenseNet121")
    print("CSV file:", CSV_FILE)
    print("Path column:", PATH_COLUMN)
    print("Model checkpoint:", MODEL_PATH)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"FedAvg checkpoint not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    test_dataset = RSNADataset(
        CSV_FILE,
        "test",
        get_transforms(train=False),
        path_column=PATH_COLUMN,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "Test images:",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # LOAD FEDAVG MODEL
    # --------------------------------------------------------

    print("\n===== LOADING FEDAVG MODEL =====")

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    # The checkpoint created by fedavg.py
    # contains model_state_dict.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        "Checkpoint round:",
        checkpoint.get("round"),
    )

    print(
        "Validation loss:",
        checkpoint.get("validation_loss"),
    )

    print(
        "Validation accuracy:",
        checkpoint.get("validation_accuracy"),
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\n===== RUNNING TEST INFERENCE =====")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Probability of pneumonia class = 1
            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    y_prob = np.array(
        all_probabilities
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    # Confusion matrix:
    #
    # [[TN, FP],
    #  [FN, TP]]
    #
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print(
        "\n===== FEDAVG TEST RESULTS ====="
    )

    print(
        f"Accuracy:           {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision:          {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Sensitivity/Recall: {sensitivity:.4f} "
        f"({sensitivity * 100:.2f}%)"
    )

    print(
        f"Specificity:        {specificity:.4f} "
        f"({specificity * 100:.2f}%)"
    )

    print(
        f"F1 Score:           {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"ROC-AUC:            {roc_auc:.4f} "
        f"({roc_auc * 100:.2f}%)"
    )

    print("\nConfusion Matrix:")

    print(
        f"TN: {tn}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"FN: {fn}"
    )

    print(
        f"TP: {tp}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "DenseNet121",
                "algorithm": "FedAvg",
                "checkpoint_round":
                    checkpoint.get("round"),
                "test_images":
                    len(test_dataset),
                "accuracy":
                    accuracy,
                "precision":
                    precision,
                "sensitivity_recall":
                    sensitivity,
                "specificity":
                    specificity,
                "f1":
                    f1,
                "roc_auc":
                    roc_auc,
                "true_negative":
                    tn,
                "false_positive":
                    fp,
                "false_negative":
                    fn,
                "true_positive":
                    tp,
            }
        ]
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved metrics:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\n===== EVALUATION COMPLETE ====="
    )


if __name__ == "__main__":
    main()