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


def load_checkpoint(model, checkpoint_path, device):
    """Load trained model weights from checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, checkpoint


def evaluate(model, loader, device):
    """Generate predictions and probabilities."""

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def calculate_metrics(
    labels,
    predictions,
    probabilities
):
    """Calculate classification metrics."""

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions
    ).ravel()

    specificity = tn / (tn + fp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def main():

    print("===== TEST SET EVALUATION =====")

    print("Device:", DEVICE)
    print("CSV file:", CSV_FILE)
    print("Path column:", PATH_COLUMN)

    # =========================
    # Test dataset
    # =========================

    print("\n===== LOADING TEST DATA =====")

    test_dataset = RSNADataset(
        CSV_FILE,
        "test",
        get_transforms(train=False),
        path_column=PATH_COLUMN
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    print(
        "Test batches:",
        len(test_loader)
    )

    # =========================
    # Model
    # =========================

    print("\n===== LOADING MODEL =====")

    model = create_model()

    model = model.to(DEVICE)

    checkpoint_path = os.path.join(
        MODEL_DIR,
        "densenet121_centralized_best.pth"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    if not os.path.exists(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    model, checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        DEVICE
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation loss:",
        checkpoint["val_loss"]
    )

    print(
        "Validation accuracy:",
        checkpoint["val_accuracy"]
    )

    # =========================
    # Evaluation
    # =========================

    print("\n===== EVALUATING =====")

    labels, predictions, probabilities = evaluate(
        model,
        test_loader,
        DEVICE
    )

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities
    )

    # =========================
    # Results
    # =========================

    print("\n===== TEST RESULTS =====")

    for name, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{name}: {value:.4f}"
            )

        else:

            print(
                f"{name}: {value}"
            )

    # =========================
    # Save metrics
    # =========================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "centralized_test_metrics.csv"
    )

    results_df = pd.DataFrame(
        [metrics]
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        "\nResults saved:",
        results_path
    )


if __name__ == "__main__":
    main()