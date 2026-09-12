# Centralized DenseNet121 Baseline

## Experiment

- Dataset: RSNA Pneumonia Detection Challenge 2018
- Task: Binary pneumonia classification
- Model: DenseNet121
- Pretrained weights: ImageNet
- Image size: 224 × 224
- Batch size: 16
- Epochs: 1
- Learning rate: 0.0001
- Weight decay: 0.0001
- Optimizer: AdamW
- Loss: Weighted Cross-Entropy Loss
- Random seed: 42
- Hardware: NVIDIA Tesla T4

## Dataset Split

The dataset was split at the patient level to prevent patient-level leakage.

| Split | Images |
|---|---:|
| Train | 16,010 |
| Validation | 3,235 |
| Test | 3,255 |
| Total | 22,500 |

Training class distribution:

| Class | Images |
|---|---:|
| No pneumonia | 5,296 |
| Pneumonia | 10,714 |

## Training Results

| Metric | Value |
|---|---:|
| Training Loss | 0.6774 |
| Training Accuracy | 59.53% |
| Validation Loss | 0.6682 |
| Validation Accuracy | 59.85% |
| Training Time | 9.09 minutes |

## Test Results

The saved best validation checkpoint was evaluated once on the held-out test set.

| Metric | Value |
|---|---:|
| Accuracy | 61.63% |
| Precision | 76.19% |
| Sensitivity / Recall | 61.59% |
| Specificity | 61.71% |
| F1-score | 68.11% |
| ROC-AUC | 65.22% |

### Confusion Matrix

| | Predicted No Pneumonia | Predicted Pneumonia |
|---|---:|---:|
| Actual No Pneumonia | 672 | 417 |
| Actual Pneumonia | 832 | 1,334 |

## Interpretation

This experiment serves as the initial centralized baseline for the project.

Only one training epoch was used, so these results should not be considered the final performance of DenseNet121. The baseline will be used as a reference for subsequent experiments involving longer training, explainability, and federated learning.

The held-out test set should not be used for hyperparameter tuning. Future model-development decisions should be based on the training and validation sets.

## Checkpoint

The trained checkpoint is stored outside GitHub in Google Drive:

`densenet121_centralized_best.pth`

The dataset and trained model weights are intentionally excluded from GitHub because of their large size.