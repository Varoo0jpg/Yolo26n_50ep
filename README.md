# YOLO26n — Training and Audit Artifacts

Repository for the YOLO26n vehicle-detection experiment and its dataset-audit materials.

## 1. Main locations

| Purpose | Location |
|---|---|
| Local model project folder (anonymized path) | `C:\\LampuS\\Smart Traffic Light\\YOLO26n_50ep` |
| Local dataset folder (anonymized path) | `C:\\LampuS\\Adalah` |
| Colab training output | `/content/runs_yolo26/colab_yolo26n_50ep/` |
| Colab dataset configuration (anonymized path) | `/content/Adalah/data_cloud.yaml` |

The local paths above are privacy-safe aliases for documentation. They do not rename folders on the user's computer. The Colab output directory is recorded in [args.yaml](args.yaml).

## 2. Visual artifacts already tracked in this repository

| Artifact | Repository path | Use |
|---|---|---|
| Dataset contact sheet | [contact_sheet.jpg](contact_sheet.jpg) | Overview of the dataset images |
| Frame verification example | [frame_verification/frame_0036.jpg](frame_verification/frame_0036.jpg) | Manual visual inspection |
| Validation confusion matrix, confidence 0.25 | [audit_error/val_conf025/confusion_matrix.png](audit_error/val_conf025/confusion_matrix.png) | Class-level validation errors at the selected confidence |
| Validation confusion matrix, confidence 0.01 | [audit_error/val_conf001/confusion_matrix.png](audit_error/val_conf001/confusion_matrix.png) | Comparison at a lower confidence threshold |

These are audit/validation artifacts. Do not describe them as training-run outputs unless their source run and settings are independently confirmed.

## 3. Training-run outputs to look for in Colab

The recorded training run directory is:

`/content/runs_yolo26/colab_yolo26n_50ep/`

If generated and retained by that run, the useful files are:

- `results.png` — training/validation loss and metric curves.
- `results.csv` — per-epoch metrics.
- `confusion_matrix.png` and `confusion_matrix_normalized.png` — validation classification outcomes.
- `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png` — detection metric curves.
- `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` — training images with augmented labels.
- `val_batch0_labels.jpg`, `val_batch0_pred.jpg` (and other numbered batches) — ground-truth and predicted validation boxes.
- `weights/best.pt` and `weights/last.pt` — model checkpoints.
- `args.yaml` — recorded training configuration.

The list above is a checklist of expected Ultralytics artifacts, not confirmation that every file is currently committed to this repository. The validation batch prediction images are the most useful visual examples for showing bounding boxes alongside ground-truth labels.

## 4. Paper usage guidance

- Use the confusion matrix and metric curves for quantitative model evaluation.
- Use paired `val_batch*_labels.jpg` and `val_batch*_pred.jpg` images to illustrate ground-truth versus predicted bounding boxes.
- Use `train_batch*.jpg` only to document training-label examples/augmentation, not as proof of final model accuracy.
- Use contact sheets and frame-verification images to explain dataset auditing, not as substitutes for held-out test results.
- Report the split, confidence threshold, IoU convention, and evaluation run associated with every result figure.

