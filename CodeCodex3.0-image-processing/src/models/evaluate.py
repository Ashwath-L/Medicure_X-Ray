import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from torch.utils.data import DataLoader

from src.config import (
    MODELS_DIR, METRICS_DIR, CONFUSION_MATRICES_DIR, SPLITS_DIR,
    CLASS_NAMES, DEVICE, NUM_WORKERS, NUM_CLASSES, XRAY_PROCESSED_DIR, PROCESSED_DATA_DIR
)
from src.models.cnn_model import build_model
from src.preprocessing.utils import MedicalImageDataset, get_transforms


def evaluate_model(model_path=None, test_csv=None, model_name="densenet121", batch_size=32, task="xray", ensemble=False):
    class_names = ["NO_CONTRAST", "CONTRAST"] if task == "ct" else ["NORMAL", "PNEUMONIA"]

    if ensemble:
        fold_paths = sorted(list(MODELS_DIR.glob(f"{task}_densenet_fold_*.pth")))
        if not fold_paths:
            print(f"No fold checkpoints found for {task}. Falling back to single model...")
            fold_paths = [Path(model_path)] if model_path else [MODELS_DIR / f"{task}_cnn_best.pth"]
        print(f"Loading {len(fold_paths)}-fold ensemble weights for [{task.upper()}] task...")
        models = []
        for fp in fold_paths:
            m = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
            m.load_state_dict(torch.load(fp, map_location=DEVICE))
            m.to(DEVICE)
            m.eval()
            models.append(m)
    else:
        print(f"Loading weights from {model_path} for [{task.upper()}] task...")
        m = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
        m.load_state_dict(torch.load(model_path, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        models = [m]

    # Discover test samples
    if test_csv and os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        test_df = df[df["split"] == "test"]
    else:
        # Fallback to scanning processed/test
        records = []
        base_dir = PROCESSED_DATA_DIR / ("ct_scans" if task == "ct" else "chest_xray")
        for cls in class_names:
            for c_dir in base_dir.rglob(cls):
                if "test" in str(c_dir).lower() and c_dir.is_dir():
                    for f in c_dir.iterdir():
                        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                            records.append({
                                "processed_path": str(f),
                                "label_idx": 0 if cls == class_names[0] else 1
                            })
        test_df = pd.DataFrame(records)

    if test_df.empty:
        print("No test samples found to evaluate.")
        return

    test_dataset = MedicalImageDataset(
        test_df["processed_path"].values,
        test_df["label_idx"].values,
        transform=get_transforms(is_train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            # Average softmax across models in ensemble
            batch_probs = torch.stack([torch.softmax(m(images), dim=1) for m in models], dim=0).mean(dim=0)
            preds = torch.argmax(batch_probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(batch_probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }

    print("\n" + "="*40)
    print("EVALUATION RESULTS:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("="*40)

    # Save metrics JSON
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(METRICS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Confusion Matrix Plot
    os.makedirs(CONFUSION_MATRICES_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({task.upper()} Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = CONFUSION_MATRICES_DIR / f"{task}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on test split")
    parser.add_argument("--task", type=str, default="xray", choices=["xray", "ct"], help="Task: xray | ct")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="densenet121")
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--ensemble", action="store_true", help="Evaluate 5-fold ensemble instead of single checkpoint")
    args = parser.parse_args()

    default_model_path = str(MODELS_DIR / ("ct_cnn_best.pth" if args.task == "ct" else "xray_cnn_best.pth"))
    model_path = args.model_path or default_model_path
    test_csv = args.test_csv or str(SPLITS_DIR / ("ct_dataset_metadata.csv" if args.task == "ct" else "xray_dataset_metadata.csv"))

    evaluate_model(model_path, test_csv=test_csv, model_name=args.model_name, task=args.task, ensemble=args.ensemble)

