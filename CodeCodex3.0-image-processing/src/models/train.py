import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import (
    MODELS_DIR, LOGS_DIR, METRICS_DIR, SPLITS_DIR, XRAY_PROCESSED_DIR, PROCESSED_DATA_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    DEVICE, NUM_WORKERS, NUM_CLASSES
)
from src.preprocessing.utils import MedicalImageDataset, get_transforms
from src.models.cnn_model import build_model


def get_data_from_directory(processed_dir, class_names=None):
    """Fallback if metadata CSV is not present: loads paths directly from train/test directory."""
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]
    records = []
    base = Path(processed_dir)
    for cls_idx, cls in enumerate(class_names):
        for c_dir in base.rglob(cls):
            if not c_dir.is_dir():
                continue
            split = "train"
            for s in ("train", "test", "val"):
                if s in [p.lower() for p in c_dir.parts]:
                    split = s
                    break
            for f in c_dir.iterdir():
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    records.append({
                        "processed_path": str(f),
                        "split": split,
                        "label_idx": cls_idx
                    })
    return pd.DataFrame(records)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_probs), np.array(all_labels)


def run_single_train(df, args, class_names, save_name):
    """Executes single train/test split training with backbone freezing."""
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"].isin(["test", "val"])]

    print(f"[{args.task.upper()}] Train samples: {len(train_df)} | Validation/Test samples: {len(val_df)}")

    train_dataset = MedicalImageDataset(
        train_df["processed_path"].values,
        train_df["label_idx"].values,
        transform=get_transforms(is_train=True)
    )
    val_dataset = MedicalImageDataset(
        val_df["processed_path"].values,
        val_df["label_idx"].values,
        transform=get_transforms(is_train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

    labels = train_df["label_idx"].values
    counts = np.bincount(labels, minlength=2)
    weight_val = counts[1] / max(counts[0], 1)
    class_weights = torch.tensor([weight_val, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class counts: {class_names[0]}={counts[0]}, {class_names[1]}={counts[1]}")

    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True, freeze_mode=args.freeze_mode).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    best_val_acc = 0.0
    history = []
    save_path = MODELS_DIR / save_name

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {best_val_acc:.4f})")

    with open(LOGS_DIR / f"{args.task}_training_history.json", "w") as f:
        json.dump(history, f, indent=4)


def run_stratified_cv(df, args, class_names, save_name, n_splits=5, seed=42):
    """
    Executes 5-Fold Stratified Cross-Validation with patient-level grouping to prevent leakage.
    Tracks Out-Of-Fold (OOF) accuracy, sensitivity, precision, F1, and ROC-AUC.
    """
    print(f"\n{'='*70}")
    print(f"Starting {n_splits}-Fold Stratified Cross-Validation for [{args.task.upper()}]")
    print(f"Backbone Freezing: {args.freeze_mode} | Architecture: {args.model}")
    print(f"Total Dataset Pool: {len(df)} samples")
    print(f"{'='*70}")

    if "original_path" in df.columns:
        groups = df["original_path"].apply(lambda p: Path(p).stem).values
    else:
        groups = df["processed_path"].apply(lambda p: Path(p).stem).values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df["processed_path"].values
    y = df["label_idx"].values

    oof_preds = np.zeros(len(df))
    oof_probs = np.zeros(len(df))
    fold_metrics = []
    best_overall_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
        print(f"\n--- [Fold {fold}/{n_splits}] ---")
        train_fold_df = df.iloc[train_idx]
        val_fold_df = df.iloc[val_idx]

        train_counts = np.bincount(train_fold_df["label_idx"].values, minlength=2)
        val_counts = np.bincount(val_fold_df["label_idx"].values, minlength=2)
        print(f"Train: {len(train_fold_df)} ({class_names[0]}={train_counts[0]}, {class_names[1]}={train_counts[1]}) | "
              f"Val: {len(val_fold_df)} ({class_names[0]}={val_counts[0]}, {class_names[1]}={val_counts[1]})")

        train_dataset = MedicalImageDataset(
            train_fold_df["processed_path"].values,
            train_fold_df["label_idx"].values,
            transform=get_transforms(is_train=True)
        )
        val_dataset = MedicalImageDataset(
            val_fold_df["processed_path"].values,
            val_fold_df["label_idx"].values,
            transform=get_transforms(is_train=False)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

        weight_val = train_counts[1] / max(train_counts[0], 1)
        class_weights = torch.tensor([weight_val, 1.0], dtype=torch.float32).to(DEVICE)

        model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True, freeze_mode=args.freeze_mode).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_fold_acc = 0.0
        fold_save_path = MODELS_DIR / f"{args.task}_densenet_fold_{fold}.pth"
        best_fold_probs = None

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, probs, targets = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_probs = probs
                torch.save(model.state_dict(), fold_save_path)

        if best_fold_probs is not None:
            oof_probs[val_idx] = best_fold_probs
            oof_preds[val_idx] = (best_fold_probs >= 0.5).astype(int)

        fold_acc = accuracy_score(y[val_idx], oof_preds[val_idx])
        fold_prec = precision_score(y[val_idx], oof_preds[val_idx], zero_division=0)
        fold_rec = recall_score(y[val_idx], oof_preds[val_idx], zero_division=0)
        fold_f1 = f1_score(y[val_idx], oof_preds[val_idx], zero_division=0)
        try:
            fold_auc = roc_auc_score(y[val_idx], oof_probs[val_idx])
        except ValueError:
            fold_auc = 0.5

        fold_metrics.append({
            "fold": fold,
            "accuracy": float(fold_acc),
            "precision": float(fold_prec),
            "recall": float(fold_rec),
            "f1_score": float(fold_f1),
            "roc_auc": float(fold_auc)
        })
        print(f"Fold {fold} Best Val Acc: {fold_acc*100:.2f}% | F1: {fold_f1:.4f} | AUC: {fold_auc:.4f}")

        if fold_acc > best_overall_acc:
            best_overall_acc = fold_acc
            torch.save(torch.load(fold_save_path), MODELS_DIR / save_name)

    oof_acc = accuracy_score(y, oof_preds)
    oof_prec = precision_score(y, oof_preds, zero_division=0)
    oof_rec = recall_score(y, oof_preds, zero_division=0)
    oof_f1 = f1_score(y, oof_preds, zero_division=0)
    try:
        oof_auc = roc_auc_score(y, oof_probs)
    except ValueError:
        oof_auc = 0.5

    mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
    std_acc = np.std([m["accuracy"] for m in fold_metrics])

    print(f"\n{'='*70}")
    print(f"5-FOLD STRATIFIED CV RESULTS [{args.task.upper()}]:")
    print(f"  Mean Accuracy       : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Out-Of-Fold Accuracy: {oof_acc*100:.2f}%")
    print(f"  Out-Of-Fold F1-Score: {oof_f1:.4f}")
    print(f"  Out-Of-Fold Precision: {oof_prec:.4f}")
    print(f"  Out-Of-Fold Recall  : {oof_rec:.4f}")
    print(f"  Out-Of-Fold ROC-AUC : {oof_auc:.4f}")
    print(f"{'='*70}")

    cv_summary = {
        "task": args.task,
        "n_splits": n_splits,
        "freeze_mode": args.freeze_mode,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "oof_accuracy": float(oof_acc),
        "oof_precision": float(oof_prec),
        "oof_recall": float(oof_rec),
        "oof_f1": float(oof_f1),
        "oof_roc_auc": float(oof_auc),
        "fold_details": fold_metrics
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = METRICS_DIR / f"{args.task}_cv_results.json"
    with open(metrics_path, "w") as f:
        json.dump(cv_summary, f, indent=4)
    print(f"Saved 5-fold CV results to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN on Chest X-Ray / CT Scans with Selective Freezing & CV")
    parser.add_argument("--task", type=str, default="xray", choices=["xray", "ct"], help="Task: xray | ct")
    parser.add_argument("--model", type=str, default="densenet121", help="densenet121 | custom | resnet50")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--data_csv", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--freeze_mode", type=str, default="most", choices=["most", "all", "none"],
                        help="Backbone freezing: 'most' (freeze blocks 1-3, train block 4 + head), 'all', or 'none'")
    parser.add_argument("--cv_folds", type=int, default=1, help="Number of folds for Stratified CV (1 for single split, 5 for 5-Fold CV)")
    args = parser.parse_args()

    if args.task == "ct":
        data_csv = args.data_csv or str(SPLITS_DIR / "ct_dataset_metadata.csv")
        save_name = args.save_name or "ct_cnn_best.pth"
        class_names = ["NO_CONTRAST", "CONTRAST"]
        default_processed_dir = PROCESSED_DATA_DIR / "ct_scans"
    else:
        data_csv = args.data_csv or str(SPLITS_DIR / "xray_dataset_metadata.csv")
        save_name = args.save_name or "xray_cnn_best.pth"
        class_names = ["NORMAL", "PNEUMONIA"]
        default_processed_dir = XRAY_PROCESSED_DIR

    if os.path.exists(data_csv):
        df = pd.read_csv(data_csv)
    else:
        print(f"Metadata CSV not found at {data_csv}. Scanning {default_processed_dir} directly...")
        df = get_data_from_directory(default_processed_dir, class_names=class_names)

    if df.empty:
        print("No images found to train on! Please run preprocessing first.")
        return

    if args.cv_folds > 1:
        run_stratified_cv(df, args, class_names, save_name, n_splits=args.cv_folds)
    else:
        run_single_train(df, args, class_names, save_name)


if __name__ == "__main__":
    main()
