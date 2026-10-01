import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import CT_RAW_DIR

VALID_EXTENSIONS = {".tif", ".tiff", ".dcm", ".dicom", ".png", ".jpg", ".jpeg"}


def split_folder_contents(source_folder, train_ratio=0.8, seed=42):
    """
    Splits all files in a folder into 80% train and 20% test subfolders.
    Preserves class subfolders (e.g. CONTRAST, NO_CONTRAST) if present.
    """
    source_path = Path(source_folder)
    if not source_path.exists():
        print(f"Directory {source_path} does not exist. Skipping.")
        return

    # Check if files are grouped by subfolders (e.g., CONTRAST, NO_CONTRAST)
    subdirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.lower() not in ("train", "test", "val")]

    if subdirs:
        # Split per class subfolder
        for class_dir in subdirs:
            cls_name = class_dir.name
            files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            if not files:
                continue

            train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=seed)

            train_dest = source_path / "train" / cls_name
            test_dest = source_path / "test" / cls_name
            os.makedirs(train_dest, exist_ok=True)
            os.makedirs(test_dest, exist_ok=True)

            for f in train_files:
                shutil.move(str(f), str(train_dest / f.name))
            for f in test_files:
                shutil.move(str(f), str(test_dest / f.name))

            print(f"[{source_path.name}/{cls_name}] Split {len(files)} files: {len(train_files)} -> train, {len(test_files)} -> test")

            # Remove empty old directory if all files moved
            try:
                class_dir.rmdir()
            except OSError:
                pass
    else:
        # Flat files directly inside folder
        files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        if not files:
            print(f"No raw files found directly in {source_path}")
            return

        train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=seed)

        train_dest = source_path / "train"
        test_dest = source_path / "test"
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        for f in train_files:
            shutil.move(str(f), str(train_dest / f.name))
        for f in test_files:
            shutil.move(str(f), str(test_dest / f.name))

        print(f"[{source_path.name}] Split {len(files)} files: {len(train_files)} (80%) -> train, {len(test_files)} (20%) -> test")


def auto_split_raw_ct(ct_raw_dir=CT_RAW_DIR, train_ratio=0.8, seed=42):
    """Splits both TIFF and DICOM folders into 80% train and 20% test."""
    print("=" * 60)
    print(f"Auto-Splitting Raw CT Scans: {train_ratio*100:.0f}% Train / {(1-train_ratio)*100:.0f}% Test")
    print(f"Target Directory: {ct_raw_dir}")
    print("=" * 60)

    for fmt in ("tiff", "dicom"):
        target_dir = Path(ct_raw_dir) / fmt
        if target_dir.exists():
            split_folder_contents(target_dir, train_ratio=train_ratio, seed=seed)
        else:
            print(f"Folder {target_dir} not found. Creating it.")
            os.makedirs(target_dir, exist_ok=True)

    print("\nSplitting finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split single raw TIFF and DICOM files into 80% train and 20% test")
    parser.add_argument("--ct_dir", type=str, default=str(CT_RAW_DIR), help="Path to raw ct_scans root")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train ratio (default: 0.8 for 80/20)")
    args = parser.parse_args()

    auto_split_raw_ct(args.ct_dir, train_ratio=args.ratio)
