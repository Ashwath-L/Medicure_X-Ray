import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CT_RAW_DIR, CT_PROCESSED_DIR, SPLITS_DIR, IMG_SIZE, CT_WINDOWS
from src.preprocessing.utils import letterbox_resize

VALID_TIFF_EXTENSIONS = {".tif", ".tiff"}
VALID_DICOM_EXTENSIONS = {".dcm", ".dicom"}
VALID_EXTENSIONS = VALID_TIFF_EXTENSIONS.union(VALID_DICOM_EXTENSIONS).union({".png", ".jpg", ".jpeg"})


def window_ct_hu(image, window_center=40, window_width=400):
    """
    Applies Hounsfield Unit (HU) windowing to a CT scan.
    For Contrast vs. Non-Contrast detection, the clinical standard is:
      - Soft Tissue Window: Center = 40 HU, Width = 400 HU (range [-160, 240])
      - Mediastinum Window: Center = 50 HU, Width = 350 HU
    For pulmonary parenchyma, use Lung Window: Center = -600 HU, Width = 1500 HU.
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    normalized = ((windowed - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    return normalized


def process_dicom_slice(dcm_path, window_name="soft_tissue", img_size=IMG_SIZE):
    """
    Reads a DICOM file, handles MONOCHROME1/2 photometric inversion,
    applies RescaleSlope/Intercept to HU, windows, and applies letterboxing.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required for DICOM processing. Run: pip install pydicom")

    try:
        dcm = pydicom.dcmread(str(dcm_path))
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation
        # MONOCHROME1: minimum value represents white (needs inversion)
        # MONOCHROME2: minimum value represents black (standard)
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixel_array = np.amax(pixel_array) - pixel_array

        # Convert to Hounsfield Units (HU)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        slope = getattr(dcm, "RescaleSlope", 1)
        hu_image = pixel_array * slope + intercept

        # Apply HU Windowing
        center, width = CT_WINDOWS.get(window_name, (40, 400))
        img = window_ct_hu(hu_image, window_center=center, window_width=width)

        # Aspect-ratio preserved letterboxing
        img = letterbox_resize(img, target_size=img_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img_rgb
    except Exception as e:
        print(f"  [skip] Failed to read DICOM {dcm_path.name}: {e}")
        return None


def process_tiff_slice(tiff_path, img_size=IMG_SIZE):
    """
    Reads 16-bit or 8-bit TIFF medical images without clipping,
    normalizes to [0, 255], and applies letterboxing.
    """
    try:
        import tifffile
        img = tifffile.imread(str(tiff_path))
    except Exception:
        img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"  [skip] Failed to read TIFF {tiff_path.name}")
        return None

    # Handle multi-channel / multi-page TIFF: take first slice/channel if 3D
    if img.ndim == 3:
        if img.shape[0] < img.shape[2]:  # (slices, H, W)
            img = img[0]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)

    # Robust percentile normalization for medical TIFFs
    p1, p99 = np.percentile(img, (1, 99))
    if p99 > p1:
        img = np.clip(img, p1, p99)
        img = ((img - p1) / (p99 - p1) * 255.0).astype(np.uint8)
    else:
        min_v, max_v = np.min(img), np.max(img)
        if max_v > min_v:
            img = ((img - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    img = letterbox_resize(img, target_size=img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb


def discover_ct_files(raw_dir, train_ratio=0.8, seed=42):
    """
    Intelligently discovers CT scans:
    1. If files already reside in train/test subfolders, respects existing splits.
    2. If files are in a single directory (e.g. all in tiff/ or dicom/),
       automatically partitions them into 80% train and 20% test.
    """
    from sklearn.model_selection import train_test_split
    raw_path = Path(raw_dir)

    unassigned_by_group = {}  # key: (fmt, label) -> list of file_paths
    final_items = []

    for file_path in raw_path.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in VALID_EXTENSIONS:
            continue

        parts = [p.upper() for p in file_path.relative_to(raw_path).parts]

        # 1. Identify format: tiff vs dicom
        if ext in VALID_TIFF_EXTENSIONS or "TIFF" in parts:
            fmt = "tiff"
        elif ext in VALID_DICOM_EXTENSIONS or "DICOM" in parts or "DCM" in parts:
            fmt = "dicom"
        else:
            fmt = "tiff" if ext in {".png", ".jpg", ".jpeg"} else "dicom"

        # 2. Identify label: CONTRAST vs NO_CONTRAST
        label = "NO_CONTRAST"
        for p in parts:
            if "CONTRAST_0" in p or "NO_CONTRAST" in p or "NON_CONTRAST" in p or "WITHOUT" in p:
                label = "NO_CONTRAST"
                break
            elif "CONTRAST_1" in p or "CONTRAST" in p:
                label = "CONTRAST"
                break

        # 3. Check for explicit train/test split in path
        has_split = False
        explicit_split = "train"
        for s in ("TRAIN", "TEST", "VAL"):
            if s in parts:
                explicit_split = s.lower()
                has_split = True
                break

        if has_split:
            final_items.append({
                "file_path": file_path,
                "format": fmt,
                "split": explicit_split,
                "label": label
            })
        else:
            key = (fmt, label)
            if key not in unassigned_by_group:
                unassigned_by_group[key] = []
            unassigned_by_group[key].append(file_path)

    # 4. Automatically split single/unassigned files into 80% train and 20% test
    for (fmt, label), file_list in unassigned_by_group.items():
        if len(file_list) >= 2:
            train_files, test_files = train_test_split(file_list, train_size=train_ratio, random_state=seed)
        elif len(file_list) == 1:
            train_files, test_files = file_list, []
        else:
            continue

        print(f"  [Auto-Split 80/20] {fmt.upper()} ({label}): {len(train_files)} -> train (80%), {len(test_files)} -> test (20%)")

        for f in train_files:
            final_items.append({
                "file_path": f,
                "format": fmt,
                "split": "train",
                "label": label
            })
        for f in test_files:
            final_items.append({
                "file_path": f,
                "format": fmt,
                "split": "test",
                "label": label
            })

    return final_items


def run_ct_preprocessing(input_dir=CT_RAW_DIR, output_dir=CT_PROCESSED_DIR, img_size=IMG_SIZE, window_name="soft_tissue", split_ratio=0.8):
    print(f"Starting Optimized CT Scan Preprocessing...")
    print(f"Source: {input_dir}")
    print(f"Destination: {output_dir}")
    print(f"HU Window: {window_name} {CT_WINDOWS.get(window_name, '')} | Target Size: {img_size}x{img_size} | Split: {split_ratio*100:.0f}% Train")

    items = discover_ct_files(input_dir, train_ratio=split_ratio)
    if not items:
        print(f"\nNo CT files found in {input_dir}.")
        print("Expected structure in data/raw/ct_scans/:")
        print("  ├── tiff/ (single folder of .tif files or train/test subfolders)")
        print("  └── dicom/ (single folder of .dcm files or train/test subfolders)")
        return

    print(f"Found {len(items)} CT files across all splits and formats.")
    records = []

    for item in tqdm(items, desc="Preprocessing CT Scans"):
        src = item["file_path"]
        fmt = item["format"]
        split = item["split"]
        label = item["label"]

        if fmt == "dicom":
            processed_img = process_dicom_slice(src, window_name=window_name, img_size=img_size)
        else:
            processed_img = process_tiff_slice(src, img_size=img_size)

        if processed_img is None:
            continue

        # Target structure: data/processed/ct_scans/{format}/{split}/{label}/{filename}.png
        out_folder = Path(output_dir) / fmt / split / label
        out_folder.makedirs_p() if hasattr(out_folder, "makedirs_p") else os.makedirs(out_folder, exist_ok=True)
        dst_path = out_folder / f"{src.stem}.png"

        # Save processed RGB image as PNG
        cv2.imwrite(str(dst_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

        records.append({
            "original_path": str(src),
            "processed_path": str(dst_path),
            "format": fmt,
            "split": split,
            "label": label,
            "label_idx": 0 if label == "NO_CONTRAST" else 1
        })

    if records:
        df = pd.DataFrame(records)
        os.makedirs(SPLITS_DIR, exist_ok=True)
        split_csv = SPLITS_DIR / "ct_dataset_metadata.csv"
        df.to_csv(split_csv, index=False)
        print(f"\nPreprocessing Complete: {len(records)}/{len(items)} processed successfully.")
        print(f"Saved CT metadata index to: {split_csv}")
        print("\nClass distribution:")
        print(df.groupby(["format", "split", "label"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CT Scans (TIFF & DICOM with Soft Tissue Windowing).")
    parser.add_argument("--input_dir", type=str, default=str(CT_RAW_DIR), help="Path to raw CT dataset")
    parser.add_argument("--output_dir", type=str, default=str(CT_PROCESSED_DIR), help="Path to save processed CT scans")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help="Target image size (default: 224)")
    parser.add_argument("--window", type=str, default="soft_tissue", choices=list(CT_WINDOWS.keys()),
                        help="CT HU window preset: 'soft_tissue' (recommended for contrast), 'mediastinum', or 'lung'")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train split ratio (default: 0.8 for 80% train, 20% test) when files are in a single folder")
    args = parser.parse_args()

    run_ct_preprocessing(args.input_dir, args.output_dir, args.size, window_name=args.window, split_ratio=args.split_ratio)

