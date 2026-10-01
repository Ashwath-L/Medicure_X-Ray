import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import XRAY_RAW_DIR, XRAY_PROCESSED_DIR, SPLITS_DIR, IMG_SIZE
from src.preprocessing.utils import letterbox_resize, apply_clahe

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def process_single_image(src_path, dst_path, img_size=IMG_SIZE, use_clahe=True):
    """Reads grayscale X-ray, enhances contrast with CLAHE, applies letterbox padding, and saves as 3-channel PNG."""
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    if use_clahe:
        img = apply_clahe(img)

    img = letterbox_resize(img, target_size=img_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    os.makedirs(dst_path.parent, exist_ok=True)
    cv2.imwrite(str(dst_path), img_rgb)
    return True


def run_xray_preprocessing(input_dir=XRAY_RAW_DIR, output_dir=XRAY_PROCESSED_DIR, img_size=IMG_SIZE, use_clahe=True):
    print(f"Starting Chest X-Ray Preprocessing...")
    print(f"Source: {input_dir}")
    print(f"Destination: {output_dir}")
    print(f"Image Size: {img_size}x{img_size} | CLAHE: {use_clahe}")

    records = []
    splits = [s for s in ("train", "test", "val") if (Path(input_dir) / s).is_dir()]
    if not splits:
        print(f"No train/test folders found in {input_dir}. Please place raw dataset in {input_dir}")
        return

    for split in splits:
        split_dir = Path(input_dir) / split
        classes = [c for c in os.listdir(split_dir) if (split_dir / c).is_dir()]

        for cls in classes:
            class_dir = split_dir / cls
            files = [f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
            print(f"Processing [{split}/{cls}]: {len(files)} images...")

            for src_file in tqdm(files, desc=f"{split}/{cls}"):
                dst_file = Path(output_dir) / split / cls / f"{src_file.stem}.png"
                success = process_single_image(src_file, dst_file, img_size=img_size, use_clahe=use_clahe)
                if success:
                    records.append({
                        "original_path": str(src_file),
                        "processed_path": str(dst_file),
                        "split": split,
                        "label": cls,
                        "label_idx": 0 if cls.upper() == "NORMAL" else 1
                    })

    if records:
        df = pd.DataFrame(records)
        os.makedirs(SPLITS_DIR, exist_ok=True)
        split_csv = SPLITS_DIR / "xray_dataset_metadata.csv"
        df.to_csv(split_csv, index=False)
        print(f"Saved dataset metadata and paths to {split_csv}")

    print("Chest X-Ray preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Chest X-Ray images (Letterbox + CLAHE).")
    parser.add_argument("--input_dir", type=str, default=str(XRAY_RAW_DIR), help="Path to raw chest X-ray dataset")
    parser.add_argument("--output_dir", type=str, default=str(XRAY_PROCESSED_DIR), help="Path to save processed images")
    parser.add_argument("--size", type=int, default=IMG_SIZE, help="Target image size")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE enhancement")
    args = parser.parse_args()

    run_xray_preprocessing(args.input_dir, args.output_dir, args.size, use_clahe=not args.no_clahe)
