import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import NORM_MEAN, NORM_STD


def letterbox_resize(img, target_size=224):
    """
    Resize image preserving aspect ratio and padding symmetrically with zeros (black).
    Prevents anatomical distortion of the thoracic cage and cardiac silhouette.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    canvas = np.zeros((target_size, target_size), dtype=img.dtype)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances subtle pulmonary infiltrates, consolidations, and reticular patterns.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def get_transforms(is_train=True, img_size=224):
    """
    Medical-grade data augmentation:
    - Conservative rotation and scale
    - CoarseDropout (Cutout) to mitigate shortcut learning (e.g. 'R'/'L' orientation markers)
    - ImageNet normalization for transfer learning backbones
    """
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.04, 0.04), rotate=(-8, 8), border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(12, 28), hole_width_range=(12, 28), fill=0, p=0.5),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2()
        ])


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for on-the-fly image loading and augmentation.
    Prevents RAM exhaustion by not loading all raw images simultaneously into memory.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
