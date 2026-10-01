import os
import torch
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Chest X-Ray Specific Paths
XRAY_RAW_DIR = RAW_DATA_DIR / "chest_xray"
XRAY_PROCESSED_DIR = PROCESSED_DATA_DIR / "chest_xray"

# CT Scan Specific Paths
CT_RAW_DIR = RAW_DATA_DIR / "ct_scans"
CT_PROCESSED_DIR = PROCESSED_DATA_DIR / "ct_scans"

# Output & Model Directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CONFUSION_MATRICES_DIR = OUTPUTS_DIR / "confusion_matrices"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Image Preprocessing Configurations
IMG_SIZE = 224
CHANNELS = 3
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# CT Scan Specific Configuration
CT_CLASS_NAMES = ["NO_CONTRAST", "CONTRAST"]
CT_WINDOWS = {
    "soft_tissue": (40, 400),    # Recommended for Contrast vs Non-Contrast (Center: 40 HU, Width: 400 HU)
    "mediastinum": (50, 350),    # Mediastinal contrast visualization
    "lung": (-600, 1500)         # Pulmonary parenchyma visualization
}

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if os.name != "nt" else 0  # Windows safe

