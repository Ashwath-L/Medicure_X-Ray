# Medical Image Classification: Chest X-Ray & CT Scans

Production-grade Deep Learning pipeline for detecting abnormalities (such as **Pneumonia**) in Chest Radiographs and CT Scans using **DenseNet-121** (CheXNet standard), **Letterbox Aspect-Ratio Padding**, and **CLAHE** contrast enhancement.

---

## 📁 Project Architecture

```text
data/
│   ├── raw/
│   │   ├── chest_xray/
│   │   │   ├── train/
│   │   │   │   ├── NORMAL/
│   │   │   │   └── PNEUMONIA/
│   │   │   └── test/
│   │   │       ├── NORMAL/
│   │   │       └── PNEUMONIA/
│   │   └── ct_scans/
│   │       ├── train/
│   │       └── test/
│   ├── processed/                 # output of preprocessing (resized/grayscale/normalized)
│   │   ├── chest_xray/
│   │   └── ct_scans/
│   └── splits/                    # train/val/test index files (csv or txt)
│
├── src/
│   ├── preprocessing/
│   │   ├── preprocess_xray.py     # resize, grayscale, letterbox, CLAHE
│   │   ├── preprocess_ct.py       # HU windowing, normalization, slice extraction
│   │   └── utils.py               # shared helpers (loading, augmentation, dataset)
│   ├── models/
│   │   ├── cnn_model.py           # DenseNet-121, ResNet-50, and Custom CNN
│   │   ├── train.py               # weighted training loop, cosine annealing
│   │   └── evaluate.py            # accuracy, recall, precision, AUC, confusion matrix
│   ├── inference/
│   │   └── predict.py             # run trained model on new images
│   └── config.py                  # centralized paths, dimensions, hyperparameters
│
├── notebooks/                     # exploratory work and validation
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_check.ipynb
│   └── 03_model_training.ipynb
│
├── models/                        # saved trained weights
│   ├── xray_cnn_best.pth
│   └── ct_cnn_best.pth
│
├── outputs/
│   ├── logs/                      # loss and accuracy training histories
│   ├── metrics/                   # evaluation reports (JSON/CSV)
│   ├── confusion_matrices/        # saved confusion matrix heatmap plots
│   └── predictions/               # inference output predictions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Install all required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart Workflow

### 1. Data Preprocessing (Letterbox + CLAHE + Auto-Split)
Preprocess raw Chest X-Rays without anatomical distortion:
```bash
python -m src.preprocessing.preprocess_xray
```

For CT Scans (automatically splits un-split TIFFs/DICOMs into **80% train / 20% test**):
```bash
python -m src.preprocessing.preprocess_ct --window soft_tissue --split_ratio 0.8
```

*(Optional) If you want to physically split your raw files into `train/` and `test/` folders beforehand:*
```bash
python -m src.preprocessing.split_ct_data --ratio 0.8
```

### 2. Model Training
Train **DenseNet-121** on Chest X-Rays:
```bash
python -m src.models.train --task xray --model densenet121 --epochs 20 --batch_size 32
```
Train **DenseNet-121** on CT Scans (Contrast vs No-Contrast):
```bash
python -m src.models.train --task ct --model densenet121 --epochs 20 --batch_size 32
```

### 3. Evaluation
Evaluate on the test set, generate ROC-AUC, sensitivity, and save the confusion matrix:
```bash
python -m src.models.evaluate --model_path models/xray_cnn_best.pth
```

### 4. Inference
Run prediction on a single image or an entire folder of scans:
```bash
python -m src.inference.predict --image "path/to/xray_image.jpeg"
```
