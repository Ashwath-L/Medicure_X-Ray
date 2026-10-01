import os
import json
import base64
from pathlib import Path
import io
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    MODELS_DIR, CONFUSION_MATRICES_DIR, METRICS_DIR,
    CLASS_NAMES, CT_CLASS_NAMES, DEVICE, NUM_CLASSES, IMG_SIZE
)
from src.models.cnn_model import build_model
from src.preprocessing.utils import letterbox_resize, apply_clahe, get_transforms
from src.preprocessing.preprocess_ct import window_ct_hu, process_tiff_slice, process_dicom_slice

app = FastAPI(title="RadPulse AI Medical Suite", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache to allow instant inference
MODEL_CACHE = {}

def get_loaded_models(task="ct", ensemble=True):
    key = f"{task}_{'ens' if ensemble else 'single'}"
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    models = []
    if ensemble and task == "ct":
        fold_paths = sorted(list(MODELS_DIR.glob("ct_densenet_fold_*.pth")))
        if not fold_paths:
            fold_paths = [MODELS_DIR / "ct_cnn_best.pth"]
        for p in fold_paths:
            m = build_model("densenet121", num_classes=NUM_CLASSES, pretrained=False)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
            m.to(DEVICE).eval()
            models.append(m)
    else:
        model_name = "ct_cnn_best.pth" if task == "ct" else "xray_cnn_best.pth"
        path = MODELS_DIR / model_name
        if path.exists():
            m = build_model("densenet121", num_classes=NUM_CLASSES, pretrained=False)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.to(DEVICE).eval()
            models.append(m)

    MODEL_CACHE[key] = models
    return models


def generate_attention_heatmap(img_rgb, model):
    """
    Computes a visual gradient-based saliency map highlighting
    anatomical regions critical to the classification decision.
    """
    transform = get_transforms(is_train=False)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True

    output = model(tensor)
    pred_idx = output.argmax(dim=1).item()
    score = output[0, pred_idx]
    score.backward()

    # Saliency via input gradients
    gradients = tensor.grad.data.abs().squeeze().cpu().numpy()
    heatmap = np.max(gradients, axis=0)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay on original 224x224 RGB
    blended = np.clip(0.6 * img_rgb + 0.4 * heatmap_color, 0, 255).astype(np.uint8)
    
    _, buffer = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/api/samples")
def get_sample_scans():
    """Provides genuine curated sample scans for instant 1-click clinical testing."""
    samples = []
    
    # Chest X-Ray samples
    xray_test = Path("data/processed/chest_xray/test")
    if xray_test.exists():
        for label, pretty in [("NORMAL", "Normal Lung (Clear)"), ("PNEUMONIA", "Bacterial Pneumonia (Infiltration)")]:
            files = list((xray_test / label).glob("*.png"))
            if files:
                samples.append({
                    "id": f"xray_{label.lower()}",
                    "modality": "xray",
                    "label": label,
                    "title": pretty,
                    "filename": files[0].name,
                    "path": str(files[0].as_posix()),
                    "desc": "224x224 CLAHE Normalized Chest Radiograph"
                })

    # CT Scan samples
    ct_test = Path("data/processed/ct_scans/dicom/test")
    if ct_test.exists():
        for label, pretty in [("CONTRAST", "Contrast-Enhanced Chest CT"), ("NO_CONTRAST", "Non-Contrast Baseline CT")]:
            files = list((ct_test / label).glob("*.png"))
            if files:
                samples.append({
                    "id": f"ct_{label.lower()}",
                    "modality": "ct",
                    "label": label,
                    "title": pretty,
                    "filename": files[0].name,
                    "path": str(files[0].as_posix()),
                    "desc": "Soft-tissue HU windowed axial tomographic slice"
                })

    return samples


@app.get("/api/sample-image")
def get_sample_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.post("/api/predict")
async def run_prediction(
    file: UploadFile = File(None),
    sample_path: str = Form(None),
    task: str = Form("xray"),
    use_ensemble: bool = Form(True)
):
    try:
        raw_bytes = None
        filename = ""
        if file:
            raw_bytes = await file.read()
            filename = file.filename
        elif sample_path and os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                raw_bytes = f.read()
            filename = Path(sample_path).name
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        ext = Path(filename).suffix.lower()

        # Medical preprocessing logic
        if ext in [".dcm", ".dicom"]:
            # DICOM parsing via pydicom
            import pydicom
            dcm = pydicom.dcmread(io.BytesIO(raw_bytes))
            pixel_array = dcm.pixel_array.astype(np.float32)
            if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_array = np.amax(pixel_array) - pixel_array
            intercept = getattr(dcm, "RescaleIntercept", 0)
            slope = getattr(dcm, "RescaleSlope", 1)
            hu_image = pixel_array * slope + intercept
            img_windowed = window_ct_hu(hu_image, window_center=40, window_width=400)
            img_letterboxed = letterbox_resize(img_windowed, target_size=IMG_SIZE)
            img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_GRAY2RGB)
        elif ext in [".tif", ".tiff"]:
            import tifffile
            img = tifffile.imread(io.BytesIO(raw_bytes))
            if img.ndim == 3:
                img = img[0] if img.shape[0] < img.shape[2] else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32)
            p1, p99 = np.percentile(img, (1, 99))
            img_norm = np.clip((img - p1) / max(p99 - p1, 1e-5) * 255.0, 0, 255).astype(np.uint8)
            img_letterboxed = letterbox_resize(img_norm, target_size=IMG_SIZE)
            img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_GRAY2RGB)
        else:
            # Standard PNG/JPEG/BMP
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError("Could not decode image format.")
            if task == "xray":
                img_gray = apply_clahe(img_gray)
            img_letterboxed = letterbox_resize(img_gray, target_size=IMG_SIZE)
            img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_GRAY2RGB)

        # Build tensor
        transform = get_transforms(is_train=False)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

        class_names = CT_CLASS_NAMES if task == "ct" else CLASS_NAMES
        models = get_loaded_models(task=task, ensemble=use_ensemble)
        if not models:
            raise HTTPException(status_code=500, detail=f"No trained model found for {task}.")

        # Multi-model inference
        fold_probs = []
        with torch.no_grad():
            for m in models:
                out = m(tensor)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                fold_probs.append(prob)

        avg_probabilities = np.mean(fold_probs, axis=0)
        pred_idx = int(np.argmax(avg_probabilities))
        confidence = float(avg_probabilities[pred_idx])

        # Generate attention heatmap using primary model
        heatmap_b64 = generate_attention_heatmap(img_rgb, models[0])

        # Encode preprocessed clean view
        _, clean_buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        clean_b64 = base64.b64encode(clean_buf).decode("utf-8")

        # Per-fold voting breakdown for full transparency
        fold_votes = []
        for i, p in enumerate(fold_probs):
            f_pred = int(np.argmax(p))
            fold_votes.append({
                "fold": i + 1,
                "prediction": class_names[f_pred],
                "confidence": float(p[f_pred]),
                "prob_positive": float(p[1])
            })

        return {
            "task": task,
            "filename": filename,
            "predicted_class": class_names[pred_idx],
            "confidence": confidence,
            "probabilities": {
                class_names[0]: float(avg_probabilities[0]),
                class_names[1]: float(avg_probabilities[1])
            },
            "ensemble_size": len(models),
            "fold_breakdown": fold_votes,
            "preprocessed_image": f"data:image/png;base64,{clean_b64}",
            "heatmap_image": f"data:image/png;base64,{heatmap_b64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
def get_system_metrics():
    """Serves cross-validation benchmarks, confusion matrix images, and test statistics."""
    xray_cm_path = CONFUSION_MATRICES_DIR / "xray_confusion_matrix.png"
    ct_cm_path = CONFUSION_MATRICES_DIR / "ct_confusion_matrix.png"

    xray_cm_b64 = None
    if xray_cm_path.exists():
        with open(xray_cm_path, "rb") as f:
            xray_cm_b64 = base64.b64encode(f.read()).decode("utf-8")

    ct_cm_b64 = None
    if ct_cm_path.exists():
        with open(ct_cm_path, "rb") as f:
            ct_cm_b64 = base64.b64encode(f.read()).decode("utf-8")

    cv_results = {}
    cv_path = METRICS_DIR / "ct_cv_results.json"
    if cv_path.exists():
        with open(cv_path, "r") as f:
            cv_results = json.load(f)

    return {
        "xray": {
            "model": "DenseNet-121 (CheXNet Transfer)",
            "accuracy": 0.8798,
            "sensitivity": 0.9923,
            "precision": 0.8431,
            "f1_score": 0.9117,
            "roc_auc": 0.9773,
            "test_samples": 624,
            "confusion_matrix_b64": xray_cm_b64
        },
        "ct": {
            "model": "DenseNet-121 (Blocks 1-3 Frozen + Soft Tissue HU)",
            "ensemble_roc_auc": 0.8125,
            "ensemble_sensitivity": 0.7000,
            "ensemble_f1": 0.6667,
            "peak_fold_accuracy": 0.7750,
            "cv_folds": cv_results.get("fold_details", []),
            "confusion_matrix_b64": ct_cm_b64
        }
    }

# Serve static frontend
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
