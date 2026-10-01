import os
import argparse
import json
from pathlib import Path
import cv2
import torch
import numpy as np

from src.config import (
    MODELS_DIR, PREDICTIONS_DIR, CLASS_NAMES, CT_CLASS_NAMES, DEVICE, NUM_CLASSES, IMG_SIZE
)
from src.models.cnn_model import build_model
from src.preprocessing.utils import letterbox_resize, apply_clahe, get_transforms


def load_inference_model(model_path, model_name="densenet121"):
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_single_image(image_path, models, class_names=CLASS_NAMES, apply_preprocessing=True):
    """Preprocesses input image and returns predicted class with probabilities (supports single model or list of ensemble models)."""
    if not isinstance(models, list):
        models = [models]

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    if apply_preprocessing:
        img = apply_clahe(img)
        img = letterbox_resize(img, target_size=IMG_SIZE)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    transform = get_transforms(is_train=False)
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs_list = [torch.softmax(m(tensor), dim=1).cpu().numpy()[0] for m in models]
        probabilities = np.mean(probs_list, axis=0)
        pred_idx = np.argmax(probabilities)

    result = {
        "image_path": str(image_path),
        "predicted_class": class_names[pred_idx],
        "confidence": float(probabilities[pred_idx]),
        "probabilities": {
            class_names[i]: float(probabilities[i]) for i in range(len(class_names))
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run inference on new Chest X-Ray / CT Scan images")
    parser.add_argument("--image", type=str, required=True, help="Path to single image or folder of images")
    parser.add_argument("--task", type=str, default="xray", choices=["xray", "ct"], help="Modality task: 'xray' or 'ct'")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--model_name", type=str, default="densenet121")
    parser.add_argument("--ensemble", action="store_true", help="Use 5-fold ensemble for highest prediction confidence")
    parser.add_argument("--output_json", type=str, default=str(PREDICTIONS_DIR / "latest_predictions.json"))
    args = parser.parse_args()

    class_names = CT_CLASS_NAMES if args.task == "ct" else CLASS_NAMES

    if args.ensemble:
        fold_paths = sorted(list(MODELS_DIR.glob(f"{args.task}_densenet_fold_*.pth")))
        if not fold_paths:
            print(f"No fold models found. Falling back to single best checkpoint...")
            default_model = MODELS_DIR / ("ct_cnn_best.pth" if args.task == "ct" else "xray_cnn_best.pth")
            fold_paths = [Path(args.model_path or default_model)]
        print(f"Using {len(fold_paths)}-model ensemble for [{args.task.upper()}] inference...")
        models = [load_inference_model(str(fp), model_name=args.model_name) for fp in fold_paths]
    else:
        default_model = MODELS_DIR / ("ct_cnn_best.pth" if args.task == "ct" else "xray_cnn_best.pth")
        model_path = args.model_path or str(default_model)
        models = [load_inference_model(model_path, model_name=args.model_name)]

    input_path = Path(args.image)

    results = []
    if input_path.is_file():
        res = predict_single_image(input_path, models, class_names=class_names)
        results.append(res)
        print(f"[{res['predicted_class']}] Confidence: {res['confidence']*100:.2f}% | Path: {res['image_path']}")
    elif input_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        files = [f for f in input_path.iterdir() if f.suffix.lower() in valid_exts]
        for f in files:
            res = predict_single_image(f, models, class_names=class_names)
            results.append(res)
            print(f"[{res['predicted_class']}] {res['confidence']*100:.2f}% | {f.name}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nInference results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
