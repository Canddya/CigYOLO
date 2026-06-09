# predict.py - Load trained YOLOv11 model and run inference on single/batch images

import os
import argparse
import csv
from pathlib import Path
from typing import Union, List, Tuple

import torch
from PIL import Image
from ultralytics import YOLO

# Default configuration (should match training settings)
DEFAULT_IMG_SIZE = 800
DEFAULT_MODEL_PATH = None  # Must be provided via argument
CLASS_NAMES = {0: "Right", 1: "Wrong"}  # 0 = non-defective, 1 = defective
CLASS_LABELS = {0: "Good", 1: "Defect"}
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


def load_model(model_path: str) -> YOLO:
    """Load trained YOLOv11 classification model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    # Set device
    model.to(DEVICE)
    print(f"Model loaded on device: {DEVICE}")
    return model


def predict_single_image(model: YOLO, img_path: str, img_size: int = DEFAULT_IMG_SIZE) -> Tuple[int, float]:
    """
    Predict a single image.
    Returns: (class_index, confidence)
    """
    results = model.predict(img_path, imgsz=img_size, device=DEVICE, verbose=False)
    probs = results[0].probs
    class_idx = int(probs.top1)
    confidence = float(probs.top1conf)
    return class_idx, confidence


def predict_batch(model: YOLO, img_paths: List[str], img_size: int = DEFAULT_IMG_SIZE) -> List[Tuple[str, int, float]]:
    """
    Predict a batch of images.
    Returns: List of (image_path, class_index, confidence)
    """
    predictions = []
    for img_path in img_paths:
        try:
            class_idx, conf = predict_single_image(model, img_path, img_size)
            predictions.append((img_path, class_idx, conf))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append((img_path, -1, 0.0))  # mark error
    return predictions


def print_results(predictions: List[Tuple[str, int, float]]):
    """Print predictions in a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Image Path':<50} {'Prediction':<12} {'Confidence':<10} {'Status'}")
    print("-" * 80)
    for img_path, class_idx, conf in predictions:
        if class_idx == -1:
            print(f"{img_path:<50} {'ERROR':<12} {'-':<10} {'Failed'}")
        else:
            class_name = CLASS_NAMES.get(class_idx, "Unknown")
            label = CLASS_LABELS.get(class_idx, "?")
            status = "Defective" if class_idx == 1 else "OK"
            print(f"{img_path:<50} {class_name:<12} {conf*100:.2f}%{'':<4} {status}")
    print("=" * 80)


def save_results_to_csv(predictions: List[Tuple[str, int, float]], output_csv: str):
    """Save predictions to CSV file."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_class", "class_name", "confidence", "status"])
        for img_path, class_idx, conf in predictions:
            if class_idx == -1:
                writer.writerow([img_path, "ERROR", "ERROR", "", "failed"])
            else:
                class_name = CLASS_NAMES[class_idx]
                status = "defective" if class_idx == 1 else "good"
                writer.writerow([img_path, class_idx, class_name, f"{conf:.4f}", status])
    print(f"Results saved to: {output_csv}")


def get_image_files(input_path: str) -> List[str]:
    """Return list of image file paths (jpg, jpeg, png, bmp)."""
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in supported_ext:
            return [str(input_path)]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    elif input_path.is_dir():
        files = []
        for ext in supported_ext:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return [str(f) for f in sorted(files)]
    else:
        raise FileNotFoundError(f"Path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Cigarette Quality Inspection - Predict")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to trained .pt model file")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to image file or folder containing images")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE,
                        help=f"Input image size (default: {DEFAULT_IMG_SIZE})")
    parser.add_argument("--output_csv", "-o", type=str, default=None,
                        help="Optional CSV file to save results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Get all image files
    try:
        img_files = get_image_files(args.input)
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    if not img_files:
        print("No supported images found.")
        return

    print(f"Found {len(img_files)} image(s) to process.")

    # Batch prediction (simple loop, model.predict can handle list but we use custom for control)
    # For better performance, we can pass list to model.predict, but need to handle paths.
    # Here we do iterative to match our result structure.
    predictions = []
    total = len(img_files)
    for i, img_path in enumerate(img_files, 1):
        print(f"Processing [{i}/{total}]: {os.path.basename(img_path)}")
        class_idx, conf = predict_single_image(model, img_path, args.imgsz)
        predictions.append((img_path, class_idx, conf))

    # Output results
    print_results(predictions)

    if args.output_csv:
        save_results_to_csv(predictions, args.output_csv)


if __name__ == "__main__":
    main()

    