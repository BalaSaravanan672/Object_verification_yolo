#!/usr/bin/env python3
"""
Bottle Verification System

Real-time application that compares a live-detected bottle against a golden
reference image using YOLOv8 for detection and MobileNetV2 embeddings for
matching. Displays MATCH / NO MATCH overlay with similarity score and FPS.

Usage:
 - Place a golden reference image at `vision_project/golden.jpg`
 - Install dependencies: `pip install -r requirements.txt`
 - Run: `python vision_project/main.py`

Requirements are in `vision_project/requirements.txt`.
"""

import os
import sys
import time
import math
import traceback

import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def get_embedding(frame, model):
    """Return a flattened MobileNetV2 embedding for a BGR image.

    Steps:
    - Convert BGR -> RGB
    - Resize to 224x224
    - preprocess_input (MobileNetV2)
    - Run through `model` and flatten output
    """
    # Convert color and resize
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))

    # Preprocess and add batch dim
    x = np.expand_dims(resized.astype("float32"), axis=0)
    x = preprocess_input(x)

    # Get pooled features (shape: (1, features))
    emb = model.predict(x, verbose=0)
    return emb.flatten()


def cosine_similarity(a, b):
    """Compute cosine similarity between two 1-D numpy arrays."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def overlay_transparent(background, overlay, x, y):
    """Overlay `overlay` onto `background` at position (x, y).

    Both images are expected in BGR. The overlay may be fully opaque.
    This function will handle bounds clipping.
    """
    h, w = overlay.shape[:2]
    bh, bw = background.shape[:2]

    if x >= bw or y >= bh:
        return background

    # Clip overlay dimensions
    w = min(w, bw - x)
    h = min(h, bh - y)
    if w <= 0 or h <= 0:
        return background

    roi = background[y : y + h, x : x + w]
    overlay_resized = cv2.resize(overlay, (w, h))
    background[y : y + h, x : x + w] = overlay_resized
    return background


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    golden_path = os.path.join(base_dir, "golden.jpg")

    # Check golden image
    if not os.path.exists(golden_path):
        print(f"Error: golden image not found at {golden_path}")
        print("Please add your golden reference image named 'golden.jpg' in the vision_project folder.")
        sys.exit(1)

    # Load golden image (BGR)
    golden_img = cv2.imread(golden_path)
    if golden_img is None:
        print(f"Error: failed to load golden image from {golden_path}")
        sys.exit(1)

    # Load MobileNetV2 (pretrained on ImageNet) for embeddings
    print("Loading MobileNetV2 model...")
    try:
        embed_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    except Exception:
        print("Error: could not load MobileNetV2. Ensure TensorFlow is installed correctly.")
        traceback.print_exc()
        sys.exit(1)

    # Precompute golden embedding
    print("Computing golden embedding...")
    golden_embedding = get_embedding(golden_img, embed_model)

    # Load YOLOv8n model (fast)
    print("Loading YOLOv8n model (yolov8n.pt)...")
    try:
        yolo = YOLO("yolov8n.pt")
    except Exception:
        print("Error: could not load YOLOv8n. Ensure the 'ultralytics' package is installed and the weights are available or will be downloaded.")
        traceback.print_exc()
        sys.exit(1)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (cv2.VideoCapture(0)).")
        sys.exit(1)

    window_name = "Bottle Verification System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    threshold = 0.65
    fps = 0.0
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: empty frame received from camera")
                time.sleep(0.1)
                continue

            # YOLO detection (returns list of results, take first)
            results = yolo(frame)
            result = results[0]

            match_text = "NO BOTTLE DETECTED"
            match_color = (0, 0, 255)  # Red by default (BGR)
            sim_score = 0.0

            # Parse detections safely
            boxes = []
            try:
                # boxes.xyxy, boxes.cls, boxes.conf are tensors usually
                if hasattr(result, "boxes") and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    boxes = list(zip(xyxy, cls_ids, confs))
            except Exception:
                # Fallback: try extracting via attributes that might be lists
                try:
                    for box in result.boxes:
                        xy = box.xyxy[0].cpu().numpy()
                        cid = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        boxes.append((xy, cid, conf))
                except Exception:
                    boxes = []

            # Filter for class 'bottle'
            bottle_indices = []
            for i, (_xy, cid, conf) in enumerate(boxes):
                # Use model names mapping
                name = yolo.model.names[int(cid)] if hasattr(yolo, "model") and hasattr(yolo.model, "names") else yolo.names.get(int(cid), str(cid))
                if str(name).lower() == "bottle":
                    bottle_indices.append((i, conf))

            if len(bottle_indices) == 0:
                # No bottle found in frame
                cv2.putText(frame, "NO BOTTLE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                # Choose the highest-confidence bottle detection
                best_idx = max(bottle_indices, key=lambda x: x[1])[0]
                xy, cid, conf = boxes[best_idx]
                x1, y1, x2, y2 = map(int, xy)

                # Clip coords to frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Crop ROI for embedding
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    cv2.putText(frame, "INVALID ROI", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Compute embedding for ROI
                    try:
                        live_emb = get_embedding(roi, embed_model)
                        sim_score = cosine_similarity(golden_embedding, live_emb)
                    except Exception:
                        sim_score = 0.0

                    # Decide match
                    if sim_score > threshold:
                        match_text = "MATCH"
                        match_color = (0, 255, 0)  # Green
                    else:
                        match_text = "NO MATCH"
                        match_color = (0, 0, 255)  # Red

                    # Display similarity and match
                    cv2.putText(frame, f"{match_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, match_color, 2)
                    cv2.putText(frame, f"Sim: {sim_score:.3f}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # FPS calculation (simple)
            now = time.time()
            dt = now - prev_time if now - prev_time > 0 else 1e-6
            prev_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps != 0 else 1.0 / dt
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show small golden preview in top-left corner
            try:
                preview = cv2.resize(golden_img, (120, 120))
                frame = overlay_transparent(frame, preview, frame.shape[1] - 130, 10)
                # Label the preview
                cv2.putText(frame, "Golden", (frame.shape[1] - 130, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            except Exception:
                pass

            # Display the resulting frame
            cv2.imshow(window_name, frame)

            # Exit on 'q' or 'Q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception:
        print("An error occurred:")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
