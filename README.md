<img width="1680" height="1050" alt="Screenshot 2026-02-18 at 11 21 22 PM" src="https://github.com/user-attachments/assets/5dc2d408-5d7f-4233-9d12-f3fbe038a0e7" />
# Real-Time Bottle Verification System (YOLOv8 + MobileNetV2)

A real-time computer vision application that detects a bottle using YOLOv8 and verifies whether it matches a predefined golden reference bottle using MobileNetV2 feature embeddings and cosine similarity. This project demonstrates object detection, transfer learning, embedding comparison, and real-time webcam processing in a single pipeline.

---

## Features

- YOLOv8 bottle detection with bounding box visualization
- MobileNetV2 embedding extraction
- Cosine similarity–based verification
- Real-time webcam processing
- MATCH / NO MATCH decision overlay
- Similarity score display
- FPS counter
- Golden reference preview
- Error handling and graceful exit

---

## How It Works

1. YOLOv8 performs object detection on each webcam frame and identifies bottles.
2. The detected bottle region is cropped from the frame.
3. MobileNetV2, pretrained on ImageNet, extracts a semantic feature embedding.
4. Cosine similarity is computed between the live embedding and the golden reference embedding.
5. A threshold-based decision determines MATCH or NO MATCH.

---

## Tech Stack

- Python 3
- OpenCV
- TensorFlow / Keras
- Ultralytics YOLOv8
- NumPy

## Project Structure

```
object-verification-yolo/
├─ main.py
├─ golden.jpg
├─ requirements.txt
└─ README.md
```

## Installation

### Clone Repository

```
git clone https://github.com/<your-username>/object-verification-yolo.git
cd object-verification-yolo
```
## Install Dependencies

```
pip install -r requirements.txt
```

## Run the Application
```
python main.py
```
Press Q to exit the application window.

## Golden Image

Place the reference bottle image in the project root directory as:
```
golden.jpg
```
## Recommendations:
	•	Use consistent lighting
	•	Ensure the label is clearly visible
	•	Avoid reflections or glare
	•	Use the same object intended for verification

## Configuration

Default similarity threshold is set to: 0.65 

You can modify this value inside main.py to adjust strictness.

## Summary

This project combines object detection and semantic verification into a lightweight, real-time system. It demonstrates practical implementation of transfer learning, embedding similarity, and end-to-end computer vision pipeline design.



