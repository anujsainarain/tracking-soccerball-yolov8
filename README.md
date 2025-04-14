# ⚽ Soccer Ball Tracking with YOLOv8

This project demonstrates how to train a YOLOv8 model on a custom dataset to detect and track a **soccer ball** in a video. It outputs an annotated video with bounding boxes and saves tracking data (center coordinates and box size) to a CSV file.

---

## 🎥 Video demonstration

![Soccer Ball Tracking Demo](output/soccer_tracking_gif.gif)

---

## 📦 Features

- ✅ Train a YOLOv8 model on a custom COCO-format dataset
- 🎯 Detect and track soccer ball in real-time video
- 💾 Save output video with bounding boxes
- 📈 Export tracking data (x/y center and box size) to a CSV

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python pandas
```

### 2. Train YOLOv8 on Your Custom Dataset
```bash
from ultralytics import YOLO

# Load the base model and train on custom soccer ball dataset
model = YOLO('yolov8m.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=960,
    batch=8,
    name='soccer_ball_test'
)
```

### 3. Run Inference and Track the Ball
```bash
python track_soccer_ball.py
```

The script loads the trained model and annotates the bounding boxes around the soccer while saving the tracked data. Once it completed, the new video is saved in the output folder along with the csv file.

