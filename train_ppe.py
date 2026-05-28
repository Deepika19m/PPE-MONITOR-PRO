#!/usr/bin/env python3
"""
YOLOv8 PPE Training Script for PPE-Detection-Project
Trains best.pt for app_ultra_fast.py / ppe_detection_engine.py
"""

import os
import shutil
from ultralytics import YOLO
import subprocess
import sys

# Dataset path
DATASET_DIR = "Ultralytics/PPE detection.yolov8"

print("🚀 Starting YOLOv8 PPE Training Pipeline")
print(f"📁 Dataset: {DATASET_DIR}")

# Verify structure
data_yaml = os.path.join(DATASET_DIR, "data.yaml")
if not os.path.exists(data_yaml):
    print("❌ data.yaml not found!")
    sys.exit(1)

print("✅ data.yaml found")
print("📊 Training model: yolov8n.pt (nano for speed)")
print("⏱️  Recommended: epochs=100, imgsz=640")
print("💾 Output: runs/detect/train/weights/best.pt")
print()

# Load model
model = YOLO("yolov8n.pt")  # nano model for speed

# Train
results = model.train(
    data=data_yaml,
    epochs=100,      # Adjust based on dataset size/time
    imgsz=640,
    batch=16,        # Auto-batch if GPU memory limited
    name="ppe_train",
    project="Ultralytics/runs/detect",
    save=True,
    plots=True,
    device=0 if os.system("nvidia-smi >nul 2>&1") == 0 else "cpu",  # GPU if available
    patience=20,     # Early stopping
    val=True,
    exist_ok=True
)

print("\n✅ Training completed!")
print("📁 Best model: Ultralytics/runs/detect/ppe_train/weights/best.pt")
print("🔄 Copy to project root:")
print("cp Ultralytics/runs/detect/ppe_train/weights/best.pt .")

# Copy best.pt to project root
best_pt = "Ultralytics/runs/detect/ppe_train/weights/best.pt"
if os.path.exists(best_pt):
    shutil.copy2(best_pt, "best.pt")
    print("✅ best.pt copied to project root!")
else:
    print("⚠️  best.pt not found - check runs/detect/ppe_train/weights/")
    print("Manual copy needed after training.")

print("\n🎉 Ready! Run app_ultra_fast.py to test (no more fallback mode).")
print("📊 View results: Ultralytics/runs/detect/ppe_train/")

