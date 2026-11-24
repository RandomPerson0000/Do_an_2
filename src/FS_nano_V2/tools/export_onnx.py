from pathlib import Path

from ultralytics import YOLO

# Load a model
root = Path(__file__).resolve().parents[3]
model = YOLO(root / "runs/train/fire_smoke_V2/fire_yolo26n/weights/best.pt")

# Export the model
model.export(format="onnx")
