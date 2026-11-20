"""Export the fire_smoke_V2 checkpoint to models/released/best.onnx."""

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[2]
    checkpoint = project_root / "runs/train/fire_smoke_V2/fire_yolo26n/weights/best.pt"
    output = project_root / "models/released/best.onnx"

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics exports next to its input checkpoint.
    with TemporaryDirectory(dir=output.parent) as temp_dir:
        temp_checkpoint = Path(temp_dir) / checkpoint.name
        copyfile(checkpoint, temp_checkpoint)
        exported = YOLO(str(temp_checkpoint)).export(
            format="onnx", imgsz=640, device="cpu", simplify=False
        )
        Path(exported).replace(output)

    print(f"ONNX saved to {output}")


if __name__ == "__main__":
    main()
