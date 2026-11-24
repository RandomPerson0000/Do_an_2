"""Pi 5 camera detection with the V2 ONNX export."""
from camera.detect import main


if __name__ == "__main__":
    raise SystemExit(main("onnx"))
