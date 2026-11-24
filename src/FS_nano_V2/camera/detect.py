import argparse
import logging
import os
from pathlib import Path
import socket
import sys

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR.parent / "firewatch" / "src"))

from firewatch.config import load_config, resolve_source
from firewatch.events import EventWriter
from messaging.mqtt_alerts import AlarmPublisher, device_name


WEIGHTS = Path("runs/train/fire_smoke_V2/fire_yolo26n/weights")
MODELS = {"pt": "best.pt", "onnx": "best_nano.onnx"}


def main(backend):
    parser = argparse.ArgumentParser(description=f"FS_nano_V2 · {backend} · Pi 5")
    parser.add_argument("--source", help="picamera2, USB index, video file or RTSP URL")
    parser.add_argument("--model", type=Path, help="Override the weight path after copying to Pi")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    cfg = load_config(APP_DIR / "config" / "camera.yaml")
    root = Path(cfg.project_root)
    model = (root / (args.model or WEIGHTS / MODELS[backend])).resolve()
    if model.suffix.lower() != "." + backend or not model.is_file():
        parser.error(f"Expected an existing .{backend} file at {model}")
    cfg.model = str(model)
    device = device_name(os.environ.get("FS_DEVICE_ID", socket.gethostname()))
    cfg.output_dir = str(Path(cfg.output_dir) / device / backend)
    if args.source is not None:
        cfg.source = resolve_source(args.source, root)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    cfg.validate()
    if args.check_config:
        print(f"FS_nano_V2 ({backend}) | {model} | source={cfg.source} | device={device}")
        return 0

    from firewatch.app import run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    publisher = AlarmPublisher(Path(cfg.output_dir), device, backend)

    class AlertWriter(EventWriter):
        def emit(self, kind, **data):
            super().emit(kind, **data)
            if kind in ("alarm_start", "alarm_end"):
                scores = [d["confidence"] for d in data["detections"] if d["label"] == data["label"]]
                publisher.enqueue(kind, data["label"], max(scores, default=None))

    try:
        return run(cfg, args.max_frames, writer_factory=AlertWriter)
    finally:
        publisher.close()
