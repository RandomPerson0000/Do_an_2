import argparse
import logging
import yaml
from pathlib import Path
from .config import load_config, resolve_source


def main(argv=None):
    parser = argparse.ArgumentParser(description="FireWatch: fire/smoke camera monitoring")
    default = Path(__file__).resolve().parents[2] / "configs" / "firewatch.yaml"
    parser.add_argument("--config", default=str(default))
    parser.add_argument("--model", help="Model path, relative to project_root")
    parser.add_argument("--source", help="Camera index, video path, RTSP URL, or picamera2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--check-config", action="store_true", help="Validate config without loading model/camera")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
        if args.source is not None:
            cfg.source = resolve_source(args.source, cfg.project_root)
        if args.model:
            cfg.model = str((Path(cfg.project_root) / args.model).resolve())
        if args.headless:
            cfg.display = False
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be > 0")
        cfg.validate()
        if args.check_config:
            print("Configuration valid (model file and camera not checked).")
            print(f"Model: {cfg.model}")
            print(f"Output: {cfg.output_dir}")
            return 0
        from .app import run
        return run(cfg, args.max_frames)
    except (OSError, ValueError, TypeError, ImportError, yaml.YAMLError) as exc:
        logging.error("%s", exc)
        return 2
