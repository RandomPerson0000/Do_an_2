"""Validated configuration. All data paths are relative to project_root."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import yaml


@dataclass
class Rule:
    conf: float = 0.335
    confirm_seconds: float = 1.0
    window_seconds: float = 2.0
    min_hits: int = 3
    hit_ratio: float = 0.6
    clear_seconds: float = 3.0
    max_gap_seconds: float = 2.0

    def validate(self):
        for key in ("conf", "hit_ratio"):
            v = getattr(self, key)
            if not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        for key in ("confirm_seconds", "window_seconds", "clear_seconds", "max_gap_seconds"):
            v = getattr(self, key)
            if not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be finite and > 0")
        if self.window_seconds < self.confirm_seconds:
            raise ValueError("window_seconds must be >= confirm_seconds")
        if type(self.min_hits) is not int or self.min_hits < 2:
            raise ValueError("min_hits must be an integer >= 2")


@dataclass
class Config:
    project_root: str = ".."
    model: str = "weights/best.pt"
    source: str | int = 0
    imgsz: int = 640
    device: str = "cpu"
    iou: float = 0.7
    max_det: int = 100
    width: int = 1280
    height: int = 720
    camera_fps: int = 30
    frame_timeout_seconds: float = 10.0
    display: bool = True
    save_snapshots: bool = True
    output_dir: str = "runs/firewatch"
    # Explicit mapping also handles ONNX files without correct class metadata.
    class_ids: dict = field(default_factory=lambda: {"fire": 0, "smoke": 1})
    rules: dict = field(default_factory=lambda: {"fire": Rule(), "smoke": Rule(confirm_seconds=1.5)})

    def validate(self):
        if set(self.class_ids) != {"fire", "smoke"} or set(self.rules) != {"fire", "smoke"}:
            raise ValueError("class_ids and rules must contain exactly fire and smoke")
        ids = list(self.class_ids.values())
        if any(type(v) is not int or v < 0 for v in ids) or len(set(ids)) != 2:
            raise ValueError("Class IDs must be two distinct nonnegative integers")
        if not isinstance(self.source, (str, int)) or isinstance(self.source, bool):
            raise ValueError("source must be a camera index, video path, URL or picamera2")
        if isinstance(self.source, str) and not self.source.strip():
            raise ValueError("source cannot be empty")
        for key in ("imgsz", "max_det", "width", "height", "camera_fps"):
            v = getattr(self, key)
            if type(v) is not int or v <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.frame_timeout_seconds) or self.frame_timeout_seconds <= 0:
            raise ValueError("frame_timeout_seconds must be finite and > 0")
        if not math.isfinite(self.iou) or not 0 < self.iou <= 1:
            raise ValueError("iou must be in (0, 1]")
        for key in ("display", "save_snapshots"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be true or false")
        if Path(self.model).suffix.lower() not in {".pt", ".onnx"}:
            raise ValueError("model must be a local .pt or .onnx file")
        for rule in self.rules.values():
            rule.validate()


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML configuration must be a mapping")
    raw = dict(raw)
    if "rules" in raw:
        raw["rules"] = {name: Rule(**values) for name, values in raw["rules"].items()}
    cfg = Config(**raw)  # Unknown keys deliberately fail instead of being ignored.
    cfg.validate()
    root = (path.parent / cfg.project_root).resolve()
    cfg.project_root = str(root)
    cfg.model = str((root / cfg.model).resolve())
    cfg.output_dir = str((root / cfg.output_dir).resolve())
    cfg.source = resolve_source(cfg.source, root)
    return cfg


def resolve_source(source, root):
    if isinstance(source, str):
        if source.isdecimal():
            return int(source)
        if source != "picamera2" and "://" not in source and not source.startswith("/dev/video"):
            return str((Path(root) / source).resolve())
    return source
