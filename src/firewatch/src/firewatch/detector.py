"""Ultralytics adapter: delegates preprocessing and output decoding to YOLO."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector:
    def __init__(self, cfg):
        if not Path(cfg.model).is_file():
            raise FileNotFoundError(f"Model not found: {cfg.model}")
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.model, task="detect")
        self.labels = {v: k for k, v in cfg.class_ids.items()}

    def predict(self, frame):
        cfg = self.cfg
        result = self.model.predict(
            source=frame, imgsz=cfg.imgsz, device=cfg.device,
            conf=min(r.conf for r in cfg.rules.values()), iou=cfg.iou,
            classes=list(self.labels), max_det=cfg.max_det,
            rect=False, verbose=False, save=False,
        )[0]
        names = result.names
        missing = set(self.labels) - set(names)
        if missing:
            raise ValueError(f"Configured class IDs absent in model: {sorted(missing)}")
        detections = []
        boxes = result.boxes
        if boxes is not None:
            for xyxy, score, class_id in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
            ):
                label = self.labels.get(int(class_id))
                if label and score >= cfg.rules[label].conf:
                    detections.append(Detection(label, float(score), tuple(xyxy)))
        return detections
