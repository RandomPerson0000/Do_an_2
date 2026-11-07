"""Local event output. Extend emit() for a buzzer or message transport later."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import uuid


class EventWriter:
    def __init__(self, output_dir, save_snapshots=True):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = Path(output_dir) / f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.file = (self.directory / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        self.save_snapshots = save_snapshots
        self.counter = 0

    def emit(self, kind, **data):
        record = {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **data}
        self.file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        logging.info("%s %s", kind, data.get("label", ""))

    def alarm(self, event, frame, detections):
        self.counter += 1
        snapshot = None
        if event.kind == "alarm_start" and self.save_snapshots:
            import cv2
            path = self.directory / f"{self.counter:06d}_{event.label}.jpg"
            if cv2.imwrite(str(path), frame):
                snapshot = path.name
            else:
                self.emit("snapshot_error", label=event.label)
        self.emit(event.kind, label=event.label, source_seconds=event.timestamp,
                  snapshot=snapshot, detections=[asdict(d) for d in detections])

    def close(self):
        self.file.close()
