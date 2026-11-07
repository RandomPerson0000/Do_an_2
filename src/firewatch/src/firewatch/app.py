import logging
import signal
import threading
import time
from .capture import open_source, VideoReader
from .detector import Detector
from .events import EventWriter
from .overlay import annotate
from .temporal import AlarmManager


def run(cfg, max_frames=None, detector_factory=Detector, source_factory=open_source):
    """Run one session; factories allow end-to-end tests without real hardware."""
    import cv2
    cfg.validate()
    stop = threading.Event()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stop.set())
    source = writer = None
    manager = AlarmManager(cfg.rules)
    reason = "stopped"
    frames = 0
    try:
        writer = EventWriter(cfg.output_dir, cfg.save_snapshots)
        writer.emit("session_start", model=cfg.model, class_ids=cfg.class_ids,
                    thresholds={k: v.conf for k, v in cfg.rules.items()})
        detector = detector_factory(cfg)
        source = source_factory(cfg)
        origin = None
        fps = 0.0
        while not stop.is_set():
            started = time.monotonic()
            packet = source.read()
            if packet is None:
                reason = "end_of_video"
                break
            if origin is None:
                origin = 0.0 if isinstance(source, VideoReader) else packet.timestamp
            timestamp = packet.timestamp - origin
            detections = detector.predict(packet.image)
            transitions = manager.update(timestamp, detections)
            current_fps = 1 / max(time.monotonic() - started, 1e-6)
            fps = current_fps if frames == 0 else 0.9 * fps + 0.1 * current_fps
            need_image = cfg.display or (cfg.save_snapshots and any(
                e.kind == "alarm_start" for e in transitions))
            image = annotate(packet.image, detections, manager.active_labels, fps) if need_image else packet.image
            for event in transitions:
                writer.alarm(event, image, detections)
            frames += 1
            if cfg.display:
                cv2.imshow("FireWatch - Q / Esc to exit", image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    reason = "user_exit"
                    break
            if max_frames is not None and frames >= max_frames:
                reason = "frame_limit"
                break
        return 0
    except Exception as exc:
        reason = "error"
        if writer:
            writer.emit("system_error", error_type=type(exc).__name__,
                        active_labels=manager.active_labels)
        # Full local diagnostic is useful for model/driver troubleshooting.
        logging.exception("FireWatch stopped due to an error")
        return 1
    finally:
        if source:
            source.close()
        if writer:
            try:
                # Closing the application is NOT an alarm_end / all-clear.
                writer.emit("session_end", reason=reason, frames=frames,
                            active_labels=manager.active_labels)
            finally:
                writer.close()
        if cfg.display:
            cv2.destroyAllWindows()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
