"""Sequential video reader and bounded, latest-frame live camera reader."""
from dataclasses import dataclass
from pathlib import Path
import math
import threading
import time


@dataclass(frozen=True)
class Frame:
    image: object
    timestamp: float
    sequence: int


class VideoReader:
    """Offline video: timestamp follows media time, not inference speed."""
    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("Cannot open input video")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.cap.release()
            raise RuntimeError("Video FPS unavailable; re-encode with a valid frame rate")
        self.index = 0
        self.last_time = -1.0

    def read(self):
        ok, image = self.cap.read()
        if not ok:
            return None
        pts = self.cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(pts) or pts <= self.last_time:
            pts = max(self.index / self.fps, self.last_time + 1 / self.fps)
        self.last_time = pts
        self.index += 1
        return Frame(image, pts, self.index)

    def close(self):
        self.cap.release()


class LiveReader:
    """Only one pending frame. Slow inference intentionally skips older frames.

    Capture resource belongs exclusively to the daemon worker. If a native
    driver blocks forever, read() still times out and the application exits.
    Restart can be handled by the provided systemd service.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.error = None
        self.consumed = 0
        self.thread = threading.Thread(target=self._worker, daemon=True, name="firewatch-camera")
        self.thread.start()

    def _worker(self):
        camera = None
        pi = self.cfg.source == "picamera2"
        try:
            if pi:
                from picamera2 import Picamera2
                camera = Picamera2()
                config = camera.create_video_configuration(
                    main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"},
                    controls={"FrameRate": self.cfg.camera_fps}, buffer_count=4,
                )
                camera.configure(config)
                camera.start()
                # Picamera2 RGB888 arrays have B,G,R byte order for OpenCV.
                def read_image():
                    return camera.capture_array("main")
            else:
                import cv2
                source = self.cfg.source
                if isinstance(source, str) and "://" in source:
                    timeout = int(self.cfg.frame_timeout_seconds * 1000)
                    camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout,
                    ])
                else:
                    camera = cv2.VideoCapture(source)
                if not camera.isOpened():
                    raise RuntimeError("Cannot open live camera; check device/URL/permissions")
                if isinstance(source, int) or str(source).startswith("/dev/video"):
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                    camera.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this.
                def read_image():
                    ok, image = camera.read()
                    if not ok:
                        raise RuntimeError("Live camera stopped returning frames")
                    return image
            sequence = 0
            while not self.stop.is_set():
                image = read_image()
                if image is None:
                    raise RuntimeError("Camera returned an empty image")
                sequence += 1
                with self.condition:
                    self.latest = Frame(image, time.monotonic(), sequence)
                    self.condition.notify_all()
        except Exception as exc:
            # Do not include the input URL, which may contain credentials.
            with self.condition:
                self.error = f"Capture failed ({type(exc).__name__}); check camera and dependencies"
                self.condition.notify_all()
        finally:
            if camera is not None:
                try:
                    camera.close() if pi else camera.release()
                except Exception:
                    pass

    def read(self):
        deadline = time.monotonic() + self.cfg.frame_timeout_seconds
        with self.condition:
            while True:
                if self.error:
                    raise RuntimeError(self.error)
                if self.latest is not None and self.latest.sequence > self.consumed:
                    if time.monotonic() - self.latest.timestamp > self.cfg.frame_timeout_seconds:
                        raise TimeoutError("Camera frame is stale")
                    self.consumed = self.latest.sequence
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No fresh camera frame before timeout")
                self.condition.wait(remaining)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)


def open_source(cfg):
    source = cfg.source
    if (isinstance(source, str) and source != "picamera2"
            and "://" not in source and not source.startswith("/dev/video")):
        if not Path(source).is_file():
            raise FileNotFoundError(f"Video not found: {source}")
        return VideoReader(source)
    return LiveReader(cfg)
