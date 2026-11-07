"""Run: python -m unittest discover -s tests -v"""
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firewatch.config import Config, Rule, load_config
from firewatch.detector import Detection, Detector
from firewatch.temporal import TemporalRule, AlarmManager


class TemporalTests(unittest.TestCase):
    def state(self):
        return TemporalRule("fire", Rule(confirm_seconds=1, window_seconds=2, min_hits=3,
                                         clear_seconds=1, max_gap_seconds=1))

    def test_single_flash_never_confirms(self):
        state = self.state()
        events = [state.update(i / 10, i == 4) for i in range(40)]
        self.assertFalse(any(events))
        self.assertFalse(state.active)

    def test_alarm_once_and_clear_only_after_negative_evidence(self):
        state = self.state()
        self.assertIsNone(state.update(0, True))
        self.assertIsNone(state.update(.5, True))
        self.assertEqual(state.update(1, True).kind, "alarm_start")
        self.assertIsNone(state.update(1.5, True))
        self.assertIsNone(state.update(2, False))
        self.assertIsNone(state.update(2.5, False))
        self.assertEqual(state.update(3, False).kind, "alarm_end")
        self.assertFalse(state.active)

    def test_gap_cannot_confirm_old_evidence(self):
        state = self.state()
        state.update(0, True)
        state.update(.5, True)
        self.assertIsNone(state.update(10, True))
        self.assertFalse(state.active)

    def test_gap_cannot_clear_existing_alarm(self):
        state = self.state()
        for t in (0, .5, 1):
            state.update(t, True)
        state.update(1.5, False)
        self.assertIsNone(state.update(10, False))
        self.assertTrue(state.active)
        state.update(10.5, False)
        self.assertEqual(state.update(11, False).kind, "alarm_end")

    def test_sparse_positives_fail_ratio(self):
        state = self.state()
        for i in range(20):
            self.assertIsNone(state.update(i / 10, i in (0, 10, 19)))

    def test_rearm_after_clear(self):
        state = self.state()
        events = []
        for t, hit in [(0, True), (.5, True), (1, True), (1.5, False),
                       (2, False), (2.5, False), (3, True), (3.5, True), (4, True)]:
            event = state.update(t, hit)
            if event:
                events.append(event.kind)
        self.assertEqual(events, ["alarm_start", "alarm_end", "alarm_start"])

    def test_two_classes_independent(self):
        manager = AlarmManager({"fire": self.state().rule, "smoke": self.state().rule})
        both = [Detection(label, .9, (0, 0, 10, 10)) for label in ("fire", "smoke")]
        for t in (0, .5):
            manager.update(t, both)
        self.assertEqual(len(manager.update(1, both)), 2)
        for t in (1.5, 2, 2.5):
            manager.update(t, both[1:])
        self.assertEqual(manager.active_labels, ["smoke"])

    def test_duplicate_and_nonfinite_time_rejected(self):
        state = self.state()
        state.update(1, True)
        for value in (1, .5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                state.update(value, True)


class ConfigTests(unittest.TestCase):
    def test_paths_resolve_from_yaml_not_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            path = root / "configs" / "settings.yaml"
            path.write_text("project_root: ..\nmodel: weights/best.onnx\nsource: videos/test.mp4\n")
            cfg = load_config(path)
            self.assertEqual(Path(cfg.model), root / "weights" / "best.onnx")
            self.assertEqual(Path(cfg.source), root / "videos" / "test.mp4")

    def test_invalid_rule_and_mapping_rejected(self):
        for cfg in (Config(class_ids={"fire": 0, "smoke": 0}), Config(imgsz=0),
                    Config(rules={"fire": Rule(conf=2), "smoke": Rule()}),
                    Config(rules={"fire": Rule(window_seconds=.5), "smoke": Rule()})):
            with self.assertRaises(ValueError):
                cfg.validate()


class DetectorTests(unittest.TestCase):
    def test_per_class_threshold_and_mapping(self):
        class Tensor:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def tolist(self): return self.values
        result = SimpleNamespace(names={0: "fire", 1: "smoke"}, boxes=SimpleNamespace(
            xyxy=Tensor([[1, 2, 3, 4]] * 3), conf=Tensor([.4, .4, .8]), cls=Tensor([0, 1, 1])))
        detector = Detector.__new__(Detector)
        detector.cfg = Config(rules={"fire": Rule(conf=.3), "smoke": Rule(conf=.6)})
        detector.labels = {0: "fire", 1: "smoke"}
        calls = []
        def predict(**kwargs):
            calls.append(kwargs)
            return [result]
        detector.model = SimpleNamespace(predict=predict)
        detections = detector.predict(object())
        self.assertEqual([(d.label, d.confidence) for d in detections], [("fire", .4), ("smoke", .8)])
        self.assertEqual(calls[0]["conf"], .3)
        self.assertFalse(calls[0]["rect"])


try:
    import cv2
    import numpy as np
    HAVE_CV = True
except ImportError:
    HAVE_CV = False


@unittest.skipUnless(HAVE_CV, "OpenCV and NumPy required")
class PipelineTests(unittest.TestCase):
    def test_real_video_decode_overlay_snapshots_and_events_with_fake_detector(self):
        from firewatch.app import run
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test.avi")
            video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
            self.assertTrue(video.isOpened())
            for i in range(60):
                video.write(np.full((120, 160, 3), 180 if i < 20 else 0, dtype=np.uint8))
            video.release()
            class FakeDetector:
                def __init__(self, cfg): pass
                def predict(self, image):
                    return [Detection("fire", .9, (10, 10, 80, 80))] if image.mean() > 100 else []
            cfg = Config(source=path, display=False, output_dir=str(Path(tmp) / "out"))
            self.assertEqual(run(cfg, detector_factory=FakeDetector), 0)
            session = next((Path(tmp) / "out").iterdir())
            events = [json.loads(line) for line in (session / "events.jsonl").read_text().splitlines()]
            alarms = [e for e in events if e["kind"].startswith("alarm_")]
            self.assertEqual([e["kind"] for e in alarms], ["alarm_start", "alarm_end"])
            self.assertGreaterEqual(alarms[0]["source_seconds"], 1)
            self.assertGreaterEqual(alarms[1]["source_seconds"], 5)
            self.assertIsNotNone(cv2.imread(str(session / alarms[0]["snapshot"])))
            self.assertEqual(events[-1]["frames"], 60)
            self.assertEqual(events[-1]["reason"], "end_of_video")

    def test_capture_failure_is_error_not_all_clear(self):
        from firewatch.app import run
        from firewatch.capture import Frame
        class FakeSource:
            index = 0
            closed = False
            def read(self):
                if self.index == 4:
                    raise TimeoutError("lost camera")
                t = self.index * .5
                self.index += 1
                return Frame(np.zeros((64, 64, 3), dtype=np.uint8), t, self.index)
            def close(self): self.closed = True
        class FakeDetector:
            def __init__(self, cfg): pass
            def predict(self, frame): return [Detection("fire", .9, (1, 1, 20, 20))]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(display=False, save_snapshots=False, output_dir=tmp)
            source = FakeSource()
            with self.assertLogs(level=logging.ERROR):
                code = run(cfg, detector_factory=FakeDetector, source_factory=lambda _: source)
            self.assertEqual(code, 1)
            self.assertTrue(source.closed)
            events = [json.loads(s) for s in next(Path(tmp).glob("*/events.jsonl")).read_text().splitlines()]
            self.assertIn("system_error", [e["kind"] for e in events])
            self.assertNotIn("alarm_end", [e["kind"] for e in events])
            self.assertEqual(events[-1]["active_labels"], ["fire"])

    def test_live_reader_consumes_latest_without_duplicate_frame(self):
        from firewatch.capture import LiveReader
        import threading
        gate = threading.Event()
        third = threading.Event()
        class Camera:
            i = 0
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self):
                self.i += 1
                if self.i > 3:
                    third.set()
                    gate.wait(2)
                    return False, None
                return True, np.zeros((10, 10, 3), np.uint8)
            def release(self): pass
        with patch("cv2.VideoCapture", return_value=Camera()):
            reader = LiveReader(Config(frame_timeout_seconds=.1))
            try:
                self.assertTrue(third.wait(1))
                self.assertEqual(reader.read().sequence, 3)
                with self.assertRaises(TimeoutError):
                    reader.read()
            finally:
                gate.set()
                reader.close()


if __name__ == "__main__":
    unittest.main()
