"""Select a video, predict fire/smoke, and save an annotated MP4 in predict/1st_predict.

Run: python script/test_fire_screen.py
Or:  python script/test_fire_screen.py --video "capture_4_test/video.mp4"
The output has no audio. Q / Esc stops and saves the processed portion.
"""

import argparse
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--weights", type=Path, help="Path to trained best.pt")
    source.add_argument("--run", default="fire_yolo26n", help="Folder under runs/train")
    parser.add_argument("--video", type=Path, help="Input video; omit to open file picker")
    parser.add_argument("--conf", type=float, default=0.335)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="cpu or 0; default: automatic")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if not args.run or args.run in (".", "..") or any(c in args.run for c in '<>:"/\\|?*') or args.run.endswith((" ", ".")):
        parser.error("--run must be a simple folder name")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    return args


def find_weights(requested, run="fire_yolo26n"):
    path = (requested.expanduser().resolve() if requested is not None
            else ROOT / "runs" / "train" / run / "weights" / "best.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Choose --run / --weights explicitly.")
    return path


def choose_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError('Video picker unavailable. Use --video "path/to/video.mp4".') from exc
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        initial_dir = ROOT / "capture_4_test"
        selected = filedialog.askopenfilename(
            parent=root, title="Select video for fire / smoke prediction",
            initialdir=str(initial_dir if initial_dir.is_dir() else ROOT),
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.webm *.m4v *.wmv *.mpeg *.mpg"),
                       ("All files", "*.*")])
    finally:
        root.destroy()
    return Path(selected) if selected else None


def predict_video(args, video, weights):
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Use the training environment with opencv-python, torch and ultralytics installed.") from exc
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    capture = cv2.VideoCapture(str(video))
    writer = None
    frame_count = 0
    stopped = False
    window = "Fire / Smoke - video (Q or Esc to stop)"
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        success, frame = capture.read()
        if not success:
            raise ValueError(f"Cannot read the first frame: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
            print("Video FPS unavailable; using 30 FPS for output.")
        height, width = frame.shape[:2]
        model = YOLO(str(weights))
        output_dir = ROOT / "predict" / "predict_model1_conf0.335"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video.stem}_predict_{time.time_ns()}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Cannot create output video: {output_path}")
        print(f"Model: {weights}\nDevice: {device}\nInput: {video}\nOutput: {output_path}")
        print(f"Output: {width}x{height}, {fps:g} FPS, without audio.")
        if not args.no_preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            scale = min(1.0, 960 / width, 640 / height)
            cv2.resizeWindow(window, max(1, round(width * scale)), max(1, round(height * scale)))
        while success:
            result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz,
                                   device=device, verbose=False, save=False)[0]
            annotated = result.plot()
            writer.write(annotated)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames.")
            if not args.no_preview:
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    stopped = True
                    break
            success, frame = capture.read()
    except KeyboardInterrupt:
        stopped = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    if frame_count:
        status = "Stopped early; saved processed portion" if stopped else "Saved"
        print(f"{status}: {output_path} ({frame_count} frames)")
    elif stopped:
        print("Stopped before any frames were saved.")


def main():
    args = parse_args()
    try:
        video = args.video if args.video is not None else choose_video()
        if video is None:
            print("Selection cancelled.")
            return
        video = video.expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Video not found: {video}")
        weights = find_weights(args.weights, args.run)
        predict_video(args, video, weights)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
