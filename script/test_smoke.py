"""Run a trained YOLO model on a live desktop capture.

Install in the environment used for training:
    python -m pip install mss opencv-python
Run and select the video area:
    python script/test_smoke.py --select
Or capture a whole monitor:
    python script/test_smoke.py --monitor 1
Press Q or Esc in the result window to stop.
"""

import argparse
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--weights", type=Path, help="Explicit path to a trained best.pt")
    model_source.add_argument("--run", default="smoke_yolo26n_V1",
                              help="Folder under runs/train (default: smoke_yolo26n_V1)")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor number, starting at 1")
    parser.add_argument("--list-monitors", action="store_true", help="List monitors and exit")
    area = parser.add_mutually_exclusive_group()
    area.add_argument("--select", action="store_true", help="Select a rectangle on a screenshot before starting")
    area.add_argument("--region", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                      help="Capture rectangle relative to the selected monitor, in pixels")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", default=None, help="For example: cpu or 0 (default: automatic)")
    parser.add_argument("--fps", type=float, default=30, help="Maximum capture rate (default: 30)")
    args = parser.parse_args()
    if not args.run or args.run in (".", "..") or any(c in args.run for c in '<>:"/\\|?*') or args.run.endswith((" ", ".")):
        parser.error("--run must be a simple folder name")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz <= 0 or args.fps <= 0:
        parser.error("--imgsz and --fps must be positive")
    return args


def find_weights(requested, run="smoke_yolo26n_V1"):
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        return path
    path = ROOT / "runs" / "train" / run / "weights" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Train this version first, or choose --run / --weights explicitly.")
    return path


def main():
    args = parse_args()
    # Report a missing version before opening screen capture or selection windows.
    weights = None
    if not args.list_monitors:
        try:
            weights = find_weights(args.weights, args.run)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        import cv2
        import mss
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing capture dependency. Run: python -m pip install mss opencv-python numpy") from exc

    try:
        with mss.mss() as capture:
            if args.list_monitors:
                for index, monitor in enumerate(capture.monitors[1:], start=1):
                    print(f"Monitor {index}: {monitor}")
                return
            if not 1 <= args.monitor < len(capture.monitors):
                raise ValueError("Invalid monitor number. Use --list-monitors to check.")
            monitor = capture.monitors[args.monitor]
            region = dict(monitor)
            rectangle = args.region
            if args.select:
                snapshot = np.asarray(capture.grab(monitor))[:, :, :3].copy()
                # Scale the selection image to fit on the selected display.
                scale = min(1.0, 1280 / monitor["width"],
                            max(1, monitor["height"] - 160) / monitor["height"])
                preview = cv2.resize(snapshot, None, fx=scale, fy=scale)
                title = "Select video area - ENTER to confirm, C to cancel"
                print("Drag around the video, then press ENTER. C cancels.")
                x, y, width, height = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow(title)
                if width == 0 or height == 0:
                    print("Selection cancelled.")
                    return
                sx = monitor["width"] / preview.shape[1]
                sy = monitor["height"] / preview.shape[0]
                left, top = round(x * sx), round(y * sy)
                right = min(monitor["width"], round((x + width) * sx))
                bottom = min(monitor["height"], round((y + height) * sy))
                rectangle = (left, top, right - left, bottom - top)
            if rectangle is not None:
                x, y, width, height = rectangle
                if (x < 0 or y < 0 or width <= 0 or height <= 0
                        or x + width > monitor["width"] or y + height > monitor["height"]):
                    raise ValueError("Capture region must fit inside the selected monitor.")
                region = dict(left=monitor["left"] + x, top=monitor["top"] + y,
                              width=width, height=height)
                print(f"Reuse selection: --monitor {args.monitor} --region {x} {y} {width} {height}")

            try:
                import torch
                from ultralytics import YOLO
            except ImportError as exc:
                raise SystemExit("Use the Python environment used for training (torch and ultralytics required).") from exc
            device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
            print(f"Model: {weights}\nDevice: {device}\nCapture: {region}")
            print("Keep the result window outside the captured area. Press Q or Esc to exit.")
            model = YOLO(str(weights))
            window = "Fire / Smoke - live screen (Q or Esc to exit)"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            display_scale = min(1.0, 960 / region["width"], 640 / region["height"])
            cv2.resizeWindow(window, max(1, round(region["width"] * display_scale)),
                             max(1, round(region["height"] * display_scale)))
            last_frame = None
            fps = 0.0
            while True:
                started = time.perf_counter()
                if last_frame is not None:
                    measured = 1.0 / max(started - last_frame, 1e-6)
                    fps = measured if fps == 0 else 0.9 * fps + 0.1 * measured
                last_frame = started
                # MSS returns BGRA; YOLO accepts a contiguous BGR numpy image.
                frame = np.asarray(capture.grab(region))[:, :, :3].copy()
                result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz,
                                       device=device, verbose=False, save=False)[0]
                annotated = result.plot()
                cv2.putText(annotated, f"FPS: {fps:.1f} | Q / Esc: quit", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window, annotated)
                remaining = 1.0 / args.fps - (time.perf_counter() - started)
                key = cv2.waitKey(max(1, round(remaining * 1000))) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
