"""YOLO fire detection: init YAML, train/resume, val/test, optional Heatmap.
Python 3.10+; recommended Python 3.12. See README_VI.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def path_of(value):
    return Path(str(value).strip().strip('"').strip("'")).expanduser().resolve()


def read_yaml(path):
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"YAML phải chứa mapping: {path}")
    return obj


def write_yaml(path, obj, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def prompt(value, message, default=None):
    if value is not None:
        return value
    answer = input(message + (f" [{default}]" if default else "") + ": ").strip()
    if not answer and default is None:
        raise ValueError("Không được để trống.")
    return answer or default


def init_configs(args):
    # Đường dẫn là của máy đang chạy script, có thể chứa dấu cách/tiếng Việt.
    root = path_of(prompt(args.root, "Thư mục gốc dataset"))
    output = path_of(prompt(args.out, "Đường dẫn YAML train mới", "configs/fire_train.yaml"))
    if output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Đường dẫn đầu ra phải kết thúc bằng .yaml hoặc .yml")
    train = prompt(args.train_images, "Thư mục ảnh train, tương đối với dataset", "train/images")
    valid = prompt(args.val_images, "Thư mục ảnh validation", "valid/images")
    test = args.test_images  # Chỉ khai báo test khi chủ động truyền tùy chọn.
    splits = {"train": train, "val": valid}
    if test:
        splits["test"] = test
    resolved = {}
    for key, value in splits.items():
        folder = (root / value).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Không có thư mục ảnh {key}: {folder}")
        if "images" not in folder.parts:
            raise ValueError(f"Đường dẫn {key} phải có thành phần 'images' để YOLO tìm labels.")
        resolved[key] = folder.as_posix()
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Train, val, test phải là các thư mục khác nhau.")
    project = path_of(prompt(args.project, "Thư mục lưu kết quả", "runs/train/fire"))
    data_file = output.with_name(output.stem + "_data.yaml")
    val_file = output.with_name(output.stem + "_val.yaml")
    solution_file = output.with_name(output.stem + "_solution.yaml")
    for file in (output, data_file, val_file, solution_file):
        if file.exists():
            raise FileExistsError(f"Không ghi đè: {file}. Hãy chọn tên YAML mới.")
    data = {"path": root.as_posix(), **resolved, "names": {0: "fire", 1:"smoke"}}
    # Cấu hình khởi đầu đề xuất; cần tinh chỉnh bằng kết quả thực nghiệm.
    train_cfg = {
        "task": "detect", "mode": "train", "model": args.model,
        "data": data_file.as_posix(), "epochs": 100, "patience": 25,
        "imgsz": 640, "batch": 4, "device": 0, "workers": 0,
        "cache": False, "pretrained": True, "optimizer": "AdamW",
        "lr0": 0.001, "lrf": 0.01, "cos_lr": True,
        "momentum": 0.9, "weight_decay": 0.0005, "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.0, "seed": 42, "deterministic": True,
        "amp": True, "single_cls": False, "rect": False,
        "fraction": 1.0, "freeze": None,
        "val": True, "conf": 0.001, "iou": 0.7, "max_det": 300,
        "hsv_h": 0.01, "hsv_s": 0.4, "hsv_v": 0.4,
        "degrees": 5.0, "translate": 0.1, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "mosaic": 0.5,
        "close_mosaic": 10, "mixup": 0.0,
        "project": project.as_posix(), "name": "fire_yolo26n",
        "exist_ok": False, "save": True, "save_period": 10,
        "plots": True, "verbose": True,
    }
    val_cfg = {
        "data": data_file.as_posix(), "split": "val", "imgsz": 640,
        "batch": 4, "device": None, "workers": 0, "conf": 0.25,
        "iou": 0.7, "max_det": 300, "augment": False,
        "plots": True, "save_json": False, "save_txt": False,
        "save_conf": False, "project": project.as_posix(),
        "name": "best_validation2_fire", "exist_ok": False, "verbose": True,
    }
    solution_cfg = {
        "kind": "heatmap", "source": "demo.mp4", "output": "fire_heatmap.mp4",
        "show": False, "model": None, "classes": [0], "conf": 0.35,
        "iou": 0.7, "imgsz": 640, "device": None, "tracker": "bytetrack.yaml",
        "line_width": 2, "colormap": "COLORMAP_JET", "verbose": False,
    }
    for file, obj in ((data_file, data), (output, train_cfg),
                      (val_file, val_cfg), (solution_file, solution_cfg)):
        write_yaml(file, obj, exclusive=True)
        print(f"Đã tạo: {file}")


def check_dataset(data_file, split="train"):
    """Kiểm tra định dạng bbox 1 class; không đánh giá độ đúng của annotation."""
    data = read_yaml(data_file)
    if data.get("names") not in ({0: "fire", 1: "smoke"}, ["fire", "smoke"]):
        raise ValueError("Dataset cần 2 class: fire=0, smoke=1.")
    root = path_of(data.get("path", Path(data_file).parent))
    keys = ("train", "val") if split == "train" else (split,)
    seen = set()
    for key in keys:
        if key not in data:
            raise ValueError(f"Dataset chưa khai báo split '{key}'.")
        folder = (root / data[key]).resolve()
        if folder in seen:
            raise ValueError("Train và val không được cùng thư mục.")
        seen.add(folder)
        images = [p for p in folder.rglob("*") if p.suffix.lower() in
                  {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}]
        if not images:
            raise ValueError(f"Không có ảnh trong {folder}")
        positive = empty = 0
        for image in images:
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError(f"Thiếu thành phần images: {image}")
            i = len(parts) - 1 - parts[::-1].index("images")
            parts[i] = "labels"
            label = Path(*parts).with_suffix(".txt")
            # Quy tắc chặt hơn YOLO: negative phải có .txt rỗng, tránh bỏ sót label.
            if not label.is_file():
                raise FileNotFoundError(f"Thiếu label: {label}. Ảnh negative cần .txt rỗng.")
            lines = label.read_text(encoding="utf-8-sig").splitlines()
            objects = 0
            for line in lines:
                if not line.strip():
                    continue
                vals = [float(v) for v in line.split()]
                if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
                    raise ValueError(f"Sai định dạng bbox: {label}: {line}")
                cls, x, y, w, h = vals
                if cls not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"Sai class hoặc tọa độ chuẩn hóa: {label}: {line}")
                objects += 1
            positive += objects > 0
            empty += objects == 0
        if positive == 0:
            raise ValueError(f"Split {key} không có ảnh được gán bbox fire.")
        print(f"{key}: {len(images)} ảnh, {positive} ảnh có lửa, {empty} negative.")


def attach_logs(model, tensorboard):
    """Callback cục bộ; không sửa thiết lập logging toàn cục của Ultralytics."""
    state = {"writer": None, "handler": None}

    def start(trainer):
        import torch
        import ultralytics
        from ultralytics.utils import LOGGER
        folder = Path(trainer.save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(folder / "train.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)
        state["handler"] = handler
        meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available()}
        (folder / "environment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=False)
        (folder / "requirements-lock.txt").write_text(freeze.stdout, encoding="utf-8")
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            state["writer"] = SummaryWriter(str(folder / "tensorboard_custom"))

    def epoch(trainer):
        record = {"epoch": trainer.epoch + 1}
        # Callback sau cả train và val; loss/metric chính vẫn nằm trong results.csv.
        for key, value in {**trainer.metrics, **trainer.lr}.items():
            try:
                number = float(value)
                if math.isfinite(number):
                    record[key] = number
            except (ValueError, TypeError):
                continue
        with (Path(trainer.save_dir) / "epochs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if state["writer"]:
            for key, value in record.items():
                if key != "epoch":
                    state["writer"].add_scalar(key, value, record["epoch"])
            state["writer"].flush()

    def close():
        from ultralytics.utils import LOGGER
        if state["writer"]:
            state["writer"].close()
        if state["handler"]:
            LOGGER.removeHandler(state["handler"])
            state["handler"].close()

    model.add_callback("on_pretrain_routine_start", start)
    model.add_callback("on_fit_epoch_end", epoch)
    return close


def validate(weights, config, split=None):
    from ultralytics import YOLO
    cfg = read_yaml(config)
    if split:
        cfg["split"] = split
    check_dataset(cfg["data"], cfg.get("split", "val"))
    metrics = YOLO(str(path_of(weights))).val(**cfg)
    result = {k: float(v) for k, v in metrics.results_dict.items()}
    dest = Path(metrics.save_dir) / "metrics_summary.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Kết quả validation: {dest}")


def train(args):
    from ultralytics import YOLO
    if args.tensorboard:
        import tensorboard  # Fail trước khi train nếu chưa cài.
    if args.resume:
        checkpoint = path_of(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
        kwargs = {"resume": True}  # Khôi phục cấu hình/optimizer/epoch từ checkpoint.
    else:
        if not args.config:
            raise ValueError("Train mới cần --config; tiếp tục cần --resume.")
        cfg = read_yaml(args.config)
        check_dataset(cfg["data"])
        model = YOLO(cfg["model"])
        kwargs = {"cfg": str(path_of(args.config))}
    close = attach_logs(model, args.tensorboard)
    try:
        model.train(**kwargs)
        best = Path(model.trainer.best)
        print(f"Thư mục kết quả: {model.trainer.save_dir}")
        print(f"Checkpoint tốt nhất: {best}")
        if args.val_config:
            if not best.is_file():
                raise FileNotFoundError(f"Chưa có best.pt: {best}")
            validate(best, args.val_config)
    finally:
        close()


def solution(args):
    import cv2
    from ultralytics import solutions
    cfg = read_yaml(args.config)
    if cfg.pop("kind") != "heatmap":
        raise ValueError("Ví dụ Solutions này hỗ trợ kind: heatmap.")
    source = args.source or cfg.pop("source")
    cfg.pop("source", None)
    output = path_of(cfg.pop("output"))
    if output.exists():
        raise FileExistsError(f"Không ghi đè video đã có: {output}")
    cfg["model"] = str(path_of(args.weights))
    cfg["colormap"] = getattr(cv2, cfg["colormap"])
    # source do OpenCV đọc, không truyền source vào constructor Solutions Python.
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
    writer = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn: {source}")
        heatmap = solutions.Heatmap(**cfg)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        output.parent.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Không tạo được video: {output}")
            result = heatmap(frame)
            writer.write(result.plot_im)
            if cfg.get("show") and cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer is None:
            raise RuntimeError("Nguồn không có frame đọc được.")
        print(f"Đã lưu: {output}")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="Tạo 4 YAML mới; không khởi chạy train")
    for flag in ("root", "out", "project", "train-images", "val-images", "test-images"):
        p.add_argument("--" + flag)
    p.add_argument("--model", default="yolo26n.pt")
    p = sub.add_parser("check", help="Kiểm tra dataset và label trước train")
    p.add_argument("--data", required=True)
    p = sub.add_parser("train")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--resume")
    p.add_argument("--val-config")
    p.add_argument("--tensorboard", action="store_true")
    p = sub.add_parser("val")
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["val", "test"])
    p = sub.add_parser("solution")
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--source")
    args = parser.parse_args()
    if args.command == "init":
        init_configs(args)
    elif args.command == "check":
        check_dataset(args.data)
    elif args.command == "train":
        train(args)
    elif args.command == "val":
        validate(args.weights, args.config, args.split)
    else:
        solution(args)


if __name__ == "__main__":
    # Bắt buộc có main guard khi dùng DataLoader multiprocessing trên Windows.
    main()
