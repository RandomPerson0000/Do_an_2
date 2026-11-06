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
        "imgsz": 640, "batch": -1, "device": 0, "workers": 0,
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


def check_dataset(data_file, split="train", *, strict_missing_labels=True,
                  require_positive=True):
    """Kiểm tra bbox 2 class và đếm ảnh/object theo annotation.

    Train giữ quy tắc cũ: kiểm tra train + val, bắt buộc có file label.
    Lệnh check dùng split=all, có thể thống kê cả ảnh thiếu file label.
    'Non-label fire/smoke' chỉ nghĩa là không có bbox của lớp đó trong nhãn;
    không chứng minh rằng ảnh thực tế không có đối tượng đó.
    """
    data = read_yaml(data_file)
    if data.get("names") not in ({0: "fire", 1: "smoke"}, ["fire", "smoke"]):
        raise ValueError("Dataset cần 2 class: fire=0, smoke=1.")
    if "nc" in data and data["nc"] != 2:
        raise ValueError("Dataset cần nc=2 nếu có khai báo nc.")
    root = path_of(data.get("path", Path(data_file).parent))
    if split == "valid":
        split = "val"
    if split == "all":
        keys = ("train", "val", "test")
    elif split == "train":
        keys = ("train", "val")  # Giữ kiểm tra trước train như bản gốc.
    elif split in ("val", "test"):
        keys = (split,)
    else:
        raise ValueError(f"Split không hợp lệ: {split}")
    seen = set()
    results = {}
    skipped = []
    for key in keys:
        display = "valid" if key == "val" else key
        if key not in data or not data[key]:
            if split == "all" and key == "test":
                skipped.append("test: chưa khai báo đường dẫn test trong YAML, không thống kê.")
                continue
            raise ValueError(f"Dataset chưa khai báo split '{key}'.")
        if not isinstance(data[key], str):
            raise ValueError(f"Check hiện hỗ trợ một thư mục ảnh cho mỗi split: {key}.")
        folder = (root / data[key]).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Không có thư mục ảnh {display}: {folder}")
        if folder in seen:
            raise ValueError("Train, valid, test không được cùng thư mục.")
        seen.add(folder)
        images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in
                        {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
        if not images:
            raise ValueError(f"Không có ảnh trong {folder}")
        stats = {"images": len(images), "fire_images": 0, "smoke_images": 0,
                 "both_images": 0, "fire_objects": 0, "smoke_objects": 0,
                 "empty_labels": 0, "missing_labels": 0}
        for image in images:
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError(f"Thiếu thành phần images: {image}")
            i = len(parts) - 1 - parts[::-1].index("images")
            parts[i] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.exists():
                if strict_missing_labels:
                    raise FileNotFoundError(f"Thiếu label: {label}. Ảnh negative cần .txt rỗng.")
                stats["missing_labels"] += 1
                continue
            if not label.is_file():
                raise ValueError(f"Đường dẫn label không phải file: {label}")
            objects = [0, 0]
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    vals = [float(v) for v in line.split()]
                except ValueError:
                    raise ValueError(f"Nhãn không phải số: {label}, dòng {line_number}: {line}") from None
                if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
                    raise ValueError(f"Sai định dạng bbox: {label}, dòng {line_number}: {line}")
                cls, x, y, w, h = vals
                if cls not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"Sai class hoặc tọa độ chuẩn hóa: {label}, dòng {line_number}: {line}")
                objects[int(cls)] += 1
            has_fire, has_smoke = objects[0] > 0, objects[1] > 0
            stats["fire_images"] += has_fire
            stats["smoke_images"] += has_smoke
            stats["both_images"] += has_fire and has_smoke
            stats["fire_objects"] += objects[0]
            stats["smoke_objects"] += objects[1]
            stats["empty_labels"] += not (has_fire or has_smoke)
        stats["fire_nonlabel_images"] = stats["images"] - stats["fire_images"]
        stats["smoke_nonlabel_images"] = stats["images"] - stats["smoke_images"]
        stats["fire_only_images"] = stats["fire_images"] - stats["both_images"]
        stats["smoke_only_images"] = stats["smoke_images"] - stats["both_images"]
        stats["background_images"] = stats["empty_labels"] + stats["missing_labels"]
        positive = stats["images"] - stats["background_images"]
        if require_positive and positive == 0:
            raise ValueError(f"Split {display} không có ảnh được gán bbox fire hoặc smoke.")
        results[key] = stats
        print(f"\n{display}: {stats['images']} ảnh, {positive} ảnh có label fire/smoke, "
              f"{stats['background_images']} ảnh không có bbox cả hai lớp.")
        print(f"  fire : {stats['fire_images']} ảnh đã label, "
              f"{stats['fire_nonlabel_images']} ảnh non-label fire, {stats['fire_objects']} objects.")
        print(f"  smoke: {stats['smoke_images']} ảnh đã label, "
              f"{stats['smoke_nonlabel_images']} ảnh non-label smoke, {stats['smoke_objects']} objects.")
        print(f"  Cả fire + smoke: {stats['both_images']} ảnh; "
              f"chỉ fire: {stats['fire_only_images']} ảnh; chỉ smoke: {stats['smoke_only_images']} ảnh.")
        print(f"  Không có bbox: {stats['empty_labels']} file label rỗng, "
              f"{stats['missing_labels']} ảnh thiếu file label.")
    for message in skipped:
        print("\n" + message)
    print("\nGhi chú: non-label của một lớp = không có annotation lớp đó, "
          "gồm cả ảnh chỉ có lớp còn lại và ảnh rỗng/thiếu label.")
    print("Ảnh có cả hai lớp được tính vào số ảnh đã label của cả fire và smoke. "
          "Thống kê không kiểm tra nội dung ảnh.")
    if any(stats["missing_labels"] for stats in results.values()):
        print("Lưu ý: thiếu file label có thể là ảnh chưa gán nhãn. "
              "Hãy kiểm tra trước khi coi là negative; kiểm tra trước train vẫn yêu cầu .txt rỗng.")
    return results


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
    if args.resume and path_of(args.resume).suffix.lower() != ".pt":
        raise ValueError("Resume cần checkpoint last.pt, không nhận YAML. Dataset mới: chọn mục 7 (finetune).")
    from ultralytics import YOLO
    if args.tensorboard:
        import tensorboard  # Fail trước khi train nếu chưa cài.
    if args.resume:
        checkpoint = path_of(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(str(checkpoint))
        ckpt = model.ckpt or {}
        if ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None:
            raise ValueError("Checkpoint không còn trạng thái resume (có thể đã train xong). "
                             "Chọn mục 7 để fine-tune từ best.pt với dataset mới.")
        kwargs = {"resume": True}  # Khôi phục cấu hình/optimizer/epoch từ checkpoint.
    else:
        if not args.config:
            raise ValueError("Train mới cần --config; tiếp tục cần --resume.")
        cfg = read_yaml(args.config)
        if "model" not in cfg or "data" not in cfg:
            raise ValueError("Mục 3 cần YAML TRAIN có model/data. Mục 7 nhận YAML DATASET có train/val/names.")
        if cfg.get("resume"):
            raise ValueError("YAML train đang bật resume. Dùng mục 4 cho resume, mục 7 cho dataset mới.")
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


def prepare_finetune(args):
    """Validate inputs without loading GPU/checkpoint; never read old run args."""
    weights = path_of(args.weights)
    if weights.suffix.lower() != ".pt" or not weights.is_file():
        raise ValueError(f"Cần checkpoint .pt có thật (khuyến nghị best.pt): {weights}")
    data_path = path_of(args.data)
    data = read_yaml(data_path)
    if not all(k in data for k in ("train", "val", "names")):
        raise ValueError("--data cần YAML DATASET có train, val, names; không dùng args.yaml/YAML train.")
    if data["names"] not in ({0: "fire", 1: "smoke"}, ["fire", "smoke"]):
        raise ValueError("Dataset mới phải giữ đúng fire=0, smoke=1. Hãy sửa cả ID trong labels nếu thứ tự khác.")
    if "nc" in data and data["nc"] != 2:
        raise ValueError("Dataset phải có nc=2.")
    # Resolve relative paths against the dataset YAML, not the current directory.
    root_value = data.get("path") or "."
    root = (data_path.parent / str(root_value)).expanduser().resolve()
    normalized = {"path": root.as_posix(), "names": {0: "fire", 1: "smoke"}}
    print("\nĐường dẫn ảnh sẽ sử dụng (chưa bắt đầu train):", flush=True)
    for split in ("train", "val", "test"):
        if split == "test" and getattr(args, "no_test", False):
            print("  test: bỏ qua theo lựa chọn của bạn", flush=True)
            continue
        override = getattr(args, split + "_images", None)
        value = override if override is not None else data.get(split)
        if value is None or value == "":
            print(f"  {split}: không khai báo", flush=True)
            continue
        if not isinstance(value, str):
            raise ValueError(f"Hiện hỗ trợ một thư mục images cho mỗi split: {split}.")
        # Explicit user paths override stale YAML roots; relative overrides use YAML folder.
        base = data_path.parent if override is not None else root
        folder = (base / Path(value).expanduser()).resolve()
        origin = "bạn nhập" if override is not None else "YAML"
        print(f"  {split} ({origin}): {folder}", flush=True)
        normalized[split] = folder.as_posix()
    for split in ("train", "val", "test"):
        if split not in normalized:
            continue
        folder = Path(normalized[split])
        if not folder.is_dir():
            flag = "val" if split == "val" else split
            raise FileNotFoundError(
                f"Không có thư mục ảnh {split}: {folder}. "
                f"Nhập lại thư mục images ở mục 7 hoặc dùng --{flag}-images. "
                f"Nếu giữ YAML, kiểm tra path/{split} trong {data_path}. "
                "Ký hiệu ../ sẽ đi lên một thư mục; script không tự đoán đường dẫn thay thế."
            )
    if len(set(normalized[s] for s in ("train", "val", "test") if s in normalized)) != sum(s in normalized for s in ("train", "val", "test")):
        raise ValueError("Train, valid, test không được dùng cùng thư mục images.")
    if args.epochs < 1 or args.patience < 0 or args.workers < 0:
        raise ValueError("epochs >= 1; patience và workers >= 0.")
    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError("imgsz phải là bội số dương của 32 (ví dụ 640).")
    if args.batch != -1 and args.batch < 1:
        raise ValueError("batch phải là -1 hoặc số nguyên dương.")
    if not math.isfinite(args.lr0) or not 0 < args.lr0 < 1:
        raise ValueError("lr0 phải nằm trong (0, 1).")
    output = path_of(args.output)
    if output == output.parent or not output.name:
        raise ValueError("Cần đường dẫn có tên thư mục run cuối cùng.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Thư mục đích không rỗng: {output}. Chọn tên run mới; resume dùng mục 4.")
    # Fresh, explicit configuration prevents checkpoint settings leaking into this run.
    cfg = {
        "task": "detect", "mode": "train", "model": weights.as_posix(),
        "data": (output / "finetune_data.yaml").as_posix(), "resume": False,
        "epochs": args.epochs, "patience": args.patience, "time": None,
        "imgsz": args.imgsz, "batch": args.batch, "device": args.device,
        "workers": args.workers, "cache": False, "pretrained": True,
        "optimizer": "AdamW", "lr0": args.lr0, "lrf": 0.01, "cos_lr": True,
        "momentum": 0.9, "weight_decay": 0.0005, "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.0, "seed": 42, "deterministic": True,
        "amp": True, "single_cls": False, "rect": False,
        "fraction": 1.0, "freeze": None, "val": True, "conf": 0.001,
        "iou": 0.7, "max_det": 300, "hsv_h": 0.01, "hsv_s": 0.4,
        "hsv_v": 0.4, "degrees": 5.0, "translate": 0.1, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.5, "close_mosaic": min(10, args.epochs), "mixup": 0.0,
        "project": output.parent.as_posix(), "name": output.name,
        "exist_ok": True, "save": True, "save_period": 10,
        "plots": True, "verbose": True,
    }
    # Validate with the same label checks as normal training, using a temporary YAML.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="fire_smoke_check_") as tmp:
        check_file = Path(tmp) / "data.yaml"
        write_yaml(check_file, normalized)
        check_dataset(check_file, strict_missing_labels=False, require_positive=False)
    print("\nFINE-TUNE — lượt train mới, resume=False", flush=True)
    print(f"Trọng số nguồn: {weights}")
    print(f"YAML dataset nguồn: {data_path}")
    print(f"Train images: {normalized['train']}")
    print(f"Val images: {normalized['val']}")
    print(f"Thư mục kết quả CHÍNH XÁC: {output}")
    print(f"Epoch tối đa: {args.epochs}; patience: {args.patience}; AdamW lr0={args.lr0}")
    print("Test không được dùng trong quá trình train/chọn best.pt.", flush=True)
    return weights, output, normalized, cfg


def finetune(args):
    weights, output, data, cfg = prepare_finetune(args)
    if args.dry_run:
        print("Kiểm tra thành công. Chưa nạp model, chưa train, chưa tạo thư mục kết quả.")
        return
    from ultralytics import YOLO
    if args.tensorboard:
        import tensorboard
    model = YOLO(str(weights))
    if model.task != "detect" or model.names != {0: "fire", 1: "smoke"}:
        raise ValueError("Checkpoint nguồn cần là detection 2 lớp fire=0, smoke=1.")
    # An existing EMPTY folder is allowed; never silently overwrite previous results.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Thư mục vừa có dữ liệu: {output}. Chọn thư mục khác.")
    write_yaml(output / "finetune_data.yaml", data, exclusive=True)
    config_path = output / "finetune_train.yaml"
    write_yaml(config_path, cfg, exclusive=True)
    state = {"epochs_completed": 0}

    def verify_run(trainer):
        if path_of(trainer.save_dir) != output:
            raise RuntimeError(f"Sai thư mục output: {trainer.save_dir}; yêu cầu: {output}")
        if trainer.args.resume or path_of(trainer.args.data) != path_of(cfg["data"]):
            raise RuntimeError("Cấu hình bị thay đổi: cần resume=False và dataset mới.")
        if trainer.start_epoch != 0:
            raise RuntimeError("Fine-tune mới phải bắt đầu từ epoch 1.")

    def count_epoch(trainer):
        state["epochs_completed"] += 1

    model.add_callback("on_pretrain_routine_start", verify_run)
    model.add_callback("on_train_epoch_end", count_epoch)
    close = attach_logs(model, args.tensorboard)
    try:
        # cfg replaces inherited training overrides; explicit args lock routing and data.
        model.train(cfg=str(config_path), data=cfg["data"], model=str(weights),
                    resume=False, project=cfg["project"], name=cfg["name"], exist_ok=True)
        if state["epochs_completed"] == 0:
            raise RuntimeError("Không có epoch train nào hoàn thành. Dòng Speed không phải thời gian train.")
        print(f"\nĐã hoàn thành {state['epochs_completed']} epoch fine-tune (tối đa {args.epochs}).")
        print(f"Kết quả: {model.trainer.save_dir}")
        print(f"Best: {model.trainer.best}")
        print(f"Last: {model.trainer.last}")
        print("Speed ... per image là tốc độ predict/validation, không phải thời gian 100 epoch.")
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
    p = sub.add_parser("check", help="Thống kê ảnh/objects fire, smoke trong train, valid, test")
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["all", "train", "val", "valid", "test"], default="all",
                   help="Mặc định kiểm tra cả ba tập; train kiểm tra train + val như trước.")
    p.add_argument("--strict-labels", action="store_true",
                   help="Báo lỗi khi thiếu .txt; mặc định check thống kê riêng các ảnh này.")
    p = sub.add_parser("train")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--resume")
    p.add_argument("--val-config")
    p.add_argument("--tensorboard", action="store_true")
    p = sub.add_parser("finetune", help="Fine-tune checkpoint với dataset mới và thư mục output chính xác")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="YAML dataset mới, không phải YAML train")
    p.add_argument("--train-images", help="Thay thư mục train trong YAML; tương đối tính từ thư mục YAML")
    p.add_argument("--val-images", help="Thay thư mục val trong YAML; tương đối tính từ thư mục YAML")
    test_group = p.add_mutually_exclusive_group()
    test_group.add_argument("--test-images", help="Thay thư mục test trong YAML; tương đối tính từ thư mục YAML")
    test_group.add_argument("--no-test", action="store_true", help="Bỏ test khỏi cấu hình lượt này")
    p.add_argument("--output", required=True, help="Toàn bộ đường dẫn thư mục run, gồm tên cuối cùng")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.0001)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra, không nạp model hoặc tạo run")
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
        check_dataset(args.data, args.split, strict_missing_labels=args.strict_labels,
                      require_positive=False)
    elif args.command == "train":
        train(args)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "val":
        validate(args.weights, args.config, args.split)
    else:
        solution(args)


if __name__ == "__main__":
    # Bắt buộc có main guard khi dùng DataLoader multiprocessing trên Windows.
    main()
