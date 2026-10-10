import argparse
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO


# V1_smoke labels use class ID 0 for smoke.
CLASS_NAMES = ["Smoke"]


def create_dataset_yaml(project_root, run_name):
    """Create a separate dataset config for this run, without overwriting one."""
    dataset_root = project_root / "datasets" / "V1_smoke"
    splits = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    for split, relative_path in splits.items():
        if not (dataset_root / relative_path).is_dir():
            raise FileNotFoundError(f"Dataset {split} folder not found: {dataset_root / relative_path}")
        labels_dir = (dataset_root / relative_path).parent / "labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Dataset {split} labels folder not found: {labels_dir}")

    config = {
        "path": dataset_root.resolve().as_posix(),
        **splits,
        "nc": len(CLASS_NAMES),
        "names": dict(enumerate(CLASS_NAMES)),
    }
    output = project_root / "configs" / "datasets" / f"{run_name}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Train a smoke detector on datasets/V1_smoke.")
    parser.add_argument("--name", default="smoke_yolo26n_V1", help="New run folder name (must not exist)")
    args = parser.parse_args()
    if not args.name or args.name in (".", "..") or any(c in args.name for c in '<>:"/\\|?*') or args.name.endswith((" ", ".")):
        parser.error("--name must be a simple folder name")
    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "runs" / "train" / args.name
    if run_dir.exists():
        parser.error(f"Run folder already exists: {run_dir}. Choose a new --name, for example smoke_yolo26n_V2.")

    config_path = project_root / "configs" / "datasets" / f"{args.name}.yaml"
    if config_path.exists():
        parser.error(f"Dataset config already exists: {config_path}. Choose a new --name to preserve it.")

    pretrained = project_root / "models" / "pretrained" / "yolo26n.pt"
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained}")
    print(f"New training run: {run_dir}")
    print(f"Starting from pretrained model: {pretrained}")
    model = YOLO(str(pretrained))
    dataset_yaml = create_dataset_yaml(project_root, args.name)
    print(f"Dataset YAML: {dataset_yaml}")
    print("Dataset: V1_smoke | Class: 0 = Smoke | Validation: valid/images")
    device = 0 if torch.cuda.is_available() else "cpu"
    batch_size = 16 if device != "cpu" else 8

    model.train(
        data=str(dataset_yaml),
        epochs=150,
        imgsz=640,
        batch=batch_size,
        device=device,
        workers=4,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        hsv_h=0.01,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.9,
        close_mosaic=10,
        amp=True,
        cache="disk",
        patience=30,
        save=True,
        save_period=10,
        project=str(project_root / "runs" / "train"),
        name=args.name,
        exist_ok=False,
        plots=True,
        deterministic=True,
        seed=42,
    )

if __name__ == "__main__":
    main()
