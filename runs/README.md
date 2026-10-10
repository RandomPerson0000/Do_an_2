# Training versions

- `train/fire_smoke_yolo26n_V1/`: existing model and training results, kept for comparison.
- `train/fire_smoke_yolo26n_V2/`: next training run; created when training starts.
- `detect/`: existing prediction and validation outputs.

Run these commands from the project root in the Python environment used for training.

## Train V2

```powershell
python script/train.py
```

Training starts from `models/pretrained/yolo26n.pt`. It does not resume V1.
The script refuses an existing run folder to protect earlier results.
Training settings and dataset are unchanged.

Before training, `train.py` generates `configs/datasets/fire_smoke_yolo26n_V2.yaml`
and passes that file to YOLO. It no longer reads `datasets/data.yaml`.
The YAML contains the absolute dataset root, train/valid/test image paths,
and class IDs `0: Fire`, `1: Smoke`. Edit `CLASS_NAMES` in `script/train.py`
only if the dataset's label ID mapping changes.

The YAML filename follows `--name`, so V3 generates
`configs/datasets/fire_smoke_yolo26n_V3.yaml`. Existing YAML files are protected
from overwriting too; choose a fresh name if retrying after a failed run.

## Test V2 on screen

```powershell
python script/test.py --select
```

Default weights: `runs/train/fire_smoke_yolo26n_V2/weights/best.pt`.
If V2 is not available, testing stops with an error instead of falling back to V1.

## Compare with V1

```powershell
python script/test.py --select --run fire_smoke_yolo26n_V1
```

## Later training runs

Use a new version name for each run, including a retry of an interrupted run:

```powershell
python script/train.py --name fire_smoke_yolo26n_V3
python script/test.py --select --run fire_smoke_yolo26n_V3
```

You can also select a checkpoint anywhere with `--weights PATH_TO_MODEL`.
Choose either `--run` or `--weights`, not both.
