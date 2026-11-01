#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tao ban sao YOLO detection 2 lop: fire=0, smoke=1; bo other=1 cu.

Yeu cau: Python >= 3.9, PyYAML (python -m pip install pyyaml).
Chay: python filter_other_labels.py --dataset "D:/FireDataset"
Hoac dat file nay canh data.yaml, train, valid, test va chay khong tham so.
Ket qua mac dinh: thu muc cung cap <ten_dataset>_2classes.
Giu nguyen anh, ke ca anh chi co other (label se rong). Khong sua du lieu goc.
Chi ho tro YOLO detection: class_id x_center y_center width height.
"""

import argparse
import math
from pathlib import Path
import shutil
import sys
import tempfile

try:
    import yaml
except ImportError:
    sys.exit("Can cai PyYAML: python -m pip install pyyaml")

SPLITS = ("train", "valid", "test")
EXPECTED_NAMES = ["fire", "other", "smoke"]


def convert_label(path):
    """Validate every box, remove other, remap smoke; preserve coordinate text."""
    kept = []
    counts = [0, 0, 0]
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        location = f"{path}, dong {line_number}"
        if len(fields) != 5:
            raise ValueError(f"{location}: can 5 cot YOLO detection, nhan duoc {len(fields)}.")
        if fields[0] not in ("0", "1", "2"):
            raise ValueError(f"{location}: class ID khong hop le: {fields[0]!r}.")
        try:
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            raise ValueError(f"{location}: toa do khong phai so.") from None
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coords):
            raise ValueError(f"{location}: toa do phai huu han va nam trong [0, 1].")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{location}: width va height phai lon hon 0.")
        old_id = int(fields[0])
        counts[old_id] += 1
        if old_id == 1:
            continue
        # Only replace the class token, never replace numbers in box coordinates.
        kept.append(("0" if old_id == 0 else "1") + " " + line.strip().split(maxsplit=1)[1])
    return "\n".join(kept) + ("\n" if kept else ""), counts


def convert_dataset(source, destination):
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Khong tim thay dataset: {source}")
    if destination == source or source in destination.parents:
        raise ValueError("Thu muc ket qua phai nam NGOAI dataset goc.")
    if destination.exists():
        raise ValueError(f"Thu muc ket qua da ton tai, khong ghi de: {destination}")

    yaml_path = source / "data.yaml"
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("data.yaml khong hop le.")
    names = config.get("names")
    if isinstance(names, dict):
        if set(names) != {0, 1, 2}:
            raise ValueError("Can names voi dung ID 0, 1, 2.")
        names = [names[index] for index in range(3)]
    if names != EXPECTED_NAMES or config.get("nc", 3) != 3:
        raise ValueError("Can YAML goc: nc=3, names=['fire', 'other', 'smoke']. "
                         "Khong chay lai tren dataset da chuyen 2 lop.")
    for split in SPLITS:
        for folder in ("images", "labels"):
            if not (source / split / folder).is_dir():
                raise ValueError(f"Thieu thu muc: {source / split / folder}")

    # Build in a temporary sibling folder. Publish only after all labels pass.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".filter_other_", dir=destination.parent))
    report = []
    try:
        for split in SPLITS:
            print(f"Dang sao chep va xu ly {split}...", flush=True)
            shutil.copytree(source / split, staging / split,
                            ignore=shutil.ignore_patterns("*.cache", "*.cache.npy"))
            totals = [0, 0, 0]
            files = empty = 0
            for label in sorted((staging / split / "labels").rglob("*")):
                if not label.is_file() or label.suffix.lower() != ".txt":
                    continue
                text, counts = convert_label(label)
                label.write_text(text, encoding="utf-8")
                files += 1
                empty += not bool(text)
                totals = [a + b for a, b in zip(totals, counts)]
            report.append(f"{split}: {files} file nhan | fire giu={totals[0]} | "
                          f"other bo={totals[1]} | smoke doi 2->1={totals[2]} | "
                          f"file nhan rong={empty}")

        # Absolute root avoids resolving the old '../train/images' outside output.
        config.update(path=destination.as_posix(), train="train/images",
                      val="valid/images", test="test/images", nc=2,
                      names=["fire", "smoke"])
        (staging / "data.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (staging / "filter_report.txt").write_text(
            "Source: " + str(source) + "\nOutput: " + str(destination) + "\n"
            + "\n".join(report) + "\n", encoding="utf-8")
        # mkdir reserves a new destination; existing output is never overwritten.
        destination.mkdir()
        try:
            for child in staging.iterdir():
                shutil.move(str(child), str(destination / child.name))
        except BaseException:
            shutil.rmtree(destination)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print("\n" + "\n".join(report))
    print(f"\nHOAN TAT. Dataset 2 lop: {destination}")
    print(f"YAML de train: {destination / 'data.yaml'}")
    print("Du lieu goc giu nguyen. Anh chi co other duoc giu lam anh nen.")
    print("Neu di chuyen dataset ket qua, cap nhat 'path' trong data.yaml.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\Do_An\Fire_Smoke_develop\datasets\FireSmokeDataset.v3i.yolo26").resolve(),
                        help="Thu muc chua data.yaml, train, valid, test; mac dinh canh script.")
    parser.add_argument("--output", type=Path,
                        help="Thu muc moi de luu ket qua; mac dinh <dataset>_2classes cung cap.")
    args = parser.parse_args()
    source = args.dataset.expanduser().resolve()
    output = args.output or source.with_name(source.name + "_2classes")
    try:
        convert_dataset(source, output)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"LOI: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Da dung. Du lieu goc khong thay doi.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
