#!/usr/bin/env python3
"""Convert normalized YOLO polygon labels to axis-aligned YOLO boxes.

Python 3.9+, standard library only. Run: python polygon_to_bbox.py
Or: python polygon_to_bbox.py --src "D:\\data\\train\\labels" --dst "D:\\data\\train\\labels_bbox"

Select a LABELS directory (not the dataset root containing README.txt).
All .txt files below it are processed; relative paths are preserved.
Only label files are exported: images, YAML and caches are not copied.
Class IDs are preserved; no assumption is made about class names.
Input must be YOLO normalized coordinates, not COCO/LabelMe/pixel coordinates.
"""

import argparse
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path


def convert_line(line):
    """Return (output_line, kind). Reject malformed labels instead of dropping them."""
    fields = line.split()
    if not fields:
        return "", "blank"
    try:
        class_id = int(fields[0])
    except ValueError:
        raise ValueError("class_id phai la so nguyen") from None
    if class_id < 0:
        raise ValueError("class_id khong duoc am")
    try:
        coords = [float(v) for v in fields[1:]]
    except ValueError:
        raise ValueError("toa do khong phai so") from None
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords):
        raise ValueError("toa do phai huu han va trong [0, 1]; khong tu dong cat/gioi han toa do")

    if len(coords) == 4:
        x, y, w, h = coords
        if w <= 0 or h <= 0:
            raise ValueError("bbox co chieu rong/cao bang 0")
        tolerance = 1e-6
        if (x - w / 2 < -tolerance or y - h / 2 < -tolerance
                or x + w / 2 > 1 + tolerance or y + h / 2 > 1 + tolerance):
            raise ValueError("bbox vuot bien anh")
        return line.strip(), "bbox"

    if len(coords) < 6 or len(coords) % 2:
        raise ValueError("can 4 toa do bbox hoac it nhat 3 cap (x, y) polygon")
    points = list(zip(coords[::2], coords[1::2]))
    if len(set(points)) < 3:
        raise ValueError("polygon phai co it nhat 3 dinh phan biet")
    area2 = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                in zip(points, points[1:] + points[:1]))
    if abs(area2) <= 1e-15:
        raise ValueError("polygon suy bien hoac co thu tu dinh khong hop le")
    xmin, xmax = min(coords[::2]), max(coords[::2])
    ymin, ymax = min(coords[1::2]), max(coords[1::2])
    box = ((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin)
    # Significant digits retain very small boxes as well as typical image labels.
    return str(class_id) + " " + " ".join(format(v, ".12g") for v in box), "polygon"


def convert_file(path):
    output = []
    counts = Counter()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            converted, kind = convert_line(line)
        except ValueError as error:
            raise ValueError(f"dong {number}: {error}") from error
        if converted:
            output.append(converted)
            counts[kind] += 1
    if not output:
        counts["empty"] += 1
    return "\n".join(output) + ("\n" if output else ""), counts


def convert_directory(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir():
        raise ValueError(f"Khong tim thay thu muc nguon: {source}")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Nguon va dich phai tach biet, khong long vao nhau.")
    if destination.exists():
        raise ValueError("Thu muc dich da ton tai. Hay chon ten thu muc MOI de tranh ghi de.")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
    if not files:
        raise ValueError("Khong co file .txt trong thu muc nguon.")

    print(f"Kiem tra {len(files)} file nhan...")
    totals, errors = Counter(), []
    for index, path in enumerate(files, 1):
        try:
            _, counts = convert_file(path)
            totals.update(counts)
        except (ValueError, OSError, UnicodeError) as error:
            errors.append(f"{path.relative_to(source)}: {error}")
        if index % 10000 == 0:
            print(f"  Da kiem tra {index}/{len(files)}")
    if errors:
        details = "\n".join(errors[:20])
        raise ValueError(f"Co {len(errors)} file loi. CHUA xuat nhan nao.\n{details}\n"
                         "(Hien toi da 20 loi; sua cac file roi chay lai.)")

    # Build a complete temporary output before exposing the destination.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="polygon_bbox_", dir=destination.parent))
    try:
        for index, path in enumerate(files, 1):
            converted, _ = convert_file(path)
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(converted, encoding="utf-8")
            if index % 10000 == 0:
                print(f"  Da chuyen {index}/{len(files)}")
        if destination.exists():
            raise ValueError("Thu muc dich vua duoc tao boi chuong trinh khac; dung de tranh ghi de.")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    totals["files"] = len(files)
    print(f"Hoan tat: {len(files)} file | {totals['polygon']} polygon da chuyen | "
          f"{totals['bbox']} bbox giu nguyen | {totals['empty']} file rong.")
    print(f"Ket qua: {destination}")
    print("Chi xuat NHAN. Giu nguyen ten file khi ghep voi anh; kiem tra class_id voi data.yaml.")
    return totals


def clean_path(value):
    return Path(value.strip().strip('\"').strip("'")).expanduser()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", help="Thu muc nhan nguon")
    parser.add_argument("--dst", help="Thu muc nhan dich MOI")
    args = parser.parse_args()
    try:
        source_text = args.src or input("Nhap thu muc LABELS nguon: ").strip()
        if not source_text:
            raise ValueError("Chua nhap thu muc nguon.")
        source = clean_path(source_text).resolve()
        default = source.with_name(source.name + "_bbox")
        target_text = args.dst or input(f"Thu muc dich moi [Enter = {default}]: ").strip()
        destination = clean_path(target_text) if target_text else default
        convert_directory(source, destination)
        return 0
    except (ValueError, OSError, UnicodeError, EOFError) as error:
        print(f"LOI: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDa huy.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
