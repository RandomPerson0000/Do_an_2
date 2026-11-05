"""Lọc label khói từ dataset riêng sang thư mục MỚI, trước khi hợp nhất dataset.

- Không sửa dataset gốc: ảnh được copy, label được ghi mới vào --dst.
- Chỉ giữ dòng thuộc class nguồn (--src-class), đổi sang class đích (--dst-class).
- Nhận cả bbox (5 giá trị) lẫn polygon (class + >=3 điểm x y); polygon -> bbox.
- Báo lỗi/bỏ qua dòng hỏng, ghi báo cáo cuối cùng.

Ví dụ (Negative_Samples chỉ có smoke = class 0, hợp nhất vào dataset smoke = 1):
  python filter_smoke_labels.py ^
      --src D:\\Do_An\\Fire_Smoke_develop\\datasets\\Negative_Samples ^
      --dst D:\\Do_An\\Fire_Smoke_develop\\datasets\\Negative_Samples_smoke ^
      --src-class 0 --dst-class 1 --dry-run
Bỏ --dry-run để chạy thật.

Cấu trúc nguồn: <src>/{train,valid,test}/{images,labels}
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "valid", "val", "test")
MIN_SIZE = 1e-4  # bỏ box nhỏ hơn ~0.01% ảnh (gần như vô hình)


def parse_line(line: str):
    """Trả về (class_id, x, y, w, h) chuẩn hóa, hoặc raise ValueError."""
    vals = [float(v) for v in line.split()]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("giá trị không hữu hạn")
    if len(vals) == 5:
        cls, x, y, w, h = vals
    elif len(vals) >= 7 and len(vals) % 2 == 1:  # polygon
        cls = vals[0]
        xs, ys = vals[1::2], vals[2::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        raise ValueError(f"số giá trị không hợp lệ ({len(vals)})")
    if cls != int(cls):
        raise ValueError("class id không phải số nguyên")
    # Cắt box vượt biên ảnh (sai số làm tròn của công cụ gán nhãn).
    x1, x2 = max(0.0, x - w / 2), min(1.0, x + w / 2)
    y1, y2 = max(0.0, y - h / 2), min(1.0, y + h / 2)
    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return int(cls), x, y, w, h


def process_split(src_split: Path, dst_split: Path, args, total: Counter):
    img_dir, lbl_dir = src_split / "images", src_split / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"  Bỏ qua {src_split.name}: thiếu images/ hoặc labels/")
        return
    images = sorted(p for p in img_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXT)
    stats = Counter()
    for image in images:
        rel = image.relative_to(img_dir)
        label = (lbl_dir / rel).with_suffix(".txt")
        stats["images"] += 1
        if not label.is_file():
            stats["missing_label"] += 1  # không tự coi là ảnh nền
            continue
        kept, seen = [], set()
        for n, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                cls, x, y, w, h = parse_line(line)
            except ValueError as e:
                stats["bad_lines"] += 1
                print(f"  ! {label.name}, dòng {n}: {e} -> bỏ dòng")
                continue
            if cls != args.src_class:
                stats["other_class_dropped"] += 1
                continue
            if w < MIN_SIZE or h < MIN_SIZE:
                stats["tiny_dropped"] += 1
                continue
            out = f"{args.dst_class} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            if out in seen:
                stats["duplicate_dropped"] += 1
                continue
            seen.add(out)
            kept.append(out)
            if line.split().__len__() > 5:
                stats["polygon_to_bbox"] += 1
        if not kept:
            stats["no_smoke_image"] += 1
            if args.drop_empty:
                stats["images_skipped"] += 1
                continue
        stats["images_written"] += 1
        stats["smoke_objects"] += len(kept)
        if args.dry_run:
            continue
        (dst_split / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst_split / "labels" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dst_split / "images" / rel)
        (dst_split / "labels" / rel).with_suffix(".txt").write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"  {src_split.name}: {stats['images']} ảnh -> ghi {stats['images_written']} ảnh, "
          f"{stats['smoke_objects']} box smoke | polygon->bbox: {stats['polygon_to_bbox']} | "
          f"ảnh không có smoke: {stats['no_smoke_image']} | thiếu label: {stats['missing_label']} | "
          f"dòng lỗi: {stats['bad_lines']} | box class khác bị bỏ: {stats['other_class_dropped']} | "
          f"box quá nhỏ bỏ: {stats['tiny_dropped']} | trùng bỏ: {stats['duplicate_dropped']}")
    total.update(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Thư mục gốc dataset nguồn")
    ap.add_argument("--dst", required=True, help="Thư mục đích MỚI (phải chưa có hoặc rỗng)")
    ap.add_argument("--src-class", type=int, default=0, help="Class id của khói trong nguồn")
    ap.add_argument("--dst-class", type=int, default=1, help="Class id của khói trong dataset đích")
    ap.add_argument("--drop-empty", action="store_true",
                    help="Bỏ ảnh không còn box smoke nào (mặc định giữ làm ảnh nền, label rỗng)")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ thống kê, không ghi file")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_dir():
        sys.exit(f"Không có thư mục nguồn: {src}")
    if dst == src or src in dst.parents:
        sys.exit("Thư mục đích không được trùng hoặc nằm trong thư mục nguồn.")
    if not args.dry_run and dst.exists() and any(dst.iterdir()):
        sys.exit(f"Thư mục đích không rỗng: {dst}. Chọn thư mục mới.")

    total = Counter()
    found = False
    for name in SPLIT_NAMES:
        split = src / name
        if split.is_dir():
            found = True
            process_split(split, dst / name, args, total)
    if not found:
        sys.exit(f"Không thấy train/valid/test trong {src}")

    print(f"\nTỔNG: {total['images_written']} ảnh, {total['smoke_objects']} box smoke "
          f"(class {args.src_class} -> {args.dst_class}).")
    if total["other_class_dropped"]:
        print(f"Cảnh báo: {total['other_class_dropped']} box class khác đã bị bỏ. "
              f"Kiểm tra --src-class có đúng không.")
    print("Dry-run: chưa ghi gì." if args.dry_run else f"Đã ghi vào: {dst}")


if __name__ == "__main__":
    main()
