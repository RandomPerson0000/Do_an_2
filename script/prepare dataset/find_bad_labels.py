"""Tìm ảnh + label bị lỗi định dạng (ví dụ label polygon thay vì bbox) và gom sang thư mục riêng.

- Mặc định COPY (không đụng dataset gốc). Thêm --move nếu muốn chuyển hẳn đi.
- Giữ nguyên cấu trúc <split>/images và <split>/labels trong thư mục đích.
- Ghi báo cáo bad_labels_report.csv (mở bằng Excel): mỗi dòng lỗi một hàng.

Các loại lỗi (error_type):
  polygon          : dòng có class + >=3 cặp x y (không phải bbox 5 giá trị)
  bad_field_count  : số giá trị không phải 5 và cũng không phải polygon hợp lệ
  not_number       : có giá trị không phải số
  not_finite       : có nan/inf
  bad_class        : class id không phải số nguyên
  class_not_allowed: class id ngoài danh sách --classes
  out_of_range     : bbox có tọa độ ngoài [0,1] hoặc w/h <= 0
  unreadable       : không đọc được file label
  missing_label    : ảnh không có file .txt (chỉ khi dùng --include-missing)

Ví dụ:
  python find_bad_labels.py --src D:\\data\\Negative_Samples --dst D:\\data\\Negative_Samples_bad --dry-run
  python find_bad_labels.py --src ... --dst ... --only polygon
  python find_bad_labels.py --src ... --dst ... --classes 0 1 --include-missing

Chế độ SỬA (--fix): ghi bản dataset ĐÃ SỬA vào --dst, dataset gốc không đổi.
  --remap CŨ:MỚI ...  đổi class id; áp dụng đồng thời nên đổi chỗ 0:1 1:0 vẫn đúng,
                      class không nêu trong quy tắc được giữ nguyên.
  --to-bbox           chuyển polygon thành bbox (đồng thời cắt box vượt biên ảnh).
  Ảnh có dòng không sửa được hoặc thiếu label -> <dst>/_unfixable/ và fix_report.csv
  để xử lý thủ công; nếu dùng --classes thì class được kiểm tra SAU khi đổi.
  python find_bad_labels.py --fix --src ... --dst ... --remap 0:1 --to-bbox --dry-run

Cấu trúc nguồn: <src>/{train,valid,val,test}/{images,labels}
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "valid", "val", "test")


def check_line(line, allowed):
    """Trả về (error_type, detail) hoặc None nếu dòng hợp lệ (bbox chuẩn)."""
    try:
        vals = [float(v) for v in line.split()]
    except ValueError:
        return "not_number", "có giá trị không phải số"
    if not all(math.isfinite(v) for v in vals):
        return "not_finite", "có giá trị nan/inf"
    n = len(vals)
    is_bbox = n == 5
    is_polygon = n >= 7 and n % 2 == 1
    if not (is_bbox or is_polygon):
        return "bad_field_count", f"{n} giá trị (cần 5, hoặc polygon: 1 + 2k, k>=3)"
    cls = vals[0]
    if cls != int(cls) or cls < 0:
        return "bad_class", f"class id = {vals[0]}"
    if allowed and int(cls) not in allowed:
        return "class_not_allowed", f"class id = {int(cls)}"
    if is_polygon:
        return "polygon", f"polygon {(n - 1) // 2} điểm"
    _, x, y, w, h = vals
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return "out_of_range", f"x={x:.4f} y={y:.4f} w={w:.4f} h={h:.4f}"
    return None


FIX_MIN_SIZE = 1e-4  # box nhỏ hơn mức này (tỉ lệ so với ảnh) coi là rỗng
FIX_EPS = 1e-6       # sai số làm tròn cho phép khi kiểm tra biên [0, 1]


def parse_remap(items):
    """['0:1', '1:0'] -> {0: 1, 1: 0}."""
    mapping = {}
    for item in items or []:
        try:
            old, new = (int(v) for v in item.split(":"))
        except ValueError:
            sys.exit(f"--remap sai định dạng: '{item}'. Dùng CŨ:MỚI, ví dụ 0:1")
        if old < 0 or new < 0:
            sys.exit(f"--remap: class id không được âm: '{item}'")
        if old in mapping:
            sys.exit(f"--remap: class {old} xuất hiện hai lần.")
        mapping[old] = new
    return mapping


def fix_line(line, remap, to_bbox, allowed):
    """Sửa một dòng label.

    Trả về (dòng_mới, tập_việc_đã_làm) nếu sửa được, hoặc
    (None, (loại_lỗi, chi_tiết)) nếu không sửa được.
    Việc đã làm: remapped, polygon_to_bbox, clamped.
    """
    try:
        vals = [float(v) for v in line.split()]
    except ValueError:
        return None, ("not_number", "có giá trị không phải số")
    if not all(math.isfinite(v) for v in vals):
        return None, ("not_finite", "có giá trị nan/inf")
    n = len(vals)
    is_bbox = n == 5
    is_polygon = n >= 7 and n % 2 == 1
    if not (is_bbox or is_polygon):
        return None, ("bad_field_count", f"{n} giá trị (cần 5, hoặc polygon: 1 + 2k, k>=3)")
    if vals[0] != int(vals[0]) or vals[0] < 0:
        return None, ("bad_class", f"class id = {vals[0]}")
    old_cls = int(vals[0])
    cls = remap.get(old_cls, old_cls)
    if allowed and cls not in allowed:
        return None, ("class_not_allowed", f"class id = {cls} (sau khi sửa)")
    notes = {"remapped"} if cls != old_cls else set()
    parts = line.split()
    parts[0] = str(cls)
    if is_polygon and not to_bbox:
        return " ".join(parts), notes  # giữ polygon, chỉ đổi class
    if is_bbox:
        _, x, y, w, h = vals
        x1, x2, y1, y2 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        if (w > 0 and h > 0 and x1 >= -FIX_EPS and x2 <= 1 + FIX_EPS
                and y1 >= -FIX_EPS and y2 <= 1 + FIX_EPS):
            return " ".join(parts), notes  # bbox hợp lệ: chỉ đổi class
    else:
        xs, ys = vals[1::2], vals[2::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        notes.add("polygon_to_bbox")
    cx1, cx2, cy1, cy2 = max(0.0, x1), min(1.0, x2), max(0.0, y1), min(1.0, y2)
    if (cx1, cx2, cy1, cy2) != (x1, x2, y1, y2):
        notes.add("clamped")
    w, h = cx2 - cx1, cy2 - cy1
    if w < FIX_MIN_SIZE or h < FIX_MIN_SIZE:
        return None, ("out_of_range", f"box rỗng/quá nhỏ sau khi cắt biên (w={w:.6f}, h={h:.6f})")
    return f"{cls} {(cx1 + cx2) / 2:.6f} {(cy1 + cy2) / 2:.6f} {w:.6f} {h:.6f}", notes


def run_fix(src, dst, args, allowed, remap):
    """Ghi bản dataset đã sửa vào dst; ảnh không sửa được -> dst/_unfixable."""
    if remap:
        print("Đổi class: " + ", ".join(f"{o}->{n}" for o, n in sorted(remap.items())))
    print("Chuyển polygon thành bbox: " + ("có" if args.to_bbox else "không"))
    total = Counter()
    problem_rows = []  # (split, image, label, line, error_type, detail)
    found = False
    for name in SPLIT_NAMES:
        split_dir = src / name
        if not split_dir.is_dir():
            continue
        found = True
        img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            print(f"Bỏ qua {name}: thiếu images/ hoặc labels/")
            continue
        images = sorted(p for p in img_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMG_EXT)
        stats = Counter()
        for image in images:
            rel = image.relative_to(img_dir)
            label = (lbl_dir / rel).with_suffix(".txt")
            stats["images"] += 1
            problems, original, fixed_lines, counts = [], [], [], Counter()
            if not label.is_file():
                problems.append((0, "missing_label", "không có file .txt"))
            else:
                try:
                    lines = label.read_text(encoding="utf-8-sig").splitlines()
                except (OSError, UnicodeDecodeError) as e:
                    problems.append((0, "unreadable", str(e)))
                    lines = []
                for n, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    original.append(line.strip())
                    fixed, info = fix_line(line.strip(), remap, args.to_bbox, allowed)
                    if fixed is None:
                        problems.append((n, *info))
                    else:
                        fixed_lines.append(fixed)
                        counts.update(info)
            if problems:
                stats["unfixable"] += 1
                for n, kind, detail in problems:
                    problem_rows.append((name, image.name, label.name, n or "", kind, detail))
                if not args.dry_run:
                    bad_img = dst / "_unfixable" / name / "images" / rel
                    bad_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image, bad_img)
                    if label.is_file():
                        bad_lbl = dst / "_unfixable" / name / "labels" / rel.with_suffix(".txt")
                        bad_lbl.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(label, bad_lbl)
                continue
            stats["written"] += 1
            stats["objects"] += len(fixed_lines)
            stats["changed_files"] += fixed_lines != original
            stats.update(counts)
            if not args.dry_run:
                out_img = dst / name / "images" / rel
                out_lbl = (dst / name / "labels" / rel).with_suffix(".txt")
                out_img.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, out_img)
                out_lbl.write_text("\n".join(fixed_lines) + ("\n" if fixed_lines else ""),
                                   encoding="utf-8")
        print(f"{name}: {stats['images']} ảnh -> ghi {stats['written']} ảnh "
              f"({stats['changed_files']} label có thay đổi), {stats['objects']} box | "
              f"đổi class: {stats['remapped']} dòng | polygon->bbox: {stats['polygon_to_bbox']} | "
              f"cắt biên: {stats['clamped']} | không sửa được: {stats['unfixable']} ảnh")
        total.update(stats)
    if not found:
        sys.exit(f"Không thấy train/valid/val/test trong {src}")
    print(f"\nTỔNG: {total['images']} ảnh đã quét, {total['written']} ảnh đã sửa, "
          f"{total['objects']} box, {total['unfixable']} ảnh không sửa được.")
    for split, image_name, label_name, n, kind, detail in problem_rows[:10]:
        print(f"  ! [{split}] {label_name} dòng {n or '-'}: {kind} - {detail}")
    if args.dry_run:
        print("\nDry-run: chưa copy/ghi gì.")
        return
    if problem_rows:
        dst.mkdir(parents=True, exist_ok=True)
        report = dst / "fix_report.csv"
        with report.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image", "label", "line", "error_type", "detail"])
            writer.writerows(problem_rows)
        print(f"\nẢnh không sửa được nằm trong: {dst / '_unfixable'}")
        print(f"Báo cáo chi tiết: {report}")
    print(f"Đã ghi dataset đã sửa vào: {dst}")


def scan_split(split_dir, args, allowed, only):
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"Bỏ qua {split_dir.name}: thiếu images/ hoặc labels/")
        return 0, []
    images = sorted(p for p in img_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXT)
    bad = []  # (image, label, [(line_no, type, detail), ...])
    for image in images:
        rel = image.relative_to(img_dir)
        label = (lbl_dir / rel).with_suffix(".txt")
        problems = []
        if not label.is_file():
            if args.include_missing:
                problems.append((0, "missing_label", "không có file .txt"))
        else:
            try:
                lines = label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeDecodeError) as e:
                problems.append((0, "unreadable", str(e)))
                lines = []
            for n, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                result = check_line(line, allowed)
                if result:
                    problems.append((n, *result))
        if only:
            problems = [p for p in problems if p[1] in only]
        if problems:
            bad.append((image, label, problems))
    return len(images), bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Thư mục gốc dataset")
    ap.add_argument("--dst", required=True, help="Thư mục đích để gom ảnh/label lỗi (phải mới hoặc rỗng)")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="Danh sách class id hợp lệ; class khác bị báo lỗi (mặc định: không kiểm tra)")
    ap.add_argument("--only", default=None,
                    help="Chỉ lấy các loại lỗi này, cách nhau bởi dấu phẩy. Ví dụ: polygon")
    ap.add_argument("--include-missing", action="store_true",
                    help="Coi ảnh không có file label là lỗi")
    ap.add_argument("--move", action="store_true", help="Chuyển file đi thay vì copy (xóa khỏi nguồn)")
    ap.add_argument("--dry-run", action="store_true", help="Chỉ thống kê, không copy/ghi file")
    ap.add_argument("--fix", action="store_true",
                    help="Chế độ sửa: ghi dataset đã sửa vào --dst (cần --remap và/hoặc --to-bbox)")
    ap.add_argument("--remap", nargs="+", metavar="CŨ:MỚI",
                    help="Quy tắc đổi class id (chỉ dùng với --fix), ví dụ: 0:1 hoặc 0:1 1:0")
    ap.add_argument("--to-bbox", action="store_true",
                    help="Chuyển polygon thành bbox và cắt box vượt biên (chỉ dùng với --fix)")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_dir():
        sys.exit(f"Không có thư mục nguồn: {src}")
    if dst == src or src in dst.parents or dst in src.parents:
        sys.exit("Thư mục đích không được trùng hoặc lồng với thư mục nguồn.")
    if not args.dry_run and dst.exists() and any(dst.iterdir()):
        sys.exit(f"Thư mục đích không rỗng: {dst}. Chọn thư mục mới.")
    allowed = set(args.classes) if args.classes else None
    only = {t.strip() for t in args.only.split(",")} if args.only else None

    if args.fix:
        if args.move or only:
            sys.exit("--move và --only chỉ dùng ở chế độ tìm lỗi, không dùng cùng --fix.")
        remap = parse_remap(args.remap)
        if not remap and not args.to_bbox:
            sys.exit("Chế độ --fix cần --remap CŨ:MỚI và/hoặc --to-bbox.")
        run_fix(src, dst, args, allowed, remap)
        return
    if args.remap or args.to_bbox:
        sys.exit("--remap và --to-bbox chỉ dùng cùng --fix.")

    total_images = 0
    all_bad = []  # (split, image, label, problems)
    found = False
    for name in SPLIT_NAMES:
        split_dir = src / name
        if not split_dir.is_dir():
            continue
        found = True
        count, bad = scan_split(split_dir, args, allowed, only)
        total_images += count
        all_bad += [(name, *b) for b in bad]
        print(f"{name}: {count} ảnh, {len(bad)} ảnh/label lỗi")
    if not found:
        sys.exit(f"Không thấy train/valid/val/test trong {src}")

    per_type = defaultdict(set)  # loại lỗi -> tập file bị ảnh hưởng
    line_count = Counter()
    for split, image, label, problems in all_bad:
        for _, kind, _ in problems:
            per_type[kind].add(image)
            line_count[kind] += 1
    print(f"\nTỔNG: {total_images} ảnh đã quét, {len(all_bad)} ảnh có label lỗi.")
    for kind in sorted(per_type):
        print(f"  {kind}: {len(per_type[kind])} ảnh ({line_count[kind]} dòng)")
    print("\nVí dụ:")
    for split, image, label, problems in all_bad[:10]:
        n, kind, detail = problems[0]
        print(f"  [{split}] {label.name} dòng {n}: {kind} - {detail}")

    if not all_bad:
        print("Không có lỗi nào.")
        return
    if args.dry_run:
        print("\nDry-run: chưa copy/ghi gì.")
        return

    dst.mkdir(parents=True, exist_ok=True)
    report = dst / "bad_labels_report.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "line", "error_type", "detail"])
        for split, image, label, problems in all_bad:
            for n, kind, detail in problems:
                writer.writerow([split, image.name, label.name, n or "", kind, detail])
    op = shutil.move if args.move else shutil.copy2
    for split, image, label, problems in all_bad:
        img_root, lbl_root = src / split / "images", src / split / "labels"
        out_img = dst / split / "images" / image.relative_to(img_root)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        op(str(image), str(out_img))
        if label.is_file():
            out_lbl = dst / split / "labels" / label.relative_to(lbl_root)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            op(str(label), str(out_lbl))
    print(f"\nĐã {'chuyển' if args.move else 'copy'} {len(all_bad)} ảnh + label lỗi vào: {dst}")
    print(f"Báo cáo chi tiết: {report}")


if __name__ == "__main__":
    main()
