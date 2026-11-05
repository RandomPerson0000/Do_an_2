"""Menu chạy find_bad_labels.py (tìm ảnh + label lỗi, gom sang thư mục riêng)
và filter_smoke_labels.py (sửa class id, chuyển polygon thành bbox).

Đặt run_find_bad_labels.py, find_bad_labels.py và filter_smoke_labels.py cùng thư mục.
Có thể dùng nút Run Python File trong editor; dùng Python/venv đang chạy menu.
Đường dẫn tương đối được tính từ thư mục cha của thư mục chứa hai script
(giống run_both.py). Nên nhập đường dẫn đầy đủ để tránh nhầm.
"""
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().with_name("find_bad_labels.py")
FILTER_SCRIPT = Path(__file__).resolve().with_name("filter_smoke_labels.py")
ERROR_TYPES = ("polygon", "bad_field_count", "not_number", "not_finite", "bad_class",
               "class_not_allowed", "out_of_range", "unreadable", "missing_label")


def ask(message, default=None, optional=False):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{message}{suffix}: ").strip().strip('"').strip("'")
        if value:
            return value
        if default is not None:
            return default
        if optional:
            return ""
        print("Vui lòng nhập giá trị.")


def choose(message, options, default=None):
    while True:
        value = ask(message, default)
        if value in options:
            return value
        print("Chọn một trong các giá trị: " + ", ".join(options))


def ask_classes():
    while True:
        raw = ask("Class id hợp lệ, cách nhau bằng dấu cách hoặc phẩy, ví dụ 0 1 "
                  "(Enter = không kiểm tra class)", optional=True)
        if not raw:
            return []
        try:
            classes = [int(v) for v in raw.replace(",", " ").split()]
        except ValueError:
            print("Chỉ nhập số nguyên, ví dụ: 0 1")
            continue
        if any(c < 0 for c in classes):
            print("Class id không được âm.")
            continue
        return [str(c) for c in classes]


def ask_only():
    print("Loại lỗi: " + ", ".join(ERROR_TYPES))
    while True:
        raw = ask("Chỉ lấy các loại lỗi này, cách nhau bằng dấu phẩy (Enter = tất cả)", optional=True)
        if not raw:
            return ""
        chosen = [v.strip() for v in raw.split(",") if v.strip()]
        unknown = [v for v in chosen if v not in ERROR_TYPES]
        if unknown:
            print("Loại lỗi không hợp lệ: " + ", ".join(unknown))
            continue
        return ",".join(chosen)


def ask_int(message, default):
    while True:
        value = ask(message, default)
        try:
            if int(value) >= 0:
                return str(int(value))
        except ValueError:
            pass
        print("Nhập số nguyên không âm, ví dụ 0 hoặc 1.")


def build_filter_args():
    print("Sửa class id + chuyển polygon thành bbox, ghi sang thư mục MỚI (không sửa dataset gốc).")
    print("Chỉ giữ box thuộc class hiện tại, đổi sang class mới; box class khác bị bỏ.")
    print("Ví dụ Negative_Samples (khói đang là class 0): class hiện tại 0, class mới 1.")
    src = ask("Thư mục gốc dataset (chứa train/valid/test)")
    path = Path(src)
    dst = ask("Thư mục đích (phải mới hoặc rỗng)", str(path.with_name(path.name + "_fixed")))
    args = [
        "--src", src, "--dst", dst,
        "--src-class", ask_int("Class id hiện tại trong label", "0"),
        "--dst-class", ask_int("Class id mới (0 = fire, 1 = smoke)", "1"),
    ]
    if choose("Bỏ ảnh không còn box nào sau khi lọc? (y/n; n = giữ làm ảnh nền)",
              ("y", "n"), "n") == "y":
        args.append("--drop-empty")
    if choose("Chạy thử, chỉ thống kê, chưa ghi file? (y/n)", ("y", "n"), "y") == "y":
        args.append("--dry-run")
    return args


def default_dst(src):
    path = Path(src)
    return str(path.with_name(path.name + "_bad"))


def build_args(mode):
    src = ask("Thư mục gốc dataset (chứa train/valid/test)")
    if mode == "1":
        print("Chạy thử: chỉ thống kê, không copy hay ghi file nào.")
        dst_message = "Thư mục đích (script vẫn kiểm tra, nhưng dry-run không tạo thư mục này)"
    else:
        dst_message = "Thư mục đích để gom ảnh/label lỗi (phải mới hoặc rỗng)"
    dst = ask(dst_message, default_dst(src))
    args = ["--src", src, "--dst", dst]

    if mode == "1":
        args.append("--dry-run")
    elif mode == "4":
        print("CẢNH BÁO: chế độ này CHUYỂN ảnh + label lỗi đi, xóa chúng khỏi dataset nguồn.")
        print("Nên chạy chế độ 1 (chạy thử) trước và/hoặc sao lưu dataset.")
        if ask("Nhập yes để xác nhận chuyển file (Enter = hủy)", optional=True).lower() != "yes":
            print("Đã hủy.")
            return None
        args.append("--move")

    if mode == "3":
        args += ["--only", "polygon"]
    else:
        only = ask_only()
        if only:
            args += ["--only", only]

    classes = ask_classes()
    if classes:
        args += ["--classes", *classes]
    if choose("Coi ảnh thiếu file label .txt là lỗi? (y/n)", ("y", "n"), "n") == "y":
        args.append("--include-missing")
    return args


def main():
    print("TÌM ẢNH + LABEL LỖI — find_bad_labels.py")
    print(f"Đường dẫn tương đối được tính từ: {PROJECT_ROOT}")
    print("\n1. Chạy thử (dry-run): chỉ thống kê, không copy/ghi file")
    print("2. Gom ảnh + label lỗi sang thư mục riêng (COPY, giữ nguyên dataset gốc)")
    print("3. Chỉ gom label dạng polygon (COPY)")
    print("4. Chuyển ảnh + label lỗi sang thư mục riêng (MOVE, xóa khỏi dataset gốc)")
    print("5. SỬA CLASS ID + chuyển polygon thành bbox, ghi sang thư mục mới (filter_smoke_labels.py)")
    print("0. Thoát")
    mode = choose("Chọn chế độ", tuple("012345"))
    if mode == "0":
        return 0
    script = FILTER_SCRIPT if mode == "5" else SCRIPT
    if not script.is_file():
        print(f"Không tìm thấy: {script}")
        print(f"Hãy đặt {script.name} cùng thư mục với run_find_bad_labels.py.")
        return 1
    args = build_filter_args() if mode == "5" else build_args(mode)
    if args is None:
        return 0
    command = [sys.executable, str(script), *args]
    print("\nChạy: " + subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode:
        print(f"Chương trình kết thúc với mã lỗi {result.returncode}. Xem thông báo phía trên.")
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nĐã dừng thao tác.")
        sys.exit(130)
