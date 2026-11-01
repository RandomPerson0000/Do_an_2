"""Menu chạy train_both.py cho fire + smoke.

Đặt run_both.py và train_both.py cùng thư mục (ví dụ <dự án>/script/).
Có thể dùng nút Run Python File trong editor; dùng Python/venv đang chạy menu.
Đường dẫn tương đối tính từ thư mục cha của thư mục chứa hai script.
"""
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().with_name("train_both.py")


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


def build_args(mode):
    if mode == "7":
        print("Fine-tune = nạp trọng số cũ, học dataset mới từ epoch 1; resume=False.")
        print("YAML dataset phải chứa train/val/names, KHÔNG phải args.yaml/YAML train.")
        print("Nhập thư mục kết quả ĐẦY ĐỦ, bao gồm tên run cuối cùng.")
        args = [
            "finetune",
            "--weights", ask("Checkpoint nguồn best.pt"),
            "--data", ask("YAML DATASET MỚI (ví dụ data.yaml)"),
        ]
        print("Nhập đường dẫn đầy đủ tới thư mục images để thay đường dẫn trong YAML.")
        print("Enter = giữ YAML. Đường dẫn tương đối nhập ở đây tính từ thư mục chứa YAML.")
        for flag, label in (("--train-images", "TRAIN"), ("--val-images", "VALID")):
            value = ask(f"Thư mục ảnh {label} (Enter = dùng YAML)", optional=True)
            if value:
                args += [flag, value]
        test = ask("Thư mục ảnh TEST (Enter = dùng YAML; nhập - = bỏ test)", optional=True)
        if test == "-":
            args.append("--no-test")
        elif test:
            args += ["--test-images", test]
        args += [
            "--output", ask("Thư mục kết quả đầy đủ", str(PROJECT_ROOT / "runs/train/fire_smoke_V2_fine-tune")),
            "--epochs", ask("Số epoch tối đa cho lượt fine-tune mới", "40"),
            "--patience", ask("Early stopping patience (0 = tắt)", "10"),
            "--lr0", ask("Learning rate ban đầu (AdamW)", "0.0001"),
            "--imgsz", ask("Kích thước ảnh", "640"),
            "--batch", ask("Batch (-1 = tự chọn; hoặc số nguyên dương)", "-1"),
            "--device", ask("Thiết bị (0 = GPU đầu tiên, hoặc cpu)", "0"),
            "--workers", ask("Số workers", "0"),
        ]
        if choose("Chỉ kiểm tra dữ liệu/cấu hình, chưa train? (y/n)", ("y", "n"), "n") == "y":
            args.append("--dry-run")
        if choose("Bật log TensorBoard? (y/n)", ("y", "n"), "n") == "y":
            args.append("--tensorboard")
        return args
    if mode == "1":
        args = [
            "init",
            "--root", ask("Thư mục gốc dataset"),
            "--out", ask("File YAML train mới", "configs/fire_train.yaml"),
            "--train-images", ask("Thư mục ảnh train tương đối với dataset", "train/images"),
            "--val-images", ask("Thư mục ảnh validation tương đối với dataset", "valid/images"),
            "--project", ask("Thư mục lưu kết quả", "runs/train/fire"),
            "--model", ask("Model pretrained", (PROJECT_ROOT / "models/pretrained/yolo26n.pt").as_posix()),
        ]
        test = ask("Thư mục ảnh test tương đối với dataset (Enter để bỏ qua)", optional=True)
        if test:
            args += ["--test-images", test]
        return args
    if mode == "2":
        args = [
            "check", "--data", ask("YAML dataset", "configs/fire_train_data.yaml"),
            "--split", choose("Tập cần kiểm tra (all/train/valid/test; train kiểm tra train + valid)",
                              ("all", "train", "valid", "test"), "all"),
        ]
        print("Mặc định thống kê riêng ảnh thiếu label; non-label được tính theo annotation.")
        if choose("Báo lỗi nếu thiếu file label .txt? (y/n)", ("y", "n"), "n") == "y":
            args.append("--strict-labels")
        return args
    if mode in {"3", "4"}:
        if mode == "3":
            args = ["train", "--config", ask("YAML train", "configs/fire_train.yaml")]
        else:
            args = ["train", "--resume", ask("Checkpoint last.pt của lượt train bị ngắt (KHÔNG nhập YAML)")]
        validation = ask("YAML đánh giá sau train (Enter để bỏ qua)", optional=True)
        if validation:
            args += ["--val-config", validation]
        if choose("Bật log TensorBoard? (y/n)", ("y", "n"), "n") == "y":
            args.append("--tensorboard")
        return args
    if mode == "5":
        return [
            "val", "--weights", ask("Checkpoint cần đánh giá, ví dụ best.pt"),
            "--config", ask("YAML đánh giá", "configs/fire_train_val.yaml"),
            "--split", choose("Tập đánh giá (val/test)", ("val", "test"), "val"),
        ]
    args = [
        "solution", "--weights", ask("Checkpoint để chạy heatmap, ví dụ best.pt"),
        "--config", ask("YAML heatmap", "configs/fire_train_solution.yaml"),
    ]
    source = ask("Video hoặc số camera (Enter để dùng nguồn trong YAML)", optional=True)
    if source:
        args += ["--source", source]
    return args


def main():
    print("YOLO FIRE + SMOKE — train_both.py")
    print(f"Thư mục dự án: {PROJECT_ROOT}")
    print("Đường dẫn tương đối được tính từ thư mục dự án.")
    print("Riêng đường dẫn ảnh ở bước init được tính từ gốc dataset.")
    print("\n1. Tạo cấu hình (init)")
    print("2. Kiểm tra dataset: ảnh có/không label fire, smoke và số objects (train/valid/test)")
    print("3. Bắt đầu train (train)")
    print("4. Tiếp tục train bị gián đoạn (resume)")
    print("5. Đánh giá model (val/test)")
    print("6. Chạy heatmap (solution)")
    print("7. Fine-tune best.pt với DATASET MỚI và thư mục kết quả tự nhập")
    print("0. Thoát")
    mode = choose("Chọn chế độ", tuple("01234567"))
    if mode == "0":
        return 0
    if not SCRIPT.is_file():
        print(f"Không tìm thấy: {SCRIPT}")
        print("Hãy đặt train_both.py bản cập nhật cùng thư mục với run_both.py.")
        return 1
    command = [sys.executable, str(SCRIPT), *build_args(mode)]
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
