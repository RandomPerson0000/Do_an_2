"""Menu chạy train_fire2.py. Có thể dùng nút Run Python File trong editor."""
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().with_name("train_fire2.py")


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
        return ["check", "--data", ask("YAML dataset", "configs/fire_train_data.yaml")]
    if mode in {"3", "4"}:
        if mode == "3":
            args = ["train", "--config", ask("YAML train", "configs/fire_train.yaml")]
        else:
            args = ["train", "--resume", ask("Checkpoint last.pt để tiếp tục train")]
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
    print(f"Thư mục dự án: {PROJECT_ROOT}")
    print("Đường dẫn tương đối được tính từ thư mục dự án.")
    print("Riêng đường dẫn ảnh ở bước init được tính từ gốc dataset.")
    print("\n1. Tạo cấu hình (init)")
    print("2. Kiểm tra dataset (check)")
    print("3. Bắt đầu train (train)")
    print("4. Tiếp tục train bị gián đoạn (resume)")
    print("5. Đánh giá model (val/test)")
    print("6. Chạy heatmap (solution)")
    print("0. Thoát")
    mode = choose("Chọn chế độ", tuple("0123456"))
    if mode == "0":
        return 0
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
