# Triển khai phát hiện lửa và khói

Để chạy trên Raspberry Pi 5 và xem cảnh báo từ laptop/PC, bắt đầu ở [hướng dẫn FS_nano_V2](FS_nano_V2/README.md). Trên Pi, mở `FS_nano_V2/dashboard.py` bằng Python, rồi chạy một trong hai script detect ở cùng thư mục.

| Thư mục | Vai trò |
|---|---|
| [FS_nano_V2/](FS_nano_V2/) | Ứng dụng V2 cho Pi. Có hai điểm chạy PT và ONNX, cấu hình camera, MQTT, dashboard web và Telegram. |
| [firewatch/](firewatch/README_FIREWATCH.md) | Thư viện dùng chung để đọc camera, suy luận, xác nhận cảnh báo và ghi nhật ký. Có CLI và bộ kiểm thử riêng. |
| [deployment/pi5/](deployment/pi5/) | Mẫu dịch vụ systemd cho CLI FireWatch độc lập. |

### Trọng số và dữ liệu khi chạy

Các đường dẫn dưới đây tính từ thư mục gốc `Fire_Smoke_develop/`.

| Đường dẫn | Nội dung |
|---|---|
| `runs/train/fire_smoke_V2/fire_yolo26n/weights/` | `best.pt` và `best_nano.onnx` dùng cho V2. |
| `runs/FS_nano_V2/<device>/<backend>/` | Nhật ký phiên, ảnh lúc bắt đầu cảnh báo và hàng đợi MQTT của từng bộ detect. |
| `runs/FS_nano_V2/dashboard.sqlite3` | Lịch sử cảnh báo và hàng đợi Telegram của dashboard. |
