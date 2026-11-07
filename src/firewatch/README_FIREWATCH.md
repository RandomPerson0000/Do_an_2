# FireWatch — module phát hiện lửa và khói

Bộ mã nguồn để ghép vào dự án `Fire_Smoke_develop`, dùng một mô hình detection có hai lớp `fire`, `smoke`. Python 3.10 trở lên; có thể dùng môi trường Python 3.12 hiện tại của bạn. Đây là bản nền để thử nghiệm ứng dụng, chưa được hiệu chỉnh hoặc đo tốc độ với trọng số/camera của bạn.

## 1. Chép vào dự án

Giải nén rồi chép **nội dung bên trong `firewatch_module`** vào thư mục gốc `D:\Do_An\Fire_Smoke_develop`. Thư mục `src` của gói cần được ghép với `src` hiện tại. Xem lại trước khi ghi đè nếu bạn đã có tệp trùng tên. Không cần di chuyển code train hoặc dataset.

| Tệp/thư mục | Công việc |
|---|---|
| `run_firewatch.py` | Điểm chạy từ thư mục gốc, tự thêm `src` vào đường dẫn import |
| `src/firewatch/config.py` | Đọc YAML, kiểm tra giá trị, quy đổi đường dẫn |
| `src/firewatch/capture.py` | Đọc video tuần tự; đọc camera bằng luồng riêng, giữ khung mới nhất |
| `src/firewatch/detector.py` | Nạp YOLO `.pt` hoặc `.onnx`, lọc confidence theo từng lớp |
| `src/firewatch/temporal.py` | Xác nhận và kết thúc cảnh báo theo thời gian, độc lập cho hai lớp |
| `src/firewatch/overlay.py` | Vẽ bbox, nhãn, trạng thái và FPS xử lý |
| `src/firewatch/events.py` | Ghi sự kiện JSONL, lưu ảnh lúc bắt đầu cảnh báo |
| `src/firewatch/app.py` | Điều phối vòng lặp, xử lý lỗi và giải phóng tài nguyên |
| `src/firewatch/cli.py` | Các tùy chọn dòng lệnh |
| `configs/firewatch.yaml` | Các tham số bạn thường cần chỉnh |
| `src/deployment/pi5/firewatch.service.example` | Mẫu dịch vụ tự khởi động trên Pi |
| `tests/test_firewatch.py` | Kiểm tra logic và pipeline bằng dữ liệu giả lập |

## 2. Chạy trên Windows trước

Trong PowerShell của VS Code, tại thư mục gốc dự án và venv đã kích hoạt:

```powershell
python -m pip install -r requirements-firewatch.txt
python run_firewatch.py --check-config
```

Giữ phiên bản Ultralytics đã train/export thành công mô hình của bạn. File requirements không cố định phiên bản vì chưa có `pip freeze` từ môi trường đó; không thêm `--upgrade` tùy tiện. Nếu tạo venv mới, cài phiên bản Ultralytics phù hợp với checkpoint YOLO26 của bạn. Cài CUDA/PyTorch phù hợp riêng nếu muốn chạy GPU; cấu hình mặc định chạy CPU.

Sửa `configs/firewatch.yaml`:

```yaml
model: runs/train/fire_smoke_V2_fine-tune/weights/best.pt
source: 0
device: cpu
class_ids:
  fire: 0
  smoke: 1
```

**Đường dẫn trên chỉ là ví dụ:** thay bằng tệp `.pt` thực tế. Kiểm tra `names` trong `data.yaml` đã dùng để train. Nếu `0` là smoke và `1` là fire thì đảo lại `class_ids`; code dùng ánh xạ này làm chuẩn. Không dùng mô hình một lớp cho cấu hình hai lớp này.

```powershell
# Webcam USB / webcam laptop
python run_firewatch.py

# Video có sẵn; đường dẫn tương đối tính từ project_root
python run_firewatch.py --source "capture_4_test/test.mp4"

# Đường dẫn tuyệt đối có khoảng trắng
python run_firewatch.py --source "D:\Videos\test fire.mp4"

# Thử giới hạn số frame, không mở cửa sổ
python run_firewatch.py --source "capture_4_test/test.mp4" --headless --max-frames 300
```

Nhấn `Q`, `Esc` trong cửa sổ hoặc `Ctrl+C` tại terminal để dừng. `--check-config` chỉ kiểm tra cấu hình, không xác nhận camera hoặc model đã hoạt động.

Với model `.pt` và PyTorch đã nhận GPU NVIDIA, có thể đặt `device: "0"`. Với ONNX trên Pi, đặt `device: cpu`. Không truyền `device: "0"` cho bản ONNX Runtime CPU trong requirements.

## 3. Chạy ONNX và Raspberry Pi 5

Code nạp ONNX **qua Ultralytics**, dùng ONNX Runtime làm backend. Cách này tái sử dụng xử lý đầu vào/đầu ra của Ultralytics và tránh tự suy đoán định dạng tensor YOLO26. Nó vẫn cần cài Ultralytics cùng các dependency của thư viện, chưa phải bản ONNX Runtime độc lập tối giản.

Nếu đã có `best.onnx` xuất từ model hiện tại với `imgsz=640`, dùng ngay:

```powershell
python run_firewatch.py --model "weights/best.onnx" --source 0
```

Nếu cần export, chạy trong môi trường đã nạp được checkpoint:

```python
from ultralytics import YOLO

model = YOLO("runs/train/fire_smoke_V2_fine-tune/weights/best.pt")
model.export(format="onnx", imgsz=640, batch=1, dynamic=False)
```

Export tạo `.onnx` cạnh checkpoint; chép tệp đó sang Pi hoặc sửa `model` cho đúng. `imgsz` trong YAML phải khớp kích thước export khi ONNX dùng shape cố định. Sau export hãy đối chiếu kết quả `.pt` và `.onnx` trên cùng ảnh/video.

**Camera USB trên Pi:** dùng `source: 0`, hoặc thiết bị `/dev/video0` nếu phù hợp.

**Camera CSI như Camera Module 3:** dùng Raspberry Pi OS 64-bit và Picamera2. Với môi trường mới trên Pi:

```bash
sudo apt install python3-picamera2 python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-firewatch.txt
```

Không chạy lệnh tạo venv trên venv đang sử dụng mà chưa kiểm tra. Với Picamera2, dùng Python hệ thống tương thích với libcamera; không ép dùng bản Python khác với gói hệ điều hành. Kiểm tra `python -c "from picamera2 import Picamera2"` trước khi chạy.

Sửa YAML:

```yaml
model: weights/best.onnx
source: picamera2
device: cpu
display: false
```

```bash
python run_firewatch.py --headless
```

Adapter Picamera2 chọn `RGB888`, vốn trả mảng byte BGR theo tài liệu Picamera2 để phù hợp với OpenCV. Kích thước camera 1280×720 được YOLO xử lý về `imgsz`, không phải kéo giãn thủ công sang 640×640. Chưa đo tốc độ trên Pi 5, chưa tích hợp điều khiển HDR/autofocus.

## 4. Cách lọc cảnh báo

Các giá trị là điểm bắt đầu để thử nghiệm, **chưa được tối ưu theo validation của bạn**. `conf=0.335` được đặt làm giá trị ban đầu, không có nghĩa là ngưỡng đã được chứng minh phù hợp cho cả fire và smoke.

| Tham số mỗi lớp | Ý nghĩa |
|---|---|
| `conf` | Chỉ nhận bbox có điểm confidence đạt ngưỡng |
| `window_seconds` | Cửa sổ lịch sử các frame đã xử lý |
| `min_hits` | Số frame dương tính tối thiểu trong cửa sổ |
| `confirm_seconds` | Khoảng cách thời gian tối thiểu giữa frame dương tính đầu và cuối trong cửa sổ |
| `hit_ratio` | Tỉ lệ frame dương tính / tổng số frame đã xử lý trong cửa sổ |
| `clear_seconds` | Thời gian liên tục có frame âm tính để kết thúc cảnh báo |
| `max_gap_seconds` | Nếu hai frame xử lý cách nhau quá lâu, xóa bằng chứng cũ và bộ đếm âm tính |

Bắt đầu cảnh báo chỉ khi frame hiện tại dương tính **và đồng thời** đạt `min_hits`, `confirm_seconds`, `hit_ratio`. Hai bbox fire trong cùng frame chỉ tính là một lần hit. `confirm_seconds` không yêu cầu mọi frame đều dương tính; các lần bỏ sót ngắn được cho phép bởi `hit_ratio`.

Ví dụ mặc định fire cần ít nhất 3 frame dương tính, trải dài ít nhất 1 giây, chiếm ít nhất 60% số frame xử lý trong cửa sổ 2 giây. Điều này tạo độ trễ xác nhận chủ ý; cần đo độ trễ và số cảnh báo sai trên tập video của bạn. Nếu tốc độ inference thấp hơn một frame mỗi 2 giây, `max_gap_seconds=2` sẽ liên tục xóa bằng chứng; phải điều chỉnh theo tốc độ thực tế hoặc tối ưu inference.

`alarm_start` chỉ phát một lần khi chuyển sang cảnh báo. `alarm_end` chỉ phát khi có đủ frame âm tính liên tục. Mất camera hoặc kết thúc chương trình không được ghi như đã hết lửa/khói.

Đây là bộ lọc theo **lớp trên toàn khung hình**, chưa phải tracking cùng một vật thể. Đèn đỏ hoặc nội dung lửa trên TV bị model nhận nhầm liên tục vẫn có thể kích hoạt cảnh báo. Bộ lọc thời gian không thay thế dữ liệu hard negative hoặc việc đánh giá model.

## 5. Luồng camera và thời gian

- **Camera/RTSP/Picamera2:** luồng đọc giữ đúng một frame mới nhất; inference chậm thì bỏ qua frame cũ để giảm tích lũy trễ. Không đảm bảo tuyệt đối không trễ vì driver/RTSP vẫn có thể buffer. Không xử lý lại một frame hai lần.
- **Video file:** đọc đủ frame tuần tự, xử lý nhanh/chậm tùy máy; không cố phát đúng tốc độ thời gian thực. Bộ lọc dùng timestamp video, dự phòng bằng FPS khi backend không trả timestamp tăng dần.
- **FPS trên overlay:** tốc độ vòng lặp đọc + inference, không phải FPS cảm biến hoặc phép đo độ trễ đầu-cuối.
- **Mất tín hiệu:** đợi tối đa `frame_timeout_seconds` trong mỗi lần đọc rồi ghi lỗi và kết thúc với exit code 1. Không tự nối lại trong cùng phiên; dịch vụ systemd có thể khởi động phiên mới. Trạng thái cảnh báo không được khôi phục qua lần khởi động lại; phiên mới xác nhận lại từ đầu.
- Video hỏng có thể được OpenCV báo như hết video; hãy dùng nguồn video hợp lệ khi đánh giá.

## 6. Kết quả đầu ra

Mỗi phiên tạo thư mục riêng trong `runs/firewatch/`, gồm:

- `events.jsonl`: mỗi dòng một JSON, có timestamp UTC, loại sự kiện và dữ liệu liên quan.
- `000001_fire.jpg` hoặc tương tự: ảnh có bbox ở thời điểm bắt đầu cảnh báo, nếu bật `save_snapshots`.

Các sự kiện gồm `session_start`, `alarm_start`, `alarm_end`, `system_error`, `snapshot_error`, `session_end`. Trường `source_seconds` là thời gian video hoặc thời gian từ frame đầu tiên của phiên camera, không phải Unix timestamp. `session_end.active_labels` ghi lại lớp còn đang cảnh báo khi dừng.

Chương trình chưa ghi toàn bộ video, chưa phát còi GPIO, chưa gửi Telegram/MQTT/email và chưa mở luồng web. Có thể bổ sung các đầu ra đó tại tầng sự kiện sau khi xác định phần cứng và giao thức. Nếu bổ sung gửi mạng, dùng queue/worker riêng để tránh chặn inference. Thư mục sự kiện chưa tự dọn; cần theo dõi dung lượng khi chạy dài ngày.

## 7. Tự khởi động trên Pi (tùy chọn)

Chạy thủ công thành công trước. Trong `src/deployment/pi5/firewatch.service.example`, thay `USERNAME` và tất cả đường dẫn bằng tài khoản/thư mục thực tế; đảm bảo tài khoản có quyền dùng camera và ghi `runs`.

```bash
sudo cp src/deployment/pi5/firewatch.service.example /etc/systemd/system/firewatch.service
sudo systemctl daemon-reload
sudo systemctl enable --now firewatch.service
systemctl status firewatch.service
journalctl -u firewatch.service -f
```

Dừng bằng `sudo systemctl stop firewatch.service`. Mẫu dùng `Restart=on-failure` và giới hạn số lần thử; sai cấu hình kéo dài có thể khiến dịch vụ ngừng thử lại. Sau khi sửa, dùng `sudo systemctl reset-failed firewatch.service` rồi `sudo systemctl start firewatch.service`. Mẫu này chưa được cài đặt hoặc chạy trên thiết bị của bạn.

## 8. Kiểm thử và giới hạn xác minh

```bash
python -m unittest discover -s tests -v
```

Kiểm thử logic không cần model. Kiểm thử pipeline cần NumPy/OpenCV và dùng video AVI tổng hợp + detector giả lập để kiểm tra đọc video, vẽ khung, lưu JPEG và các chuyển trạng thái. Đây **không phải** phép đo độ chính xác YOLO.

Chưa có trọng số của bạn, webcam thật, Pi 5 hoặc Camera Module 3 trong môi trường soạn code. Vì vậy việc nạp checkpoint thực tế, inference `.pt`/`.onnx`, CUDA, RTSP và Picamera2 cần kiểm tra trên máy của bạn. Chưa tuyên bố FPS hay độ chính xác thực tế. Xem `TEST_RESULTS.txt` để biết kết quả kiểm thử kèm gói.

## 9. Nguồn API tham khảo

- Ultralytics Predict: https://docs.ultralytics.com/modes/predict/
- Ultralytics ONNX: https://docs.ultralytics.com/integrations/onnx/
- Picamera2 Manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

Đối chiếu tài liệu ngày 07/11/2026.
