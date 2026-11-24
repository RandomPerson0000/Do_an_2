# FS_nano_V2 trên Raspberry Pi 5

**Pi 5 chạy toàn bộ hệ thống.** Camera → bộ detect PT hoặc ONNX → Mosquitto trên Pi → `dashboard.py`. Bộ nhận Python lưu sự kiện vào SQLite, cung cấp dashboard web qua mạng LAN và gửi Telegram khi đã cấu hình bot. Chỉ cần **một** bộ nhận cho cả hai bản.

| Thiết bị | Cần làm gì |
|---|---|
| Raspberry Pi 5 | Gắn camera, chạy broker, dashboard và một bộ detect. Pi cần Internet khi gửi Telegram. |
| Laptop/PC | Dùng trình duyệt trong cùng mạng LAN/Wi-Fi với Pi. Có thể tắt máy tính hoặc đóng tab trong khi Pi tiếp tục xử lý. |
| Điện thoại (tùy chọn) | Nhận tin qua ứng dụng Telegram trên Wi-Fi hoặc 4G/5G. Khi ở cùng LAN, điện thoại cũng mở được web của Pi. |

Chọn một bản cho mỗi camera. Hai bản dùng chung cách lọc cảnh báo và định dạng MQTT.

| Lệnh chạy từ thư mục gốc repo | Trọng số trong `runs/train/fire_smoke_V2/fire_yolo26n/weights/` |
|---|---|
| `python src/FS_nano_V2/FS_nano_V2_pt.py` | `best.pt` |
| `python src/FS_nano_V2/FS_nano_V2_onnx.py` | `best_nano.onnx` |

### Tìm đúng tệp

| Việc cần làm | Vị trí trong `FS_nano_V2/` |
|---|---|
| Chạy trên Pi | `dashboard.py` và một trong hai script `FS_nano_V2_*.py` ngay cạnh README |
| Đổi nguồn camera, độ phân giải hoặc ngưỡng lửa/khói | [config/camera.yaml](config/camera.yaml) |
| Xem cách chọn trọng số và phát cảnh báo theo kết quả detect | [camera/detect.py](camera/detect.py) |
| Chỉnh kết nối broker, topic hoặc hàng đợi gửi cảnh báo | [messaging/mqtt_alerts.py](messaging/mqtt_alerts.py) |
| Chỉnh bộ nhận MQTT, lưu lịch sử và gửi Telegram | [web/server.py](web/server.py) |
| Chỉnh giao diện cho trình duyệt | [web/dashboard.html](web/dashboard.html) |
| Cài thư viện đúng backend | [requirements/](requirements/) |
| Xuất lại ONNX từ checkpoint V2 khi cần | [tools/export_onnx.py](tools/export_onnx.py) (Ultralytics tạo `best.onnx` cạnh checkpoint) |

## 1. Chuẩn bị Pi

Chép hai thư mục `src/FS_nano_V2`, `src/firewatch` và file trọng số tương ứng sang Pi, giữ cấu trúc trên. Nếu đặt trọng số ở nơi khác, truyền đường dẫn tuyệt đối bằng `--model /duong/dan/den/model` với đúng đuôi của phiên bản. Bản ONNX dùng ONNX Runtime CPU qua Ultralytics.

Trên Raspberry Pi OS 64-bit, chạy từ thư mục gốc repo. Ví dụ dưới đây cài bản ONNX và broker Mosquitto ngay trên Pi. Với bản PT, dùng `src/FS_nano_V2/requirements/requirements-pt.txt` ở dòng cài pip.

```bash
sudo apt install python3-venv python3-picamera2 mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
python3 -m venv --system-site-packages .venv-pi
source .venv-pi/bin/activate
python -m pip install -r src/FS_nano_V2/requirements/requirements-onnx.txt
export FS_DEVICE_ID=pi5-01
python src/FS_nano_V2/FS_nano_V2_onnx.py --check-config
```

Hai script mặc định dùng camera CSI qua Picamera2 ở 1280×720, chạy CPU và không cần màn hình. Với webcam USB, thêm `--source 0`. Để thử bằng video, thêm `--source /duong/dan/video.mp4`. Video thử cũng có thể phát cảnh báo đến người nhận.

## 2. Thiết lập Telegram trên Pi (tùy chọn)

Bỏ qua mục này để chạy riêng dashboard. Để bật thông báo, tạo bot bằng `/newbot` với **@BotFather**. Người nhận mở chat với bot và gửi `/start`. Nếu dùng nhóm, thêm bot vào nhóm và gửi lệnh `/start@ten_bot` trong nhóm.

Trong terminal sẽ chạy dashboard, nhập token ở lời nhắc ẩn. Với bot mới tạo, đoạn Python sau lấy ID của cuộc trò chuyện vừa gửi lệnh. Đối chiếu tên chat trước khi chọn ID (ID của nhóm thường là số âm). Biến token và chat ID chỉ cần đặt trên Pi.

```bash
read -rsp 'Bot token: ' TELEGRAM_BOT_TOKEN; echo
export TELEGRAM_BOT_TOKEN
python - <<'PY'
import json, os, urllib.request
url = "https://api.telegram.org/bot" + os.environ["TELEGRAM_BOT_TOKEN"] + "/getUpdates"
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        updates = json.load(response)["result"]
except OSError as exc:
    raise SystemExit(f"getUpdates failed ({type(exc).__name__}, HTTP {getattr(exc, 'code', '?')})")
for update in updates:
    chat = update.get("message", {}).get("chat", {})
    if chat:
        print(chat["id"], chat.get("title") or chat.get("username") or chat.get("first_name"))
PY
read -rp 'Chat ID: ' TELEGRAM_CHAT_ID
export TELEGRAM_CHAT_ID
```

Nếu danh sách trống, gửi lại lệnh trong chat và chạy lại đoạn Python. Với bot đang dùng webhook hoặc đã có dịch vụ đọc `getUpdates`, lấy chat ID từ dịch vụ đó. Bộ nhận chỉ gọi `sendMessage`. Điện thoại và laptop đăng nhập cùng tài khoản Telegram có thể xem cùng cuộc trò chuyện. Bật thông báo cho chat nếu muốn nhận thông báo bật lên. [Telegram Bot API](https://core.telegram.org/bots/api#sendmessage)

## 3. Chạy dashboard trước, rồi chạy camera

**Terminal A trên Pi** — ở thư mục gốc repo, dùng cùng terminal đã khai báo Telegram ở mục 2. Bản cài PT và ONNX đều đã có thư viện MQTT cho dashboard.

```bash
source .venv-pi/bin/activate
python src/FS_nano_V2/dashboard.py
```

**Terminal B trên Pi** — cũng mở tại thư mục gốc repo. Xem danh sách camera CSI, lấy IP của Pi rồi chạy một trong hai script.

```bash
source .venv-pi/bin/activate
export FS_DEVICE_ID=pi5-01
rpicam-hello --list-cameras
hostname -I
python src/FS_nano_V2/FS_nano_V2_onnx.py
# Nếu chọn PT, dùng dòng sau thay cho lệnh ONNX
# python src/FS_nano_V2/FS_nano_V2_pt.py
```

Giữ cả hai tiến trình Python chạy trong khi giám sát. Nhấn `Ctrl+C` ở từng terminal để dừng. Các lệnh trên chạy thủ công. Sau khi khởi động lại Pi, chạy lại cả hai và đặt lại các biến môi trường đã dùng. Mosquitto được systemd khởi động theo mục 1.

## 4. Xem trên laptop/PC hoặc điện thoại

Nếu `hostname -I` cho địa chỉ LAN của Pi là `192.168.1.50`, mở **`http://192.168.1.50:8080`** trên trình duyệt. Thay IP ví dụ bằng IP thực. Khi mở ngay trên Pi, dùng `http://127.0.0.1:8080`. Laptop/PC chỉ cần trình duyệt. Các thiết bị truy cập web phải kết nối được đến cổng TCP 8080 của Pi.

Dashboard tự làm mới khoảng hai giây một lần. Thẻ **Lửa** và **Khói** đếm các cảnh báo đang mở theo thiết bị và backend. Nhật ký hiển thị tối đa 100 sự kiện gần nhất, theo giờ của thiết bị xem. Bản tin kết thúc đóng cảnh báo tương ứng. Đây là trạng thái theo **sự kiện cuối đã nhận**. Chỉ báo MQTT xác nhận dashboard đã đăng ký topic với broker. Để kiểm tra camera còn chạy, xem terminal của bộ detect.

Lịch sử và hàng chờ Telegram được lưu ở `runs/FS_nano_V2/dashboard.sqlite3`. Đóng trình duyệt hoặc khởi động lại dashboard vẫn giữ dữ liệu. Nếu Pi mất Internet, web trong LAN vẫn hoạt động và bộ gửi sẽ thử lại các tin đang chờ. Sự kiện nhận trong thời gian chưa bật Telegram chỉ được lưu để xem trên web. Dashboard phục vụ HTTP trong LAN, không có đăng nhập. Dùng trong mạng tin cậy và giữ cổng 8080 trong LAN.

## 5. Kiểm tra toàn bộ luồng trước khi gắn camera

Giữ broker và dashboard đang chạy, mở web trên laptop/PC. Trong một terminal khác trên Pi, chạy đoạn sau từ thư mục gốc repo với venv đã kích hoạt. Hai sự kiện thử mang tên thiết bị `demo-pi`. Nếu đã bật Telegram, **chat đã cấu hình cũng sẽ nhận hai tin thử**.

```bash
python - <<'PY'
from pathlib import Path
import sys, time
sys.path.insert(0, "src/FS_nano_V2")
from messaging.mqtt_alerts import AlarmPublisher
sender = AlarmPublisher(Path("runs/FS_nano_V2/demo-pi/onnx"), "demo-pi", "onnx")
try:
    sender.enqueue("alarm_start", "smoke", 0.87)
    time.sleep(6)
    sender.enqueue("alarm_end", "smoke", None)
finally:
    sender.close()
PY
```

Trên web, số cảnh báo khói tăng một rồi trở về mức trước đó. Nhật ký giữ hai dòng bắt đầu/kết thúc của `demo-pi`. Chat nhận tin ngay cả khi tab web đã đóng. Sau bước này, bật camera theo mục 3 và kiểm tra với video mẫu hoặc camera của bạn.

| Hiện tượng | Kiểm tra |
|---|---|
| Laptop không mở được web | Mở `http://127.0.0.1:8080` ngay trên Pi. Nếu được, kiểm tra IP LAN, Wi-Fi khách có cô lập thiết bị hay không và quy tắc firewall cho TCP 8080 từ LAN. |
| Web hiện “Đang chờ MQTT” | Chạy `systemctl status mosquitto`. Kiểm tra địa chỉ, cổng và quyền subscribe của bộ nhận. |
| MQTT đã kết nối nhưng không thấy sự kiện | Chạy bài thử ở trên và lệnh `mosquitto_sub` bên dưới. Bộ detect chỉ phát khi có chuyển trạng thái đã xác nhận. |
| Web có sự kiện nhưng Telegram chưa có tin | Xem trạng thái Telegram cuối trang và log ở terminal A. Kiểm tra Internet trên Pi, token, chat ID, quyền gửi vào nhóm và việc người nhận đã mở chat với bot. Sau khi đổi biến môi trường, khởi động lại dashboard. |

## Giao thức MQTT và các tùy chọn

Mặc định bộ detect kết nối `127.0.0.1:1883`. Có thể đặt `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME` và `MQTT_PASSWORD` trong môi trường. Đặt `MQTT_CA_FILE` thành đường dẫn PEM của CA để bật TLS (cổng mặc định khi đó là 8883). Các tiến trình kết nối cùng broker phải có quyền publish hoặc subscribe tương ứng.

Chạy một tiến trình `dashboard.py` cho mỗi broker. Bộ nhận dùng client ID cố định `FS_nano_V2-dashboard` với phiên MQTT bền vững. Hãy mở bộ nhận và đợi “MQTT đã kết nối” trước lần phát cảnh báo đầu tiên. Sau lần đăng ký đầu, việc nhận bù lúc dashboard tạm dừng phụ thuộc vào thời hạn phiên, giới hạn hàng đợi và cấu hình lưu trữ của broker. [Tài liệu Mosquitto](https://mosquitto.org/man/mosquitto-conf-5.html)

Mỗi lần bắt đầu hoặc kết thúc một cảnh báo đã xác nhận, bộ detect gửi một JSON lên `FS_nano_V2/<device>/<pt|onnx>/alerts` với QoS 1. Ví dụ payload khi phát hiện khói

```json
{"event_id":"dffcb43ac1b041e89c07377cf16728ab","version":"FS_nano_V2","device":"pi5-01","backend":"onnx","utc":"2026-11-24T08:30:00+00:00","kind":"alarm_start","label":"smoke","confidence":0.87}
```

`confidence` là điểm cao nhất của lớp đó trong khung xác nhận. Khi bộ lọc phát `alarm_end`, trường này là `null`. Sự kiện được xếp hàng trên đĩa cho tới khi broker xác nhận. Khi chạy lại cùng thiết bị và backend, các sự kiện đang chờ được gửi tiếp. Bên nhận dùng `event_id` để xử lý bản tin QoS 1 bị giao lặp. Để kiểm tra topic, mở một terminal trên Pi trước khi chạy detect

```bash
mosquitto_sub -h 127.0.0.1 -t 'FS_nano_V2/+/+/alerts' -q 1 -v
```

Chỉnh ngưỡng riêng cho lửa và khói trong `config/camera.yaml`. Giá trị 0.335 kế thừa từ FireWatch là điểm khởi đầu để thử camera. Bộ lọc xác nhận theo nhiều khung hình và kết thúc sau một khoảng liên tục không phát hiện lớp đó. Nếu mất camera, sự kiện lỗi được ghi vào nhật ký. Mỗi phiên lưu nhật ký và ảnh lúc bắt đầu cảnh báo trong `runs/FS_nano_V2/<device>/<backend>/`.
