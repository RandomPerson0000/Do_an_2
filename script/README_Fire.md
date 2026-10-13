# Bộ code huấn luyện YOLO phát hiện lửa

Mặc định: YOLO26n pretrained, object detection, một class `0: fire`.
Các giá trị là cấu hình khởi đầu đề xuất, chưa phải cấu hình đã tối ưu trên dataset của bạn.
Không cần chạy train trên Raspberry Pi; có thể train trên laptop/GPU rồi triển khai riêng.

## 1. Cài đặt

Khuyên dùng Python 3.12 trong virtual environment. Trong terminal tại thư mục bộ code:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, ultralytics; print(ultralytics.__version__); print(torch.__version__); print(torch.cuda.is_available())"
```

Nếu dùng NVIDIA mà dòng cuối là `False`, kiểm tra bản PyTorch/CUDA và driver theo bộ chọn cài đặt chính thức https://pytorch.org/get-started/locally/ . Không có CUDA, `device: null` có thể chuyển sang CPU và train chậm. Sửa `device: 0` khi muốn yêu cầu GPU 0 rõ ràng. Nếu thư viện không nhận `yolo26n.pt`, cập nhật Ultralytics trong môi trường ảo hoặc chọn một checkpoint tương thích đã có. Lần đầu tải pretrained cần internet; cũng có thể truyền đường dẫn `.pt` cục bộ bằng `--model`.

## 2. Dataset đầu vào

Chuẩn bị `train/images`, `train/labels`, `valid/images`, `valid/labels`; nếu có test thì thêm `test/images`, `test/labels`. Cũng hỗ trợ kiểu `images/train` và `labels/train`, miễn nhập đúng đường dẫn.

Mỗi ảnh có file `.txt` cùng tên: `a.jpg` đi với `a.txt`. Mỗi dòng bbox:

```text
0 x_center y_center width height
```

Tọa độ chuẩn hóa theo kích thước ảnh trong [0,1]; width và height phải lớn hơn 0. Ảnh negative (flash, đèn đỏ/cam, nắng...) dùng `.txt` rỗng, không thêm class `negative`. YOLO có thể chấp nhận thiếu label như background, nhưng script chủ động yêu cầu file rỗng để hạn chế nhầm ảnh chưa label thành negative.

Script chỉ dành cho label `0: fire`. Không dùng `single_cls=True` để biến cả smoke thành fire. Dataset nhiều class cần lọc/chuyển label có chủ đích trước khi dùng. Bộ kiểm tra chỉ kiểm tra đường dẫn và cú pháp label, chưa phát hiện ảnh hỏng, nhãn sai nội dung, ảnh trùng hoặc rò rỉ dữ liệu.

Chia train/val/test theo video hoặc phiên quay trước khi trích frame. Không đưa frame gần giống nhau từ cùng một video vào nhiều split. Test phải có nhãn nếu muốn tính metric; không dùng test để chọn augmentation hay ngưỡng cảnh báo.

## 3. Tạo YAML theo đường dẫn tự nhập

Chạy chế độ hỏi từng đường dẫn:

```powershell
python train_fire.py init
```

Hoặc truyền trực tiếp để không phải nhập tương tác:

```powershell
python train_fire.py init --root "D:/FireProject/dataset" --out "D:/FireProject/configs/fire_train.yaml" --project "D:/FireProject/runs/fire" --train-images "train/images" --val-images "valid/images"
```

Nếu có test, thêm `--test-images "test/images"`. Các thư mục ảnh phải tồn tại; script không tự chia hoặc di chuyển dữ liệu. `--out` là đường dẫn đầy đủ gồm tên file YAML mới. Dùng `/` trong đường dẫn Windows dễ đọc hơn. Script từ chối ghi đè file cấu hình đã có.

Từ tên `fire_train.yaml`, script tạo bốn file cạnh nhau:

| File | Vai trò |
|---|---|
| `fire_train.yaml` | Cấu hình train và augmentation dạng phẳng, dùng được với `model.train(cfg=...)` |
| `fire_train_data.yaml` | Đường dẫn train/val/test và `names: {0: fire}` |
| `fire_train_val.yaml` | Đánh giá checkpoint bằng `model.val(...)` |
| `fire_train_solution.yaml` | Cấu hình riêng cho ví dụ Heatmap sau train |

Các đường dẫn dataset và project do `init` tạo được đổi thành tuyệt đối. Khi chuyển máy phải tạo lại YAML hoặc sửa các đường dẫn đó. File YAML là cấu hình, không chứa ảnh, label hay trọng số model.

## 4. Kiểm tra rồi train

```powershell
python train_fire.py check --data "D:/FireProject/configs/fire_train_data.yaml"
python train_fire.py train --config "D:/FireProject/configs/fire_train.yaml" --val-config "D:/FireProject/configs/fire_train_val.yaml" --tensorboard
```

`train` cũng kiểm tra dataset trước khi bắt đầu. `--val-config` chạy thêm một lần đánh giá `best.pt` sau khi train xong; không cần nó để bật validation mỗi epoch. `val: true` trong YAML train đảm nhiệm phần đó. Có thể bỏ `--tensorboard` nếu chỉ cần CSV/JSON/PNG.

Các nhóm cấu hình quan trọng:

| Nhóm | Giá trị khởi đầu trong script | Cách điều chỉnh |
|---|---|---|
| Tài nguyên | `imgsz: 640`, `batch: 4`, `workers: 0`, `cache: false`, `amp: true` | GPU 4 GB có thể thử batch 4; nếu OOM giảm 2 hoặc 1. Chưa bảo đảm vừa VRAM. Workers 0 dễ chạy Windows; tăng sau khi ổn định. |
| Thời gian | `epochs: 100`, `patience: 25` | Epoch là mức tối đa; early stopping có thể kết thúc sớm. |
| Tối ưu | `optimizer: AdamW`, `lr0: 0.001`, `lrf: 0.01`, `cos_lr: true` | Chọn optimizer cụ thể để kiểm soát learning rate; `auto` có thể tự chọn và bỏ qua `lr0`. |
| Warmup | `warmup_epochs: 3`, `warmup_bias_lr: 0` | Cấu hình đề xuất cho thử nghiệm AdamW này. |
| Tái lập | `seed: 42`, `deterministic: true` | Giảm biến thiên, không bảo đảm giống bit trên mọi GPU/phiên bản. |
| Validation | `val: true`, `conf: 0.001`, `iou: 0.7`, `max_det: 300` | `conf` thấp để đánh giá PR/mAP; không dùng làm ngưỡng cảnh báo trực tiếp. |
| Lưu kết quả | `save: true`, `save_period: 10`, `plots: true` | Có checkpoint định kỳ và biểu đồ. |

`iou` trong cấu hình inference/validation là ngưỡng NMS khi sử dụng nhánh có NMS; không phải tham số đổi định nghĩa mAP50–95. Với nhánh NMS-free, ý nghĩa áp dụng khác; luôn ghi lại phiên bản và chế độ inference khi so sánh.

Augmentation được chọn vừa phải cho phép thử đầu tiên với ảnh lửa:

| Tham số | Giá trị | Ý đồ của cấu hình đề xuất |
|---|---:|---|
| `hsv_h` | 0.01 | Biến thiên hue nhỏ |
| `hsv_s`, `hsv_v` | 0.4, 0.4 | Thử độ bão hòa và ánh sáng khác nhau |
| `degrees` | 5 | Xoay nhẹ |
| `translate`, `scale` | 0.1, 0.3 | Thay vị trí và kích thước |
| `fliplr`, `flipud` | 0.5, 0 | Lật ngang, không lật dọc |
| `mosaic`, `close_mosaic` | 0.5, 10 | Ghép ảnh một phần quá trình, tắt 10 epoch cuối |
| `mixup`, `shear`, `perspective` | 0, 0, 0 | Chưa thêm các biến đổi này vào baseline |

Đây là lựa chọn thử nghiệm của bộ code, không phải cấu hình tối ưu được Ultralytics công bố cho fire. Xem các ảnh train batch để phát hiện augmentation làm cảnh thiếu thực tế. Những augmentation này áp dụng lúc train; `augment: false` trong validation tắt TTA. `copy_paste`, `auto_augment`, `erasing` không được thêm như các tùy chọn hữu ích cho bài toán detect này. Nếu cài Albumentations, phiên bản Ultralytics có thể tự thêm biến đổi nhẹ; cần giữ môi trường nhất quán giữa các lần so sánh.

## 5. Log, checkpoint, biểu đồ

Mỗi run nằm trong `project/name`. `exist_ok: false` cho phép tạo tên tăng dần; xem đường dẫn thực tế in ở cuối, không luôn giả định run tên `fire_yolo26n`.

| Kết quả | Mục đích |
|---|---|
| `weights/best.pt` | Checkpoint có fitness validation tốt nhất theo phiên bản thư viện |
| `weights/last.pt` | Checkpoint gần nhất; dùng resume khi run bị ngắt |
| Checkpoint epoch trong `weights/` | Lưu định kỳ theo `save_period`; tên phụ thuộc phiên bản |
| `args.yaml` | Cấu hình thực tế Ultralytics sử dụng |
| `results.csv`, `results.png` | Loss/metric theo epoch và biểu đồ tổng hợp |
| Các biểu đồ PR/F1/P/R, confusion matrix, ảnh batch | Kiểm tra chất lượng; tên file cụ thể có thể khác giữa phiên bản |
| `train.log` | Thông điệp qua logger Ultralytics từ lúc callback bắt đầu; không thu toàn bộ progress bar hoặc mọi stdout |
| `epochs.jsonl` | Callback ghi metric và learning rate theo epoch; có thể có dòng lặp khi final validation/resume |
| `environment.json`, `requirements-lock.txt` | Phiên bản môi trường lúc chạy; được cập nhật khi resume |
| `tensorboard_custom/` | Metric và learning rate do callback ghi khi dùng `--tensorboard` |
| `metrics_summary.json` | Metric từ lần gọi `val` riêng, lưu trong thư mục validation |

Xem TensorBoard:

```powershell
tensorboard --logdir "D:/FireProject/runs/fire"
```

Mở địa chỉ local được terminal thông báo. Cơ chế TensorBoard này là callback riêng; nếu đã bật tích hợp TensorBoard của Ultralytics, có thể thấy thêm bộ log của thư viện.

## 6. Resume, validation, test

Tiếp tục một run bị ngắt sau khi đã lưu checkpoint:

```powershell
python train_fire.py train --resume "D:/FireProject/runs/fire/fire_yolo26n/weights/last.pt" --tensorboard
```

Resume khôi phục trạng thái run từ checkpoint; script không trộn cấu hình train mới vào. Muốn fine-tune sau khi run đã hoàn tất, tạo YAML mới với `--model ".../best.pt"` rồi train mới. Không coi resume của run đã hoàn tất là cách mặc định để tăng epochs.

Đánh giá best trên validation:

```powershell
python train_fire.py val --weights "D:/FireProject/runs/fire/fire_yolo26n/weights/best.pt" --config "D:/FireProject/configs/fire_train_val.yaml"
```

Chỉ khi đã chốt cấu hình, đánh giá test có nhãn:

```powershell
python train_fire.py val --weights "D:/FireProject/runs/fire/fire_yolo26n/weights/best.pt" --config "D:/FireProject/configs/fire_train_val.yaml" --split test
```

## 7. Solutions: chạy riêng sau huấn luyện

Solutions là nhóm ứng dụng như đếm đối tượng, heatmap, quản lý vùng... Không phải một nhóm siêu tham số để truyền tất cả vào `model.train()`.

Bộ code kèm ví dụ chính thức `solutions.Heatmap`, áp dụng trên checkpoint fire của bạn. YAML chứa các tùy chọn model, classes, conf, iou, imgsz, device, tracker, colormap, line_width, show; các khóa `kind`, `source`, `output` do wrapper xử lý. Không truyền file này qua `train(cfg=...)`.

```powershell
python train_fire.py solution --weights "D:/FireProject/runs/fire/fire_yolo26n/weights/best.pt" --config "D:/FireProject/configs/fire_train_solution.yaml" --source "D:/FireProject/videos/demo.mp4"
```

Sửa `output` trong YAML nếu muốn chọn nơi lưu video. Đường dẫn source/output tương đối được hiểu từ thư mục terminal hiện tại. `--source 0` mở camera 0, nhưng nên thử video trước. Nếu bật `show: true`, nhấn Q để thoát; nếu không, Ctrl+C. Video đầu ra dùng FPS khai báo của nguồn, không phải số FPS inference đạt được.

Heatmap chỉ là hình tích lũy vị trí xuất hiện của detection/tracking, **không phải ảnh nhiệt hay phép đo nhiệt độ lửa**, cũng không tự tạo thuật toán cảnh báo theo thời gian. Tracking có thể mất ID; không dùng số ID để suy ra số đám cháy. `conf: 0.35` là điểm bắt đầu thử inference, cần chọn lại trên validation/video thực tế. Script không gửi email, bật còi hoặc chạy Solutions tự động sau train.

## 8. Phạm vi kiểm thử

Đã kiểm tra cú pháp Python, tạo/đọc YAML tại đường dẫn có khoảng trắng, bảo vệ ghi đè, kiểm tra label hợp lệ/không hợp lệ và kết nối tham số train/val/resume bằng API giả lập. Môi trường soạn code không có Ultralytics/PyTorch/GPU và chưa có dataset của bạn, nên chưa chạy huấn luyện thật hoặc kiểm chứng đầu ra Heatmap. Sau khi cài môi trường, nên thử một run 1–2 epoch trên bản sao cấu hình trước khi train dài.

## Nguồn chính thức đã đối chiếu

- https://docs.ultralytics.com/vi/usage/cfg/
- https://docs.ultralytics.com/modes/train/
- https://docs.ultralytics.com/modes/val/
- https://docs.ultralytics.com/guides/yolo-data-augmentation/
- https://docs.ultralytics.com/usage/callbacks/
- https://docs.ultralytics.com/datasets/detect/
- https://docs.ultralytics.com/guides/heatmaps/

Tài liệu online thay đổi theo phiên bản. Khi đã chạy ổn định, lưu `requirements-lock.txt` và `args.yaml` cùng kết quả để truy vết. Không cập nhật thư viện giữa một run đang train dở và lần resume nếu không cần thiết.
