# Nhận diện và đọc biển số xe Việt Nam

Pipeline: **YOLOv8** tìm khung biển số → **EasyOCR** (model `plate` tự train) đọc chữ → vẽ khung kèm kết quả lên ảnh/video.

```
anh/video --> plate_detector.py (tim khung) --> plate_reader.py (doc chu + bo phieu) --> main.py (ve khung, luu ket qua)
```

## 1. Cấu trúc thư mục

```
du_an/
├── main.py                                  # file chay chinh
├── plate_detector.py                        # phat hien bien so (YOLOv8)
├── plate_reader.py                          # doc bien so (EasyOCR) + theo doi/bo phieu
├── setup_model.py                           # cai model OCR vao EasyOCR (chay 1 lan)
├── best_640.pt / best_320.pt                # model YOLOv8 phat hien bien so (cua ban)
└── best_accuracy.pth                        # model OCR sau khi train
```

Các file phục vụ train OCR (chỉ cần khi train lại): `split_dataset.py`, `convert_easyocr.py`, `run_train.py`, `patch_next.py`, `patch_index.py`, `patch_test.py`.

## 2. Cài đặt thư viện

```bash
pip install ultralytics easyocr opencv-python matplotlib
```

- Máy Linux không có màn hình: dùng `opencv-python-headless` thay cho `opencv-python`.
- Có GPU NVIDIA thì cài `torch` bản CUDA trước, rồi mới cài các gói trên.
- Kiểm tra GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

## 3. Setup OCR lần đầu

### Đảm bảo đã có `best_accuracy.pth` (chỉ cần cài model)

**Windows:** và **Linux:**
```bash
python3 setup_model.py --pth /duong/dan/best_accuracy.pth
```

Thấy dòng `Nap model plate thanh cong` là xong. Script tạo 3 file trong thư mục `~/.EasyOCR` (Windows: `C:\Users\<ten>\.EasyOCR`):

```
model/plate.pth               <- trong so da train
user_network/plate.py         <- dinh nghia mang (VGG + BiLSTM + CTC)
user_network/plate.yaml       <- cau hinh va bo ky tu
```

Tuỳ chọn: `--chars` (bộ ký tự, phải đúng thứ tự `number` trong `plate_config.yaml`), `--name` (tên model), `--home` (thư mục EasyOCR), `--no-test` (bỏ bước thử nạp).

## 4. Tác dụng của từng file

### Chạy khi dùng (nhận diện)

| File | Tác dụng |
|---|---|
| `main.py` | File chạy chính. Đọc ảnh/video, gọi detector + reader, vẽ khung kèm chữ, lưu ảnh/video kết quả và crop biển số. |
| `plate_detector.py` | Lớp `PlateDetector` (YOLOv8) trả về danh sách khung `(x1, y1, x2, y2, conf)`. Hàm `cat_bien` cắt biển số khỏi frame. Có đoạn vá `torch.load` cho PyTorch ≥ 2.6. |
| `plate_reader.py` | Lớp `PlateOCR` đọc biển (biển 2 dòng cắt đôi, đọc từng dòng, thử nhiều mức cắt lề, ưu tiên kết quả đúng định dạng biển VN). Lớp `Tracker` ghép khung giữa các frame và bỏ phiếu để chữ không nhấp nháy. |
| `setup_model.py` | Cài model OCR vào `~/.EasyOCR` (chạy 1 lần sau khi train). |

## 5. Chạy main.py

```bash
python main.py --source <anh_hoac_video> --weights <best.pt>
```

Ví dụ:
```bash
# anh
python main.py --source anh.jpg --weights best.pt

# video
python main.py --source video.mp4 --weights runs/detect/train5/weights/best.pt

# video, uu tien toc do
python main.py --source video.mp4 --weights best.pt --ocr-every 5 --ocr-passes 2

# chi detect, khong doc chu
python main.py --source anh.jpg --weights best.pt --no-ocr

# ep OCR chay CPU
python main.py --source anh.jpg --weights best.pt --cpu
```

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | (bắt buộc) | Đường dẫn ảnh hoặc video (`jpg png bmp webp`, `mp4 avi mov mkv wmv flv`) |
| `--weights` | `best_640.pt` | Model YOLOv8 phát hiện biển số (imgsz 640), còn model `best_320.pt` có imgsz 320 |
| `--out` | `output` | Thư mục kết quả |
| `--imgsz` | `640` | Kích thước ảnh đưa vào YOLO |
| `--conf` | `0.3` | Ngưỡng tin cậy của detector |
| `--pad` | `5` | Số pixel nới thêm quanh khung khi crop |
| `--min-size` | `20` | Bỏ qua crop nhỏ hơn số pixel này |
| `--crop-every` | `1` | Video: lưu crop mỗi N frame |
| `--no-ocr` | tắt | Chỉ detect, không đọc biển |
| `--ocr-every` | `3` | Video: đọc biển mỗi N frame, các frame còn lại dùng kết quả đã bỏ phiếu |
| `--ocr-passes` | `3` | Số mức cắt lề thử khi đọc (1 đến 4). Nhiều hơn thì chính xác hơn nhưng chậm hơn |
| `--cpu` | tắt | Ép OCR chạy CPU |

### Kết quả

```
output/
├── <ten>_detected.jpg | <ten>_detected.mp4   <- khung xanh kem chu, vi du "59G163188 0.87"
└── crops/                                      <- anh bien so da cat, ten file chua bien doc duoc
```

## 6. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Khong thay file model` khi chạy `setup_model.py` | Sai đường dẫn `--pth`. |
| `size mismatch` / `Missing keys` khi nạp model | `--chars` không khớp bộ ký tự lúc train (phải đúng thứ tự `number` trong `plate_config.yaml`), hoặc trainer đã đổi kiến trúc. |
| Chữ đọc ra lệch ký tự, đọc cả `-`, `.`, ốc vít | Thử `--ocr-passes 4`, hoặc train lại với dữ liệu augment (`convert_easyocr.py` bản mới đã có). |
| Nhầm `G`/`6`, `B`/`8`, `D`/`0` | Ảnh quá nhỏ/mờ. Thêm ảnh biển thật từ camera của bạn vào dataset và gán nhãn. |
| Video chạy chậm | Tăng `--ocr-every 5`, giảm `--ocr-passes 2`, hoặc kiểm tra GPU đã bật chưa. |
| `ImportError: cannot import name '_accumulate'` khi train | Trainer cũ: sửa dòng import trong `dataset.py` thành `from itertools import accumulate as _accumulate`. |
| `train.py` chạy xong im lặng, không in gì | Dùng `run_train.py`, không chạy trực tiếp `train.py`. |
| `MapFullError` (lmdb) | Bản `convert_easyocr.py` mới ghi thẳng ảnh + csv, không còn dùng lmdb. |

## 7. Ghi chú

- `main.py` import `plate_detector` trước `plate_reader` vì `plate_detector` có đoạn vá `torch.load`. Đừng đảo thứ tự.
- Biển kiểu khác (quân đội, ngoại giao...) không khớp mẫu regex `MAU_*` ở đầu `plate_reader.py`. Muốn hỗ trợ thì sửa regex đó.
- Dùng GPU: bỏ `--cpu`. Chạy trên CPU vẫn được nhưng chậm với video.
