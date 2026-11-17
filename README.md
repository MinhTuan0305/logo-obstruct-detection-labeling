# Phát hiện vùng che logo / nhãn mác (thử nghiệm)

Phát hiện các vùng người bán cố ý che trên ảnh sản phẩm: dán sticker/emoji, tô trơn, quẹt cọ xóa logo, làm mờ nhãn, xóa chữ trên mác, chữ đè lên logo.
Kết quả chỉ là **tín hiệu sàng lọc** "có dấu hiệu che", không phải kết luận hàng thật/giả.

Việc đang làm: **gán nhãn tay ảnh che thật** (đã có khung gợi ý sẵn) để fine-tune lại model.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `labelstudio/nguoi_1/`, `labelstudio/nguoi_2/` | **Phần việc gán nhãn của từng người**: `images/` (ảnh), `tasks.json` (ảnh + khung gợi ý). Gán xong thì thêm `export.json` vào đây |
| `labelstudio/label_config.xml` | Cấu hình giao diện gán nhãn cho Label Studio |
| `data_real_obstruct/`, `data_real_obstruct_2/` | Ảnh che thật gốc (nguồn) |
| `data_real_train/manifest.csv` | Danh sách ảnh train sau khi lọc trùng (ảnh trong `data_real_train/images/` sinh lại bằng bước 1–3) |
| `dataset/`, `dataset2/` | **Tập test** (giày, áo) – không dùng để train; khung gán tay ở `annotations/` |
| `chuan_bi_du_lieu_that.ipynb` | Hash + lọc trùng → gom ảnh train → nhãn gợi ý → chia việc → gộp nhãn thành COCO |
| `obstruct_detection.ipynb` | Sinh dữ liệu che tổng hợp, train probe DINOv2, đo trên ảnh thật |
| `kiem_tra_model.ipynb` | Thử model (probe / RF-DETR / kết hợp) trên ảnh bất kỳ |
| `rfdetr_train.py`, `rfdetr_eval.py` | Fine-tune và đo RF-DETR |
| `obstruct_models.py` | Nạp và chạy model, dùng chung |
| `setup/requirements-*.txt` | Danh sách thư viện đã ghim phiên bản, giống máy gốc |

Model và dữ liệu tổng hợp (`work/`, khoảng 3 GB) **không** có trong repo. Người chỉ gán nhãn không cần chúng.

---

## 1. Cài đặt (Windows, PowerShell)

Cần **Python 3.11** và **uv** (trình cài gói nhanh, tạo venv giống hệt máy gốc):

```powershell
winget install Python.Python.3.11
```

```powershell
pip install uv
```

Clone repo rồi mở PowerShell tại thư mục repo.

### 1a. Chỉ để gán nhãn: venv `.venv-labelstudio`

```powershell
uv venv --python 3.11 .venv-labelstudio
```

```powershell
uv pip install --python .venv-labelstudio\Scripts\python.exe -r setup\requirements-labelstudio.txt
```

(Không có uv thì dùng: `py -3.11 -m venv .venv-labelstudio`, rồi `.venv-labelstudio\Scripts\pip install -r setup\requirements-labelstudio.txt`.)

### 1b. Để chạy model và notebook: venv `.venv` (cần GPU NVIDIA, CUDA 12.4)

```powershell
uv venv --python 3.11 .venv
```

```powershell
uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r setup\requirements-ml.txt
```

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name logo-obstruct --display-name "logo-obstruct (.venv)"
```

Chọn kernel **logo-obstruct (.venv)** khi mở notebook. Ngoài thư viện, còn cần các model sau (không có trong repo):
- **DINOv2-base:** tải `facebook/dinov2-base` từ Hugging Face, rồi đặt biến môi trường `DINO_DIR` trỏ tới thư mục đó.
- `work/probe.pt`, `work/rfdetr_out/checkpoint_best_total.pth`: xin từ người giữ repo.

---

## 2. Gán nhãn trên Label Studio

Mỗi người làm **một folder riêng**: người 1 dùng `labelstudio/nguoi_1`, người 2 dùng `labelstudio/nguoi_2`. Trong các bước dưới, thay `nguoi_k` bằng folder của bạn.

### Khởi động (mỗi lần làm)

Mở PowerShell **tại thư mục gốc của repo**:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"; $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = (Get-Location).Path; .\.venv-labelstudio\Scripts\label-studio.exe start --port 8080 --internal-host localhost
```

Đợi khoảng 1 phút (lần đầu), rồi mở http://localhost:8080. Dữ liệu Label Studio lưu trên máy bạn (`%LOCALAPPDATA%\label-studio`), không nằm trong repo.

### Tạo project (chỉ làm lần đầu)

1. Tạo tài khoản (chỉ lưu trên máy) → **Create Project**, đặt tên tùy ý.
2. Tab **Labeling Setup** → **Custom template** / **Code** → xóa nội dung mẫu, dán toàn bộ `labelstudio/label_config.xml` → **Save**.
3. Vào project → **Settings** → **Cloud Storage** → **Add Source Storage** → chọn **Local files**:
   - **Absolute local path:** `<thư mục repo>\labelstudio\nguoi_k\images`, ví dụ `D:\Code\logo-obstruct-detection\labelstudio\nguoi_1\images`;
   - **tắt** *Treat every bucket object as a source file*;
   - bấm **Add Storage**, **không** bấm *Sync*. Bước này chỉ cấp quyền cho Label Studio đọc ảnh.
4. **Import** → chọn file `labelstudio/nguoi_k/tasks.json` → **Import**. Mỗi ảnh có sẵn các khung gợi ý (màu tím) do model tạo.

Nếu ảnh không hiện: kiểm tra lại đã chạy lệnh khởi động **tại thư mục gốc repo**, và đường dẫn ở bước 3 đúng là thư mục `images` của bạn.

### Quy tắc vẽ khung

**Khoanh VÙNG CHE**, không khoanh cái logo nằm bên dưới. Mỗi vùng che một khung, khung sát mép vùng che.

| Khoanh (vùng che) | **Không** khoanh |
|---|---|
| Sticker, emoji, hình dán (sao, tim, "HOT SALE", bong bóng…) đè lên sản phẩm | Watermark tên shop, dòng chữ chạy ngang ảnh |
| Mảng tô trơn, vết quẹt cọ, nét gạch xóa logo hoặc chữ | Banner, khung giá, ô voucher **nằm ngoài sản phẩm** |
| Vùng bị làm mờ hoặc làm nhòe trên logo, nhãn | Logo in, thêu thật; mác treo còn nguyên chữ |
| Nhãn cổ, mác treo bị xóa chữ: khoanh phần bị tô hoặc xóa | Số size viết tay, giá viết tay **không** đè lên logo |
| Chữ, số điện thoại đè **lên** logo | |

Thao tác:
- **Khung gợi ý sai** (watermark, logo thật…): chọn khung → phím `Delete`.
- **Vùng che bị sót:** chọn nhãn **obstruction** (phím `1`) rồi kéo chuột vẽ.
- **Khung lệch:** kéo mép khung cho khớp.
- **Kiểu che** (không bắt buộc, nên chọn): bấm vào khung, chọn `sticker` / `paint` (tô, quẹt, gạch, xóa) / `blur` / `text_overlay`.
- **Ảnh hỏng hoặc không liên quan:** chọn **bỏ ảnh này**.
- Bấm **Submit** cho **mọi** ảnh, kể cả ảnh không còn khung nào (nghĩa là "ảnh này không có che"). Ảnh chưa Submit sẽ không được dùng.

### Nộp kết quả

1. Trong project: **Export** → định dạng **JSON** → lưu thành `labelstudio/nguoi_k/export.json`. Ghi đè file cũ nếu đã có, và giữ đúng tên file.
2. Chỉ commit **file export của mình** để tránh xung đột với người khác:

```powershell
git pull
```

```powershell
git add labelstudio/nguoi_k/export.json
```

```powershell
git commit -m "nhan nguoi_k"
```

```powershell
git push
```

Có thể export và push nhiều lần trong lúc làm: mỗi lần export đều chứa toàn bộ những gì đã Submit.

---

## 3. Sau khi có nhãn (người train)

1. `chuan_bi_du_lieu_that.ipynb`:
   - nếu vừa clone repo, chạy bước 1–3 để sinh lại `data_real_train/images/`;
   - sau đó chạy **bước 5**: gộp mọi `labelstudio/nguoi_*/export.json` thành `data_real_train/coco_{train,val}.json`.
   - Bước 4 (nhãn gợi ý) và 4b (chia việc) **không chạy lại**, vì sẽ tạo lại folder `nguoi_*`.
2. Train tiếp RF-DETR với ảnh thật (val bằng ảnh thật):

```powershell
.venv\Scripts\python.exe -X utf8 rfdetr_train.py --real data_real_train --init work/rfdetr_out/checkpoint_best_total.pth
```

3. Đo trên tập test (`dataset/`, `dataset2/`, ảnh shop sạch):

```powershell
.venv\Scripts\python.exe -X utf8 rfdetr_eval.py --ckpt work/rfdetr_real_out/checkpoint_best_total.pth
```

Khi có thêm folder ảnh che mới (ví dụ của người khác), thêm nó vào `NGUON_CHE` trong notebook rồi chạy bước 1–4b. Ảnh trùng với ảnh đã có, hoặc trùng với tập test, sẽ tự bị loại.

## Lưu ý

- Ảnh lấy từ trang bán hàng công khai, chỉ dùng cho nghiên cứu nội bộ. **Giữ repo ở chế độ private.**
- Model dùng: DINOv2 (Apache-2.0), RF-DETR (Apache-2.0), OWLv2 (Apache-2.0), Label Studio Community (Apache-2.0).
