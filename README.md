# Phát Hiện Trộm Điện Bằng Học Máy

Dự án phát hiện khách hàng có khả năng trộm điện dựa trên lịch sử tiêu thụ điện hằng ngày. Bài toán được mô hình hóa thành phân loại nhị phân có giám sát:

- `0`: khách hàng bình thường
- `1`: khách hàng có khả năng trộm điện

Bộ dữ liệu sử dụng: [SGCC Electricity Theft Detection Dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset).

## Mục Tiêu

- Làm sạch dữ liệu tiêu thụ điện theo ngày và hạn chế target leakage.
- Tạo đặc trưng cấp khách hàng từ chuỗi tiêu thụ điện.
- So sánh 3 mô hình: Logistic Regression + L2 + balanced, Random Forest, LightGBM.
- Chọn threshold trên validation set, đánh giá cuối trên test set.
- Lưu LightGBM inference bundle để phục vụ web demo.

## Kết Quả Chính

Mô hình tốt nhất hiện tại là **LightGBM benchmark**.

| Model | Threshold | Precision | Recall | F2 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM benchmark | 0.4368 | 0.2885 | 0.6458 | 0.5176 | 0.4489 | 0.8387 |
| Random Forest benchmark | 0.2987 | 0.2477 | 0.6421 | 0.4870 | 0.3701 | 0.8082 |
| LR + L2 + balanced | 0.5440 | 0.2184 | 0.5701 | 0.4312 | 0.3117 | 0.7771 |

Accuracy không phải chỉ số chính vì dữ liệu mất cân bằng mạnh. Báo cáo ưu tiên Recall, F2, PR-AUC, ROC-AUC và confusion matrix.

## Cấu Trúc Dự Án

```text
.
├── data/
│   ├── raw/                 # raw CSV tải từ Kaggle, không commit
│   ├── processed/           # cleaned.csv, quality_features.csv, features.csv
│   ├── test/                # test_raw_15_percent.csv cho web demo
│   └── README.md
├── figures/                 # hình EDA/model dùng cho báo cáo
├── models/                  # model bundle và các bảng metric sinh từ train.py
├── notebooks/               # notebook phân tích và huấn luyện
├── Project Description/     # nội dung slide và báo cáo LaTeX
├── src/
│   ├── preprocessing_v2.py
│   ├── feature.py
│   ├── train.py
│   └── evaluate.py
├── web/
│   ├── backend/
│   └── frontend/
├── package.json
├── README.md
└── requirements.txt
```

Web demo là phần phụ nên project structure chỉ liệt kê `web/backend` và `web/frontend`. Chi tiết chạy web nằm trong `web/README.md`.

## Cài Đặt Môi Trường

Tạo và kích hoạt môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài thư viện cho pipeline machine learning:

```powershell
python -m pip install -r requirements.txt
```

Nếu cần chạy web demo:

```powershell
npm run install:web
```

## Dữ Liệu

Đặt file raw đầy đủ vào:

```text
data/raw/data set.csv
```

File raw đầy đủ có 42,372 khách hàng, 1,034 cột ngày tiêu thụ, `CONS_NO` và `FLAG`.

Ví dụ tải dữ liệu bằng `kagglehub`:

```python
import kagglehub

kagglehub.dataset_download(
    "bensalem14/sgcc-dataset",
    output_dir="data/raw"
)
```

Không commit file trong `data/raw/` vì kích thước lớn.

## Quy Trình Chạy Pipeline

Chạy preprocessing:

```powershell
python src/preprocessing_v2.py
```

Chạy feature engineering:

```powershell
python src/feature.py
```

Train và lưu artifact:

```powershell
python src/train.py
```

Sau khi chạy `train.py`, các artifact chính gồm:

```text
models/energy_theft_model_bundle.pkl
models/model_metadata.json
models/training_summary.csv
models/test_comparison.csv
models/threshold_report.csv
data/test/test_raw_15_percent.csv
```

`energy_theft_model_bundle.pkl` chỉ lưu LightGBM inference bundle để web demo sử dụng. Các file CSV/JSON trong `models/` dùng để xem metric, báo cáo và kiểm tra lại kết quả.

## Web Demo

Web demo dùng Next.js frontend và FastAPI backend. Backend load sẵn LightGBM bundle, nhận CSV raw theo format dataset gốc, chạy preprocessing + feature engineering, rồi trả kết quả dự đoán.

Chạy từ root repo:

```powershell
npm run dev:all
```

Địa chỉ mặc định:

- Frontend: `http://127.0.0.1:3000`
- Backend API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

File CSV mẫu cho demo:

```text
data/test/test_raw_15_percent.csv
```

File này là đúng 15% test split được tạo bằng cùng logic trong `train.py`, không lấy từ train/validation.

## Ghi Chú Đánh Giá

- Split hiện tại: train/validation/test = 70/15/15, stratified, `random_state=42`.
- Threshold cuối của từng model được chọn trên validation set theo Best F2.
- Test set chỉ dùng cho đánh giá cuối và file demo raw.
- `FLAG` trong CSV demo chỉ dùng để hiển thị ground truth, không dùng làm input predict.
