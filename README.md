# Phát hiện trộm cắp điện bằng học máy

Dự án phát hiện khách hàng có khả năng trộm cắp điện dựa trên lịch sử tiêu thụ điện hằng ngày. Bài toán được mô hình hóa thành bài toán phân loại nhị phân có giám sát:

- `0`: khách hàng bình thường
- `1`: khách hàng có khả năng trộm điện

Bộ dữ liệu sử dụng: [SGCC Electricity Theft Detection Dataset](https://www.kaggle.com/datasets/bensalem14/sgcc-dataset).

## Mục tiêu dự án

- Làm sạch dữ liệu tiêu thụ điện theo ngày.
- Trích xuất các đặc trưng thống kê và hành vi từ chuỗi tiêu thụ điện.
- Huấn luyện hồi quy logistic làm mô hình chính.
- So sánh với hồi quy logistic có điều chuẩn L2, phân tích phân biệt Gaussian và hồi quy tuyến tính làm mô hình so sánh phụ.
- Đánh giá bằng các chỉ số phù hợp với dữ liệu mất cân bằng, ưu tiên Recall, F1-score và PR-AUC.

## Cấu trúc dự án

```text
.
├── data/
│   ├── README.md
│   └── preview.ipynb
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── README.md
└── requirements.txt
```

Các file dữ liệu cục bộ không được đưa lên Git. Đặt các file CSV đã tải vào thư mục `data/raw/`.

## Cài đặt môi trường

Tạo và kích hoạt môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài đặt thư viện cần thiết:

```powershell
python -m pip install -r requirements.txt
```

## Bộ dữ liệu

Tải bộ dữ liệu trực tiếp vào thư mục dữ liệu của dự án:

```python
import kagglehub

kagglehub.dataset_download(
    "bensalem14/sgcc-dataset",
    output_dir="data/raw"
)
```

Các file dữ liệu thô dự kiến:

```text
data/raw/data set.csv
data/raw/datasetsmall.csv
```

Nếu file đầy đủ được đổi tên cục bộ để tránh dấu cách, dùng:

```text
data/raw/data_set.csv
```

## Quy trình thực hiện

1. Khám phá bộ dữ liệu và phân phối nhãn trong `notebooks/eda.ipynb`.
2. Làm sạch dữ liệu thô trong `src/preprocessing.py`.
3. Xây dựng đặc trưng thống kê và hành vi trong `src/features.py`.
4. Huấn luyện mô hình chính và các mô hình so sánh trong `src/train.py`.
5. Đánh giá mô hình trong `src/evaluate.py`.

## Chỉ số đánh giá

Vì bộ dữ liệu bị mất cân bằng, Accuracy chỉ nên được dùng để tham khảo. Các chỉ số chính gồm:

- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion Matrix

Âm tính giả là loại lỗi cần chú ý nhất trong bài toán này, vì đó là trường hợp khách hàng thật sự trộm điện nhưng mô hình dự đoán là bình thường.
