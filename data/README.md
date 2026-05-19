# Dữ liệu

Thư mục này chứa notebook xem trước dữ liệu và các file dữ liệu cục bộ của dự án.

## Dữ liệu thô

Đặt các file CSV tải từ Kaggle vào:

```text
data/raw/
```

Các file dự kiến:

```text
data/raw/data set.csv
data/raw/datasetsmall.csv
```

File dữ liệu đầy đủ cũng có thể được đổi tên cục bộ thành:

```text
data/raw/data_set.csv
```

Dữ liệu thô được bỏ qua bởi Git vì file CSV đầy đủ vượt quá giới hạn kích thước file thông thường của GitHub.

## Ví dụ tải dữ liệu

```python
import kagglehub

kagglehub.dataset_download(
    "bensalem14/sgcc-dataset",
    output_dir="data/raw"
)
```

Không commit các file trong `data/raw/` hoặc các thư mục tải lồng nhau như `data/data/raw/`.

## Dữ liệu xử lý

Pipeline hiện tại tạo các file sau trong `data/processed/`:

```text
data/processed/cleaned.csv
data/processed/quality_features.csv
data/processed/features.csv
```

Trong đó `features.csv` là bảng feature cấp khách hàng dùng để train model.

## Dữ liệu test cho web demo

`src/train.py` tạo thêm:

```text
data/test/test_raw_15_percent.csv
```

Đây là đúng 15% test split được tách bằng cùng logic train/validation/test trong pipeline hiện tại: stratified 70/15/15 với `random_state=42`. File này giữ format raw 1,034 cột ngày, `CONS_NO`, `FLAG` để web demo có thể upload/import và predict mà không dùng nhầm dữ liệu train hoặc validation.
