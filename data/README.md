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
