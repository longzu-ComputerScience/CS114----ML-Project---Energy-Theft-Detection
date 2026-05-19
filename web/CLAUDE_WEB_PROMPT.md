# Prompt for Claude Opus 4.6 - Build Web Demo

Bạn là Claude Opus 4.6 đang làm việc trong repo Energy Theft Detection. Trước khi sửa code, hãy đọc kỹ `CLAUDE.md` ở root repo để nắm quy tắc làm việc, sau đó đọc các file liên quan bên dưới để hiểu đúng context.

## Context bắt buộc phải hiểu

Repo hiện tại đã có pipeline machine learning đầy đủ:

- Raw dataset chính: `data/raw/data set.csv`
  - Mỗi dòng là một khách hàng.
  - Có `1034` cột ngày tiêu thụ từ `1/1/2014` đến `10/31/2016`.
  - Có `CONS_NO` và `FLAG`.
- Processed features: `data/processed/features.csv`
  - Shape logic: `CONS_NO`, `FLAG`, và feature engineering.
  - Khi train, 2 feature hằng bị bỏ: `negative_count_raw`, `negative_ratio_raw`.
  - Model dùng `159` engineered features.
- Exact test split cho demo: `data/test/test_raw_15_percent.csv`
  - Đây là đúng 15% test split được tách bằng cùng logic trong notebook/train.py.
  - Không lấy lại từ train/validation để tránh leakage.
  - Format giống raw dataset: 1034 cột ngày, sau đó `CONS_NO`, `FLAG`.
- Model artifact chính: `models/energy_theft_model_bundle.pkl`
  - File này hiện chỉ chứa LightGBM inference bundle, không chứa LR/RF.
  - Các key quan trọng: `estimator`, `threshold`, `active_feature_cols`, `constant_cols`, `id_col`, `target_col`, `score_kind`.
  - Threshold LightGBM hiện tại khoảng `0.4368`, policy là Best F2 trên validation.
- Metadata hỗ trợ: `models/model_metadata.json`
  - Dùng để hiển thị thông tin model, split và metric, không dùng thay cho model.

Các file nên đọc:

- `CLAUDE.md`
- `src/train.py`
- `src/preprocessing_v2.py`
- `src/feature.py`
- `models/model_metadata.json`
- `data/test/test_raw_15_percent.csv` chỉ đọc header/shape/sample, không cần load toàn bộ nếu không cần.

## Mục tiêu

Xây dựng web demo trong folder `web/`. Tất cả code web, cấu hình web, README web, và file liên quan đến app web phải nằm trong `web/`.

Kiến trúc mong muốn:

```text
web/
  package.json
  README.md
  frontend/
    ... Next.js app ...
  backend/
    ... FastAPI app ...
```

Lưu ý: nếu thấy cấu trúc hiện tại có `web/frontend/backend` do tạo nhầm, hãy tạo đúng `web/backend` và không dùng thư mục nested sai đó.

## Yêu cầu chạy local

Trong `web/package.json`, cần có script để khi đứng tại `web/` chạy:

```bash
npm run dev:all
```

thì chạy đồng thời frontend và backend.

Đề xuất script:

```json
{
  "scripts": {
    "dev:frontend": "npm --prefix frontend run dev -- --hostname 127.0.0.1 --port 3000",
    "dev:backend": "python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 --app-dir backend",
    "dev:all": "concurrently \"npm run dev:backend\" \"npm run dev:frontend\""
  },
  "devDependencies": {
    "concurrently": "latest"
  }
}
```

Có thể điều chỉnh nếu cần, nhưng vẫn phải giữ mục tiêu: `npm run dev:all` chạy được cả frontend và backend từ `web/`.

## Backend

Dùng **FastAPI** thay vì Flask vì API rõ, có validation tốt, dễ có docs tự động. Backend không được train model lại.

Backend cần:

1. Load model từ:

```text
../models/energy_theft_model_bundle.pkl
```

tính từ repo root, hoặc `../../models/energy_theft_model_bundle.pkl` nếu code chạy trong `web/backend`.

2. Accept input là file CSV upload, mẫu là:

```text
data/test/test_raw_15_percent.csv
```

CSV input có thể có nhiều khách hàng. Format cần hỗ trợ:

- Bắt buộc có đủ 1034 cột ngày đúng tên như full dataset.
- Nên hỗ trợ `CONS_NO`.
- `FLAG` là optional. Nếu có thì dùng để hiển thị ground truth và TP/TN/FP/FN; tuyệt đối không dùng `FLAG` làm feature hoặc để xử lý dữ liệu.

3. Vì model nhận 159 engineered features chứ không nhận 1034 ngày raw, backend phải chạy pipeline inference:

```text
uploaded raw CSV
-> validate columns
-> preprocessing logic tương thích preprocessing_v2.py
-> feature engineering logic tương thích feature.py
-> chọn đúng active_feature_cols trong bundle
-> LightGBM predict_proba
-> so score với threshold
-> trả JSON
```

Ưu tiên reuse các function hiện có trong `src/preprocessing_v2.py` và `src/feature.py`. Nếu các function hiện tại quá gắn với file path global, tạo wrapper trong `web/backend/inference_pipeline.py` bằng cách import các hàm cấp thấp như `prepare_columns`, `create_quality_features`, `clean_negative_values`, `clip_outliers`, `fill_missing_values`, `create_statistical_features`, `add_*_features`, `merge_quality_features`, `add_interaction_features`, `validate_features`. Không sửa bừa core pipeline nếu không cần.

4. API endpoints đề xuất:

- `GET /health`
- `GET /model-info`
- `GET /sample-info`
- `POST /predict/upload`

`POST /predict/upload` trả về:

```json
{
  "model": "LightGBM benchmark",
  "threshold": 0.4368,
  "rows": 6356,
  "summary": {
    "predicted_theft": 1213,
    "predicted_normal": 5143,
    "average_score": 0.18
  },
  "records": [
    {
      "cons_no": "...",
      "score": 0.62,
      "threshold": 0.4368,
      "prediction": "Suspected theft",
      "risk_level": "High",
      "actual_label": 1,
      "outcome": "TP",
      "feature_summary": {
        "mean_consumption": 31.2,
        "zero_ratio_clean": 0.05,
        "missing_ratio_raw": 0.12,
        "recent_90_mean": 28.7,
        "mean_abs_daily_change": 8.4
      }
    }
  ]
}
```

Risk level gợi ý:

- `Low`: score < 0.30
- `Medium`: 0.30 <= score < threshold
- `High`: threshold <= score < 0.70
- `Critical`: score >= 0.70

Có thể điều chỉnh wording nhưng phải nhất quán.

5. Backend phải có CORS cho frontend local `http://127.0.0.1:3000` và `http://localhost:3000`.

6. Backend dependencies nên có file riêng, ví dụ:

```text
web/backend/requirements.txt
```

bao gồm tối thiểu: `fastapi`, `uvicorn`, `python-multipart`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`.

## Frontend

Dùng **Next.js** trong `web/frontend`. UI cần đẹp, chuyên nghiệp, đầy đủ, không thừa. Không làm landing page marketing; màn hình đầu tiên phải là web demo thật.

Yêu cầu UI:

- Upload CSV panel rõ ràng.
- Có nút dùng sample/test CSV nếu backend hỗ trợ đọc sample từ `data/test/test_raw_15_percent.csv`.
- Hiển thị model card:
  - Model: LightGBM benchmark.
  - Threshold: khoảng `0.4368`.
  - Feature count: `159`.
  - Test metric: F2 `0.5176`, Recall `0.6458`, PR-AUC `0.4489`, ROC-AUC `0.8387`.
- Sau khi predict, có dashboard:
  - Tổng số khách hàng trong file.
  - Số predicted theft / normal.
  - Average score.
  - Nếu CSV có `FLAG`: confusion summary TP/TN/FP/FN.
- Bảng kết quả từng khách hàng:
  - `CONS_NO`
  - score
  - risk level
  - prediction
  - actual label nếu có
  - outcome nếu có
- Khi chọn một khách hàng:
  - Hiển thị consumption line chart từ 1034 ngày raw.
  - Hiển thị score/risk card.
  - Hiển thị feature summary dễ hiểu.
- Có minh họa/visual:
  - score distribution chart
  - consumption time-series chart
  - confusion mini chart nếu có ground truth

Design:

- Giao diện sáng, chuyên nghiệp, giống dashboard phân tích rủi ro.
- Không dùng quá nhiều màu; dùng màu theo risk level có kiểm soát.
- Responsive desktop trước, mobile vẫn không vỡ layout.
- Dùng icon nếu có thư viện như `lucide-react`.
- Text ngắn, tập trung vào thao tác và kết quả; không nhồi lý thuyết dài.

## Ràng buộc quan trọng

- Web chỉ dùng LightGBM, không cho chọn LR/RF.
- Không train lại model trong web runtime.
- Không dùng `FLAG` trong preprocessing/feature engineering/predict; chỉ dùng để hiển thị ground truth nếu upload CSV có nhãn.
- Không sửa kết quả model, không tune threshold.
- Không copy toàn bộ file lớn vào `web/` nếu không cần. Backend có thể đọc `data/test/test_raw_15_percent.csv` từ project root làm sample.
- Nếu cần tạo sample nhỏ cho UI, chỉ tạo một file nhỏ vài dòng trong `web/backend/data/`, nhưng vẫn giữ input chính là upload CSV.
- Nếu phải sửa `src/preprocessing_v2.py` hoặc `src/feature.py` để reuse sạch hơn, hãy giữ backward compatibility với pipeline hiện tại và giải thích trong final response.

## Verification bắt buộc

Sau khi implement, hãy chạy hoặc hướng dẫn rõ nếu thiếu dependency:

```bash
cd web
npm install
npm run dev:all
```

Kiểm tra:

- Backend health endpoint hoạt động.
- Upload `../data/test/test_raw_15_percent.csv` predict được.
- Frontend gọi được backend.
- Kết quả không bị crash khi CSV có nhiều dòng.
- Nếu upload CSV thiếu cột ngày, backend trả lỗi rõ ràng.

Cuối cùng, trả lời ngắn gọn:

- Đã tạo/sửa file nào trong `web/`.
- Cách chạy.
- Endpoint chính.
- Những hạn chế còn lại nếu có.
