# Energy Theft Detection - Web Demo

Web demo là phần phụ của repo, dùng để minh họa mô hình LightGBM đã train sẵn.

- Frontend: Next.js
- Backend: FastAPI
- Model: `../models/energy_theft_model_bundle.pkl`
- CSV mẫu: `../data/test/test_raw_15_percent.csv`

## Chạy Nhanh

Từ root repo:

```powershell
npm run install:web
npm run dev:all
```

Hoặc từ thư mục `web/`:

```powershell
npm install
npm --prefix frontend install
python -m pip install -r backend/requirements.txt
npm run dev:all
```

Địa chỉ mặc định:

- Frontend: `http://127.0.0.1:3000`
- Backend API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

## Input Và Output

Input chính là CSV raw theo format dataset gốc:

- 1 dòng = 1 khách hàng
- 1,034 cột ngày tiêu thụ
- `CONS_NO` nên có để định danh khách hàng
- `FLAG` là optional, chỉ dùng để hiển thị ground truth nếu có

Backend sẽ chạy:

```text
raw CSV -> preprocessing -> feature engineering -> LightGBM predict
```

Output gồm theft score, threshold, prediction, risk level, feature summary và ground truth/outcome nếu CSV có `FLAG`.

## API Chính

| Method | Path | Mục đích |
|---|---|---|
| `GET` | `/health` | Kiểm tra backend |
| `GET` | `/model-info` | Thông tin model và metric |
| `GET` | `/sample-info` | Thông tin CSV mẫu |
| `POST` | `/predict/upload` | Upload CSV để predict |
| `POST` | `/predict/sample` | Predict trên CSV mẫu |

Backend không train lại model ở runtime.
