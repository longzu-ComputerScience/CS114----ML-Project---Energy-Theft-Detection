"""
Thay đổi:
- Không xóa duplicate comsumption series: Có thể là 2 khách hàng khác nhau có chuỗi tiêu thụ khác nhau, 
nhất là là khách hàng inactive hoặc nhiều ngày bằng 0. Nếu mà xóa tự động hết thì có thể làm mất dữ liệu 
thật và làm lệch phân phối normal/theft. 
- Thêm quality_features.csv: Theo như EDA, thì thấy missing, zero, outlier không chỉ là dữ liệu bẩn. Nó
có thể là tín hiệu quan trọng cho bài toán. Nên có thể lưu ra riêng rồi clean sau 
- Giữ lại thông tin missing trước khi fill: thêm missing_count_raw, missing_ratio_raw, max_missing_streak_raw, 
vì theo tui nghĩ có thể một khách hàng có nhiều ngày mất dữ liệu hoặc mất dữ liệu liên tục dài ngày có thể là
một pattern. Nên cứ lưu trước khi fill. 
- Không dùng FLAG để fill missing: fill theo thời gian phải bằng median của chính khách hàng, hoặc global median,
bởi vì nếu dùng FLAG (tức là target) để điền missing thì chẳng khác gì là biết trước đáp án và dataset còn có thể 
được chia ra train/test thì việc này có thể khiến cho model có thể nhìn trước được đáp án?
- Zero - ratio thì chia ra 2 loại: bản cũ thì có zero_ratio (zero / tổng số ngày), thì bản mới sẽ có thêm 
zero_ratio_observed tức là (zero / tổng số ngày thực sự có dữ liệu (không tính missing)) vì 2 tỷ lệ này khác nhau
và có thể khai thác được
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data set.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLEANED_OUTPUT_PATH = PROCESSED_DIR / "cleaned.csv"
QUALITY_OUTPUT_PATH = PROCESSED_DIR / "quality_features.csv"

OUTLIER_UPPER = 500
ZERO_RATIO_INACTIVE = 0.80
FILL_LIMIT_DAYS = 30


def log(message: str) -> None:
    """In thông báo theo format thống nhất."""
    print(f"[PREPROCESS] {message}")


def print_step(step: int, title: str) -> None:
    """In tiêu đề cho từng bước xử lý."""
    print(f"\n{'=' * 70}")
    print(f"STEP {step}: {title}")
    print(f"{'=' * 70}")


def get_consumption_columns(df: pd.DataFrame) -> list[str]:
    """Lấy danh sách các cột ngày tiêu thụ, bỏ qua cột định danh, nhãn và cột phụ."""
    ignored_cols = {
        "CONS_NO",
        "FLAG",
        "missing_count_raw",
        "missing_ratio_raw",
        "zero_count_raw",
        "zero_ratio_total_raw",
        "zero_ratio_observed_raw",
        "negative_count_raw",
        "negative_ratio_raw",
        "outlier_count_raw",
        "outlier_ratio_raw",
        "max_consumption_raw",
        "max_missing_streak_raw",
        "max_zero_streak_raw",
        "is_inactive_raw",
    }

    return [col for col in df.columns if col not in ignored_cols]


def sort_date_columns(consumption_cols: list[str]) -> list[str]:
    """Sắp xếp các cột ngày theo đúng thứ tự thời gian."""
    parsed_cols = []

    for col in consumption_cols:
        date = pd.to_datetime(col, format="%m/%d/%Y", errors="coerce")
        if pd.notna(date):
            parsed_cols.append((col, date))

    parsed_cols.sort(key=lambda x: x[1])
    return [col for col, _ in parsed_cols]


def max_streak(values: np.ndarray) -> int:
    """Tính độ dài chuỗi True liên tiếp lớn nhất."""
    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu thô từ file CSV."""
    print_step(1, "Đọc dữ liệu")

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu thô: {path}")

    df = pd.read_csv(path)

    log(f"Path: {path}")
    log(f"Shape: {df.shape[0]:,} dòng x {df.shape[1]:,} cột")
    log(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def prepare_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Kiểm tra cột bắt buộc, ép kiểu numeric và sắp xếp các cột ngày."""
    print_step(2, "Chuẩn hóa cột")

    required_cols = {"CONS_NO", "FLAG"}
    missing_required = required_cols - set(df.columns)

    if missing_required:
        raise ValueError(f"Thiếu cột bắt buộc: {missing_required}")

    consumption_cols = get_consumption_columns(df)
    consumption_cols = sort_date_columns(consumption_cols)

    if not consumption_cols:
        raise ValueError("Không tìm thấy cột ngày tiêu thụ hợp lệ.")

    for col in consumption_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["CONS_NO", "FLAG"] + consumption_cols].copy()

    log(f"Số cột ngày tiêu thụ: {len(consumption_cols):,}")
    log(f"Ngày đầu: {consumption_cols[0]}")
    log(f"Ngày cuối: {consumption_cols[-1]}")

    return df, consumption_cols


def remove_duplicate_customers(df: pd.DataFrame, consumption_cols: list[str]) -> pd.DataFrame:
    """Xóa CONS_NO trùng, nhưng chỉ báo cáo chuỗi tiêu thụ trùng."""
    print_step(3, "Kiểm tra dữ liệu trùng")

    duplicated_cons_no = df["CONS_NO"].duplicated().sum()
    log(f"CONS_NO bị trùng: {duplicated_cons_no:,}")

    if duplicated_cons_no > 0:
        df = df.drop_duplicates(subset="CONS_NO", keep="first").reset_index(drop=True)
        log(f"Đã giữ dòng đầu tiên cho mỗi CONS_NO. Shape mới: {df.shape}")

    duplicated_series = df.duplicated(subset=consumption_cols).sum()
    log(f"Chuỗi tiêu thụ bị trùng hoàn toàn: {duplicated_series:,}")
    log("Không tự động xóa chuỗi tiêu thụ trùng để tránh làm lệch phân phối dữ liệu.")

    return df


def create_quality_features(df: pd.DataFrame, consumption_cols: list[str]) -> pd.DataFrame:
    """Tạo các tín hiệu chất lượng dữ liệu thô trước khi làm sạch."""
    print_step(4, "Tạo quality features từ dữ liệu thô")

    consumption = df[consumption_cols]

    missing_mask = consumption.isna()
    zero_mask = consumption.eq(0)
    negative_mask = consumption.lt(0)
    outlier_mask = consumption.gt(OUTLIER_UPPER)

    observed_count = consumption.notna().sum(axis=1)
    zero_count = zero_mask.sum(axis=1)

    quality_df = pd.DataFrame(
        {
            "CONS_NO": df["CONS_NO"],
            "FLAG": df["FLAG"],
            "missing_count_raw": missing_mask.sum(axis=1),
            "missing_ratio_raw": missing_mask.mean(axis=1),
            "zero_count_raw": zero_count,
            "zero_ratio_total_raw": zero_count / len(consumption_cols),
            "zero_ratio_observed_raw": zero_count / observed_count.replace(0, np.nan),
            "negative_count_raw": negative_mask.sum(axis=1),
            "negative_ratio_raw": negative_mask.mean(axis=1),
            "outlier_count_raw": outlier_mask.sum(axis=1),
            "outlier_ratio_raw": outlier_mask.mean(axis=1),
            "max_consumption_raw": consumption.max(axis=1, skipna=True),
            "max_missing_streak_raw": missing_mask.apply(
                lambda row: max_streak(row.to_numpy()), axis=1
            ),
            "max_zero_streak_raw": zero_mask.apply(
                lambda row: max_streak(row.to_numpy()), axis=1
            ),
        }
    )

    quality_df["zero_ratio_observed_raw"] = quality_df["zero_ratio_observed_raw"].fillna(0)
    quality_df["is_inactive_raw"] = (
        quality_df["zero_ratio_total_raw"] >= ZERO_RATIO_INACTIVE
    ).astype(np.int8)

    log(f"Missing trung bình: {quality_df['missing_ratio_raw'].mean():.3f}")
    log(f"Zero trung bình: {quality_df['zero_ratio_total_raw'].mean():.3f}")
    log(f"Số khách hàng inactive: {quality_df['is_inactive_raw'].sum():,}")
    log(f"Số khách hàng có giá trị âm: {(quality_df['negative_count_raw'] > 0).sum():,}")
    log(f"Số khách hàng có outlier > {OUTLIER_UPPER}: {(quality_df['outlier_count_raw'] > 0).sum():,}")

    return quality_df


def clean_negative_values(df: pd.DataFrame, consumption_cols: list[str]) -> pd.DataFrame:
    """Chuyển giá trị tiêu thụ âm thành NaN."""
    print_step(5, "Xử lý giá trị âm")

    negative_mask = df[consumption_cols].lt(0)
    negative_count = negative_mask.sum().sum()

    log(f"Số ô có giá trị âm: {negative_count:,}")

    if negative_count > 0:
        df[consumption_cols] = df[consumption_cols].where(~negative_mask)

    return df


def clip_outliers(df: pd.DataFrame, consumption_cols: list[str]) -> pd.DataFrame:
    """Giới hạn các giá trị tiêu thụ quá lớn về ngưỡng cho trước."""
    print_step(6, "Xử lý outlier")

    outlier_count = df[consumption_cols].gt(OUTLIER_UPPER).sum().sum()

    log(f"Số ô > {OUTLIER_UPPER}: {outlier_count:,}")

    if outlier_count > 0:
        df[consumption_cols] = df[consumption_cols].clip(upper=OUTLIER_UPPER)

    stats = df[consumption_cols].stack().describe()
    log(f"Min: {stats['min']:.3f}")
    log(f"Median: {stats['50%']:.3f}")
    log(f"Mean: {stats['mean']:.3f}")
    log(f"Max: {stats['max']:.3f}")

    return df


def fill_missing_values(df: pd.DataFrame, consumption_cols: list[str]) -> pd.DataFrame:
    """Điền missing value theo chuỗi thời gian của từng khách hàng."""
    print_step(7, "Xử lý missing value")

    missing_before = df[consumption_cols].isna().sum().sum()
    total_cells = df[consumption_cols].size

    log(f"Missing trước khi fill: {missing_before:,} ({100 * missing_before / total_cells:.2f}%)")

    transposed = df[consumption_cols].T
    filled = transposed.ffill(limit=FILL_LIMIT_DAYS).bfill(limit=FILL_LIMIT_DAYS)

    missing_after_time_fill = filled.isna().sum().sum()
    log(f"Còn NaN sau ffill/bfill: {missing_after_time_fill:,}")

    if missing_after_time_fill > 0:
        filled_customer_view = filled.T
        customer_median = filled_customer_view.median(axis=1)
        filled_customer_view = filled_customer_view.T.fillna(customer_median).T
        filled = filled_customer_view.T

    missing_after_customer_median = filled.isna().sum().sum()
    log(f"Còn NaN sau median từng khách hàng: {missing_after_customer_median:,}")

    if missing_after_customer_median > 0:
        global_median = df[consumption_cols].stack().median()
        filled = filled.fillna(global_median)
        log(f"Global median dùng để fill phần còn lại: {global_median:.3f}")

    df[consumption_cols] = filled.T

    missing_after = df[consumption_cols].isna().sum().sum()
    log(f"Missing sau khi fill: {missing_after:,}")

    return df


def validate_cleaned_data(df: pd.DataFrame, consumption_cols: list[str]) -> None:
    """Kiểm tra dữ liệu sau preprocessing."""
    print_step(8, "Kiểm tra dữ liệu sau xử lý")

    missing_count = df[consumption_cols].isna().sum().sum()
    negative_count = df[consumption_cols].lt(0).sum().sum()
    over_upper_count = df[consumption_cols].gt(OUTLIER_UPPER).sum().sum()

    log(f"Missing còn lại: {missing_count:,}")
    log(f"Giá trị âm còn lại: {negative_count:,}")
    log(f"Giá trị > {OUTLIER_UPPER} còn lại: {over_upper_count:,}")

    if missing_count > 0:
        raise ValueError("Dữ liệu vẫn còn missing value.")

    if negative_count > 0:
        raise ValueError("Dữ liệu vẫn còn giá trị âm.")

    if over_upper_count > 0:
        raise ValueError("Dữ liệu vẫn còn outlier vượt ngưỡng.")


def save_outputs(
    cleaned_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    cleaned_path: Path = CLEANED_OUTPUT_PATH,
    quality_path: Path = QUALITY_OUTPUT_PATH,
) -> None:
    """Lưu dữ liệu đã xử lý và quality features ra file CSV."""
    print_step(9, "Lưu kết quả")

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(cleaned_path, index=False)
    quality_df.to_csv(quality_path, index=False)

    log(f"Saved cleaned data: {cleaned_path}")
    log(f"Saved quality features: {quality_path}")
    log(f"Cleaned shape: {cleaned_df.shape}")
    log(f"Quality features shape: {quality_df.shape}")


def run_preprocessing() -> None:
    """pipeline preprocessing"""
    df = load_data()
    df, consumption_cols = prepare_columns(df)
    df = remove_duplicate_customers(df, consumption_cols)

    quality_df = create_quality_features(df, consumption_cols)

    df = clean_negative_values(df, consumption_cols)
    df = clip_outliers(df, consumption_cols)
    df = fill_missing_values(df, consumption_cols)

    validate_cleaned_data(df, consumption_cols)
    save_outputs(df, quality_df)

    log("Hoàn tất preprocessing.")


if __name__ == "__main__":
    run_preprocessing()
