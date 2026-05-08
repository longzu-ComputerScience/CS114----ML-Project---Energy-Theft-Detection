from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLEANED_PATH = PROCESSED_DIR / "cleaned.csv"
QUALITY_PATH = PROCESSED_DIR / "quality_features.csv"
FEATURE_OUTPUT_PATH = PROCESSED_DIR / "features.csv"

ID_COL = "CONS_NO"
TARGET_COL = "FLAG"

LOW_CONSUMPTION_THRESHOLD = 1.0
ABS_CHANGE_THRESHOLD = 5.0
REL_CHANGE_THRESHOLD = 0.5

RAW_QUALITY_COLS = {
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


def log(message: str) -> None:
    """In thông báo theo format thống nhất."""
    print(f"[FEATURE] {message}")


def print_step(step: int, title: str) -> None:
    """In tiêu đề cho từng bước."""
    print(f"\n{'=' * 70}")
    print(f"STEP {step}: {title}")
    print(f"{'=' * 70}")


def get_consumption_columns(df: pd.DataFrame) -> list[str]:
    """Lấy và sắp xếp các cột ngày tiêu thụ."""
    ignored_cols = {ID_COL, TARGET_COL} | RAW_QUALITY_COLS
    candidate_cols = [col for col in df.columns if col not in ignored_cols]

    parsed = []
    for col in candidate_cols:
        date = pd.to_datetime(col, format="%m/%d/%Y", errors="coerce")
        if pd.notna(date):
            parsed.append((col, date))

    parsed.sort(key=lambda x: x[1])
    return [col for col, _ in parsed]


def max_streak_1d(mask: np.ndarray) -> int:
    """Tính độ dài chuỗi True liên tiếp lớn nhất."""
    best = 0
    current = 0

    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def max_streak_2d(mask: np.ndarray) -> np.ndarray:
    """Tính streak lớn nhất cho từng khách hàng."""
    return np.array([max_streak_1d(row) for row in mask], dtype=np.int16)


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    eps: float = 1e-6,
    clip_upper: float | None = None,
) -> np.ndarray:
    """Chia an toàn và có thể giới hạn giá trị quá lớn."""
    ratio = numerator / (denominator + eps)

    if clip_upper is not None:
        ratio = np.clip(ratio, 0, clip_upper)

    return ratio


def log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Tính log-ratio ổn định hơn ratio thường khi mẫu số nhỏ."""
    return np.log1p(numerator) - np.log1p(denominator)


def trend_slope(values: np.ndarray) -> np.ndarray:
    """Tính slope xu hướng tiêu thụ theo thời gian cho từng khách hàng."""
    n_days = values.shape[1]
    t = np.arange(n_days, dtype=np.float32)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)

    return values @ t_centered / denom


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DatetimeIndex]:
    """Đọc cleaned data, quality features và lấy danh sách cột ngày."""
    print_step(1, "Đọc dữ liệu đầu vào")

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {CLEANED_PATH}")

    if not QUALITY_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {QUALITY_PATH}")

    cleaned_df = pd.read_csv(CLEANED_PATH)
    quality_df = pd.read_csv(QUALITY_PATH)

    consumption_cols = get_consumption_columns(cleaned_df)
    dates = pd.to_datetime(consumption_cols, format="%m/%d/%Y")

    log(f"cleaned_df shape: {cleaned_df.shape}")
    log(f"quality_df shape: {quality_df.shape}")
    log(f"Số ngày tiêu thụ: {len(consumption_cols):,}")

    return cleaned_df, quality_df, consumption_cols, dates


def create_statistical_features(
    cleaned_df: pd.DataFrame,
    consumption_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Tạo feature thống kê cơ bản và log-transform từ chuỗi tiêu thụ."""
    print_step(2, "Tạo statistical features")

    X = cleaned_df[consumption_cols].to_numpy(dtype=np.float32)
    X_log = np.log1p(X)

    features = cleaned_df[[ID_COL, TARGET_COL]].copy()

    features["mean_consumption"] = X.mean(axis=1)
    features["median_consumption"] = np.median(X, axis=1)
    features["std_consumption"] = X.std(axis=1)
    features["min_consumption"] = X.min(axis=1)
    features["max_consumption"] = X.max(axis=1)
    features["sum_consumption"] = X.sum(axis=1)

    q25 = np.percentile(X, 25, axis=1)
    q75 = np.percentile(X, 75, axis=1)
    q90 = np.percentile(X, 90, axis=1)
    q95 = np.percentile(X, 95, axis=1)

    features["q25_consumption"] = q25
    features["q75_consumption"] = q75
    features["q90_consumption"] = q90
    features["q95_consumption"] = q95
    features["iqr_consumption"] = q75 - q25
    features["range_consumption"] = features["max_consumption"] - features["min_consumption"]

    features["cv_consumption"] = safe_ratio(
        features["std_consumption"].to_numpy(),
        features["mean_consumption"].to_numpy(),
        clip_upper=50,
    )

    features["mean_log1p_consumption"] = X_log.mean(axis=1)
    features["std_log1p_consumption"] = X_log.std(axis=1)
    features["max_log1p_consumption"] = X_log.max(axis=1)

    log(f"Feature shape: {features.shape}")
    return features, X


def add_zero_low_features(features: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Tạo feature về ngày zero và ngày tiêu thụ thấp."""
    print_step(3, "Tạo zero và low-consumption features")

    zero_mask = X == 0
    low_mask = X <= LOW_CONSUMPTION_THRESHOLD

    features["zero_count_clean"] = zero_mask.sum(axis=1)
    features["zero_ratio_clean"] = zero_mask.mean(axis=1)

    features["low_consumption_count"] = low_mask.sum(axis=1)
    features["low_consumption_ratio"] = low_mask.mean(axis=1)

    features["max_zero_streak_clean"] = max_streak_2d(zero_mask)
    features["max_low_streak_clean"] = max_streak_2d(low_mask)

    log(f"Feature shape: {features.shape}")
    return features


def add_temporal_features(features: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Tạo feature so sánh tiêu thụ gần đây với quá khứ."""
    print_step(4, "Tạo temporal features")

    def window_mean(values: np.ndarray, n_recent_days: int) -> np.ndarray:
        n_recent_days = min(n_recent_days, values.shape[1])
        return values[:, -n_recent_days:].mean(axis=1)

    features["first_30_mean"] = X[:, :30].mean(axis=1)
    features["first_90_mean"] = X[:, :90].mean(axis=1)

    features["recent_30_mean"] = window_mean(X, 30)
    features["recent_90_mean"] = window_mean(X, 90)
    features["recent_180_mean"] = window_mean(X, 180)

    mid = X.shape[1] // 2
    features["first_half_mean"] = X[:, :mid].mean(axis=1)
    features["second_half_mean"] = X[:, mid:].mean(axis=1)

    features["recent30_to_first90_ratio"] = safe_ratio(
        features["recent_30_mean"].to_numpy(),
        features["first_90_mean"].to_numpy(),
        clip_upper=50,
    )

    features["second_half_to_first_half_ratio"] = safe_ratio(
        features["second_half_mean"].to_numpy(),
        features["first_half_mean"].to_numpy(),
        clip_upper=50,
    )

    features["recent30_first90_log_ratio"] = log_ratio(
        features["recent_30_mean"].to_numpy(),
        features["first_90_mean"].to_numpy(),
    )

    features["second_first_half_log_ratio"] = log_ratio(
        features["second_half_mean"].to_numpy(),
        features["first_half_mean"].to_numpy(),
    )

    features["recent30_minus_first90"] = (
        features["recent_30_mean"] - features["first_90_mean"]
    )

    features["second_half_minus_first_half"] = (
        features["second_half_mean"] - features["first_half_mean"]
    )

    features["trend_slope"] = trend_slope(X)

    log(f"Feature shape: {features.shape}")
    return features


def add_volatility_features(features: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Tạo feature về dao động, spike và drop."""
    print_step(5, "Tạo volatility, spike và drop features")

    prev = X[:, :-1]
    nxt = X[:, 1:]

    diff = nxt - prev
    abs_diff = np.abs(diff)

    features["mean_daily_change"] = diff.mean(axis=1)
    features["mean_abs_daily_change"] = abs_diff.mean(axis=1)
    features["max_abs_daily_change"] = abs_diff.max(axis=1)
    features["std_daily_change"] = diff.std(axis=1)

    spike_mask = (
        (diff >= ABS_CHANGE_THRESHOLD)
        & (diff >= REL_CHANGE_THRESHOLD * np.maximum(prev, 1.0))
    )

    drop_mask = (
        (diff <= -ABS_CHANGE_THRESHOLD)
        & (-diff >= REL_CHANGE_THRESHOLD * np.maximum(prev, 1.0))
    )

    features["spike_count"] = spike_mask.sum(axis=1)
    features["spike_ratio"] = spike_mask.mean(axis=1)
    features["drop_count"] = drop_mask.sum(axis=1)
    features["drop_ratio"] = drop_mask.mean(axis=1)

    features["max_spike_streak"] = max_streak_2d(spike_mask)
    features["max_drop_streak"] = max_streak_2d(drop_mask)

    features["drop_to_spike_ratio"] = safe_ratio(
        features["drop_count"].to_numpy(),
        features["spike_count"].to_numpy(),
        clip_upper=50,
    )

    log(f"Feature shape: {features.shape}")
    return features


def add_monthly_features(
    features: pd.DataFrame,
    X: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Tạo feature tiêu thụ trung bình và độ lệch chuẩn theo từng tháng."""
    print_step(6, "Tạo monthly features")

    month_values = dates.month

    for month in range(1, 13):
        month_mask = month_values == month

        if month_mask.sum() > 0:
            features[f"month_{month:02d}_mean"] = X[:, month_mask].mean(axis=1)
            features[f"month_{month:02d}_std"] = X[:, month_mask].std(axis=1)

    monthly_mean_cols = [
        f"month_{month:02d}_mean"
        for month in range(1, 13)
        if f"month_{month:02d}_mean" in features.columns
    ]

    features["monthly_mean_range"] = (
        features[monthly_mean_cols].max(axis=1)
        - features[monthly_mean_cols].min(axis=1)
    )

    features["monthly_mean_std"] = features[monthly_mean_cols].std(axis=1)

    log(f"Feature shape: {features.shape}")
    return features


def merge_quality_features(
    features: pd.DataFrame,
    quality_df: pd.DataFrame,
    n_days: int,
) -> pd.DataFrame:
    """Merge và bổ sung quality features từ dữ liệu thô."""
    print_step(7, "Merge quality features")

    quality_to_merge = quality_df.copy()

    quality_to_merge["is_all_missing_raw"] = (
        quality_to_merge["missing_count_raw"] == n_days
    ).astype(np.int8)

    quality_to_merge["is_high_missing_raw"] = (
        quality_to_merge["missing_ratio_raw"] >= 0.50
    ).astype(np.int8)

    quality_to_merge["is_very_high_missing_raw"] = (
        quality_to_merge["missing_ratio_raw"] >= 0.80
    ).astype(np.int8)

    quality_to_merge["has_outlier_raw"] = (
        quality_to_merge["outlier_count_raw"] > 0
    ).astype(np.int8)

    quality_to_merge["max_consumption_raw_missing"] = (
        quality_to_merge["max_consumption_raw"].isna()
    ).astype(np.int8)

    quality_to_merge["max_consumption_raw"] = (
        quality_to_merge["max_consumption_raw"].fillna(0)
    )

    quality_to_merge["log1p_outlier_count_raw"] = np.log1p(
        quality_to_merge["outlier_count_raw"]
    )

    quality_to_merge["log1p_max_consumption_raw"] = np.log1p(
        quality_to_merge["max_consumption_raw"]
    )

    quality_to_merge["log1p_missing_count_raw"] = np.log1p(
        quality_to_merge["missing_count_raw"]
    )

    quality_to_merge["log1p_zero_count_raw"] = np.log1p(
        quality_to_merge["zero_count_raw"]
    )

    quality_feature_cols = [col for col in quality_to_merge.columns if col != TARGET_COL]

    features = features.merge(
        quality_to_merge[quality_feature_cols],
        on=ID_COL,
        how="left",
    )

    log(f"Feature shape sau merge: {features.shape}")
    return features


def validate_features(features: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra và xử lý NaN/inf trong feature cuối."""
    print_step(8, "Kiểm tra feature cuối")

    feature_cols = [col for col in features.columns if col not in [ID_COL, TARGET_COL]]

    n_missing = features[feature_cols].isna().sum().sum()
    n_inf = np.isinf(features[feature_cols].to_numpy(dtype=np.float64)).sum()

    log(f"Số feature: {len(feature_cols):,}")
    log(f"Missing trong features: {n_missing:,}")
    log(f"Inf trong features: {n_inf:,}")

    if n_missing > 0:
        features[feature_cols] = features[feature_cols].fillna(0)

    if n_inf > 0:
        features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], 0)

    return features


def save_features(features: pd.DataFrame) -> None:
    """Lưu features ra file CSV."""
    print_step(9, "Lưu features")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_OUTPUT_PATH, index=False)

    log(f"Saved: {FEATURE_OUTPUT_PATH}")
    log(f"Shape: {features.shape}")


def run_feature_engineering() -> None:
    """Chạy toàn bộ pipeline Feature Engineering."""
    cleaned_df, quality_df, consumption_cols, dates = load_inputs()

    features, X = create_statistical_features(cleaned_df, consumption_cols)
    features = add_zero_low_features(features, X)
    features = add_temporal_features(features, X)
    features = add_volatility_features(features, X)
    features = add_monthly_features(features, X, dates)
    features = merge_quality_features(features, quality_df, len(consumption_cols))
    features = validate_features(features)
    save_features(features)

    log("Hoàn tất Feature Engineering.")


if __name__ == "__main__":
    run_feature_engineering()
