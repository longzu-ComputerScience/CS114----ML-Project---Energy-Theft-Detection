"""
Inference pipeline for the web demo.
Reuses low-level functions from src/preprocessing_v2.py and src/feature.py
to replicate the exact same transformations applied during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the project root's `src/` is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # web/backend -> web -> repo root
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import low-level helpers from the existing pipeline
from preprocessing_v2 import (
    get_consumption_columns as _preprocess_get_consumption_columns,
    sort_date_columns,
    max_streak,
    clean_negative_values,
    clip_outliers,
    fill_missing_values,
    OUTLIER_UPPER,
    ZERO_RATIO_INACTIVE,
)
from feature import (
    get_consumption_columns as _feature_get_consumption_columns,
    max_streak_1d,
    max_streak_2d,
    safe_ratio,
    log_ratio,
    trend_slope,
    rolling_mean_2d,
    rolling_std_2d,
    ID_COL,
    TARGET_COL,
    LOW_CONSUMPTION_THRESHOLD,
    ABS_CHANGE_THRESHOLD,
    REL_CHANGE_THRESHOLD,
    ROLLING_WINDOWS,
    N_SEGMENTS,
    RAW_QUALITY_COLS,
)


def validate_upload_columns(df: pd.DataFrame) -> list[str]:
    """Validate that the uploaded CSV has the required date columns (1034 days)."""
    if "CONS_NO" not in df.columns:
        raise ValueError("CSV must contain a CONS_NO column.")

    # Get consumption columns using the preprocessing logic
    ignored = {"CONS_NO", "FLAG"} | RAW_QUALITY_COLS
    candidate_cols = [c for c in df.columns if c not in ignored]
    date_cols = []
    for c in candidate_cols:
        d = pd.to_datetime(c, format="%m/%d/%Y", errors="coerce")
        if pd.notna(d):
            date_cols.append((c, d))
    date_cols.sort(key=lambda x: x[1])
    consumption_cols = [c for c, _ in date_cols]

    if len(consumption_cols) < 100:
        raise ValueError(
            f"Expected ~1034 date columns but found only {len(consumption_cols)}. "
            "Please upload a CSV with the full daily consumption columns."
        )

    return consumption_cols


def run_preprocessing(df: pd.DataFrame, consumption_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replicate preprocessing_v2 steps on the uploaded dataframe:
    1. Coerce numeric
    2. Create quality features (before cleaning)
    3. Clean negatives -> clip outliers -> fill missing
    Returns (cleaned_df, quality_df)
    """
    # Coerce to numeric
    for col in consumption_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Create quality features from raw data BEFORE cleaning ---
    consumption = df[consumption_cols]
    missing_mask = consumption.isna()
    zero_mask = consumption.eq(0)
    negative_mask = consumption.lt(0)
    outlier_mask = consumption.gt(OUTLIER_UPPER)
    observed_count = consumption.notna().sum(axis=1)
    zero_count = zero_mask.sum(axis=1)

    quality_df = pd.DataFrame({
        "CONS_NO": df["CONS_NO"],
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
    })
    quality_df["zero_ratio_observed_raw"] = quality_df["zero_ratio_observed_raw"].fillna(0)
    quality_df["is_inactive_raw"] = (
        quality_df["zero_ratio_total_raw"] >= ZERO_RATIO_INACTIVE
    ).astype(np.int8)

    # If FLAG is present, add it to quality_df for later display (NOT for feature use)
    if "FLAG" in df.columns:
        quality_df["FLAG"] = df["FLAG"]

    # --- Clean ---
    df = clean_negative_values(df, consumption_cols)
    df = clip_outliers(df, consumption_cols)
    df = fill_missing_values(df, consumption_cols)

    return df, quality_df


def run_feature_engineering(
    cleaned_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    consumption_cols: list[str],
) -> pd.DataFrame:
    """
    Replicate feature.py logic to produce the same 161 features.
    """
    dates = pd.to_datetime(consumption_cols, format="%m/%d/%Y")
    X = cleaned_df[consumption_cols].to_numpy(dtype=np.float32)
    X_log = np.log1p(X)
    n_days = len(consumption_cols)

    features = pd.DataFrame({"CONS_NO": cleaned_df["CONS_NO"]})
    if "FLAG" in cleaned_df.columns:
        features["FLAG"] = cleaned_df["FLAG"]

    # --- Statistical features ---
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

    # --- Zero / low features ---
    zero_mask = X == 0
    low_mask = X <= LOW_CONSUMPTION_THRESHOLD
    features["zero_count_clean"] = zero_mask.sum(axis=1)
    features["zero_ratio_clean"] = zero_mask.mean(axis=1)
    features["low_consumption_count"] = low_mask.sum(axis=1)
    features["low_consumption_ratio"] = low_mask.mean(axis=1)
    features["max_zero_streak_clean"] = max_streak_2d(zero_mask)
    features["max_low_streak_clean"] = max_streak_2d(low_mask)

    # --- Temporal features ---
    def window_mean(values, n_recent_days):
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
        features["recent_30_mean"].to_numpy(), features["first_90_mean"].to_numpy(), clip_upper=50,
    )
    features["second_half_to_first_half_ratio"] = safe_ratio(
        features["second_half_mean"].to_numpy(), features["first_half_mean"].to_numpy(), clip_upper=50,
    )
    features["recent30_first90_log_ratio"] = log_ratio(
        features["recent_30_mean"].to_numpy(), features["first_90_mean"].to_numpy(),
    )
    features["second_first_half_log_ratio"] = log_ratio(
        features["second_half_mean"].to_numpy(), features["first_half_mean"].to_numpy(),
    )
    features["recent30_minus_first90"] = features["recent_30_mean"] - features["first_90_mean"]
    features["second_half_minus_first_half"] = features["second_half_mean"] - features["first_half_mean"]
    features["trend_slope"] = trend_slope(X)

    # --- Volatility / spike / drop ---
    prev = X[:, :-1]
    nxt = X[:, 1:]
    diff = nxt - prev
    abs_diff = np.abs(diff)
    features["mean_daily_change"] = diff.mean(axis=1)
    features["mean_abs_daily_change"] = abs_diff.mean(axis=1)
    features["max_abs_daily_change"] = abs_diff.max(axis=1)
    features["std_daily_change"] = diff.std(axis=1)
    spike_mask = (diff >= ABS_CHANGE_THRESHOLD) & (diff >= REL_CHANGE_THRESHOLD * np.maximum(prev, 1.0))
    drop_mask = (diff <= -ABS_CHANGE_THRESHOLD) & (-diff >= REL_CHANGE_THRESHOLD * np.maximum(prev, 1.0))
    features["spike_count"] = spike_mask.sum(axis=1)
    features["spike_ratio"] = spike_mask.mean(axis=1)
    features["drop_count"] = drop_mask.sum(axis=1)
    features["drop_ratio"] = drop_mask.mean(axis=1)
    features["max_spike_streak"] = max_streak_2d(spike_mask)
    features["max_drop_streak"] = max_streak_2d(drop_mask)
    features["drop_to_spike_ratio"] = safe_ratio(
        features["drop_count"].to_numpy(), features["spike_count"].to_numpy(), clip_upper=50,
    )

    # --- Monthly features ---
    month_values = dates.month
    for month in range(1, 13):
        month_mask = month_values == month
        if month_mask.sum() > 0:
            features[f"month_{month:02d}_mean"] = X[:, month_mask].mean(axis=1)
            features[f"month_{month:02d}_std"] = X[:, month_mask].std(axis=1)
    monthly_mean_cols = [f"month_{m:02d}_mean" for m in range(1, 13) if f"month_{m:02d}_mean" in features.columns]
    features["monthly_mean_range"] = features[monthly_mean_cols].max(axis=1) - features[monthly_mean_cols].min(axis=1)
    features["monthly_mean_std"] = features[monthly_mean_cols].std(axis=1)

    # --- Rolling features ---
    for window in ROLLING_WINDOWS:
        roll_mean = rolling_mean_2d(X, window)
        roll_std = rolling_std_2d(X, window)
        roll_mean_avg = roll_mean.mean(axis=1)
        roll_mean_min = roll_mean.min(axis=1)
        roll_mean_max = roll_mean.max(axis=1)
        features[f"roll{window}_mean_avg"] = roll_mean_avg
        features[f"roll{window}_mean_std"] = roll_mean.std(axis=1)
        features[f"roll{window}_mean_min"] = roll_mean_min
        features[f"roll{window}_mean_max"] = roll_mean_max
        features[f"roll{window}_mean_range"] = roll_mean_max - roll_mean_min
        features[f"roll{window}_std_avg"] = roll_std.mean(axis=1)
        features[f"roll{window}_std_max"] = roll_std.max(axis=1)
        features[f"roll{window}_max_to_avg_ratio"] = safe_ratio(roll_mean_max, roll_mean_avg, clip_upper=50)
        del roll_mean, roll_std

    # --- Segment features ---
    segments = np.array_split(X, N_SEGMENTS, axis=1)
    segment_means_list = []
    segment_stds_list = []
    for idx, segment in enumerate(segments, start=1):
        seg_mean = segment.mean(axis=1)
        seg_std = segment.std(axis=1)
        features[f"segment_{idx}_mean"] = seg_mean
        features[f"segment_{idx}_std"] = seg_std
        segment_means_list.append(seg_mean)
        segment_stds_list.append(seg_std)
    segment_means = np.vstack(segment_means_list).T
    segment_stds = np.vstack(segment_stds_list).T
    features["segment_mean_range"] = segment_means.max(axis=1) - segment_means.min(axis=1)
    features["segment_mean_std"] = segment_means.std(axis=1)
    features["segment_std_avg"] = segment_stds.mean(axis=1)
    features["segment_std_max"] = segment_stds.max(axis=1)
    features["segment4_to_segment1_ratio"] = safe_ratio(
        features["segment_4_mean"].to_numpy(), features["segment_1_mean"].to_numpy(), clip_upper=50,
    )
    features["segment4_segment1_log_ratio"] = log_ratio(
        features["segment_4_mean"].to_numpy(), features["segment_1_mean"].to_numpy(),
    )
    features["segment4_minus_segment1"] = features["segment_4_mean"] - features["segment_1_mean"]
    segment_diffs = np.diff(segment_means, axis=1)
    features["segment_largest_drop"] = segment_diffs.min(axis=1)
    features["segment_largest_increase"] = segment_diffs.max(axis=1)
    features["n_segment_drops"] = (segment_diffs < 0).sum(axis=1)
    features["n_segment_increases"] = (segment_diffs > 0).sum(axis=1)

    # --- Weekday/weekend features ---
    day_of_week = dates.dayofweek
    weekday_mask = day_of_week < 5
    weekend_mask = day_of_week >= 5
    X_weekday = X[:, weekday_mask]
    X_weekend = X[:, weekend_mask]
    features["weekday_mean"] = X_weekday.mean(axis=1)
    features["weekday_std"] = X_weekday.std(axis=1)
    features["weekend_mean"] = X_weekend.mean(axis=1)
    features["weekend_std"] = X_weekend.std(axis=1)
    features["weekend_to_weekday_ratio"] = safe_ratio(
        features["weekend_mean"].to_numpy(), features["weekday_mean"].to_numpy(), clip_upper=50,
    )
    features["weekend_weekday_log_ratio"] = log_ratio(
        features["weekend_mean"].to_numpy(), features["weekday_mean"].to_numpy(),
    )
    features["weekend_minus_weekday"] = features["weekend_mean"] - features["weekday_mean"]
    features["weekend_std_to_weekday_std_ratio"] = safe_ratio(
        features["weekend_std"].to_numpy(), features["weekday_std"].to_numpy(), clip_upper=50,
    )

    # --- Merge quality features ---
    quality_to_merge = quality_df.copy()
    quality_to_merge["is_all_missing_raw"] = (quality_to_merge["missing_count_raw"] == n_days).astype(np.int8)
    quality_to_merge["is_high_missing_raw"] = (quality_to_merge["missing_ratio_raw"] >= 0.50).astype(np.int8)
    quality_to_merge["is_very_high_missing_raw"] = (quality_to_merge["missing_ratio_raw"] >= 0.80).astype(np.int8)
    quality_to_merge["has_outlier_raw"] = (quality_to_merge["outlier_count_raw"] > 0).astype(np.int8)
    quality_to_merge["max_consumption_raw_missing"] = quality_to_merge["max_consumption_raw"].isna().astype(np.int8)
    quality_to_merge["max_consumption_raw"] = quality_to_merge["max_consumption_raw"].fillna(0)
    quality_to_merge["log1p_outlier_count_raw"] = np.log1p(quality_to_merge["outlier_count_raw"])
    quality_to_merge["log1p_max_consumption_raw"] = np.log1p(quality_to_merge["max_consumption_raw"])
    quality_to_merge["log1p_missing_count_raw"] = np.log1p(quality_to_merge["missing_count_raw"])
    quality_to_merge["log1p_zero_count_raw"] = np.log1p(quality_to_merge["zero_count_raw"])

    # Drop FLAG from quality before merge (FLAG is not a feature)
    quality_merge_cols = [c for c in quality_to_merge.columns if c not in {"FLAG"}]
    features = features.merge(quality_to_merge[quality_merge_cols], on="CONS_NO", how="left")

    # --- Interaction features ---
    features["max_to_mean_ratio"] = safe_ratio(
        features["max_consumption"].to_numpy(), features["mean_consumption"].to_numpy(), clip_upper=100,
    )
    features["q95_to_median_ratio"] = safe_ratio(
        features["q95_consumption"].to_numpy(), features["median_consumption"].to_numpy(), clip_upper=100,
    )
    features["iqr_to_median_ratio"] = safe_ratio(
        features["iqr_consumption"].to_numpy(), features["median_consumption"].to_numpy(), clip_upper=100,
    )
    features["std_to_median_ratio"] = safe_ratio(
        features["std_consumption"].to_numpy(), features["median_consumption"].to_numpy(), clip_upper=100,
    )
    features["recent90_to_total_mean_ratio"] = safe_ratio(
        features["recent_90_mean"].to_numpy(), features["mean_consumption"].to_numpy(), clip_upper=50,
    )
    features["recent30_to_total_mean_ratio"] = safe_ratio(
        features["recent_30_mean"].to_numpy(), features["mean_consumption"].to_numpy(), clip_upper=50,
    )
    features["zero_to_low_ratio"] = safe_ratio(
        features["zero_count_clean"].to_numpy(), features["low_consumption_count"].to_numpy(), clip_upper=50,
    )
    features["missing_x_zero_ratio"] = features["missing_ratio_raw"] * features["zero_ratio_clean"]
    features["missing_x_low_ratio"] = features["missing_ratio_raw"] * features["low_consumption_ratio"]
    features["missing_x_recent_drop"] = features["missing_ratio_raw"] * (-features["recent30_first90_log_ratio"])
    features["outlier_x_volatility"] = features["log1p_outlier_count_raw"] * features["mean_abs_daily_change"]
    features["outlier_x_max_change"] = features["log1p_outlier_count_raw"] * features["max_abs_daily_change"]
    features["drop_x_recent_drop"] = features["drop_ratio"] * (-features["recent30_first90_log_ratio"])
    features["drop_x_segment_drop"] = features["drop_ratio"] * (-features["segment_largest_drop"])
    features["missing_x_outlier"] = features["missing_ratio_raw"] * features["log1p_outlier_count_raw"]

    # --- Validate / clean NaN/Inf ---
    feature_cols = [c for c in features.columns if c not in ["CONS_NO", "FLAG"]]
    if features[feature_cols].isna().sum().sum() > 0:
        features[feature_cols] = features[feature_cols].fillna(0)
    arr = features[feature_cols].to_numpy(dtype=np.float64)
    if np.isinf(arr).sum() > 0:
        features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], 0)

    return features


def predict_from_raw_csv(
    df_raw: pd.DataFrame,
    bundle: dict,
) -> list[dict]:
    """
    Full inference pipeline: raw CSV -> preprocessing -> feature engineering -> predict.
    Returns list of per-customer result dicts.
    """
    df = df_raw.copy()

    # Extract FLAG if present (for display only)
    has_flag = "FLAG" in df.columns
    flag_series = df["FLAG"].copy() if has_flag else None

    # Validate columns
    consumption_cols = validate_upload_columns(df)

    # Keep only CONS_NO + FLAG (if present) + date columns
    keep_cols = ["CONS_NO"] + (["FLAG"] if has_flag else []) + consumption_cols
    df = df[keep_cols].copy()

    # Run preprocessing
    cleaned_df, quality_df = run_preprocessing(df, consumption_cols)

    # Run feature engineering
    features = run_feature_engineering(cleaned_df, quality_df, consumption_cols)

    # Select active features from bundle
    active_cols = bundle["active_feature_cols"]
    missing_features = set(active_cols) - set(features.columns)
    if missing_features:
        # If features are missing, fill with 0 (shouldn't happen if pipeline is correct)
        for col in missing_features:
            features[col] = 0

    X_predict = features[active_cols]

    # Predict
    estimator = bundle["estimator"]
    score_kind = bundle.get("score_kind", "predict_proba")
    threshold = bundle["threshold"]

    if score_kind == "predict_proba":
        scores = estimator.predict_proba(X_predict)[:, 1]
    else:
        raise ValueError(f"Unsupported score_kind: {score_kind}")

    # Build results
    records = []
    for i in range(len(features)):
        score = float(scores[i])
        if score >= 0.70:
            risk_level = "Critical"
        elif score >= threshold:
            risk_level = "High"
        elif score >= 0.30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        prediction = "Suspected theft" if score >= threshold else "Normal"

        record = {
            "cons_no": str(features.iloc[i]["CONS_NO"]),
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "prediction": prediction,
            "risk_level": risk_level,
            "feature_summary": {
                "mean_consumption": round(float(features.iloc[i].get("mean_consumption", 0)), 2),
                "zero_ratio_clean": round(float(features.iloc[i].get("zero_ratio_clean", 0)), 4),
                "missing_ratio_raw": round(float(features.iloc[i].get("missing_ratio_raw", 0)), 4),
                "recent_90_mean": round(float(features.iloc[i].get("recent_90_mean", 0)), 2),
                "mean_abs_daily_change": round(float(features.iloc[i].get("mean_abs_daily_change", 0)), 2),
            },
        }

        if has_flag and flag_series is not None:
            actual = int(flag_series.iloc[i])
            record["actual_label"] = actual
            pred_label = 1 if score >= threshold else 0
            if pred_label == 1 and actual == 1:
                record["outcome"] = "TP"
            elif pred_label == 0 and actual == 0:
                record["outcome"] = "TN"
            elif pred_label == 1 and actual == 0:
                record["outcome"] = "FP"
            else:
                record["outcome"] = "FN"

        # Add raw consumption for time-series chart
        raw_row = df_raw.iloc[i]
        consumption_data = []
        for col in consumption_cols:
            val = raw_row.get(col, None)
            try:
                val = float(val) if pd.notna(val) else None
            except (TypeError, ValueError):
                val = None
            consumption_data.append({"date": col, "value": val})
        record["consumption_timeseries"] = consumption_data

        records.append(record)

    return records
