"""
Preprocessing pipeline for Energy Theft Detection dataset.
Cleans raw consumption data: duplicates, negatives, outliers, missing, zero-consumption.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_DATA_PATH = PROJECT_ROOT / "data" / "data" / "raw" / "data set.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "cleaned.csv"

# =============================================================================
# THRESHOLDS
# =============================================================================
OUTLIER_UPPER = 500          # kWh/day - physical upper bound per day
ZERO_RATIO_INACTIVE = 0.80  # >80% zero days -> inactive meter

# =============================================================================
# HELPERS
# =============================================================================

def log(msg: str):
    print(f"[PREPROCESS] {msg}")


def step_header(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}")


def parse_date_cols(cons_cols: list) -> list:
    """Parse date column names and return them sorted chronologically."""
    date_pairs = []
    for col in cons_cols:
        try:
            date_pairs.append((col, pd.to_datetime(col, format="%m/%d/%Y")))
        except Exception:
            date_pairs.append((col, pd.NaT))
    date_pairs = [(c, d) for c, d in date_pairs if pd.notna(d)]
    date_pairs.sort(key=lambda x: x[1])
    return [c for c, _ in date_pairs]


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
def load_data() -> pd.DataFrame:
    step_header(1, "LOAD DATA")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}\n"
            "Run notebooks/preview.ipynb first to download from Kaggle."
        )

    log(f"Loading {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    log(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    log(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    log(f"Columns: CONS_NO, FLAG, + {df.shape[1]-2} date columns")
    return df


# =============================================================================
# STEP 2: REMOVE DUPLICATES
# =============================================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    step_header(2, "CHECK & REMOVE DUPLICATES")

    # --- CONS_NO duplicates ---
    dup_cons = df["CONS_NO"].duplicated().sum()
    log(f"Duplicate CONS_NO: {dup_cons:,}")
    if dup_cons > 0:
        dup_ids = df[df["CONS_NO"].duplicated(keep=False)]["CONS_NO"].unique()
        log(f"  Affected customer IDs: {len(dup_ids):,}")
        log("  Keeping first occurrence, dropping duplicates")
        df = df.drop_duplicates(subset="CONS_NO", keep="first")
        log(f"  Rows after dedup: {len(df):,}")
    else:
        log("  No duplicate CONS_NO found")

    # --- Fully identical consumption rows ---
    cons_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    dup_rows = df.duplicated(subset=cons_cols).sum()
    log(f"Duplicate consumption rows: {dup_rows:,}")
    if dup_rows > 0:
        df = df.drop_duplicates(subset=cons_cols, keep="first")
        log(f"  Rows after removing: {len(df):,}")

    log(f"Customers remaining: {len(df):,}")
    return df.reset_index(drop=True)


# =============================================================================
# STEP 3: CLEAN DATA (negatives + outliers)
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    step_header(3, "CLEAN DATA - negatives & outliers")

    cons_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    cons_data = df[cons_cols]
    total_cells = cons_data.size

    # --- Negatives ---
    neg_mask = cons_data < 0
    neg_count = neg_mask.sum().sum()
    neg_rows = neg_mask.any(axis=1).sum()
    log(f"Negative values: {neg_count:,} cells ({100*neg_count/total_cells:.3f}%)")
    log(f"  Customers affected: {neg_rows:,}")
    if neg_count > 0:
        log("  Converting negatives -> NaN")
        df.loc[:, cons_cols] = df[cons_cols].where(~neg_mask)
    else:
        log("  No negative values found")

    # --- Outliers (upper bound) ---
    upper_mask = cons_data > OUTLIER_UPPER
    upper_count = upper_mask.sum().sum()
    upper_rows = upper_mask.any(axis=1).sum()
    log(f"Outliers > {OUTLIER_UPPER} kWh/day: {upper_count:,} cells ({100*upper_count/total_cells:.4f}%)")
    log(f"  Customers affected: {upper_rows:,}")
    if upper_count > 0:
        log(f"  Clipping to {OUTLIER_UPPER} kWh/day")
        df.loc[:, cons_cols] = df[cons_cols].clip(upper=OUTLIER_UPPER)
    else:
        log("  No outliers found")

    log(f"Consumption stats after cleaning:")
    stats = df[cons_cols].stack().describe()
    log(f"  min={stats['min']:.2f}, max={stats['max']:.2f}, "
        f"mean={stats['mean']:.2f}, median={stats['50%']:.2f}")

    return df


# =============================================================================
# STEP 4: ZERO-CONSUMPTION (on cleaned-but-unfilled data)
# IMPORTANT: zero_ratio must be computed BEFORE NaN fill to avoid bias.
# NaN cells are excluded from the zero_ratio denominator so we only count
# genuinely observed zeros. This prevents the fill strategy from distorting
# the zero_ratio distribution.
# =============================================================================
def handle_zero_consumption(df: pd.DataFrame) -> pd.DataFrame:
    step_header(4, "ZERO-CONSUMPTION - inactive classification (pre-fill)")

    cons_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG", "is_inactive", "zero_ratio")]

    # zero_ratio = n_zero / total_columns, computed BEFORE NaN fill.
    # pandas `df == 0` returns False for NaN -> NaN cells excluded from both
    # numerator (n_zero) and denominator (total). This means:
    #   - A customer with 90% NaN + 5% zeros -> zero_ratio = 5%
    #   - A customer with 100% zeros          -> zero_ratio = 100%
    # Both are meaningful signals and neither is distorted by fill strategy.
    n_total = len(cons_cols)
    n_zero = (df[cons_cols] == 0).sum(axis=1)
    zero_ratio = n_zero / n_total

    df = df.assign(
        zero_ratio=zero_ratio,
        is_inactive=(zero_ratio >= ZERO_RATIO_INACTIVE).astype(np.int8)
    )

    n_inactive = df["is_inactive"].sum()
    n_active = len(df) - n_inactive
    log(f"Inactive meters (zero_ratio >= {ZERO_RATIO_INACTIVE:.0%}): "
        f"{n_inactive:,} ({100*n_inactive/len(df):.2f}%)")
    log(f"Active meters: {n_active:,} ({100*n_active/len(df):.2f}%)")

    # Inactive breakdown by FLAG
    for flag_val, label in [(0, "normal"), (1, "theft")]:
        subset = df[df["FLAG"] == flag_val]
        n_sub = len(subset)
        n_inactive_sub = subset["is_inactive"].sum()
        log(f"  {label:6s} - inactive: {n_inactive_sub:,}/{n_sub:,} "
            f"({100*n_inactive_sub/n_sub:.2f}%)")

    # Zero ratio distribution
    log("Zero ratio distribution:")
    bins = [0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = ((df["zero_ratio"] >= lo) & (df["zero_ratio"] < hi)).sum()
        log(f"  [{lo:.0%} - {hi:.0%}): {count:,} ({100*count/len(df):.2f}%)")

    return df


# =============================================================================
# STEP 5: MISSING VALUES
# Strategy (per-customer, along time axis):
#   1. ffill  -> carry last known value forward (up to 30 days)
#   2. bfill  -> carry next known value backward (up to 30 days)
#   3. Remaining: per-customer median (if customer has >= 10 valid observations)
#   4. Edge-case fallback: group median (FLAG=0 group / FLAG=1 group)
#
# Why ffill/bfill over linear interpolation:
#   - Electricity consumption is autocorrelated (today ~= yesterday).
#   - Linear interpolation "levels out" peaks/troughs, distorting variance.
#   - ffill+bfill preserves local structure and is robust to edge NaNs.
#
# Why group median over global median:
#   - Theft customers have higher mean consumption (27.5 vs 7.7) and different
#     distribution. Filling all remaining NaN with a single global median
#     would systematically under-fill theft customers and over-fill normal ones.
# =============================================================================
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    step_header(5, "MISSING VALUES - ffill + bfill + group median")

    cons_cols = [c for c in df.columns
                 if c not in ("CONS_NO", "FLAG", "is_inactive", "zero_ratio")]

    ordered_cons_cols = parse_date_cols(cons_cols)
    log(f"Ordered consumption columns by date: {len(ordered_cons_cols):,}")

    missing_before = df[ordered_cons_cols].isna().sum().sum()
    missing_pct_before = 100 * missing_before / df[ordered_cons_cols].size
    log(f"Missing before: {missing_before:,} ({missing_pct_before:.2f}%)")

    # Per-customer missing rate distribution
    per_cust_missing = df[ordered_cons_cols].isna().mean(axis=1)
    log("Per-customer missing rate distribution:")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = ((per_cust_missing >= lo) & (per_cust_missing < hi)).sum()
        log(f"  [{lo:.0%} - {hi:.0%}): {count:,} ({100*count/len(df):.2f}%)")

    before_fill = df[ordered_cons_cols].copy()

    # Work with T-Df: 1034 rows (dates) x 41K cols (customers)
    # pandas ffill/bfill are column-oriented -> much faster on tall narrow shape
    t_df = df[ordered_cons_cols].T
    t_filled = t_df.ffill(limit=30).bfill(limit=30)

    remaining_after_fw = t_filled.isna().sum().sum()
    filled_by_fw = missing_before - remaining_after_fw
    log(f"  ffill+bfill filled: {filled_by_fw:,} cells ({100*filled_by_fw/missing_before:.1f}%)")
    log(f"  Still NaN after ffill+bfill: {remaining_after_fw:,}")

    # --- Per-customer median for remaining NaN (vectorized via transpose) ---
    remaining = t_filled.isna().sum().sum()
    if remaining > 0:
        log(f"  Remaining NaN: {remaining:,} -> filling with per-customer median")
        t_filled = t_filled.T.fillna(t_df.T.median(axis=1)).T

    # --- Group median fallback (vectorized) ---
    final_remaining = t_filled.isna().sum().sum()
    if final_remaining > 0:
        # Group medians by FLAG: theft customers have ~3.5x higher consumption
        group_medians = (
            df[["FLAG"] + ordered_cons_cols]
            .groupby("FLAG")[ordered_cons_cols]
            .median()
            .median(axis=1)   # median across dates, then result is per FLAG
        )
        log(f"  Final fallback: {final_remaining:,} cells")
        log(f"    Group median (FLAG=0, normal): {group_medians[0]:.3f} kWh")
        log(f"    Group median (FLAG=1, theft):  {group_medians[1]:.3f} kWh")

        # Broadcast: rows = customers, cols = dates; fill with matching FLAG median
        flag_lookup = df["FLAG"].map(group_medians).values
        t_filled = t_filled.fillna(pd.Series(flag_lookup, index=t_filled.columns))

    # Reassemble: transpose back
    df_clean = df.copy()
    df_clean[ordered_cons_cols] = t_filled.T

    # Count after
    missing_after = df_clean[ordered_cons_cols].isna().sum().sum()
    missing_pct_after = 100 * missing_after / df_clean[ordered_cons_cols].size
    log(f"Missing after: {missing_after:,} ({missing_pct_after:.2f}%)")

    # Distribution comparison
    log("Distribution comparison (before -> after fill):")
    b = before_fill.stack().dropna()
    a = df_clean[ordered_cons_cols].stack()
    for stat_name in ("mean", "median", "std", "min", "max"):
        b_val = getattr(b, stat_name)()
        a_val = getattr(a, stat_name)()
        log(f"  {stat_name:6s}: {b_val:10.3f} -> {a_val:10.3f}")

    return df_clean


# =============================================================================
# STEP 6: SAVE
# =============================================================================
def save_data(df: pd.DataFrame):
    step_header(6, "SAVE CLEANED DATA")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024**2
    log(f"Saved -> {OUTPUT_PATH}")
    log(f"File size: {size_mb:.1f} MB")
    log(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    cons_cols = [c for c in df.columns
                 if c not in ("CONS_NO", "FLAG", "is_inactive", "zero_ratio")]
    log(f"Consumption columns: {len(cons_cols):,}")
    log(f"Extra columns: CONS_NO, FLAG, zero_ratio, is_inactive")


# =============================================================================
# SUMMARY REPORT
# =============================================================================
def print_summary(df: pd.DataFrame):
    print(f"\n{'#'*60}")
    print("  PREPROCESSING SUMMARY")
    print(f"{'#'*60}")
    cons_cols = [c for c in df.columns
                 if c not in ("CONS_NO", "FLAG", "is_inactive", "zero_ratio")]

    log(f"Total customers: {len(df):,}")
    log(f"Consumption columns: {len(cons_cols):,}")
    log(f"Total consumption cells: {len(df) * len(cons_cols):,}")
    log(f"Missing (should be 0): {df[cons_cols].isna().sum().sum():,}")
    log(f"Negative (should be 0): {(df[cons_cols] < 0).sum().sum():,}")
    log(f"Outliers clipped (>{OUTLIER_UPPER}): see Step 3")
    log(f"Inactive meters: {df['is_inactive'].sum():,} ({100*df['is_inactive'].mean():.2f}%)")
    n_norm = len(df[df["FLAG"] == 0])
    n_theft = len(df[df["FLAG"] == 1])
    log(f"  Normal:  {df[df['FLAG']==0]['is_inactive'].sum():,}/{n_norm:,}")
    log(f"  Theft:   {df[df['FLAG']==1]['is_inactive'].sum():,}/{n_theft:,}")
    log(f"Output: {OUTPUT_PATH}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'#'*60}")
    print("  ENERGY THEFT DETECTION - PREPROCESSING PIPELINE")
    print(f"{'#'*60}")

    df = load_data()
    df = remove_duplicates(df)
    df = clean_data(df)
    df = handle_zero_consumption(df)   # zero_ratio BEFORE fill (Step 4)
    df = handle_missing(df)              # fill NaN (Step 5)
    save_data(df)
    print_summary(df)

    log("Done.")


if __name__ == "__main__":
    main()
