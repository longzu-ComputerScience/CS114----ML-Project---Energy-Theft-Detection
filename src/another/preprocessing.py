"""
Pipeline tiền xử lý cho bộ dữ liệu Energy Theft Detection.
Mục tiêu của file này là làm sạch chuỗi tiêu thụ thô: trùng lặp,
giá trị âm, outlier, missing value và các trường hợp tiêu thụ bằng 0.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_DATA_PATH = PROJECT_ROOT / "data" / "data" / "raw" / "data set.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "cleaned.csv"

# =============================================================================
# NGƯỠNG XỬ LÝ
# =============================================================================
OUTLIER_UPPER = 500          # kWh/ngày - ngưỡng vật lý tạm dùng để chặn giá trị lỗi quá lớn
ZERO_RATIO_INACTIVE = 0.80  # >80% ngày bằng 0 -> xem là công tơ gần như không hoạt động

# =============================================================================
# HÀM HỖ TRỢ
# =============================================================================

def log(msg: str):
    print(f"[PREPROCESS] {msg}")


def step_header(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}")


def parse_date_cols(cons_cols: list) -> list:
    """Chuyển tên cột ngày sang datetime và trả về danh sách cột đã sắp theo thời gian."""
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
# BƯỚC 1: ĐỌC DỮ LIỆU
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
# BƯỚC 2: KIỂM TRA VÀ XÓA DỮ LIỆU TRÙNG
# =============================================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    step_header(2, "CHECK & REMOVE DUPLICATES")

    # --- Trùng mã khách hàng CONS_NO ---
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

    # --- Các dòng có toàn bộ chuỗi tiêu thụ giống nhau ---
    cons_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    dup_rows = df.duplicated(subset=cons_cols).sum()
    log(f"Duplicate consumption rows: {dup_rows:,}")
    if dup_rows > 0:
        df = df.drop_duplicates(subset=cons_cols, keep="first")
        log(f"  Rows after removing: {len(df):,}")

    log(f"Customers remaining: {len(df):,}")
    return df.reset_index(drop=True)


# =============================================================================
# BƯỚC 3: LÀM SẠCH GIÁ TRỊ ÂM VÀ OUTLIER
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    step_header(3, "CLEAN DATA - negatives & outliers")

    cons_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    cons_data = df[cons_cols]
    total_cells = cons_data.size

    # --- Giá trị âm: không hợp lý với dữ liệu tiêu thụ nên chuyển thành NaN ---
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

    # --- Outlier quá lớn: clip về ngưỡng trên để tránh một vài lỗi kéo lệch thống kê ---
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
# BƯỚC 4: TIÊU THỤ BẰNG 0 (tính trên dữ liệu đã clean nhưng chưa fill NaN)
# Quan trọng: zero_ratio được tính TRƯỚC khi fill NaN để chiến lược fill
# không tạo thêm ngày bằng 0 giả. Code hiện tại tính:
#   zero_ratio = số ngày có giá trị 0 / tổng số cột ngày theo dõi.
# NaN không được tính là 0 ở tử số, nhưng vẫn nằm trong mẫu số vì ta muốn
# đo tỷ lệ ngày bằng 0 trên toàn bộ giai đoạn theo dõi.
# =============================================================================
def handle_zero_consumption(df: pd.DataFrame) -> pd.DataFrame:
    step_header(4, "ZERO-CONSUMPTION - inactive classification (pre-fill)")

    cons_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG", "is_inactive", "zero_ratio")]

    # pandas `df == 0` trả False với NaN, nên NaN không làm tăng số ngày bằng 0.
    # Nếu sau này muốn tỷ lệ zero chỉ trên các ngày quan sát được, cần đổi mẫu số
    # thành số ô không NaN của từng khách hàng.
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

    # Thống kê công tơ không hoạt động theo từng nhãn để phục vụ EDA/báo cáo
    for flag_val, label in [(0, "normal"), (1, "theft")]:
        subset = df[df["FLAG"] == flag_val]
        n_sub = len(subset)
        n_inactive_sub = subset["is_inactive"].sum()
        log(f"  {label:6s} - inactive: {n_inactive_sub:,}/{n_sub:,} "
            f"({100*n_inactive_sub/n_sub:.2f}%)")

    # Phân phối zero_ratio theo các khoảng dễ đọc
    log("Zero ratio distribution:")
    bins = [0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = ((df["zero_ratio"] >= lo) & (df["zero_ratio"] < hi)).sum()
        log(f"  [{lo:.0%} - {hi:.0%}): {count:,} ({100*count/len(df):.2f}%)")

    return df


# =============================================================================
# BƯỚC 5: XỬ LÝ MISSING VALUE
# Chiến lược theo từng khách hàng, dọc theo trục thời gian:
#   1. ffill  -> lấy giá trị gần nhất trước đó để điền về sau, tối đa 30 ngày
#   2. bfill  -> lấy giá trị gần nhất sau đó để điền ngược lại, tối đa 30 ngày
#   3. Phần còn lại: điền bằng median của chính khách hàng đó nếu tính được
#   4. Fallback cuối: median theo nhóm FLAG
#
# Vì sao dùng ffill/bfill thay vì nội suy tuyến tính:
#   - Tiêu thụ điện có tính tự tương quan: hôm nay thường gần với hôm qua.
#   - Nội suy tuyến tính có thể làm phẳng đỉnh/đáy, làm sai lệch độ dao động.
#   - ffill+bfill giữ cấu trúc cục bộ tốt hơn, đặc biệt ở đầu/cuối chuỗi.
#
# Lưu ý về fallback theo FLAG:
#   - Cách này dùng nhãn để điền phần NaN còn lại, phù hợp cho file cleaning/EDA.
#   - Nếu xây pipeline train nghiêm ngặt trước khi split train/test, nên thay bằng
#     median không dùng nhãn để tránh label leakage.
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

    # Phân phối tỷ lệ thiếu theo từng khách hàng
    per_cust_missing = df[ordered_cons_cols].isna().mean(axis=1)
    log("Per-customer missing rate distribution:")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        count = ((per_cust_missing >= lo) & (per_cust_missing < hi)).sum()
        log(f"  [{lo:.0%} - {hi:.0%}): {count:,} ({100*count/len(df):.2f}%)")

    before_fill = df[ordered_cons_cols].copy()

    # Chuyển vị dữ liệu: hàng = ngày, cột = khách hàng.
    # pandas ffill/bfill chạy theo cột, nên cách này nhanh hơn cho chuỗi thời gian.
    t_df = df[ordered_cons_cols].T
    t_filled = t_df.ffill(limit=30).bfill(limit=30)

    remaining_after_fw = t_filled.isna().sum().sum()
    filled_by_fw = missing_before - remaining_after_fw
    log(f"  ffill+bfill filled: {filled_by_fw:,} cells ({100*filled_by_fw/missing_before:.1f}%)")
    log(f"  Still NaN after ffill+bfill: {remaining_after_fw:,}")

    # --- Điền NaN còn lại bằng median của từng khách hàng ---
    remaining = t_filled.isna().sum().sum()
    if remaining > 0:
        log(f"  Remaining NaN: {remaining:,} -> filling with per-customer median")
        t_filled = t_filled.T.fillna(t_df.T.median(axis=1)).T

    # --- Fallback cuối bằng median theo nhóm FLAG ---
    final_remaining = t_filled.isna().sum().sum()
    if final_remaining > 0:
        # Median theo nhãn: chỉ dùng khi vẫn còn khách hàng không có median riêng.
        group_medians = (
            df[["FLAG"] + ordered_cons_cols]
            .groupby("FLAG")[ordered_cons_cols]
            .median()
            .median(axis=1)   # median across dates, then result is per FLAG
        )
        log(f"  Final fallback: {final_remaining:,} cells")
        log(f"    Group median (FLAG=0, normal): {group_medians[0]:.3f} kWh")
        log(f"    Group median (FLAG=1, theft):  {group_medians[1]:.3f} kWh")

        # Broadcast giá trị median theo nhãn vào các ô còn thiếu của khách hàng tương ứng.
        flag_lookup = df["FLAG"].map(group_medians).values
        t_filled = t_filled.fillna(pd.Series(flag_lookup, index=t_filled.columns))

    # Ghép lại DataFrame theo chiều ban đầu: hàng = khách hàng, cột = ngày.
    df_clean = df.copy()
    df_clean[ordered_cons_cols] = t_filled.T

    # Kiểm tra số missing sau khi fill.
    missing_after = df_clean[ordered_cons_cols].isna().sum().sum()
    missing_pct_after = 100 * missing_after / df_clean[ordered_cons_cols].size
    log(f"Missing after: {missing_after:,} ({missing_pct_after:.2f}%)")

    # So sánh phân phối trước/sau fill để phát hiện fill làm lệch dữ liệu quá mạnh không.
    log("Distribution comparison (before -> after fill):")
    b = before_fill.stack().dropna()
    a = df_clean[ordered_cons_cols].stack()
    for stat_name in ("mean", "median", "std", "min", "max"):
        b_val = getattr(b, stat_name)()
        a_val = getattr(a, stat_name)()
        log(f"  {stat_name:6s}: {b_val:10.3f} -> {a_val:10.3f}")

    return df_clean


# =============================================================================
# BƯỚC 6: LƯU DỮ LIỆU ĐÃ LÀM SẠCH
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
# BÁO CÁO TÓM TẮT
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
# HÀM MAIN
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
