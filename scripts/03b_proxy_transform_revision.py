# 03b_proxy_transform_revision.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
PROCESSED_DIR = BASE_DIR / "data_processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = PROCESSED_DIR / "proxy_dataset_raw.csv"
REVISED_PROFILE_OUT = PROCESSED_DIR / "proxy_dataset_profiles_revised.csv"
SUMMARY_OUT = TABLES_DIR / "Table_proxy_transform_revision.xlsx"
LOG_OUT = DIAG_DIR / "proxy_transform_revision_log.txt"

RAW_PROXY_COLS = [f"C{i}_raw" for i in range(1, 9)]
TRANS_PROXY_COLS = [f"C{i}_trans" for i in range(1, 9)]
P95_SCALED_COLS = [f"C{i}_p95scaled_rev" for i in range(1, 9)]
W_REV_COLS = [f"W{i}_rev" for i in range(1, 9)]

EPS = 1e-12


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def log1p_nonnegative(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    s = s.clip(lower=0)
    return np.log1p(s)


def clip_upper_quantile(series: pd.Series, q: float = 0.95) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    upper = s.quantile(q)
    if pd.isna(upper):
        return s
    return s.clip(upper=upper)


def p95_scale(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    p95 = s.quantile(0.95)
    if pd.isna(p95) or p95 <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s / p95).clip(upper=1.0)


def row_unit_sum(df: pd.DataFrame, cols: List[str], out_cols: List[str], sum_col: str) -> pd.DataFrame:
    row_sum = df[cols].sum(axis=1)
    for src, dst in zip(cols, out_cols):
        df[dst] = np.where(row_sum > EPS, df[src] / row_sum, np.nan)
    df[sum_col] = df[out_cols].sum(axis=1)
    return df


def second_moment_skewness(series: pd.Series) -> float:
    s = safe_numeric(series).dropna()
    if len(s) < 3:
        return np.nan
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return np.nan
    mean = s.mean()
    return float((((s - mean) / std) ** 3).mean())


def top1_counts(df: pd.DataFrame, cols: List[str], label: str) -> pd.DataFrame:
    idx = df[cols].values.argmax(axis=1)
    counts = pd.Series(idx).value_counts().sort_index()
    rows = []
    for i in range(8):
        cnt = int(counts.get(i, 0))
        rows.append({
            "normalization": label,
            "proxy": f"W{i+1}",
            "top1_count": cnt,
            "top1_pct": float(cnt / len(df) * 100) if len(df) else np.nan,
        })
    return pd.DataFrame(rows)


def spearman_corr_safe(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp.iloc[:, 0].nunique() <= 1 or tmp.iloc[:, 1].nunique() <= 1:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="spearman")


def top1_match_rate(df_a: pd.DataFrame, cols_a: List[str], df_b: pd.DataFrame, cols_b: List[str]) -> float:
    idx_a = df_a[cols_a].values.argmax(axis=1)
    idx_b = df_b[cols_b].values.argmax(axis=1)
    if len(idx_a) == 0:
        return np.nan
    return float((idx_a == idx_b).mean())


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_cols = ["invalid_for_profile"] + RAW_PROXY_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in proxy_dataset_raw.csv: {missing}")

    work = df[df["invalid_for_profile"] == 0].copy()

    if len(work) == 0:
        raise RuntimeError("No valid rows available after invalid_for_profile filter.")

    # Keep original raw proxies numeric
    for col in RAW_PROXY_COLS:
        work[col] = safe_numeric(work[col]).fillna(0.0).clip(lower=0)

    # -------------------------
    # Original reference normalization from raw proxies: p95 + row sum
    # -------------------------
    orig_scaled_cols = []
    orig_w_cols = []
    for i in range(1, 9):
        s_col = f"C{i}_orig_p95scaled"
        w_col = f"W{i}_orig_p95"
        work[s_col] = p95_scale(work[f"C{i}_raw"])
        orig_scaled_cols.append(s_col)
        orig_w_cols.append(w_col)

    work = row_unit_sum(work, orig_scaled_cols, orig_w_cols, "profile_sum_orig_p95")

    # -------------------------
    # Revised transformed proxies
    # Rules:
    # C1, C2, C3, C4, C7 -> log1p
    # C5, C6 -> keep raw
    # C8 -> clip upper 95%, then keep in original scale
    # -------------------------
    work["C1_trans"] = log1p_nonnegative(work["C1_raw"])
    work["C2_trans"] = log1p_nonnegative(work["C2_raw"])
    work["C3_trans"] = log1p_nonnegative(work["C3_raw"])
    work["C4_trans"] = log1p_nonnegative(work["C4_raw"])
    work["C5_trans"] = work["C5_raw"]
    work["C6_trans"] = work["C6_raw"]
    work["C7_trans"] = log1p_nonnegative(work["C7_raw"])
    work["C8_trans"] = clip_upper_quantile(work["C8_raw"], q=0.95)

    # Revised main normalization: p95 + row sum
    for i in range(1, 9):
        work[f"C{i}_p95scaled_rev"] = p95_scale(work[f"C{i}_trans"])

    work = row_unit_sum(work, P95_SCALED_COLS, W_REV_COLS, "profile_sum_rev")

    # -------------------------
    # Diagnostics tables
    # -------------------------
    transform_rows = []
    for i in range(1, 9):
        raw_col = f"C{i}_raw"
        trans_col = f"C{i}_trans"

        transform_rows.append({
            "proxy": f"C{i}",
            "raw_mean": work[raw_col].mean(),
            "raw_std": work[raw_col].std(),
            "raw_p95": work[raw_col].quantile(0.95),
            "raw_max": work[raw_col].max(),
            "raw_skewness_approx": second_moment_skewness(work[raw_col]),
            "trans_mean": work[trans_col].mean(),
            "trans_std": work[trans_col].std(),
            "trans_p95": work[trans_col].quantile(0.95),
            "trans_max": work[trans_col].max(),
            "trans_skewness_approx": second_moment_skewness(work[trans_col]),
        })
    transform_df = pd.DataFrame(transform_rows)

    orig_dom_df = top1_counts(work, orig_w_cols, "orig_p95_raw")
    rev_dom_df = top1_counts(work, W_REV_COLS, "revised_p95_transformed")
    dominance_df = pd.concat([orig_dom_df, rev_dom_df], ignore_index=True)

    weight_comp_rows = []
    for i in range(1, 9):
        weight_comp_rows.append({
            "proxy_weight": f"W{i}",
            "spearman_orig_vs_revised": spearman_corr_safe(work[f"W{i}_orig_p95"], work[f"W{i}_rev"]),
            "mean_orig": work[f"W{i}_orig_p95"].mean(),
            "mean_revised": work[f"W{i}_rev"].mean(),
            "std_orig": work[f"W{i}_orig_p95"].std(),
            "std_revised": work[f"W{i}_rev"].std(),
        })
    weight_comp_df = pd.DataFrame(weight_comp_rows)

    profile_comparison_df = pd.DataFrame([{
        "comparison": "orig_p95_raw_vs_revised_p95_transformed",
        "mean_l1_distance_per_row": np.abs(work[orig_w_cols].values - work[W_REV_COLS].values).sum(axis=1).mean(),
        "top1_match_rate": top1_match_rate(work, orig_w_cols, work, W_REV_COLS),
        "profile_sum_fail_orig_abs_gt_1e_6": int((work["profile_sum_orig_p95"].sub(1).abs() > 1e-6).sum()),
        "profile_sum_fail_rev_abs_gt_1e_6": int((work["profile_sum_rev"].sub(1).abs() > 1e-6).sum()),
    }])

    invalidity_check_df = pd.DataFrame([{
        "n_rows_used": len(work),
        "profile_sum_fail_rev_abs_gt_1e_6": int((work["profile_sum_rev"].sub(1).abs() > 1e-6).sum()),
        "revised_top1_W8_pct": float((work[W_REV_COLS].values.argmax(axis=1) == 7).mean() * 100),
        "orig_top1_W8_pct": float((work[orig_w_cols].values.argmax(axis=1) == 7).mean() * 100),
    }])

    # -------------------------
    # Save revised profile dataset
    # -------------------------
    keep_meta = [
        "id", "acronym", "title", "status", "frameworkProgramme",
        "startDate", "endDate", "start_year", "start_month",
        "ecSignature_year", "duration_months", "ecMaxContribution_num",
        "fundingScheme", "masterCall", "subCall", "call_id", "call_id_source",
        "legalBasis", "keyword_count", "objective_token_count",
        "objective_unique_token_count", "objective_lexical_diversity"
    ]
    keep_meta = [c for c in keep_meta if c in work.columns]

    revised_cols = (
        keep_meta
        + RAW_PROXY_COLS
        + TRANS_PROXY_COLS
        + P95_SCALED_COLS
        + W_REV_COLS
        + ["profile_sum_rev"]
    )
    work[revised_cols].to_csv(REVISED_PROFILE_OUT, index=False, encoding="utf-8-sig")

    # -------------------------
    # Save summary workbook
    # -------------------------
    with pd.ExcelWriter(SUMMARY_OUT, engine="openpyxl") as writer:
        transform_df.to_excel(writer, sheet_name="transform_summary", index=False)
        dominance_df.to_excel(writer, sheet_name="dominance_comparison", index=False)
        weight_comp_df.to_excel(writer, sheet_name="weight_comparison", index=False)
        profile_comparison_df.to_excel(writer, sheet_name="profile_comparison", index=False)
        invalidity_check_df.to_excel(writer, sheet_name="revision_check", index=False)

    # -------------------------
    # Logging
    # -------------------------
    log_lines = [
        "03b_proxy_transform_revision.py completed",
        f"Input file: {INPUT_FILE}",
        f"Rows used: {len(work)}",
        "",
        "=== TRANSFORM RULES ===",
        "C1 -> log1p",
        "C2 -> log1p",
        "C3 -> log1p",
        "C4 -> log1p",
        "C5 -> raw",
        "C6 -> raw",
        "C7 -> log1p",
        "C8 -> upper 95% clip",
        "",
        "=== KEY CHECKS ===",
        f"orig_top1_W8_pct: {float((work[orig_w_cols].values.argmax(axis=1) == 7).mean() * 100):.6f}",
        f"revised_top1_W8_pct: {float((work[W_REV_COLS].values.argmax(axis=1) == 7).mean() * 100):.6f}",
        f"profile_sum_fail_orig_abs_gt_1e_6: {int((work['profile_sum_orig_p95'].sub(1).abs() > 1e-6).sum())}",
        f"profile_sum_fail_rev_abs_gt_1e_6: {int((work['profile_sum_rev'].sub(1).abs() > 1e-6).sum())}",
        f"mean_l1_distance_per_row: {profile_comparison_df.loc[0, 'mean_l1_distance_per_row']:.6f}",
        f"top1_match_rate: {profile_comparison_df.loc[0, 'top1_match_rate']:.6f}",
    ]
    write_log(log_lines)

    # -------------------------
    # Terminal summary
    # -------------------------
    orig_top1_w8_pct = float((work[orig_w_cols].values.argmax(axis=1) == 7).mean() * 100)
    rev_top1_w8_pct = float((work[W_REV_COLS].values.argmax(axis=1) == 7).mean() * 100)

    print("=" * 72)
    print("03b_proxy_transform_revision.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Revised profile output: {REVISED_PROFILE_OUT}")
    print(f"Summary workbook: {SUMMARY_OUT}")
    print(f"Log: {LOG_OUT}")
    print("-" * 72)
    print(f"Rows used: {len(work)}")
    print(f"orig_top1_W8_pct: {orig_top1_w8_pct:.6f}")
    print(f"revised_top1_W8_pct: {rev_top1_w8_pct:.6f}")
    print(f"mean_l1_distance_per_row: {profile_comparison_df.loc[0, 'mean_l1_distance_per_row']:.6f}")
    print(f"top1_match_rate: {profile_comparison_df.loc[0, 'top1_match_rate']:.6f}")
    print(f"profile_sum_fail_orig_abs_gt_1e_6: {int((work['profile_sum_orig_p95'].sub(1).abs() > 1e-6).sum())}")
    print(f"profile_sum_fail_rev_abs_gt_1e_6: {int((work['profile_sum_rev'].sub(1).abs() > 1e-6).sum())}")
    print("=" * 72)


if __name__ == "__main__":
    main()