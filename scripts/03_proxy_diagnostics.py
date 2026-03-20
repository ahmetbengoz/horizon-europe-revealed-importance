# 03_proxy_diagnostics.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
PROCESSED_DIR = BASE_DIR / "data_processed"
OUTPUT_TABLES_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = PROCESSED_DIR / "proxy_dataset_raw.csv"
TABLE_OUT = OUTPUT_TABLES_DIR / "Table_proxy_distribution.xlsx"
ROBUSTNESS_OUT = OUTPUT_TABLES_DIR / "Table_normalization_robustness.xlsx"
FIG_OUT = OUTPUT_FIGURES_DIR / "Figure_proxy_correlation.png"
LOG_OUT = OUTPUT_DIAG_DIR / "proxy_diagnostics_log.txt"

RAW_PROXY_COLS = [f"C{i}_raw" for i in range(1, 9)]
SCALED_MAX_COLS = [f"C{i}_maxscaled" for i in range(1, 9)]
W_MAX_COLS = [f"W{i}_max" for i in range(1, 9)]
SCALED_MINMAX_COLS = [f"C{i}_minmax" for i in range(1, 9)]
W_MINMAX_COLS = [f"W{i}_minmax" for i in range(1, 9)]
SCALED_P95_COLS = [f"C{i}_p95scaled" for i in range(1, 9)]
W_P95_COLS = [f"W{i}_p95" for i in range(1, 9)]

EPS = 1e-12


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def max_scale(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    max_val = s.max()
    if pd.isna(max_val) or max_val <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return s / max_val


def minmax_scale(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val <= min_val:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - min_val) / (max_val - min_val)


def p95_scale(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    p95 = s.quantile(0.95)
    if pd.isna(p95) or p95 <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    scaled = s / p95
    return scaled.clip(upper=1.0)


def row_unit_sum(df: pd.DataFrame, cols: List[str], out_prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    row_sum = df[cols].sum(axis=1)
    out_cols = []
    for i, col in enumerate(cols, start=1):
        w_col = f"W{i}_{out_prefix}"
        df[w_col] = np.where(row_sum > EPS, df[col] / row_sum, np.nan)
        out_cols.append(w_col)
    df[f"profile_sum_{out_prefix}"] = df[out_cols].sum(axis=1)
    return df, out_cols


def spearman_corr_safe(a: pd.Series, b: pd.Series) -> float:
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp.iloc[:, 0].nunique() <= 1 or tmp.iloc[:, 1].nunique() <= 1:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="spearman")


def top1_match_rate(w_a: pd.DataFrame, w_b: pd.DataFrame) -> float:
    idx_a = w_a.values.argmax(axis=1)
    idx_b = w_b.values.argmax(axis=1)
    if len(idx_a) == 0:
        return np.nan
    return float((idx_a == idx_b).mean())


def second_moment_skewness(series: pd.Series) -> float:
    s = safe_numeric(series).dropna()
    if len(s) < 3:
        return np.nan
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return np.nan
    mean = s.mean()
    return float((((s - mean) / std) ** 3).mean())


def outlier_counts_iqr(series: pd.Series) -> Tuple[int, float]:
    s = safe_numeric(series).dropna()
    if len(s) == 0:
        return 0, np.nan
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0, 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n = int(((s < lower) | (s > upper)).sum())
    pct = float(n / len(s) * 100)
    return n, pct


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    if "invalid_for_profile" not in df.columns:
        raise RuntimeError("Column 'invalid_for_profile' not found in proxy_dataset_raw.csv")

    for col in RAW_PROXY_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Missing proxy column: {col}")

    diag = df[df["invalid_for_profile"] == 0].copy()

    log_lines = []
    log_lines.append(f"Input rows raw: {len(df)}")
    log_lines.append(f"Rows used for diagnostics (valid profiles): {len(diag)}")

    # -------------------------
    # Proxy distribution table
    # -------------------------
    dist_rows = []
    for col in RAW_PROXY_COLS:
        s = safe_numeric(diag[col])
        n = s.notna().sum()
        missing = int(s.isna().sum())
        missing_pct = round(missing / len(diag) * 100, 4) if len(diag) else np.nan
        out_n, out_pct = outlier_counts_iqr(s)
        row = {
            "proxy": col,
            "n_non_null": int(n),
            "missing_count": missing,
            "missing_pct": missing_pct,
            "min": s.min(),
            "p1": s.quantile(0.01),
            "p5": s.quantile(0.05),
            "p25": s.quantile(0.25),
            "median": s.quantile(0.50),
            "p75": s.quantile(0.75),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "mean": s.mean(),
            "std": s.std(),
            "cv": s.std() / s.mean() if pd.notna(s.mean()) and s.mean() != 0 else np.nan,
            "skewness_approx": second_moment_skewness(s),
            "outlier_count_iqr": out_n,
            "outlier_pct_iqr": out_pct,
        }
        dist_rows.append(row)

    dist_df = pd.DataFrame(dist_rows)

    # -------------------------
    # Correlation matrix
    # -------------------------
    corr_df = diag[RAW_PROXY_COLS].apply(pd.to_numeric, errors="coerce")
    corr_matrix = corr_df.corr(method="spearman")

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr_matrix.values, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(RAW_PROXY_COLS)), RAW_PROXY_COLS, rotation=45, ha="right")
    plt.yticks(range(len(RAW_PROXY_COLS)), RAW_PROXY_COLS)
    plt.title("Spearman Correlation of Raw Proxies")
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close()

    high_corr_pairs = []
    for i in range(len(RAW_PROXY_COLS)):
        for j in range(i + 1, len(RAW_PROXY_COLS)):
            a = RAW_PROXY_COLS[i]
            b = RAW_PROXY_COLS[j]
            v = corr_matrix.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.80:
                high_corr_pairs.append({
                    "proxy_a": a,
                    "proxy_b": b,
                    "spearman_corr": v,
                })
    high_corr_df = pd.DataFrame(high_corr_pairs)

    # -------------------------
    # Normalization robustness
    # -------------------------
    for i in range(1, 9):
        raw_col = f"C{i}_raw"
        diag[f"C{i}_maxscaled"] = max_scale(diag[raw_col])
        diag[f"C{i}_minmax"] = minmax_scale(diag[raw_col])
        diag[f"C{i}_p95scaled"] = p95_scale(diag[raw_col])

    diag, w_max_cols = row_unit_sum(diag, SCALED_MAX_COLS, "max")
    diag, w_minmax_cols = row_unit_sum(diag, SCALED_MINMAX_COLS, "minmax")
    diag, w_p95_cols = row_unit_sum(diag, SCALED_P95_COLS, "p95")

    robustness_rows = []

    for i in range(1, 9):
        r1 = spearman_corr_safe(diag[f"W{i}_max"], diag[f"W{i}_minmax"])
        r2 = spearman_corr_safe(diag[f"W{i}_max"], diag[f"W{i}_p95"])
        r3 = spearman_corr_safe(diag[f"W{i}_minmax"], diag[f"W{i}_p95"])
        robustness_rows.append({
            "proxy_weight": f"W{i}",
            "spearman_max_vs_minmax": r1,
            "spearman_max_vs_p95": r2,
            "spearman_minmax_vs_p95": r3,
            "mean_max": diag[f"W{i}_max"].mean(),
            "mean_minmax": diag[f"W{i}_minmax"].mean(),
            "mean_p95": diag[f"W{i}_p95"].mean(),
            "std_max": diag[f"W{i}_max"].std(),
            "std_minmax": diag[f"W{i}_minmax"].std(),
            "std_p95": diag[f"W{i}_p95"].std(),
        })

    profile_level_rows = [
        {
            "comparison": "max_vs_minmax",
            "mean_abs_sum_diff_per_row": (diag[w_max_cols].values - diag[w_minmax_cols].values).astype(float).sum(axis=1).mean(),
            "mean_l1_distance_per_row": np.abs(diag[w_max_cols].values - diag[w_minmax_cols].values).sum(axis=1).mean(),
            "top1_match_rate": top1_match_rate(diag[w_max_cols], diag[w_minmax_cols]),
            "profile_sum_fail_count_a": int((diag["profile_sum_max"].sub(1).abs() > 1e-6).sum()),
            "profile_sum_fail_count_b": int((diag["profile_sum_minmax"].sub(1).abs() > 1e-6).sum()),
        },
        {
            "comparison": "max_vs_p95",
            "mean_abs_sum_diff_per_row": (diag[w_max_cols].values - diag[w_p95_cols].values).astype(float).sum(axis=1).mean(),
            "mean_l1_distance_per_row": np.abs(diag[w_max_cols].values - diag[w_p95_cols].values).sum(axis=1).mean(),
            "top1_match_rate": top1_match_rate(diag[w_max_cols], diag[w_p95_cols]),
            "profile_sum_fail_count_a": int((diag["profile_sum_max"].sub(1).abs() > 1e-6).sum()),
            "profile_sum_fail_count_b": int((diag["profile_sum_p95"].sub(1).abs() > 1e-6).sum()),
        },
        {
            "comparison": "minmax_vs_p95",
            "mean_abs_sum_diff_per_row": (diag[w_minmax_cols].values - diag[w_p95_cols].values).astype(float).sum(axis=1).mean(),
            "mean_l1_distance_per_row": np.abs(diag[w_minmax_cols].values - diag[w_p95_cols].values).sum(axis=1).mean(),
            "top1_match_rate": top1_match_rate(diag[w_minmax_cols], diag[w_p95_cols]),
            "profile_sum_fail_count_a": int((diag["profile_sum_minmax"].sub(1).abs() > 1e-6).sum()),
            "profile_sum_fail_count_b": int((diag["profile_sum_p95"].sub(1).abs() > 1e-6).sum()),
        },
    ]
    profile_level_df = pd.DataFrame(profile_level_rows)

    dominant_proxy_rows = []
    for label, cols in [("max", w_max_cols), ("minmax", w_minmax_cols), ("p95", w_p95_cols)]:
        dominant_idx = diag[cols].values.argmax(axis=1)
        counts = pd.Series(dominant_idx).value_counts().sort_index()
        row = {"normalization": label}
        for i in range(8):
            row[f"top1_W{i+1}_count"] = int(counts.get(i, 0))
            row[f"top1_W{i+1}_pct"] = float(counts.get(i, 0) / len(diag) * 100) if len(diag) else np.nan
        dominant_proxy_rows.append(row)
    dominant_proxy_df = pd.DataFrame(dominant_proxy_rows)

    robustness_df = pd.DataFrame(robustness_rows)

    # -------------------------
    # Save outputs
    # -------------------------
    with pd.ExcelWriter(TABLE_OUT, engine="openpyxl") as writer:
        dist_df.to_excel(writer, sheet_name="proxy_distribution", index=False)
        corr_matrix.to_excel(writer, sheet_name="proxy_correlation")
        high_corr_df.to_excel(writer, sheet_name="high_corr_pairs", index=False)

    with pd.ExcelWriter(ROBUSTNESS_OUT, engine="openpyxl") as writer:
        robustness_df.to_excel(writer, sheet_name="weight_level_robustness", index=False)
        profile_level_df.to_excel(writer, sheet_name="profile_level_robustness", index=False)
        dominant_proxy_df.to_excel(writer, sheet_name="dominant_proxy_by_norm", index=False)

    # -------------------------
    # Logging
    # -------------------------
    log_lines.append("")
    log_lines.append("=== HIGH CORRELATION PAIRS (|rho| >= 0.80) ===")
    if len(high_corr_df) == 0:
        log_lines.append("None")
    else:
        for _, r in high_corr_df.iterrows():
            log_lines.append(f"{r['proxy_a']} vs {r['proxy_b']}: {r['spearman_corr']:.4f}")

    log_lines.append("")
    log_lines.append("=== PROFILE-LEVEL ROBUSTNESS ===")
    for _, r in profile_level_df.iterrows():
        log_lines.append(
            f"{r['comparison']}: "
            f"mean_l1_distance_per_row={r['mean_l1_distance_per_row']:.6f}, "
            f"top1_match_rate={r['top1_match_rate']:.6f}"
        )

    write_log(log_lines)

    # -------------------------
    # Terminal summary
    # -------------------------
    print("=" * 72)
    print("03_proxy_diagnostics.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Distribution table: {TABLE_OUT}")
    print(f"Robustness table: {ROBUSTNESS_OUT}")
    print(f"Correlation figure: {FIG_OUT}")
    print(f"Log: {LOG_OUT}")
    print("-" * 72)
    print(f"Rows used for diagnostics: {len(diag)}")
    print(f"High-correlation proxy pairs (|rho| >= 0.80): {len(high_corr_df)}")
    for _, r in profile_level_df.iterrows():
        print(
            f"{r['comparison']}: "
            f"mean_l1_distance_per_row={r['mean_l1_distance_per_row']:.6f}, "
            f"top1_match_rate={r['top1_match_rate']:.6f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()