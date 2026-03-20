# 05b_effect_size_kruskal.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import kruskal


BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
PROCESSED_DIR = BASE_DIR / "data_processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = PROCESSED_DIR / "proxy_dataset_profiles_revised.csv"
OUTPUT_FILE = TABLES_DIR / "Table_kruskal_effect_size.xlsx"
LOG_FILE = DIAG_DIR / "kruskal_effect_size_log.txt"

TARGET_COLS = [f"W{i}_rev" for i in range(1, 9)]
GROUP_COL = "fundingScheme"


def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def interpret_epsilon_squared(eps2: float) -> str:
    if pd.isna(eps2):
        return "NA"
    if eps2 < 0.01:
        return "negligible"
    if eps2 < 0.08:
        return "small"
    if eps2 < 0.26:
        return "moderate"
    return "large"


def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required = [GROUP_COL] + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    work = df[[GROUP_COL] + TARGET_COLS].copy()
    work = work.dropna(subset=[GROUP_COL])

    rows = []
    log_lines = []

    group_counts = work[GROUP_COL].value_counts(dropna=False).sort_values(ascending=False)
    valid_groups = group_counts[group_counts >= 2].index.tolist()

    log_lines.append(f"Total rows: {len(work)}")
    log_lines.append(f"Total groups before filtering: {work[GROUP_COL].nunique()}")
    log_lines.append(f"Groups retained with n>=2: {len(valid_groups)}")

    work = work[work[GROUP_COL].isin(valid_groups)].copy()

    for target in TARGET_COLS:
        tmp = work[[GROUP_COL, target]].dropna().copy()

        grouped = [g[target].to_numpy() for _, g in tmp.groupby(GROUP_COL)]
        grouped = [x for x in grouped if len(x) >= 2]

        n = sum(len(x) for x in grouped)
        k = len(grouped)

        if k < 2 or n <= k:
            H = np.nan
            p = np.nan
            eps2 = np.nan
        else:
            H, p = kruskal(*grouped)
            eps2 = (H - k + 1) / (n - k)

        rows.append({
            "criterion": target,
            "n_total": n,
            "k_groups": k,
            "H_statistic": H,
            "p_value": p,
            "epsilon_squared": eps2,
            "effect_size_label": interpret_epsilon_squared(eps2),
        })

    result_df = pd.DataFrame(rows)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="kruskal_effect_size", index=False)
        group_counts.reset_index().rename(
            columns={"index": GROUP_COL, GROUP_COL: "count"}
        ).to_excel(writer, sheet_name="group_counts", index=False)

    log_lines.append("")
    log_lines.append("=== RESULTS ===")
    for _, r in result_df.iterrows():
        log_lines.append(
            f"{r['criterion']}: H={r['H_statistic']:.6f}, "
            f"p={r['p_value']:.6g}, eps2={r['epsilon_squared']:.6f}, "
            f"label={r['effect_size_label']}"
        )

    write_log(log_lines)

    print("=" * 72)
    print("05b_effect_size_kruskal.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Log: {LOG_FILE}")
    print("-" * 72)
    for _, r in result_df.iterrows():
        print(
            f"{r['criterion']}: "
            f"H={r['H_statistic']:.6f}, "
            f"p={r['p_value']:.6g}, "
            f"eps2={r['epsilon_squared']:.6f}, "
            f"label={r['effect_size_label']}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()