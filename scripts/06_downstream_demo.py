# 06_downstream_demo.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
PROCESSED_DIR = BASE_DIR / "data_processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = PROCESSED_DIR / "proxy_dataset_profiles_revised.csv"
RANKING_TABLE_OUT = TABLES_DIR / "Table_ranking_comparison.xlsx"
RANK_SHIFT_FIG_OUT = FIGURES_DIR / "Figure_rank_shift.png"
LOG_OUT = DIAG_DIR / "downstream_demo_log.txt"

RAW_PROXY_COLS = [f"C{i}_raw" for i in range(1, 9)]
WEIGHT_COLS = [f"W{i}_rev" for i in range(1, 9)]

SUBSET_SIZE = 25
RANDOM_STATE = 42


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def minmax_scale(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0.0)
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val <= min_val:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - min_val) / (max_val - min_val)


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Entropy weighting on benefit-type normalized criteria matrix.
    """
    X = np.asarray(matrix, dtype=float)
    X = np.clip(X, 0, None)

    col_sums = X.sum(axis=0)
    n, m = X.shape

    # avoid division by zero
    P = np.divide(X, col_sums, out=np.zeros_like(X), where=col_sums > 0)

    k = 1.0 / np.log(n) if n > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
        e = -k * np.sum(P * logP, axis=0)

    d = 1 - e
    if np.allclose(d.sum(), 0):
        return np.ones(m) / m
    return d / d.sum()


def weighted_sum_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.dot(X, w)


def rank_desc(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores)
    return s.rank(method="min", ascending=False).astype(int).to_numpy()


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() <= 1 or b.nunique() <= 1:
        return np.nan
    return float(a.corr(b, method="spearman"))


def kendall_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() <= 1 or b.nunique() <= 1:
        return np.nan
    return float(a.corr(b, method="kendall"))


def top_k_overlap(rank_a: pd.Series, rank_b: pd.Series, k: int = 5) -> float:
    set_a = set(rank_a.nsmallest(k).index)
    set_b = set(rank_b.nsmallest(k).index)
    return len(set_a & set_b) / k


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required = RAW_PROXY_COLS + WEIGHT_COLS + ["id", "title", "fundingScheme", "start_year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # -------------------------
    # Real-data subset selection rule
    # Rule:
    # 1) keep rows with complete raw proxies
    # 2) sort by most recent start_year, then highest ec contribution if available
    # 3) sample across funding schemes if possible, else take top N
    # -------------------------
    work = df.copy()

    for col in RAW_PROXY_COLS + WEIGHT_COLS:
        work[col] = safe_numeric(work[col])

    work = work.dropna(subset=RAW_PROXY_COLS + WEIGHT_COLS).copy()

    if "ecMaxContribution_num" in work.columns:
        work["ecMaxContribution_num"] = safe_numeric(work["ecMaxContribution_num"]).fillna(0.0)
    else:
        work["ecMaxContribution_num"] = 0.0

    work["start_year"] = safe_numeric(work["start_year"])
    work = work.sort_values(
        by=["start_year", "ecMaxContribution_num"],
        ascending=[False, False]
    ).copy()

    # diversify by funding scheme if possible
    subset_parts = []
    if "fundingScheme" in work.columns and work["fundingScheme"].nunique() > 1:
        grouped = work.groupby("fundingScheme", dropna=False)
        # first pass: take top 1 from each scheme
        first_pass = grouped.head(1)
        subset_parts.append(first_pass)

        remaining_needed = max(0, SUBSET_SIZE - len(first_pass))
        if remaining_needed > 0:
            remaining = work.drop(index=first_pass.index, errors="ignore")
            subset_parts.append(remaining.head(remaining_needed))

        subset = pd.concat(subset_parts, axis=0).drop_duplicates().head(SUBSET_SIZE).copy()
    else:
        subset = work.head(SUBSET_SIZE).copy()

    if len(subset) < 10:
        raise RuntimeError("Downstream subset too small after filtering.")

    subset = subset.reset_index(drop=True)

    # -------------------------
    # Decision matrix from real proxy values
    # Normalize within subset for downstream comparison
    # -------------------------
    X = pd.DataFrame(index=subset.index)
    for i, col in enumerate(RAW_PROXY_COLS, start=1):
        X[f"C{i}"] = minmax_scale(subset[col])

    X_mat = X.to_numpy(dtype=float)

    # -------------------------
    # Weight scenarios
    # 1) equal weights
    # 2) learned weights = average real learned weights from selected subset
    # 3) entropy weights computed from the same real subset matrix
    # -------------------------
    equal_w = np.ones(8) / 8

    learned_w = subset[WEIGHT_COLS].mean(axis=0).to_numpy(dtype=float)
    learned_w = learned_w / learned_w.sum()

    entropy_w = entropy_weights(X_mat)

    # -------------------------
    # Weighted Sum scores and ranks
    # -------------------------
    score_equal = weighted_sum_scores(X_mat, equal_w)
    score_learned = weighted_sum_scores(X_mat, learned_w)
    score_entropy = weighted_sum_scores(X_mat, entropy_w)

    rank_equal = rank_desc(score_equal)
    rank_learned = rank_desc(score_learned)
    rank_entropy = rank_desc(score_entropy)

    results = subset[[
        "id", "title", "fundingScheme", "start_year", "ecMaxContribution_num"
    ]].copy()

    for i in range(1, 9):
        results[f"C{i}"] = X[f"C{i}"]

    results["score_equal"] = score_equal
    results["score_learned"] = score_learned
    results["score_entropy"] = score_entropy

    results["rank_equal"] = rank_equal
    results["rank_learned"] = rank_learned
    results["rank_entropy"] = rank_entropy

    results["shift_learned_vs_equal"] = results["rank_equal"] - results["rank_learned"]
    results["shift_learned_vs_entropy"] = results["rank_entropy"] - results["rank_learned"]

    results = results.sort_values("rank_learned").reset_index(drop=True)

    # -------------------------
    # Ranking comparison summary
    # -------------------------
    rank_equal_s = results["rank_equal"]
    rank_learned_s = results["rank_learned"]
    rank_entropy_s = results["rank_entropy"]

    comparison_df = pd.DataFrame([
        {
            "comparison": "learned_vs_equal",
            "spearman": spearman_corr(rank_learned_s, rank_equal_s),
            "kendall": kendall_corr(rank_learned_s, rank_equal_s),
            "top5_overlap": top_k_overlap(rank_learned_s, rank_equal_s, k=min(5, len(results))),
            "mean_abs_rank_diff": float(np.mean(np.abs(rank_learned_s - rank_equal_s))),
        },
        {
            "comparison": "learned_vs_entropy",
            "spearman": spearman_corr(rank_learned_s, rank_entropy_s),
            "kendall": kendall_corr(rank_learned_s, rank_entropy_s),
            "top5_overlap": top_k_overlap(rank_learned_s, rank_entropy_s, k=min(5, len(results))),
            "mean_abs_rank_diff": float(np.mean(np.abs(rank_learned_s - rank_entropy_s))),
        },
        {
            "comparison": "equal_vs_entropy",
            "spearman": spearman_corr(rank_equal_s, rank_entropy_s),
            "kendall": kendall_corr(rank_equal_s, rank_entropy_s),
            "top5_overlap": top_k_overlap(rank_equal_s, rank_entropy_s, k=min(5, len(results))),
            "mean_abs_rank_diff": float(np.mean(np.abs(rank_equal_s - rank_entropy_s))),
        },
    ])

    weights_df = pd.DataFrame({
        "criterion": [f"C{i}" for i in range(1, 9)],
        "equal_weight": equal_w,
        "learned_weight": learned_w,
        "entropy_weight": entropy_w,
    })

    # -------------------------
    # Rank shift figure
    # -------------------------
    plot_df = results.nsmallest(min(15, len(results)), "rank_learned").copy()
    y = np.arange(len(plot_df))

    plt.figure(figsize=(10, 7))
    plt.plot(plot_df["rank_equal"], y, marker="o", label="Equal")
    plt.plot(plot_df["rank_learned"], y, marker="o", label="Learned")
    plt.plot(plot_df["rank_entropy"], y, marker="o", label="Entropy")

    for i in range(len(plot_df)):
        plt.plot(
            [plot_df.loc[plot_df.index[i], "rank_equal"], plot_df.loc[plot_df.index[i], "rank_learned"]],
            [y[i], y[i]],
            linewidth=1
        )
        plt.plot(
            [plot_df.loc[plot_df.index[i], "rank_learned"], plot_df.loc[plot_df.index[i], "rank_entropy"]],
            [y[i], y[i]],
            linewidth=1,
            linestyle="--"
        )

    labels = [
        str(v)[:45] + ("..." if len(str(v)) > 45 else "")
        for v in plot_df["title"]
    ]
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Rank position")
    plt.ylabel("Project")
    plt.title("Ranking shifts under alternative weighting schemes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RANK_SHIFT_FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Save outputs
    # -------------------------
    with pd.ExcelWriter(RANKING_TABLE_OUT, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="ranking_results", index=False)
        comparison_df.to_excel(writer, sheet_name="ranking_comparison", index=False)
        weights_df.to_excel(writer, sheet_name="weight_scenarios", index=False)

    # -------------------------
    # Logging
    # -------------------------
    log_lines = [
        f"Input rows after complete-case filter: {len(work)}",
        f"Selected subset size: {len(subset)}",
        "Subset selection rule: real projects sorted by recent year and contribution, diversified by funding scheme where possible",
        "",
        "=== WEIGHT SCENARIOS ===",
    ]
    for _, r in weights_df.iterrows():
        log_lines.append(
            f"{r['criterion']}: "
            f"equal={r['equal_weight']:.6f}, "
            f"learned={r['learned_weight']:.6f}, "
            f"entropy={r['entropy_weight']:.6f}"
        )

    log_lines.append("")
    log_lines.append("=== RANKING COMPARISON ===")
    for _, r in comparison_df.iterrows():
        log_lines.append(
            f"{r['comparison']}: "
            f"spearman={r['spearman']:.6f}, "
            f"kendall={r['kendall']:.6f}, "
            f"top5_overlap={r['top5_overlap']:.6f}, "
            f"mean_abs_rank_diff={r['mean_abs_rank_diff']:.6f}"
        )

    write_log(log_lines)

    # -------------------------
    # Terminal summary
    # -------------------------
    print("=" * 72)
    print("06_downstream_demo.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Ranking table: {RANKING_TABLE_OUT}")
    print(f"Rank shift figure: {RANK_SHIFT_FIG_OUT}")
    print(f"Log: {LOG_OUT}")
    print("-" * 72)
    print(f"Selected subset size: {len(subset)}")
    print("Ranking comparison:")
    for _, r in comparison_df.iterrows():
        print(
            f"{r['comparison']}: "
            f"spearman={r['spearman']:.6f}, "
            f"kendall={r['kendall']:.6f}, "
            f"top5_overlap={r['top5_overlap']:.6f}, "
            f"mean_abs_rank_diff={r['mean_abs_rank_diff']:.6f}"
        )
    print("-" * 72)
    print("Weight scenarios:")
    for _, r in weights_df.iterrows():
        print(
            f"{r['criterion']}: "
            f"equal={r['equal_weight']:.6f}, "
            f"learned={r['learned_weight']:.6f}, "
            f"entropy={r['entropy_weight']:.6f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()