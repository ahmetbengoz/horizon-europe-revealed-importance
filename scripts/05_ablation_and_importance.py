# 05_ablation_and_importance_fast.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
PROCESSED_DIR = BASE_DIR / "data_processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = PROCESSED_DIR / "proxy_dataset_profiles_revised.csv"
ABLATION_OUT = TABLES_DIR / "Table_ablation.xlsx"
IMPORTANCE_TABLE_OUT = TABLES_DIR / "Table_feature_importance.xlsx"
IMPORTANCE_FIG_OUT = FIGURES_DIR / "Figure_feature_importance.png"
LOG_OUT = DIAG_DIR / "ablation_and_importance_fast_log.txt"

TARGET_COLS = [f"W{i}_rev" for i in range(1, 9)]
RANDOM_STATE = 42
N_SPLITS = 3

BASE_NUMERIC_FEATURES = [
    "start_year",
    "start_month",
    "ecSignature_year",
    "duration_months",
    "ecMaxContribution_num",
    "keyword_count",
    "objective_token_count",
    "objective_unique_token_count",
    "objective_lexical_diversity",
]

BASE_CATEGORICAL_FEATURES = [
    "fundingScheme",
    "subCall",
    "legalBasis",
    "status",
    "frameworkProgramme",
]


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


def spearman_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(len(y_true)):
        a = pd.Series(y_true[i])
        b = pd.Series(y_pred[i])
        if a.nunique() <= 1 or b.nunique() <= 1:
            continue
        vals.append(a.corr(b, method="spearman"))
    return float(np.nanmean(vals)) if vals else np.nan


def kendall_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = []
    for i in range(len(y_true)):
        a = pd.Series(y_true[i])
        b = pd.Series(y_pred[i])
        if a.nunique() <= 1 or b.nunique() <= 1:
            continue
        vals.append(a.corr(b, method="kendall"))
    return float(np.nanmean(vals)) if vals else np.nan


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman_row_avg": spearman_per_row(y_true, y_pred),
        "kendall_row_avg": kendall_per_row(y_true, y_pred),
    }


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])


def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", model),
    ])


def get_feature_sets(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    num = [c for c in BASE_NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in BASE_CATEGORICAL_FEATURES if c in df.columns]

    sets = {
        "full": {
            "num": num,
            "cat": cat,
        },
        "no_year": {
            "num": [c for c in num if c not in {"start_year", "ecSignature_year"}],
            "cat": cat,
        },
        "no_funding_scheme": {
            "num": num,
            "cat": [c for c in cat if c != "fundingScheme"],
        },
        "no_call_vars": {
            "num": num,
            "cat": [c for c in cat if c != "subCall"],
        },
        "reduced_institutional": {
            "num": [c for c in num if c not in {"start_month", "ecSignature_year"}],
            "cat": [c for c in cat if c not in {"legalBasis", "status"}],
        },
    }
    return sets


def temporal_split_late_period(df: pd.DataFrame):
    years = sorted(df["start_year"].dropna().unique().tolist())
    if len(years) < 3:
        raise RuntimeError("Need at least 3 distinct years for late-period temporal split.")

    test_years = years[-2:]
    train_years = years[:-2]

    train_df = df[df["start_year"].isin(train_years)].copy()
    test_df = df[df["start_year"].isin(test_years)].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Temporal late-period split failed.")

    return train_df, test_df, train_years, test_years


def aggregate_feature_importance(pipe: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> pd.DataFrame:
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]

    importances = model.feature_importances_

    names_num = numeric_features

    cat_names = []
    if categorical_features:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out(categorical_features))

    all_names = names_num + cat_names
    imp_df = pd.DataFrame({
        "encoded_feature": all_names,
        "importance": importances,
    })

    def group_name(name: str) -> str:
        for c in categorical_features:
            if name.startswith(c + "_"):
                return c
        return name

    imp_df["feature_group"] = imp_df["encoded_feature"].apply(group_name)

    grouped = (
        imp_df.groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    return grouped


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required = TARGET_COLS + ["start_year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    feature_sets = get_feature_sets(df)
    y = df[TARGET_COLS].copy()

    # -------------------------
    # CV Ablation
    # -------------------------
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    ablation_rows = []

    for set_name, feats in feature_sets.items():
        numeric_features = feats["num"]
        categorical_features = feats["cat"]
        X = df[numeric_features + categorical_features].copy()

        fold_metrics = []
        for train_idx, test_idx in kf.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            pipe = build_pipeline(numeric_features, categorical_features)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            fold_metrics.append(evaluate_predictions(y_test.to_numpy(), np.asarray(y_pred)))

        ablation_rows.append({
            "feature_set": set_name,
            "n_numeric": len(numeric_features),
            "n_categorical": len(categorical_features),
            "mse_mean": float(np.mean([m["mse"] for m in fold_metrics])),
            "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "spearman_mean": float(np.nanmean([m["spearman_row_avg"] for m in fold_metrics])),
            "kendall_mean": float(np.nanmean([m["kendall_row_avg"] for m in fold_metrics])),
        })

    ablation_df = pd.DataFrame(ablation_rows).sort_values(
        by=["mse_mean", "mae_mean"], ascending=[True, True]
    )

    # -------------------------
    # Temporal Ablation
    # -------------------------
    train_df, test_df, train_years, test_years = temporal_split_late_period(df)
    temporal_rows = []

    temporal_variants = ["full", "no_year", "no_funding_scheme", "no_call_vars"]

    for set_name in temporal_variants:
        feats = feature_sets[set_name]
        numeric_features = feats["num"]
        categorical_features = feats["cat"]

        X_train = train_df[numeric_features + categorical_features].copy()
        X_test = test_df[numeric_features + categorical_features].copy()
        y_train = train_df[TARGET_COLS].copy()
        y_test = test_df[TARGET_COLS].copy()

        pipe = build_pipeline(numeric_features, categorical_features)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate_predictions(y_test.to_numpy(), np.asarray(y_pred))
        temporal_rows.append({
            "feature_set": set_name,
            "train_years": ",".join(map(str, train_years)),
            "test_years": ",".join(map(str, test_years)),
            "train_n": len(train_df),
            "test_n": len(test_df),
            **metrics,
        })

    temporal_ablation_df = pd.DataFrame(temporal_rows).sort_values(
        by=["mse", "mae"], ascending=[True, True]
    )

    # -------------------------
    # Fast importance on full model
    # -------------------------
    full_feats = feature_sets["full"]
    full_num = full_feats["num"]
    full_cat = full_feats["cat"]

    X_full = df[full_num + full_cat].copy()
    y_full = df[TARGET_COLS].copy()

    full_pipe = build_pipeline(full_num, full_cat)
    full_pipe.fit(X_full, y_full)

    importance_df = aggregate_feature_importance(full_pipe, full_num, full_cat)

    top_plot = importance_df.head(12).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(top_plot["feature_group"], top_plot["importance"])
    plt.xlabel("Aggregated feature importance")
    plt.ylabel("Feature group")
    plt.title("Feature Importance (ExtraTrees, aggregated)")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_FIG_OUT, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Save outputs
    # -------------------------
    with pd.ExcelWriter(ABLATION_OUT, engine="openpyxl") as writer:
        ablation_df.to_excel(writer, sheet_name="cv_ablation", index=False)
        temporal_ablation_df.to_excel(writer, sheet_name="temporal_ablation", index=False)

    with pd.ExcelWriter(IMPORTANCE_TABLE_OUT, engine="openpyxl") as writer:
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)

    # -------------------------
    # Logging
    # -------------------------
    log_lines = [
        f"Input rows: {len(df)}",
        f"Targets: {TARGET_COLS}",
        "",
        "=== CV ABLATION ===",
    ]
    for _, r in ablation_df.iterrows():
        log_lines.append(
            f"{r['feature_set']}: "
            f"mse_mean={r['mse_mean']:.6f}, "
            f"mae_mean={r['mae_mean']:.6f}, "
            f"spearman_mean={r['spearman_mean']:.6f}, "
            f"kendall_mean={r['kendall_mean']:.6f}"
        )

    log_lines.append("")
    log_lines.append("=== TEMPORAL ABLATION ===")
    for _, r in temporal_ablation_df.iterrows():
        log_lines.append(
            f"{r['feature_set']}: "
            f"mse={r['mse']:.6f}, "
            f"mae={r['mae']:.6f}, "
            f"spearman_row_avg={r['spearman_row_avg']:.6f}, "
            f"kendall_row_avg={r['kendall_row_avg']:.6f}"
        )

    log_lines.append("")
    log_lines.append("=== TOP FEATURE IMPORTANCE ===")
    for _, r in importance_df.head(12).iterrows():
        log_lines.append(f"{r['feature_group']}: importance={r['importance']:.6f}")

    write_log(log_lines)

    # -------------------------
    # Terminal summary
    # -------------------------
    print("=" * 72)
    print("05_ablation_and_importance_fast.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Ablation table: {ABLATION_OUT}")
    print(f"Importance table: {IMPORTANCE_TABLE_OUT}")
    print(f"Importance figure: {IMPORTANCE_FIG_OUT}")
    print(f"Log: {LOG_OUT}")
    print("-" * 72)
    print("CV ablation:")
    for _, r in ablation_df.iterrows():
        print(
            f"{r['feature_set']}: "
            f"mse_mean={r['mse_mean']:.6f}, "
            f"mae_mean={r['mae_mean']:.6f}, "
            f"spearman_mean={r['spearman_mean']:.6f}, "
            f"kendall_mean={r['kendall_mean']:.6f}"
        )
    print("-" * 72)
    print("Temporal ablation:")
    for _, r in temporal_ablation_df.iterrows():
        print(
            f"{r['feature_set']}: "
            f"mse={r['mse']:.6f}, "
            f"mae={r['mae']:.6f}, "
            f"spearman_row_avg={r['spearman_row_avg']:.6f}, "
            f"kendall_row_avg={r['kendall_row_avg']:.6f}"
        )
    print("-" * 72)
    print("Top 10 feature importance:")
    for _, r in importance_df.head(10).iterrows():
        print(f"{r['feature_group']}: importance={r['importance']:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()