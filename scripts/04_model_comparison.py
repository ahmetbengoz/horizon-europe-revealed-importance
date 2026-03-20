# 04_model_comparison.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
PROCESSED_DIR = BASE_DIR / "data_processed"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = PROCESSED_DIR / "proxy_dataset_profiles_revised.csv"
MODEL_TABLE_OUT = TABLES_DIR / "Table_model_comparison.xlsx"
TEMPORAL_TABLE_OUT = TABLES_DIR / "Table_temporal_holdout.xlsx"
LOG_OUT = DIAG_DIR / "model_comparison_log.txt"

TARGET_COLS = [f"W{i}_rev" for i in range(1, 9)]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "fundingScheme",
    "subCall",
    "legalBasis",
    "status",
    "frameworkProgramme",
]

N_SPLITS = 3
RANDOM_STATE = 42


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def safe_rank(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr)
    return s.rank(method="average").to_numpy()


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

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    return preprocessor


def build_models() -> Dict[str, object]:
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=120,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return models


def make_pipeline(model, numeric_features, categorical_features) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(df["start_year"].dropna().unique().tolist())
    if len(years) < 2:
        raise RuntimeError("Temporal holdout requires at least 2 distinct start_year values.")

    max_year = max(years)
    train_df = df[df["start_year"] < max_year].copy()
    test_df = df[df["start_year"] == max_year].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Temporal split failed: empty train or test set.")

    return train_df, test_df


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_cols = TARGET_COLS + ["start_year"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if not numeric_features and not categorical_features:
        raise RuntimeError("No usable features found.")

    X = df[numeric_features + categorical_features].copy()
    y = df[TARGET_COLS].copy()

    log_lines = []
    log_lines.append(f"Input rows: {len(df)}")
    log_lines.append(f"Numeric features: {numeric_features}")
    log_lines.append(f"Categorical features: {categorical_features}")
    log_lines.append(f"Targets: {TARGET_COLS}")
    log_lines.append("")

    models = build_models()

    # -------------------------
    # Random CV
    # -------------------------
    cv_rows = []
    fold_rows = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in models.items():
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            pipe = make_pipeline(clone(model), numeric_features, categorical_features)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            metrics = evaluate_predictions(y_test.to_numpy(), np.asarray(y_pred))
            metrics["model"] = model_name
            metrics["fold"] = fold_idx
            fold_rows.append(metrics)
            fold_metrics.append(metrics)

        summary = {
            "model": model_name,
            "cv_folds": N_SPLITS,
            "mse_mean": float(np.mean([m["mse"] for m in fold_metrics])),
            "mse_std": float(np.std([m["mse"] for m in fold_metrics])),
            "mae_mean": float(np.mean([m["mae"] for m in fold_metrics])),
            "mae_std": float(np.std([m["mae"] for m in fold_metrics])),
            "spearman_mean": float(np.nanmean([m["spearman_row_avg"] for m in fold_metrics])),
            "spearman_std": float(np.nanstd([m["spearman_row_avg"] for m in fold_metrics])),
            "kendall_mean": float(np.nanmean([m["kendall_row_avg"] for m in fold_metrics])),
            "kendall_std": float(np.nanstd([m["kendall_row_avg"] for m in fold_metrics])),
        }
        cv_rows.append(summary)

    cv_summary_df = pd.DataFrame(cv_rows).sort_values(
        by=["mse_mean", "mae_mean"], ascending=[True, True]
    )
    cv_folds_df = pd.DataFrame(fold_rows)

    # -------------------------
    # Temporal holdout
    # -------------------------
    train_df, test_df = temporal_split(df)

    X_train = train_df[numeric_features + categorical_features].copy()
    X_test = test_df[numeric_features + categorical_features].copy()
    y_train = train_df[TARGET_COLS].copy()
    y_test = test_df[TARGET_COLS].copy()

    temporal_rows = []
    for model_name, model in models.items():
        pipe = make_pipeline(clone(model), numeric_features, categorical_features)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate_predictions(y_test.to_numpy(), np.asarray(y_pred))
        metrics["model"] = model_name
        metrics["train_year_min"] = int(train_df["start_year"].min())
        metrics["train_year_max"] = int(train_df["start_year"].max())
        metrics["test_year"] = int(test_df["start_year"].iloc[0])
        metrics["train_n"] = len(train_df)
        metrics["test_n"] = len(test_df)
        temporal_rows.append(metrics)

    temporal_df = pd.DataFrame(temporal_rows).sort_values(
        by=["mse", "mae"], ascending=[True, True]
    )

    # -------------------------
    # Save outputs
    # -------------------------
    with pd.ExcelWriter(MODEL_TABLE_OUT, engine="openpyxl") as writer:
        cv_summary_df.to_excel(writer, sheet_name="cv_summary", index=False)
        cv_folds_df.to_excel(writer, sheet_name="cv_fold_details", index=False)

    with pd.ExcelWriter(TEMPORAL_TABLE_OUT, engine="openpyxl") as writer:
        temporal_df.to_excel(writer, sheet_name="temporal_holdout", index=False)

    # -------------------------
    # Log
    # -------------------------
    log_lines.append("=== CV SUMMARY ===")
    for _, r in cv_summary_df.iterrows():
        log_lines.append(
            f"{r['model']}: "
            f"mse_mean={r['mse_mean']:.6f}, "
            f"mae_mean={r['mae_mean']:.6f}, "
            f"spearman_mean={r['spearman_mean']:.6f}, "
            f"kendall_mean={r['kendall_mean']:.6f}"
        )

    log_lines.append("")
    log_lines.append("=== TEMPORAL HOLDOUT ===")
    log_lines.append(
        f"train_year_range={int(train_df['start_year'].min())}-{int(train_df['start_year'].max())}, "
        f"test_year={int(test_df['start_year'].iloc[0])}, "
        f"train_n={len(train_df)}, test_n={len(test_df)}"
    )
    for _, r in temporal_df.iterrows():
        log_lines.append(
            f"{r['model']}: "
            f"mse={r['mse']:.6f}, "
            f"mae={r['mae']:.6f}, "
            f"spearman_row_avg={r['spearman_row_avg']:.6f}, "
            f"kendall_row_avg={r['kendall_row_avg']:.6f}"
        )

    write_log(log_lines)

    # -------------------------
    # Terminal summary
    # -------------------------
    print("=" * 72)
    print("04_model_comparison.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Model comparison table: {MODEL_TABLE_OUT}")
    print(f"Temporal holdout table: {TEMPORAL_TABLE_OUT}")
    print(f"Log: {LOG_OUT}")
    print("-" * 72)
    print("CV summary:")
    for _, r in cv_summary_df.iterrows():
        print(
            f"{r['model']}: "
            f"mse_mean={r['mse_mean']:.6f}, "
            f"mae_mean={r['mae_mean']:.6f}, "
            f"spearman_mean={r['spearman_mean']:.6f}, "
            f"kendall_mean={r['kendall_mean']:.6f}"
        )
    print("-" * 72)
    print(
        f"Temporal split: train_year_range={int(train_df['start_year'].min())}-"
        f"{int(train_df['start_year'].max())}, "
        f"test_year={int(test_df['start_year'].iloc[0])}, "
        f"train_n={len(train_df)}, test_n={len(test_df)}"
    )
    for _, r in temporal_df.iterrows():
        print(
            f"{r['model']}: "
            f"mse={r['mse']:.6f}, "
            f"mae={r['mae']:.6f}, "
            f"spearman_row_avg={r['spearman_row_avg']:.6f}, "
            f"kendall_row_avg={r['kendall_row_avg']:.6f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()