# 01_data_audit.py

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
RAW_DIR = BASE_DIR / "data_raw"
OUT_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = RAW_DIR / "cordis_he_projects.csv"
EXCEL_OUT = OUT_DIR / "data_audit_summary.xlsx"
COLUMN_LIST_OUT = OUT_DIR / "column_list.csv"
YEAR_COVERAGE_OUT = OUT_DIR / "year_coverage.csv"
READ_LOG_OUT = OUT_DIR / "read_log.txt"

REQUIRED_COLUMNS = [
    "startDate",
    "endDate",
    "ecMaxContribution",
    "topics",
    "masterCall",
    "subCall",
    "fundingScheme",
    "legalBasis",
    "status",
    "frameworkProgramme",
    "objective",
]

# Alias map: canonical_name -> possible alternatives
ALIASES = {
    "startDate": ["startdate", "start_date", "projectStartDate"],
    "endDate": ["enddate", "end_date", "projectEndDate"],
    "ecMaxContribution": ["ecContribution", "ec_max_contribution", "ecmaxcontribution"],
    "topics": ["topic", "topicList", "topic_list"],
    "masterCall": ["call", "master_call", "callProgramme"],
    "subCall": ["callIdentifier", "callId", "sub_call", "call_id"],
    "fundingScheme": ["fundingscheme", "funding_scheme"],
    "legalBasis": ["legalbasis", "legal_basis"],
    "status": ["projectStatus"],
    "frameworkProgramme": ["frameworkprogramme", "framework_programme", "programme"],
    "objective": ["objectives", "projectObjective", "description"],
}


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(READ_LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def normalize_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())


def token_count(text: object) -> int:
    if pd.isna(text):
        return 0
    s = str(text).strip()
    if not s:
        return 0
    tokens = re.findall(r"\b\w+\b", s.lower(), flags=re.UNICODE)
    return len(tokens)


def char_count(text: object) -> int:
    if pd.isna(text):
        return 0
    return len(str(text).strip())


def safe_to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("EUR", "", regex=False)
        .str.strip()
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def classify_missing_severity(missing_pct: float) -> str:
    if missing_pct < 1:
        return "low"
    if missing_pct < 5:
        return "moderate"
    if missing_pct < 20:
        return "high"
    return "critical"


def detect_topics_delimiter(sample_values: pd.Series) -> str:
    candidates = [",", ";", "|"]
    counts = {c: 0 for c in candidates}
    json_like = 0

    for val in sample_values.dropna().astype(str).head(500):
        v = val.strip()
        if v.startswith("[") and v.endswith("]"):
            json_like += 1
        for c in candidates:
            if c in v:
                counts[c] += 1

    if json_like > max(counts.values(), default=0):
        return "json_like"
    if max(counts.values(), default=0) == 0:
        return "unknown_or_single_topic"
    return max(counts, key=counts.get)


def count_topics(value: object, detected_delimiter: str) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if not s:
        return 0

    if detected_delimiter == "json_like":
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return len([x for x in parsed if str(x).strip()])
        except Exception:
            return 0

    if detected_delimiter in [",", ";", "|"]:
        return len([x.strip() for x in s.split(detected_delimiter) if x.strip()])

    # fallback: single string counts as 1 if non-empty
    return 1


def read_csv_with_fallback(file_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    log_lines = []
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    delimiters = [",", ";", "\t"]

    best_df = None
    best_meta = None

    for enc in encodings:
        for delim in delimiters:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=enc,
                    sep=delim,
                    engine="python",
                    on_bad_lines="skip",
                )
                score = (df.shape[0], df.shape[1])
                log_lines.append(
                    f"TRY encoding={enc}, delimiter={repr(delim)} -> rows={df.shape[0]}, cols={df.shape[1]}"
                )
                if best_df is None or score > (best_df.shape[0], best_df.shape[1]):
                    best_df = df
                    best_meta = (enc, delim)
            except Exception as e:
                log_lines.append(f"FAIL encoding={enc}, delimiter={repr(delim)} -> {e}")

    if best_df is None:
        raise RuntimeError("CSV could not be read with fallback settings.")

    enc, delim = best_meta
    log_lines.append(f"SELECTED encoding={enc}, delimiter={repr(delim)}")
    return best_df, log_lines


def resolve_required_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    current_cols = list(df.columns)
    normalized_map = {normalize_col(c): c for c in current_cols}

    resolved = {}
    resolution_log = []

    for canonical in REQUIRED_COLUMNS:
        canonical_norm = normalize_col(canonical)

        # exact normalized match
        if canonical_norm in normalized_map:
            resolved[canonical] = normalized_map[canonical_norm]
            resolution_log.append(
                {"required": canonical, "matched_column": normalized_map[canonical_norm], "match_type": "exact"}
            )
            continue

        # alias match
        found = None
        for alias in ALIASES.get(canonical, []):
            alias_norm = normalize_col(alias)
            if alias_norm in normalized_map:
                found = normalized_map[alias_norm]
                break

        if found is not None:
            resolved[canonical] = found
            resolution_log.append(
                {"required": canonical, "matched_column": found, "match_type": "alias"}
            )
        else:
            resolution_log.append(
                {"required": canonical, "matched_column": "", "match_type": "missing"}
            )

    return resolved, resolution_log


def summarize_numeric(series: pd.Series, name: str) -> pd.DataFrame:
    s = safe_to_numeric(series)
    non_null = s.notna().sum()
    total = len(s)
    missing = total - non_null

    if non_null == 0:
        row = {
            "variable": name,
            "non_null": non_null,
            "missing_pct": round(missing / total * 100, 4) if total else np.nan,
            "min": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "n_zero": np.nan,
            "n_negative": np.nan,
        }
        return pd.DataFrame([row])

    row = {
        "variable": name,
        "non_null": non_null,
        "missing_pct": round(missing / total * 100, 4),
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
        "n_zero": int((s == 0).sum()),
        "n_negative": int((s < 0).sum()),
    }
    return pd.DataFrame([row])


def main() -> None:
    ensure_dirs()

    # -------------------------
    # Read CSV
    # -------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df, read_logs = read_csv_with_fallback(INPUT_FILE)

    duplicate_columns = int(df.columns.duplicated().sum())
    read_logs.append(f"Duplicate column names: {duplicate_columns}")
    write_log(read_logs)

    # -------------------------
    # Required columns resolution
    # -------------------------
    resolved_cols, resolution_log = resolve_required_columns(df)
    resolution_df = pd.DataFrame(resolution_log)

    missing_required = resolution_df[resolution_df["match_type"] == "missing"]["required"].tolist()
    alias_required = resolution_df[resolution_df["match_type"] == "alias"]["required"].tolist()

    # -------------------------
    # Column inventory
    # -------------------------
    column_inventory = pd.DataFrame({
        "column_name": df.columns,
        "dtype_raw": [str(t) for t in df.dtypes],
        "non_null_count": [df[c].notna().sum() for c in df.columns],
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_pct": [round(df[c].isna().mean() * 100, 4) for c in df.columns],
    }).sort_values(["missing_pct", "column_name"], ascending=[False, True])

    column_inventory.to_csv(COLUMN_LIST_OUT, index=False, encoding="utf-8-sig")

    # -------------------------
    # Dataset overview
    # -------------------------
    n_rows, n_cols = df.shape
    n_duplicate_rows = int(df.duplicated().sum())

    # try to detect likely project id column
    candidate_id_cols = [c for c in df.columns if normalize_col(c) in {"id", "projectid", "rcn"}]
    project_id_col = candidate_id_cols[0] if candidate_id_cols else None
    n_unique_projects = int(df[project_id_col].nunique()) if project_id_col else np.nan

    dataset_overview_rows = [
        {"metric": "n_rows", "value": n_rows},
        {"metric": "n_columns", "value": n_cols},
        {"metric": "n_duplicate_rows", "value": n_duplicate_rows},
        {"metric": "project_id_column", "value": project_id_col if project_id_col else ""},
        {"metric": "n_unique_projects", "value": n_unique_projects},
        {"metric": "n_missing_required_columns", "value": len(missing_required)},
        {"metric": "n_alias_resolutions", "value": len(alias_required)},
    ]

    # frameworkProgramme and status distributions
    for key in ["frameworkProgramme", "status"]:
        if key in resolved_cols:
            col = resolved_cols[key]
            top_counts = df[col].astype(str).value_counts(dropna=False).head(10)
            for cat, cnt in top_counts.items():
                dataset_overview_rows.append({"metric": f"top_{key}_{cat}", "value": int(cnt)})

    dataset_overview = pd.DataFrame(dataset_overview_rows)

    # -------------------------
    # Date audit
    # -------------------------
    date_audit_rows = []
    year_coverage = pd.DataFrame(columns=["year", "n_projects_started", "n_projects_ended"])

    duration_series = pd.Series(dtype=float)
    date_status = "NO-GO"

    if "startDate" in resolved_cols and "endDate" in resolved_cols:
        start_col = resolved_cols["startDate"]
        end_col = resolved_cols["endDate"]

        start_dt = pd.to_datetime(df[start_col], errors="coerce")
        end_dt = pd.to_datetime(df[end_col], errors="coerce")

        start_fail = int(start_dt.isna().sum())
        end_fail = int(end_dt.isna().sum())

        duration_days = (end_dt - start_dt).dt.days
        duration_months = duration_days / 30.44

        negative_duration = int((duration_days < 0).sum())
        zero_duration = int((duration_days == 0).sum())
        very_long_duration = int((duration_days > 3650).sum())  # >10 years
        valid_duration_count = int(duration_days.notna().sum())

        date_audit_rows.extend([
            {"metric": "start_parse_failures", "value": start_fail},
            {"metric": "end_parse_failures", "value": end_fail},
            {"metric": "valid_duration_count", "value": valid_duration_count},
            {"metric": "negative_duration_count", "value": negative_duration},
            {"metric": "zero_duration_count", "value": zero_duration},
            {"metric": "very_long_duration_count_gt_10y", "value": very_long_duration},
            {"metric": "min_start_date", "value": str(start_dt.min()) if start_dt.notna().any() else ""},
            {"metric": "max_end_date", "value": str(end_dt.max()) if end_dt.notna().any() else ""},
            {"metric": "min_start_year", "value": int(start_dt.dt.year.min()) if start_dt.notna().any() else np.nan},
            {"metric": "max_end_year", "value": int(end_dt.dt.year.max()) if end_dt.notna().any() else np.nan},
        ])

        start_year_counts = start_dt.dt.year.value_counts().sort_index()
        end_year_counts = end_dt.dt.year.value_counts().sort_index()
        all_years = sorted(set(start_year_counts.index.dropna().tolist()) | set(end_year_counts.index.dropna().tolist()))

        year_coverage = pd.DataFrame({
            "year": all_years,
            "n_projects_started": [int(start_year_counts.get(y, 0)) for y in all_years],
            "n_projects_ended": [int(end_year_counts.get(y, 0)) for y in all_years],
        })
        year_coverage.to_csv(YEAR_COVERAGE_OUT, index=False, encoding="utf-8-sig")

        start_fail_pct = start_fail / n_rows * 100 if n_rows else 100
        end_fail_pct = end_fail / n_rows * 100 if n_rows else 100

        if valid_duration_count == 0:
            date_status = "NO-GO"
        elif start_fail_pct > 5 or end_fail_pct > 5:
            date_status = "REVISION REQUIRED"
        elif negative_duration > 0:
            date_status = "REVISION REQUIRED"
        else:
            date_status = "GO"
    else:
        date_audit_rows.append({"metric": "date_status_note", "value": "Missing startDate/endDate"})
        year_coverage.to_csv(YEAR_COVERAGE_OUT, index=False, encoding="utf-8-sig")

    date_audit = pd.DataFrame(date_audit_rows)

    # -------------------------
    # Numeric audit
    # -------------------------
    numeric_audit_frames = []
    numeric_status = "NO-GO"

    if "ecMaxContribution" in resolved_cols:
        contrib_col = resolved_cols["ecMaxContribution"]
        num_df = summarize_numeric(df[contrib_col], "ecMaxContribution")
        numeric_audit_frames.append(num_df)

        parsed = safe_to_numeric(df[contrib_col])
        if parsed.notna().sum() == 0:
            numeric_status = "NO-GO"
        elif (parsed < 0).any():
            numeric_status = "REVISION REQUIRED"
        else:
            numeric_status = "GO"
    else:
        numeric_audit_frames.append(pd.DataFrame([{
            "variable": "ecMaxContribution",
            "non_null": 0,
            "missing_pct": 100.0,
            "min": np.nan, "p1": np.nan, "p5": np.nan, "p25": np.nan, "median": np.nan,
            "p75": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan,
            "mean": np.nan, "std": np.nan, "n_zero": np.nan, "n_negative": np.nan,
        }]))

    numeric_audit = pd.concat(numeric_audit_frames, ignore_index=True)

    # -------------------------
    # Text audit
    # -------------------------
    text_rows = []
    text_status = "NO-GO"
    objective_ok = False
    topics_ok = False
    detected_topics_delim = "unknown_or_single_topic"

    for key in ["objective", "topics", "masterCall", "subCall", "fundingScheme", "legalBasis", "status", "frameworkProgramme"]:
        if key not in resolved_cols:
            text_rows.append({
                "variable": key,
                "non_null_count": 0,
                "missing_pct": 100.0,
                "empty_string_count": np.nan,
                "avg_char_count": np.nan,
                "median_char_count": np.nan,
                "avg_token_count": np.nan,
                "median_token_count": np.nan,
                "short_text_count_lt_5_tokens": np.nan,
                "long_text_count_gt_500_tokens": np.nan,
                "note": "missing column",
            })
            continue

        col = resolved_cols[key]
        s = df[col]
        s_str = s.fillna("").astype(str)
        non_null_count = int(s.notna().sum())
        missing_pct = round(s.isna().mean() * 100, 4)
        empty_string_count = int((s_str.str.strip() == "").sum())
        char_counts = s_str.apply(char_count)
        tok_counts = s_str.apply(token_count)

        note = ""
        if key == "topics":
            detected_topics_delim = detect_topics_delimiter(s)
            topic_counts = s.apply(lambda x: count_topics(x, detected_topics_delim))
            note = f"detected_delimiter={detected_topics_delim}; avg_topic_count={round(topic_counts.mean(), 4)}"
            topics_ok = non_null_count > 0 and topic_counts.mean() > 0

        if key == "objective":
            objective_ok = non_null_count > 0 and tok_counts.mean() > 0

        text_rows.append({
            "variable": key,
            "non_null_count": non_null_count,
            "missing_pct": missing_pct,
            "empty_string_count": empty_string_count,
            "avg_char_count": round(char_counts.mean(), 4),
            "median_char_count": round(char_counts.median(), 4),
            "avg_token_count": round(tok_counts.mean(), 4),
            "median_token_count": round(tok_counts.median(), 4),
            "short_text_count_lt_5_tokens": int((tok_counts < 5).sum()),
            "long_text_count_gt_500_tokens": int((tok_counts > 500).sum()),
            "note": note,
        })

    text_audit = pd.DataFrame(text_rows)

    if objective_ok and topics_ok:
        text_status = "GO"
    elif objective_ok or topics_ok:
        text_status = "REVISION REQUIRED"
    else:
        text_status = "NO-GO"

    # -------------------------
    # Categorical audit
    # -------------------------
    categorical_rows = []
    categorical_status = "NO-GO"

    cat_ok_count = 0
    for key in ["fundingScheme", "masterCall", "subCall", "legalBasis", "status", "frameworkProgramme"]:
        if key not in resolved_cols:
            categorical_rows.append({
                "variable": key,
                "non_null_count": 0,
                "missing_pct": 100.0,
                "n_unique": 0,
                "n_rare_lt_5": 0,
                "n_singletons": 0,
                "top_1_category": "",
                "top_1_count": 0,
                "top_2_category": "",
                "top_2_count": 0,
                "top_3_category": "",
                "top_3_count": 0,
                "note": "missing column",
            })
            continue

        col = resolved_cols[key]
        s = df[col].fillna("<<MISSING>>").astype(str).str.strip()
        vc = s.value_counts(dropna=False)

        rare_lt_5 = int((vc < 5).sum())
        singletons = int((vc == 1).sum())

        tops = vc.head(3).to_dict()
        top_items = list(tops.items()) + [("", 0)] * (3 - len(tops))

        categorical_rows.append({
            "variable": key,
            "non_null_count": int(df[col].notna().sum()),
            "missing_pct": round(df[col].isna().mean() * 100, 4),
            "n_unique": int(s.nunique()),
            "n_rare_lt_5": rare_lt_5,
            "n_singletons": singletons,
            "top_1_category": top_items[0][0],
            "top_1_count": int(top_items[0][1]),
            "top_2_category": top_items[1][0],
            "top_2_count": int(top_items[1][1]),
            "top_3_category": top_items[2][0],
            "top_3_count": int(top_items[2][1]),
            "note": "",
        })

        if key in {"fundingScheme", "subCall"} and s.nunique() > 1:
            cat_ok_count += 1

    categorical_audit = pd.DataFrame(categorical_rows)

    if cat_ok_count == 2:
        categorical_status = "GO"
    elif cat_ok_count == 1:
        categorical_status = "REVISION REQUIRED"
    else:
        categorical_status = "NO-GO"

    # -------------------------
    # Missingness risk
    # -------------------------
    missing_rows = []
    for req in REQUIRED_COLUMNS:
        if req in resolved_cols:
            col = resolved_cols[req]
            missing_pct = round(df[col].isna().mean() * 100, 4)
            severity = classify_missing_severity(missing_pct)
            note = "alias matched" if req in alias_required else "exact matched"
        else:
            missing_pct = 100.0
            severity = "critical"
            note = "missing required column"

        missing_rows.append({
            "variable": req,
            "missing_pct": missing_pct,
            "severity": severity,
            "note": note,
        })

    missingness_risk = pd.DataFrame(missing_rows)

    # -------------------------
    # Decision summary
    # -------------------------
    required_columns_status = "GO" if len(missing_required) == 0 else ("REVISION REQUIRED" if len(missing_required) <= 2 else "NO-GO")
    readability_status = "GO"

    decision_rows = [
        {"component": "readability_status", "status": readability_status, "note": "CSV readable with fallback parser"},
        {"component": "required_columns_status", "status": required_columns_status, "note": f"missing={missing_required}; alias={alias_required}"},
        {"component": "date_status", "status": date_status, "note": "startDate/endDate audit completed"},
        {"component": "numeric_status", "status": numeric_status, "note": "ecMaxContribution audit completed"},
        {"component": "text_status", "status": text_status, "note": f"topics_delimiter={detected_topics_delim}"},
        {"component": "categorical_status", "status": categorical_status, "note": "fundingScheme/subCall suitability checked"},
    ]

    component_statuses = [required_columns_status, date_status, numeric_status, text_status, categorical_status]

    if "NO-GO" in component_statuses:
        overall_status = "NO-GO"
    elif "REVISION REQUIRED" in component_statuses:
        overall_status = "REVISION REQUIRED"
    else:
        overall_status = "GO"

    decision_rows.append({"component": "overall_status", "status": overall_status, "note": "aggregate decision"})
    decision_summary = pd.DataFrame(decision_rows)

    # -------------------------
    # Excel output
    # -------------------------
    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        dataset_overview.to_excel(writer, sheet_name="dataset_overview", index=False)
        resolution_df.to_excel(writer, sheet_name="required_columns_map", index=False)
        date_audit.to_excel(writer, sheet_name="date_audit", index=False)
        numeric_audit.to_excel(writer, sheet_name="numeric_audit", index=False)
        text_audit.to_excel(writer, sheet_name="text_audit", index=False)
        categorical_audit.to_excel(writer, sheet_name="categorical_audit", index=False)
        missingness_risk.to_excel(writer, sheet_name="missingness_risk", index=False)
        decision_summary.to_excel(writer, sheet_name="decision_summary", index=False)

    # -------------------------
    # Terminal summary
    # -------------------------
    print("=" * 70)
    print("01_data_audit.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Excel output: {EXCEL_OUT}")
    print(f"Column list: {COLUMN_LIST_OUT}")
    print(f"Year coverage: {YEAR_COVERAGE_OUT}")
    print(f"Read log: {READ_LOG_OUT}")
    print("-" * 70)
    print(f"Rows: {n_rows}")
    print(f"Columns: {n_cols}")
    print(f"Missing required columns: {missing_required}")
    print(f"Alias-resolved columns: {alias_required}")
    print(f"Topics delimiter detected: {detected_topics_delim}")
    print("-" * 70)
    print(f"required_columns_status: {required_columns_status}")
    print(f"date_status: {date_status}")
    print(f"numeric_status: {numeric_status}")
    print(f"text_status: {text_status}")
    print(f"categorical_status: {categorical_status}")
    print(f"OVERALL STATUS: {overall_status}")
    print("=" * 70)


if __name__ == "__main__":
    main()