# 02_proxy_construction.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent / "dss_revision"
RAW_DIR = BASE_DIR / "data_raw"
PROCESSED_DIR = BASE_DIR / "data_processed"
DIAG_DIR = BASE_DIR / "outputs" / "diagnostics"

INPUT_FILE = RAW_DIR / "cordis_he_projects.csv"
RAW_OUT = PROCESSED_DIR / "proxy_dataset_raw.csv"
PROFILE_OUT = PROCESSED_DIR / "proxy_dataset_profiles.csv"
LOG_OUT = DIAG_DIR / "proxy_construction_log.txt"
MISSING_SUMMARY_OUT = DIAG_DIR / "proxy_missing_summary.xlsx"

EPS = 1e-12


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_log(lines: List[str]) -> None:
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def normalize_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())


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


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    aliases = {
        "id": ["projectid", "rcn", "id"],
        "acronym": ["acronym"],
        "title": ["title"],
        "startDate": ["startdate", "start_date", "projectStartDate"],
        "endDate": ["enddate", "end_date", "projectEndDate"],
        "ecMaxContribution": ["ecContribution", "ec_max_contribution", "ecmaxcontribution"],
        "keywords": ["keywords", "keyword"],
        "masterCall": ["call", "master_call", "callProgramme"],
        "subCall": ["callIdentifier", "callId", "sub_call", "call_id"],
        "fundingScheme": ["fundingscheme", "funding_scheme"],
        "legalBasis": ["legalbasis", "legal_basis"],
        "status": ["projectStatus", "status"],
        "frameworkProgramme": ["frameworkprogramme", "framework_programme", "programme"],
        "objective": ["objectives", "projectObjective", "description", "objective"],
        "ecSignatureDate": ["ecsignaturedate", "ec_signature_date"],
        "grantDoi": ["grantdoi", "grant_doi"],
        "contentUpdateDate": ["contentupdatedate", "content_update_date"],
    }

    normalized_map = {normalize_col(c): c for c in df.columns}
    resolved = {}

    for canonical, alias_list in aliases.items():
        candidates = [canonical] + alias_list
        found = None
        for cand in candidates:
            norm = normalize_col(cand)
            if norm in normalized_map:
                found = normalized_map[norm]
                break
        if found is not None:
            resolved[canonical] = found

    return resolved


def clean_text_basic(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def token_list(text: object) -> List[str]:
    s = clean_text_basic(text)
    if not s:
        return []
    return re.findall(r"\b\w+\b", s, flags=re.UNICODE)


def token_count(text: object) -> int:
    return len(token_list(text))


def unique_token_count(text: object) -> int:
    return len(set(token_list(text)))


def lexical_diversity(text: object) -> float:
    tokens = token_list(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def char_count(text: object) -> int:
    return len(clean_text_basic(text))


def safe_numeric_eu(series: pd.Series) -> pd.Series:
    """
    Handles mixed numeric strings such as:
    3608915,55
    1,500,000
    150000
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()

    # remove spaces and currency markers
    s = (
        s.str.replace("€", "", regex=False)
         .str.replace("EUR", "", regex=False)
         .str.replace(" ", "", regex=False)
    )

    def convert_one(v: str):
        if v in {"", "nan", "None", "NaN"}:
            return np.nan

        # both comma and dot present
        if "," in v and "." in v:
            last_comma = v.rfind(",")
            last_dot = v.rfind(".")
            if last_comma > last_dot:
                # European style: 1.234.567,89
                v2 = v.replace(".", "").replace(",", ".")
            else:
                # US style: 1,234,567.89
                v2 = v.replace(",", "")
            try:
                return float(v2)
            except Exception:
                return np.nan

        # only comma present
        if "," in v:
            # single comma with 1-2 digits after -> decimal comma
            if v.count(",") == 1 and len(v.split(",")[-1]) in {1, 2}:
                v2 = v.replace(",", ".")
            else:
                # likely thousands separators
                v2 = v.replace(",", "")
            try:
                return float(v2)
            except Exception:
                return np.nan

        # only dot present or plain integer
        try:
            return float(v)
        except Exception:
            return np.nan

    return s.apply(convert_one)


def detect_keyword_delimiter(sample_values: pd.Series) -> str:
    candidates = [",", ";", "|"]
    counts = {c: 0 for c in candidates}
    json_like = 0

    for val in sample_values.dropna().astype(str).head(1000):
        v = val.strip()
        if not v:
            continue
        if v.startswith("[") and v.endswith("]"):
            json_like += 1
        for c in candidates:
            if c in v:
                counts[c] += 1

    if json_like > max(counts.values(), default=0):
        return "json_like"
    if max(counts.values(), default=0) == 0:
        return "unknown_or_single_keyword"
    return max(counts, key=counts.get)


def count_keywords(value: object, delimiter: str) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if not s:
        return 0

    if delimiter == "json_like":
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return len([x for x in parsed if str(x).strip()])
        except Exception:
            return 0

    if delimiter in {",", ";", "|"}:
        return len([x.strip() for x in s.split(delimiter) if x.strip()])

    return 1 if s else 0


def choose_call_id(subcall: object, mastercall: object) -> Tuple[str, str]:
    sub = "" if pd.isna(subcall) else str(subcall).strip()
    master = "" if pd.isna(mastercall) else str(mastercall).strip()

    if sub:
        return sub, "subCall"
    if master:
        return master, "masterCall"
    return "", "missing"


def scale_by_max(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    max_val = s.max()
    if pd.isna(max_val) or max_val <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return s / max_val


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df, read_logs = read_csv_with_fallback(INPUT_FILE)
    resolved = resolve_columns(df)

    required_for_this_script = [
        "startDate", "endDate", "ecMaxContribution", "keywords",
        "masterCall", "subCall", "fundingScheme", "objective"
    ]
    missing = [c for c in required_for_this_script if c not in resolved]
    if missing:
        raise RuntimeError(f"Missing required columns for proxy construction: {missing}")

    log_lines = []
    log_lines.extend(read_logs)
    log_lines.append("")
    log_lines.append("=== COLUMN RESOLUTION ===")
    for k, v in resolved.items():
        log_lines.append(f"{k} -> {v}")

    # -------------------------
    # Build working dataframe
    # -------------------------
    out = pd.DataFrame(index=df.index)

    # carry-through columns if available
    carry_cols = [
        "id", "acronym", "title", "status", "frameworkProgramme",
        "startDate", "endDate", "ecMaxContribution", "fundingScheme",
        "masterCall", "subCall", "legalBasis", "objective", "keywords",
        "ecSignatureDate", "grantDoi", "contentUpdateDate"
    ]
    for col in carry_cols:
        if col in resolved:
            out[col] = df[resolved[col]]
        else:
            out[col] = np.nan

    # -------------------------
    # Dates and duration
    # -------------------------
    start_dt = pd.to_datetime(out["startDate"], errors="coerce", dayfirst=True)
    end_dt = pd.to_datetime(out["endDate"], errors="coerce", dayfirst=True)

    out["start_year"] = start_dt.dt.year
    out["start_month"] = start_dt.dt.month
    out["ecSignature_year"] = pd.to_datetime(out["ecSignatureDate"], errors="coerce", dayfirst=True).dt.year

    duration_days = (end_dt - start_dt).dt.days
    out["duration_days"] = duration_days
    out["duration_months"] = duration_days / 30.44

    # -------------------------
    # Numeric cleaning
    # -------------------------
    out["ecMaxContribution_num"] = safe_numeric_eu(out["ecMaxContribution"])

    # -------------------------
    # Text features
    # -------------------------
    keyword_delim = detect_keyword_delimiter(out["keywords"])
    out["keyword_count"] = out["keywords"].apply(lambda x: count_keywords(x, keyword_delim))

    out["objective_clean"] = out["objective"].apply(clean_text_basic)
    out["objective_token_count"] = out["objective"].apply(token_count)
    out["objective_unique_token_count"] = out["objective"].apply(unique_token_count)
    out["objective_lexical_diversity"] = out["objective"].apply(lexical_diversity)
    out["objective_char_count"] = out["objective"].apply(char_count)

    out["title_token_count"] = out["title"].apply(token_count)
    out["has_grant_doi"] = out["grantDoi"].notna().astype(int)

    # -------------------------
    # Call ID
    # -------------------------
    call_id_values = []
    call_id_source = []
    for sub, master in zip(out["subCall"], out["masterCall"]):
        cid, src = choose_call_id(sub, master)
        call_id_values.append(cid)
        call_id_source.append(src)

    out["call_id"] = call_id_values
    out["call_id_source"] = call_id_source

    # -------------------------
    # Frequency-based components
    # -------------------------
    call_freq = out["call_id"].replace("", np.nan).value_counts(dropna=True)
    scheme_freq = out["fundingScheme"].astype(str).replace("nan", np.nan).replace("", np.nan).value_counts(dropna=True)

    out["call_frequency"] = out["call_id"].map(call_freq).fillna(0)
    out["fundingScheme_frequency"] = out["fundingScheme"].map(scheme_freq).fillna(0)

    # -------------------------
    # Raw proxies
    # -------------------------
    out["C1_raw"] = out["ecMaxContribution_num"]

    valid_duration = out["duration_months"] > 0
    out["C2_raw"] = np.where(
        valid_duration,
        out["ecMaxContribution_num"] / out["duration_months"],
        np.nan
    )

    out["C3_raw"] = out["duration_months"]
    out["C4_raw"] = out["keyword_count"]
    out["C5_raw"] = np.where(out["call_frequency"] > 0, 1 / out["call_frequency"], np.nan)
    out["C6_raw"] = np.where(out["fundingScheme_frequency"] > 0, 1 / out["fundingScheme_frequency"], np.nan)
    out["C7_raw"] = out["objective_token_count"]
    out["C8_raw"] = out["objective_lexical_diversity"]

    # -------------------------
    # Invalid row flag
    # -------------------------
    invalid_conditions = {
        "invalid_ecMaxContribution": out["ecMaxContribution_num"].isna(),
        "invalid_duration": out["duration_months"].isna() | (out["duration_months"] <= 0),
        "invalid_objective": out["objective_clean"].eq("") | (out["objective_token_count"] == 0),
        "invalid_fundingScheme": out["fundingScheme"].isna() | out["fundingScheme"].astype(str).str.strip().eq(""),
        "invalid_call": out["call_id"].astype(str).str.strip().eq(""),
    }

    for name, cond in invalid_conditions.items():
        out[name] = cond.astype(int)

    out["invalid_for_profile"] = (
        invalid_conditions["invalid_ecMaxContribution"]
        | invalid_conditions["invalid_duration"]
        | invalid_conditions["invalid_objective"]
        | invalid_conditions["invalid_fundingScheme"]
        | invalid_conditions["invalid_call"]
    ).astype(int)

    # -------------------------
    # Save raw dataset
    # -------------------------
    raw_cols = [
        "id", "acronym", "title", "status", "frameworkProgramme",
        "startDate", "endDate", "start_year", "start_month",
        "ecSignature_year", "duration_days", "duration_months",
        "ecMaxContribution", "ecMaxContribution_num",
        "fundingScheme", "masterCall", "subCall", "call_id", "call_id_source",
        "legalBasis", "objective", "objective_clean", "keywords",
        "objective_char_count", "objective_token_count",
        "objective_unique_token_count", "objective_lexical_diversity",
        "title_token_count", "keyword_count", "has_grant_doi",
        "call_frequency", "fundingScheme_frequency",
        "C1_raw", "C2_raw", "C3_raw", "C4_raw", "C5_raw", "C6_raw", "C7_raw", "C8_raw",
        "invalid_ecMaxContribution", "invalid_duration", "invalid_objective",
        "invalid_fundingScheme", "invalid_call", "invalid_for_profile"
    ]
    raw_cols = [c for c in raw_cols if c in out.columns]
    out[raw_cols].to_csv(RAW_OUT, index=False, encoding="utf-8-sig")

    # -------------------------
    # Build profiles dataset
    # -------------------------
    profile_df = out[out["invalid_for_profile"] == 0].copy()

    raw_proxy_cols = [f"C{i}_raw" for i in range(1, 9)]

    # fill any residual NaN in valid rows with 0 only for scaling stage
    for col in raw_proxy_cols:
        profile_df[col] = pd.to_numeric(profile_df[col], errors="coerce").fillna(0.0)

    for i in range(1, 9):
        raw_col = f"C{i}_raw"
        scaled_col = f"C{i}_scaled"
        profile_df[scaled_col] = scale_by_max(profile_df[raw_col])

    scaled_cols = [f"C{i}_scaled" for i in range(1, 9)]
    row_sum = profile_df[scaled_cols].sum(axis=1)

    # avoid division by zero
    zero_sum_mask = row_sum <= EPS
    for i in range(1, 9):
        scaled_col = f"C{i}_scaled"
        w_col = f"W{i}"
        profile_df[w_col] = np.where(~zero_sum_mask, profile_df[scaled_col] / row_sum, np.nan)

    weight_cols = [f"W{i}" for i in range(1, 9)]
    profile_df["profile_sum_check"] = profile_df[weight_cols].sum(axis=1)

    profile_cols = [
        "id", "acronym", "title", "status", "frameworkProgramme",
        "startDate", "endDate", "start_year", "start_month",
        "ecSignature_year", "duration_months", "ecMaxContribution_num",
        "fundingScheme", "masterCall", "subCall", "call_id", "call_id_source",
        "legalBasis", "keyword_count", "objective_token_count",
        "objective_unique_token_count", "objective_lexical_diversity",
    ] + raw_proxy_cols + scaled_cols + weight_cols + ["profile_sum_check"]

    profile_cols = [c for c in profile_cols if c in profile_df.columns]
    profile_df[profile_cols].to_csv(PROFILE_OUT, index=False, encoding="utf-8-sig")

    # -------------------------
    # Missing summary
    # -------------------------
    proxy_missing_rows = []
    for col in raw_proxy_cols:
        proxy_missing_rows.append({
            "proxy": col,
            "missing_count_raw_dataset": int(out[col].isna().sum()),
            "missing_pct_raw_dataset": round(out[col].isna().mean() * 100, 4),
            "zero_count_profile_dataset": int((profile_df[col] == 0).sum()),
            "mean_profile_dataset": float(profile_df[col].mean()) if len(profile_df) else np.nan,
            "std_profile_dataset": float(profile_df[col].std()) if len(profile_df) else np.nan,
        })

    invalid_summary_rows = []
    for name in [
        "invalid_ecMaxContribution",
        "invalid_duration",
        "invalid_objective",
        "invalid_fundingScheme",
        "invalid_call",
        "invalid_for_profile",
    ]:
        invalid_summary_rows.append({
            "flag": name,
            "count": int(out[name].sum()),
            "pct": round(out[name].mean() * 100, 4),
        })

    normalization_summary = pd.DataFrame([{
        "n_rows_raw": len(out),
        "n_rows_valid_profiles": len(profile_df),
        "n_rows_invalid_profiles": int(out["invalid_for_profile"].sum()),
        "keyword_delimiter_detected": keyword_delim,
        "call_id_from_subCall": int((out["call_id_source"] == "subCall").sum()),
        "call_id_from_masterCall": int((out["call_id_source"] == "masterCall").sum()),
        "call_id_missing": int((out["call_id_source"] == "missing").sum()),
        "profile_sum_fail_count_abs_gt_1e_6": int((profile_df["profile_sum_check"].sub(1).abs() > 1e-6).sum()) if len(profile_df) else 0,
    }])

    with pd.ExcelWriter(MISSING_SUMMARY_OUT, engine="openpyxl") as writer:
        pd.DataFrame(proxy_missing_rows).to_excel(writer, sheet_name="proxy_missing", index=False)
        pd.DataFrame(invalid_summary_rows).to_excel(writer, sheet_name="invalid_flags", index=False)
        normalization_summary.to_excel(writer, sheet_name="normalization_summary", index=False)

    # -------------------------
    # Logging
    # -------------------------
    log_lines.append("")
    log_lines.append("=== PROXY CONSTRUCTION SUMMARY ===")
    log_lines.append(f"Input rows: {len(out)}")
    log_lines.append(f"Valid profile rows: {len(profile_df)}")
    log_lines.append(f"Invalid profile rows: {int(out['invalid_for_profile'].sum())}")
    log_lines.append(f"Keyword delimiter detected: {keyword_delim}")
    log_lines.append(f"call_id from subCall: {int((out['call_id_source'] == 'subCall').sum())}")
    log_lines.append(f"call_id from masterCall: {int((out['call_id_source'] == 'masterCall').sum())}")
    log_lines.append(f"call_id missing: {int((out['call_id_source'] == 'missing').sum())}")
    log_lines.append("")

    log_lines.append("=== RAW PROXY MISSING COUNTS ===")
    for row in proxy_missing_rows:
        log_lines.append(
            f"{row['proxy']}: missing={row['missing_count_raw_dataset']} "
            f"({row['missing_pct_raw_dataset']}%)"
        )

    log_lines.append("")
    log_lines.append("=== INVALID FLAG COUNTS ===")
    for row in invalid_summary_rows:
        log_lines.append(f"{row['flag']}: {row['count']} ({row['pct']}%)")

    fail_count = int((profile_df["profile_sum_check"].sub(1).abs() > 1e-6).sum()) if len(profile_df) else 0
    log_lines.append("")
    log_lines.append(f"profile_sum_fail_count_abs_gt_1e_6: {fail_count}")

    write_log(log_lines)

    # -------------------------
    # Terminal summary
    # -------------------------
    print("=" * 72)
    print("02_proxy_construction.py completed")
    print(f"Input: {INPUT_FILE}")
    print(f"Raw output: {RAW_OUT}")
    print(f"Profile output: {PROFILE_OUT}")
    print(f"Diagnostics: {MISSING_SUMMARY_OUT}")
    print(f"Log: {LOG_OUT}")
    print("-" * 72)
    print(f"Input rows: {len(out)}")
    print(f"Valid profile rows: {len(profile_df)}")
    print(f"Invalid profile rows: {int(out['invalid_for_profile'].sum())}")
    print(f"Keyword delimiter detected: {keyword_delim}")
    print(f"call_id from subCall: {int((out['call_id_source'] == 'subCall').sum())}")
    print(f"call_id from masterCall: {int((out['call_id_source'] == 'masterCall').sum())}")
    print(f"profile_sum_fail_count_abs_gt_1e_6: {fail_count}")
    print("=" * 72)


if __name__ == "__main__":
    main()